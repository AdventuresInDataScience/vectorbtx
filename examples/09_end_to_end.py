"""09 — End to end.

The whole runtime flow in one file: describe a universe, load rates, prepare data, build
targets, resolve arguments, simulate, analyse.

Run: python examples/09_end_to_end.py
"""

import numpy as np
import pandas as pd

from _bootstrap import HAS_VBT, header, x

Instrument = x.Instrument

INDEX = pd.date_range("2023-01-02", periods=520, freq="B")
rng = np.random.default_rng(11)


# ==================================================================================
header("STEP 1 — Describe the universe once")
# ==================================================================================

universe = x.load_universe({
    # A UK index spread bet: no contract size, £1/pt minimum, cost in the spread.
    "UKX": Instrument(
        mode="spreadbet", point_size=1.0, asset_class="index",
        spread_points=4.0, leverage=10.0,
    ),
    # A US equity held by a GBP investor: prices convert, but the $ fee does not.
    "MSFT": Instrument(
        mode="equity", point_size=0.01,
        quote_currency="USD", cost_currency="USD",
        commission_pct=0.001, commission_min=1.0, spread_points=2.0,
    ),
    # An FX spot position: carries regardless of leverage. base_currency is the other
    # side of the pair, and is meaningful only for fx_spot.
    "EURUSD": Instrument(
        mode="fx_spot", point_size=0.0001, contract_size=100_000, asset_class="fx",
        quote_currency="USD", base_currency="EUR",
    ),
})

print(universe.to_frame()[
    ["mode", "asset_class", "point_size", "contract_size", "increment",
     "min_size", "leverage", "price_field"]
].to_string())
print()
print("Three different products, three different unit systems, one description.")


# ==================================================================================
header("STEP 2 — Load rates (needed for financing and carry)")
# ==================================================================================

rates = x.static_rate_table(["GBP", "USD", "EUR"], INDEX,
                            {"GBP": 0.0525, "USD": 0.0533, "EUR": 0.0400})
print(rates.head(3).to_string())
print()
print("In production this would be vbt.x.rate_table([...], index) pulling SONIA, SOFR")
print("and ESTR from FRED, cached and refreshed on load. The static path exists so the")
print("library works without an API key.")


# ==================================================================================
header("STEP 3 — Prepare data, converting to base currency")
# ==================================================================================

prices = pd.DataFrame({
    "UKX": 7500 * np.exp(np.cumsum(rng.normal(0.0002, 0.009, len(INDEX)))),
    "MSFT": 240 * np.exp(np.cumsum(rng.normal(0.0005, 0.016, len(INDEX)))),
    "EURUSD": 1.09 * np.exp(np.cumsum(rng.normal(0.0000, 0.004, len(INDEX)))),
}, index=INDEX)

usd_in_gbp = pd.Series(
    0.79 * np.exp(np.cumsum(rng.normal(0.0, 0.003, len(INDEX)))), index=INDEX
)
fx_table = x.FXTable.from_canonical(
    pd.DataFrame({"USD": usd_in_gbp, "GBP": 1.0, "EUR": usd_in_gbp * 1.09}, index=INDEX),
    hub="GBP",
)

msft_gbp = x.to_base_currency(prices[["MSFT"]], "GBP", fx_table,
                              quote_currency="USD", fx_markup=0.005)

converted = prices.copy()
converted["MSFT"] = msft_gbp["MSFT"]

print(f"MSFT in USD : {prices['MSFT'].iloc[0]:8.2f} -> {prices['MSFT'].iloc[-1]:8.2f}"
      f"  ({prices['MSFT'].iloc[-1] / prices['MSFT'].iloc[0] - 1:+.2%})")
print(f"USD in GBP  : {usd_in_gbp.iloc[0]:8.4f} -> {usd_in_gbp.iloc[-1]:8.4f}"
      f"  ({usd_in_gbp.iloc[-1] / usd_in_gbp.iloc[0] - 1:+.2%})")
print(f"MSFT in GBP : {converted['MSFT'].iloc[0]:8.2f} -> {converted['MSFT'].iloc[-1]:8.2f}"
      f"  ({converted['MSFT'].iloc[-1] / converted['MSFT'].iloc[0] - 1:+.2%})")
print()
decomp = msft_gbp.decompose()
total = float(np.nansum(np.log1p(decomp["total"]["MSFT"].to_numpy(dtype=float))))
asset = float(np.nansum(np.log1p(decomp["asset"]["MSFT"].to_numpy(dtype=float))))
fx = float(np.nansum(np.log1p(decomp["fx"].to_numpy(dtype=float).ravel())))
print(f"log-return decomposition: total {total:+.4f} = asset {asset:+.4f} + fx {fx:+.4f}"
      f"   (residual {total - asset - fx:+.1e})")
print()
print("The converted series conflates asset return with FX return. Correct for total")
print("return — and the original is retained so it can be split apart again.")
print()
print("The UKX spread bet is NOT converted: spread bets are account-currency")
print("denominated by construction. Pre-conversion applies to equity and FX spot only.")


# ==================================================================================
header("STEP 4 — Build targets")
# ==================================================================================

weights = x.equal_weight(converted.columns, INDEX)
schedule = x.calendar_schedule(INDEX, freq="M")
targets, drift = x.to_targets(weights, schedule, universe=universe,
                              price=converted, equity=100_000.0)

rows = targets.notna().all(axis=1)
print(f"rebalance bars      : {int(schedule.sum())}")
print(f"non-NaN target rows : {int(rows.sum())}  (equal, as it must be)")
print()
print("first three rebalances, after quantization:")
print(targets[rows].head(3).round(6).to_string())
print()
print("drift introduced by rounding to fillable sizes:")
print(drift[rows].head(3).round(6).to_string())
print()
print(f"mean absolute drift across the backtest: {float(drift[rows].abs().mean().mean()):.6f}")


# ==================================================================================
header("STEP 5 — Resolve arguments")
# ==================================================================================

args = universe.resolve(converted, fx_table=fx_table, base_currency="GBP",
                        notional=converted * 100)

def describe(value):
    """Summarise an Order argument without dumping 520 numbers."""
    if isinstance(value, pd.DataFrame):
        return f"<DataFrame {value.shape}> first row {value.iloc[0].round(6).to_dict()}"
    if isinstance(value, list):
        return "[" + ", ".join(describe(v) for v in value) + "]"
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return f"{float(arr):g}"
    return f"<series n={arr.size} {arr.min():.4g}..{arr.max():.4g}>"


for key, value in args.items():
    print(f"  {key:18s} {describe(value)}")

print()
print("Per-column arrays for every Order field, in the wrapper's column order:")
print("  - size_granularity and min_size differ per mode: £0.01/pt, 1 share, 1000 units")
print("  - spread and slippage SUMMED into one field, because only one field exists")
print("  - MSFT's fixed_fees vary per bar: max(0.1% x notional, $1 minimum), with the")
print("    dollar minimum FX-converted to GBP via spec.cost_currency")
print("  - lock_cash set, so no leveraged position borrows another's collateral")
print()
print("Note what is NOT here: fees is 0 for MSFT because the minimum absorbed it. A")
print("percentage plus a minimum cannot be expressed as fees + fixed_fees, so the")
print("conditional is pre-computed per bar instead.")


# ==================================================================================
header("STEP 6 — Simulate")
# ==================================================================================

print("""
Three entry points, and the fork between them is the key structural fact:

  investing / rebalancing        no holding costs              with holding costs
  ---------------------------    --------------------------    ----------------------
  Portfolio.from_weights(...)    Portfolio.from_orders(...)    Portfolio.from_holding_costs(...)
       |                              |                              |
       +-- wraps steps 4-5 and        +-- the simple, fast,          +-- from_order_func +
           calls from_orders with         stable path                    post_segment_func_nb
           TargetPercent                                                 mutating last_cash

Holding costs do NOT force that third path. Charges are computed from the realised
positions afterwards and folded into reported cash, so every strategy type gets correct
financing and carry on the simple path. The replay is reserved for the one case where a
charge must feed back into the cash available to the next order in the same run — and it
needs explicit sizes, so it cannot express entries and exits.

Keeping that fork narrow is the central architectural goal, and it is now narrower than
the plan assumed.
""")

if HAS_VBT:
    import vectorbtX as vbt

    pf = vbt.Portfolio.from_weights(
        converted, weights=weights, schedule=schedule,
        universe=universe, init_cash=100_000, freq="D",
    )
    print(pf.stats())
    print()
    print(pf.trades.stats())
    print()
    print(pf.returns_stats())
else:
    print("""
    pf = vbt.Portfolio.from_weights(
        converted, weights=weights, schedule=schedule,
        universe=universe, init_cash=100_000, freq="D",
    )

[vectorbt not installed here, so the simulation is not executed. Everything above
 this point ran for real.]
""")


# ==================================================================================
header("STEP 7 — Analyse")
# ==================================================================================

print("""
  pf.stats()            native + Total Net Profit, Recovery Factor,
                        Absolute Drawdown, GHPR
  pf.trades.stats()     the full MT4/MT5 trade set (tags="mt5" selects just those)
  pf.returns_stats()    native + Ulcer, UPI, M2, Treynor, SNR, K-ratio,
                        geometric Sharpe
  pf.x.drift()          what quantization actually cost
  pf.x.borrowed_portion()   borrowed vs cash-covered, per bar
  pf.x.min_margin_level()   the closest this came to a margin call

  vbt.x.aggregate([pf1.stats(), pf2.stats()], how="geometric")
""")


# ==================================================================================
header("A closing note on what this does NOT do")
# ==================================================================================

print("""
Seven limitations, documented rather than hidden:

 1. Symmetric spread only — no bid/ask asymmetry.
 2. Holding costs land on the next bar (shifts WHEN, not the total).
 3. Return decomposition needs the retained native series.
 4. No dividend treatment for spread bets and CFDs.
 5. portfolio_level leverage is exact only via from_weights.
 6. End-of-day time is inert on daily bars.
 7. Day count follows the market: /365 for GBP, SGD, ZAR; /360 elsewhere.
 8. Taxes are out of scope — no stamp duty, no CGT. Cross-mode comparison is PRE-TAX.
 9. Short share borrow charge is not modelled (understates short-side cost).

Deferred deliberately: maintenance margin, margin calls, forced liquidation.
""")
