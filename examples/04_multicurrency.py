"""04 — Multi-currency.

Holding foreign-currency instruments in a base-currency portfolio, with correct P&L and
without touching the engine. Pre-conversion is exact, not an approximation.

Run: python examples/04_multicurrency.py
"""

import numpy as np
import pandas as pd

from _bootstrap import header, x

Instrument = x.Instrument

INDEX = pd.date_range("2024-01-01", periods=5, freq="D")


# --- 1. Building a rate table -----------------------------------------------------------
header("1. The rate table — direction lives in the KEY")

usd_gbp = pd.Series([0.7874, 0.7900, 0.7950, 0.7820, 0.7750], index=INDEX)
eur_gbp = pd.Series([0.8583, 0.8600, 0.8650, 0.8520, 0.8450], index=INDEX)

table = x.FXTable.from_long(
    pd.concat([
        pd.DataFrame({"date": INDEX, "quote": "USD", "base": "GBP", "value": usd_gbp.to_numpy()}),
        pd.DataFrame({"date": INDEX, "quote": "EUR", "base": "GBP", "value": eur_gbp.to_numpy()}),
    ])
)
print("currencies:", sorted(table.currencies))
print("USD->GBP  :", round(float(table.get("USD", "GBP").iloc[0]), 6))
print("GBP->USD  :", round(float(table.get("GBP", "USD").iloc[0]), 6), "(derived reciprocal)")
print("EUR->USD  :", round(float(table.get("EUR", "USD").iloc[0]), 6), "(derived by triangulation)")
print()
print("`value` is what you MULTIPLY a `quote`-currency price by to get `base`.")
print("There is no multiply-or-divide decision left to get wrong.")


# --- 2. Why reciprocals are derived, never fetched ----------------------------------------
header("2. Derive reciprocals — never fetch both directions")

a = float(table.get("USD", "GBP").iloc[0])
b = float(table.get("GBP", "USD").iloc[0])
print(f"value(USD,GBP) x value(GBP,USD) = {a * b:.15f}")
print()
print("Exactly 1, by construction. Fetching both directions independently gives")
print("non-exact reciprocals — different timestamps, mid vs bid/ask capture — and a")
print("round-trip conversion then produces phantom P&L. That noise is RANDOM IN SIGN,")
print("so it inflates measured volatility rather than acting as a cost, quietly")
print("degrading every risk-adjusted metric with a standard deviation in the denominator.")


# --- 3. Validation catches a corrupted table -----------------------------------------------
header("3. Reciprocal and triangulation checks catch bad data")

import warnings

# Build a grid by hand with a deliberately inconsistent cross rate.
grid = pd.DataFrame(
    {("USD", "GBP"): usd_gbp, ("GBP", "USD"): 1 / usd_gbp,
     ("EUR", "GBP"): eur_gbp, ("GBP", "EUR"): 1 / eur_gbp,
     ("USD", "USD"): 1.0, ("GBP", "GBP"): 1.0, ("EUR", "EUR"): 1.0,
     ("EUR", "USD"): eur_gbp / usd_gbp,
     ("USD", "EUR"): 1.5},                       # <- wrong: should be usd_gbp / eur_gbp
    index=INDEX,
)
grid.columns = pd.MultiIndex.from_tuples(grid.columns, names=["quote", "base"])
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    x.FXTable(grid, hub="GBP")
    print("warnings raised:", len(caught))
    for w in caught[:2]:
        print("  -", str(w.message)[:130])
print()
print("Note: a table built through from_canonical or from_long CANNOT fail these checks,")
print("because the grid is derived rather than fetched. The validation exists for tables")
print("assembled from an external source, where stale rows and bad joins are real.")


# --- 4. The conversion identity, verified numerically ----------------------------------------
header("4. Pre-conversion is EXACT — the D4 identity")

msft_usd = pd.Series([400.0, 405.0, 410.0, 395.0, 390.0], index=INDEX)
shares = 100

# Route A: convert prices, then compute P&L in GBP
msft_gbp = msft_usd * usd_gbp
pnl_converted = shares * (msft_gbp.iloc[-1] - msft_gbp.iloc[0])

# Route B: the ground truth a GBP investor actually experiences
pnl_truth = shares * (msft_usd.iloc[-1] * usd_gbp.iloc[-1] - msft_usd.iloc[0] * usd_gbp.iloc[0])

print(f"convert-then-simulate : {pnl_converted:.10f}")
print(f"ground truth          : {pnl_truth:.10f}")
print(f"difference            : {abs(pnl_converted - pnl_truth):.2e}")
print()
print("shares x (P_usd[t1]·fx[t1] - P_usd[t0]·fx[t0])  ==  shares x (P_gbp[t1] - P_gbp[t0])")
print("Same expression. So cash sharing works, cross-currency rebalancing works, and FX")
print("moves correctly affect buying power mid-backtest — without engine changes.")


# --- 5. Price and FX moving in OPPOSITE directions ----------------------------------------------
header("5. Price and FX moving together — and against each other")

print(f"MSFT in USD : {msft_usd.iloc[0]:.2f} -> {msft_usd.iloc[-1]:.2f} "
      f"({msft_usd.iloc[-1] / msft_usd.iloc[0] - 1:+.2%})")
print(f"USD in GBP  : {usd_gbp.iloc[0]:.4f} -> {usd_gbp.iloc[-1]:.4f} "
      f"({usd_gbp.iloc[-1] / usd_gbp.iloc[0] - 1:+.2%})")
print(f"MSFT in GBP : {msft_gbp.iloc[0]:.2f} -> {msft_gbp.iloc[-1]:.2f} "
      f"({msft_gbp.iloc[-1] / msft_gbp.iloc[0] - 1:+.2%})")
print()
print()
# Now the same holding over a window where they move in OPPOSITE directions.
window = slice(0, 3)          # USD price rises 400 -> 410, GBP rate rises then...
opp_usd = pd.Series([400.0, 405.0, 410.0], index=INDEX[:3])
opp_fx = pd.Series([0.7874, 0.7800, 0.7700], index=INDEX[:3])
opp_gbp = opp_usd * opp_fx
print(f"opposite directions: USD price {opp_usd.iloc[-1] / opp_usd.iloc[0] - 1:+.2%}, "
      f"FX {opp_fx.iloc[-1] / opp_fx.iloc[0] - 1:+.2%} "
      f"-> GBP {opp_gbp.iloc[-1] / opp_gbp.iloc[0] - 1:+.2%}")
truth = 100 * (opp_usd.iloc[-1] * opp_fx.iloc[-1] - opp_usd.iloc[0] * opp_fx.iloc[0])
print(f"identity still exact: {100 * (opp_gbp.iloc[-1] - opp_gbp.iloc[0]) - truth:.2e}")
print()
print("Same-direction moves can hide a sign error. Opposite directions cannot, which")
print("is why the worked example uses them.")


# --- 6. Converting OHLC — one rate per bar --------------------------------------------------------
header("6. ONE rate per bar, applied to all of OHLC")

ohlc = pd.DataFrame({
    "open":  [400.0, 405.0, 410.0, 395.0, 390.0],
    "high":  [403.0, 408.0, 412.0, 400.0, 393.0],
    "low":   [398.0, 402.0, 405.0, 390.0, 386.0],
    "close": [402.0, 407.0, 407.0, 392.0, 391.0],
}, index=INDEX)

converted = x.to_base_currency(ohlc, "GBP", table, quote_currency="USD", fx_markup=0.005)
print(converted.round(4).to_string())
print()
ok = ((converted["high"] >= converted["close"]) & (converted["close"] >= converted["low"])).all()
print("H >= C >= L survives conversion:", bool(ok))
print()
print("Different rates for open/high/low/close would break that invariant, and native")
print("SL/TP would then fire at prices that never existed.")


# --- 7. The original series is retained --------------------------------------------------------------
header("7. Decomposition needs the original series — so it is kept")

print("converted.native is the pre-conversion frame:", converted.native is not None)
print("converted.fx_markup                          :", converted.fx_markup)
decomp = converted.decompose()
print()
for name, part in decomp.items():
    col = part["close"] if "close" in getattr(part, "columns", []) else part.squeeze()
    print(f"  {name:6s}", np.round(np.asarray(col, dtype=float)[1:], 6))
print()
print("The converted series conflates asset return with FX return. Correct for total")
print("return; decomposition needs the original, which is cheap to keep.")


# --- 8. fx_markup is adverse in BOTH directions -----------------------------------------------------------
header("8. fx_markup — adverse converting in AND converting out")

amount = 100.0
mid = amount * float(table.get("USD", "GBP").iloc[-1])
into = x.convert_amount(amount, "USD", "GBP", table, markup=0.005)
back = x.convert_amount(into, "GBP", "USD", table, markup=0.005)

print(f"$100 -> GBP at mid      : {mid:.4f}")
print(f"$100 -> GBP with markup : {into:.4f}   (worse)")
print(f"back  -> USD with markup: {back:.4f}   (worse AGAIN, not reversed)")
print(f"round-trip cost         : {amount - back:.4f}"
      f"   expected 100 x (1 - 0.995^2) = {100 * (1 - 0.995 ** 2):.4f}")
print()
print("It is applied as slippage on the trade rather than baked into the price series,")
print("because a UNIFORM rescaling of one series would cancel out and cost nothing.")
print("As slippage it is genuinely adverse on entry and again on exit.")


# --- 9. Foreign fixed costs -------------------------------------------------------------------------------
header("9. Percentage costs are FX-invariant; fixed costs are NOT")

fee_usd = 1.0        # a $1 minimum commission
fee_gbp = x.convert_fixed_costs(fee_usd, "USD", "GBP", table, INDEX)
print("a $1 minimum commission, in GBP, per bar:")
print(np.round(np.asarray(fee_gbp, dtype=float), 6))
print()
print("This is what spec.cost_currency exists for. D4 converts a US equity's PRICES to")
print("GBP, but a $1 minimum commission is still quoted in dollars — and a percentage")
print("commission needs no conversion at all.")


# --- 10. Calendar mismatches warn ---------------------------------------------------------------------------
header("10. A calendar mismatch warns rather than silently forward-filling")

sparse = INDEX[[0, 2, 4]]
sparse_table = x.FXTable.from_long(pd.DataFrame({
    "date": sparse, "quote": "USD", "base": "GBP", "value": [0.7874, 0.7950, 0.7750],
}))
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    sparse_table.get("USD", "GBP", INDEX)
    if caught:
        print("->", str(caught[0].message)[:160])
    else:
        print("(no gaps on this index)")
print()
print("FX trades ~24/5, equities have exchange hours, holidays differ by country.")
print("Mismatches are guaranteed, so the fill policy is explicit rather than assumed.")
