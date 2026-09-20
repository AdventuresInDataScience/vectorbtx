"""05 — Costs and leverage.

Transaction cost modelled composably, and positions a broker would refuse prevented before
they are placed.

Run: python examples/05_costs_margin.py
"""

import numpy as np
import pandas as pd

from _bootstrap import header, x

Instrument = x.Instrument

INDEX = pd.date_range("2024-01-01", periods=30, freq="D")
PRICES = pd.DataFrame({"MSFT": np.linspace(400, 430, 30), "VOD": np.linspace(75, 71, 30)},
                      index=INDEX)


# --- 1. The default cost stacks -------------------------------------------------------
header("1. Default cost stack per mode")

for mode, asset_class in [("equity", None), ("fx_spot", None), ("cfd", "fx"),
                          ("cfd", "stocks"), ("spreadbet", "fx"), ("spreadbet", "stocks")]:
    stack = sorted(x.default_cost_stack(mode, asset_class))
    label = f"{mode}/{asset_class or '-'}"
    print(f"  {label:20s} {' + '.join(stack)}")

print()
print("Spread bets carry NO commission — the fee is baked into a wider spread.")
print("Anything with a currency pair underneath it carries, leveraged or not.")


# --- 2. Where each cost lands ------------------------------------------------------------
header("2. Cost -> Order field mapping")

print("""
  percentage commission   -> fees
  flat / per-unit         -> fixed_fees
  minimum commission      -> fixed_fees, PRE-COMPUTED, with fees zeroed
  spread                  -> slippage
  slippage                -> slippage        <- both land here, so they are SUMMED
""")


# --- 3. Percentage commission --------------------------------------------------------------
header("3. Percentage commission -> fees")

from vectorbtx.costs.commission import COMMISSION

spec = Instrument(mode="equity", point_size=0.01, commission_pct=0.001).filled()
print(COMMISSION.contribute(spec, PRICES["MSFT"], {"column": "MSFT"}))


# --- 4. Flat commission, and the equivalence helper --------------------------------------------
header("4. Retail equity commission is often FLAT")

spec = Instrument(mode="equity", point_size=0.01, commission_flat=10.0).filled()
print("contribution:", COMMISSION.contribute(spec, PRICES["MSFT"], {"column": "MSFT"}))
print()
print("£10 on £10,000 notional ->", x.equivalent_pct(10.0, 10_000), "as a percentage")
print("0.1% on £10,000 notional ->", x.equivalent_flat(0.001, 10_000), "as a flat fee")
print("The helper exists because users often want one number, not two.")


# --- 5. A minimum commission cannot be expressed additively ----------------------------------------
header("5. Minimum commission is pre-computed per order")

spec = Instrument(mode="equity", point_size=0.01,
                  commission_pct=0.001, commission_min=6.0).filled()
notional = pd.Series([2_000.0, 6_000.0, 20_000.0], index=INDEX[:3])
out = COMMISSION.contribute(spec, PRICES["MSFT"].iloc[:3],
                            {"column": "MSFT", "notional": notional})
print("notional  :", list(notional))
print("0.1% of it:", list((notional * 0.001).round(2)))
print("fixed_fees:", list(np.round(np.asarray(out["fixed_fees"], dtype=float), 2)))
print("fees      :", out["fees"], "(zeroed — the minimum absorbed it)")
print()
print("max(pct x notional, minimum) is a CONDITIONAL. fees and fixed_fees are additive,")
print("so it genuinely cannot be expressed as one plus the other.")

from vectorbtx.costs.base import CostError

try:
    COMMISSION.contribute(spec, PRICES["MSFT"], {"column": "MSFT"})
except CostError as exc:
    print()
    print("without notional ->", exc)


# --- 6. Spread ---------------------------------------------------------------------------------------
header("6. Spread — the full gap, halved per side")

from vectorbtx.costs.spread import SPREAD

spec = Instrument(mode="spreadbet", point_size=1.0, spread_points=4.0).filled()
out = SPREAD.contribute(spec, pd.Series([7500.0], index=INDEX[:1]), {"column": "UKX"})
frac = float(np.ravel(out["slippage"])[0])
print(f"4-point spread on 7500 -> slippage {frac:.8f}  = 2 points / 7500")
print(f"round trip costs the full 4 points: {2 * frac * 7500:.2f} points")
print()
print("Symmetric only. vectorbt has one slippage field per order, so asymmetric")
print("bid/ask would drag ALL cost modelling into simulation callbacks.")


# --- 7. Auto slippage -----------------------------------------------------------------------------------
header("7. Auto slippage — sqrt market impact")

from vectorbtx.costs.slippage import SLIPPAGE, auto_slippage

rng = np.random.default_rng(0)
price = pd.Series(400 + np.cumsum(rng.normal(0, 2, 30)), index=INDEX)
volume = pd.Series(1_000_000.0, index=INDEX)

for size in (1_000, 10_000, 100_000):
    series = auto_slippage(price, volume, size, coef=1.0, slippage_max=0.005, window=20)
    print(f"  size {size:7,d} -> slippage {float(series.iloc[-1]):.6f}")

print()
print("coef x sigma x sqrt(size / ADV), clipped to [0, slippage_max].")
print("Square-root impact is the standard empirical result and beats linear.")
print("Doubling size raises cost by ~sqrt(2), not 2x.")

spec = Instrument(mode="equity", point_size=0.01, slippage="auto").filled()
try:
    SLIPPAGE.contribute(spec, price, {"column": "MSFT", "size": 1000})
except CostError as exc:
    print()
    print("no volume column ->", exc)
    print("It fails at RESOLVE time rather than silently falling back to zero cost.")


# --- 8. Composition, and the summation that matters ---------------------------------------------------------
header("8. resolve_costs — spread and slippage SUMMED, not overwritten")

universe = x.load_universe({
    "MSFT": Instrument(mode="equity", point_size=0.01, commission_pct=0.001,
                       spread_points=2.0, slippage=0.0005),
    "VOD": Instrument(mode="spreadbet", point_size=0.01, spread_points=1.0),
})
out = x.resolve_costs(universe, PRICES)
print("fees      :", out["fees"])
print("fixed_fees:", out["fixed_fees"])
slip = out["slippage"]
print("slippage  :")
print(slip.head(3).round(8).to_string())
print()
spread_only = 1.0 * 0.01 / 2 / PRICES["MSFT"].iloc[0]
print(f"MSFT bar 0: spread {2 * 0.01 / 2 / 400:.8f} + slippage 0.00050000 "
      f"= {float(slip['MSFT'].iloc[0]):.8f}")
print("Only ONE slippage field exists. Summing before assignment is not optional.")


# --- 9. Margin and notional ----------------------------------------------------------------------------------
header("9. Notional, margin required, borrowed portion")

cfd = Instrument(mode="cfd", point_size=0.01, contract_size=1.0, leverage=5.0,
                 asset_class="stocks").filled()
size, price = 500.0, 100.0
notional = x.notional(size, price, cfd)
margin = x.margin_required(size, price, cfd)
print(f"500 lots at 100, 5x leverage")
print(f"  notional        : {notional:,.2f}")
print(f"  margin required : {margin:,.2f}")
print(f"  borrowed        : {notional - margin:,.2f}")
print()
print("A 5x leveraged £10,000 margin position has £50,000 notional. Financing is")
print("charged on the borrowed £40,000 only — never on the whole position.")


# --- 10. The leverage cap ------------------------------------------------------------------------------------------
header("10. max_size clips rather than rejects")

equity = 10_000.0
cap = x.leverage_cap(cfd, equity, price)
print(f"equity {equity:,.0f} at {price} with 5x -> max_size {cap:,.2f} engine units")
print(f"that is {cap * price:,.0f} of notional = {cap * price / equity:.0f}x equity")
print()
print("Order.max_size PARTIALLY FILLS rather than rejecting, so we get clip semantics")
print("for free. For reject semantics instead, pass raise_reject=True.")
print()
print("instrument_level is a per-column cap and is exact.")
print("portfolio_level is a JOINT constraint — max_size is per-column and knows nothing")
print("about its siblings — so it is enforced on the weight matrix by from_weights,")
print("where weights are known ahead of time and scaling to fit is exact and cheap.")
