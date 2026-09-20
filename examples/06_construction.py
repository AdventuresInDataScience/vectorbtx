"""06 — Portfolio construction and rebalancing.

Investing as a first-class use case. vectorbt's native constructors all assume
signal-driven entry and exit; a rebalanced portfolio instead holds continuously and
periodically adjusts weights.

Run: python examples/06_construction.py
"""

import numpy as np
import pandas as pd

from _bootstrap import HAS_VBT, header, x

Instrument = x.Instrument

INDEX = pd.date_range("2024-01-01", periods=180, freq="D")
rng = np.random.default_rng(7)
PRICES = pd.DataFrame({
    "AAA": 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.011, 180))),
    "BBB": 50 * np.exp(np.cumsum(rng.normal(0.0002, 0.008, 180))),
    "CCC": 20 * np.exp(np.cumsum(rng.normal(0.0006, 0.015, 180))),
}, index=INDEX)


# --- 1. Weighting schemes ---------------------------------------------------------------
header("1. Weighting schemes")

equal = x.equal_weight(PRICES.columns, INDEX)
price_w = x.price_weight(PRICES)
caps = pd.DataFrame({"AAA": 900.0, "BBB": 300.0, "CCC": 100.0}, index=INDEX)
invvol = x.cap_weight(caps, INDEX)   # plan specifies equal, price and cap schemes

print("equal        :", equal.iloc[-1].round(4).to_dict())
print("price-weight :", price_w.iloc[-1].round(4).to_dict())
print("cap-weight   :", invvol.iloc[-1].round(4).to_dict())


print()
print("Cap weights are RECOMPUTED at each rebalance. Fixed at construction is not cap")
print("weighting — it is an initial-cap-weighted buy-and-hold.")


# --- 2. Calendar schedules ----------------------------------------------------------------
header("2. Calendar schedules")

for freq, label in [("ME", "monthly"), ("QE", "quarterly"), ("W", "weekly")]:
    sched = x.calendar_schedule(INDEX, freq)
    print(f"  {label:10s} {int(sched.sum()):3d} rebalance bars out of {len(INDEX)}")

monthly = x.calendar_schedule(INDEX, "ME")
print()
print("first four monthly rebalance dates:", [str(d.date()) for d in INDEX[monthly][:4]])


# --- 3. Threshold schedules ------------------------------------------------------------------
header("3. Threshold rebalancing uses RELATIVE drift")

target = pd.Series({"AAA": 0.50, "BBB": 0.49, "CCC": 0.01})
current = pd.Series({"AAA": 0.52, "BBB": 0.465, "CCC": 0.015})

print("target :", target.to_dict())
print("current:", current.to_dict())
print()
for name, value in current.items():
    rel = abs(value - target[name]) / target[name]
    absd = abs(value - target[name])
    print(f"  {name}: absolute drift {absd:.4f}   relative drift {rel:.1%}")
print()
print("CCC drifted 0.005 absolutely — trivial — but 50% relatively, which is what it")
print("actually is. Absolute thresholds under-trigger on small positions.")


# --- 4. Schedules compose --------------------------------------------------------------------
header("4. Schedules compose with & and |")

drifted = x.drifted_weights(equal, PRICES, schedule=monthly)
threshold = x.threshold_schedule(drifted, equal, tolerance=0.05)
combined = monthly & threshold

print(f"monthly alone            : {int(monthly.sum()):3d} rebalances")
print(f"threshold alone (5%)     : {int(threshold.sum()):3d} rebalances")
print(f"monthly AND drift > 5%   : {int(combined.sum()):3d} rebalances")
print()
print("'Monthly, but only if drift exceeds 5%' is the common real-world rule, and it is")
print("just an `&`. Supported explicitly rather than left for users to build.")


# --- 5. NaN off-schedule — the single most important line in the section -----------------------
header("5. NaN off-schedule, NEVER forward-filled")

# Use a weight matrix that genuinely varies, so the forward-fill check stays quiet
# (section 6 demonstrates that check deliberately).
targets, drift = x.to_targets(invvol, monthly)
print(targets.iloc[26:32].round(4).to_string())
print()
print("Non-NaN rows:", int(targets.notna().all(axis=1).sum()), "= the number of rebalances")
print()
print("NaN means 'no order', and the position drifts as intended between rebalances.")
print("Forward-filling the weights rebalances EVERY BAR — which still runs, still looks")
print("plausible, and quietly accrues transaction costs while producing a suspiciously")
print("smooth equity curve. It is the single most likely thing to get written backwards.")


# --- 6. So a forward-filled input warns ----------------------------------------------------------
header("6. A forward-filled input is caught")

import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    x.to_targets(equal, monthly)          # equal weights are constant by construction
    if caught:
        print("->", str(caught[0].message)[:200])


# --- 7. Weights summing above 1 -------------------------------------------------------------------
header("7. Weights summing above 1 route through the leverage budget")

from vectorbtx.construct.rebalance import validate_weights

under = pd.DataFrame({"A": [0.3], "B": [0.4]})
over = pd.DataFrame({"A": [0.8], "B": [0.9]})

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    validate_weights(under)
    print("gross 0.70 ->", str(caught[0].message)[:130] if caught else "(no warning)")

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    validate_weights(over)
    print("gross 1.70 ->", str(caught[0].message)[:170] if caught else "(no warning)")

print()
print("Under 1 leaves the remainder in cash — fine, and often intended.")
print("Over 1 is implicit leverage, and must route through the leverage_type budget")
print("rather than entering through the back door.")


# --- 8. The portfolio-level leverage budget -----------------------------------------------------------
header("8. portfolio_level scales the weight matrix to fit the budget")

universe = x.load_universe({
    "A": Instrument(mode="cfd", point_size=0.01, contract_size=1.0,
                    leverage=2.0, leverage_type="portfolio_level"),
    "B": Instrument(mode="cfd", point_size=0.01, contract_size=1.0,
                    leverage=2.0, leverage_type="portfolio_level"),
})

for gross, w in [("2.4x (over budget)", pd.DataFrame({"A": [1.2], "B": [1.2]})),
                 ("1.4x (within 2x)", pd.DataFrame({"A": [0.7], "B": [0.7]}))]:
    scaled = x.apply_leverage_budget(w, universe)
    print(f"  {gross:20s} -> {scaled.iloc[0].round(4).to_dict()}  "
          f"gross {float(scaled.iloc[0].abs().sum()):.2f}x")

print()
print("Within budget, weights are untouched. Over budget, they scale proportionally —")
print("exact and cheap, because weights are known before the simulation starts.")


# --- 9. Quantization happens BEFORE emission -----------------------------------------------------------
header("9. Quantize before emitting targets, and report the drift")

small = x.load_universe({
    c: Instrument(mode="equity", point_size=0.01) for c in PRICES.columns
})
targets, drift = x.to_targets(invvol, monthly, universe=small,
                              price=PRICES, equity=5_000.0)
rows = targets.notna().all(axis=1)
print("quantized targets on rebalance bars:")
print(targets[rows].head(3).round(6).to_string())
print()
print("drift introduced by whole-share rounding:")
print(drift[rows].head(3).round(6).to_string())
print()
print("Order matters: quantizing AFTER the engine computes target percentages loses")
print("the drift information entirely.")


# --- 10. The constructor -------------------------------------------------------------------------------------
header("10. Portfolio.from_weights")

print("""
    pf = vbt.Portfolio.from_weights(
        close,
        weights   = vbt.x.equal_weight(close.columns, close.index),
        schedule  = vbt.x.calendar_schedule(close.index, "M"),
        universe  = universe,
        init_cash = 100_000,
    )
    pf.x.drift()        # what quantization and suppression actually cost

Three defaults are load-bearing, not niceties:

  size_type="targetpercent"   Percent is NOT a substitute — upstream warns it misbehaves
                              with call_seq="auto" and cannot reverse a position.
  cash_sharing=True           the cash-sharing unit is the only coherent budget boundary
  call_seq="auto"             sells are ordered before buys so the sells FUND the buys;
                              without it a rebalance can fail for lack of cash despite
                              the weights being perfectly fundable

You never compute the buy/sell orders. TargetPercent derives them.
""")

if not HAS_VBT:
    print("[vectorbt not installed here — the call above is not executed]")
