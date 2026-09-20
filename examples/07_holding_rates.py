"""07 — Holding costs and rate data.

Charging for time. The rules here differ by **product**, and getting that wrong is silent
— which is why `docs/dev/AUDIT-G.md` exists. Every claim in this file is checked against a
broker's published costs schedule rather than against intuition.

Run: python examples/07_holding_rates.py
"""

import numpy as np
import pandas as pd

from _bootstrap import HAS_VBT, header, x

Instrument = x.Instrument
from vectorbtx.instruments.defaults import financing_basis, triple_day
from vectorbtx.rates.table import DEFAULT_SERIES, day_count_for


# --- 1. Which cost applies at all ------------------------------------------------------
header("1. Financing OR carry — never both")

print("An FX-underlying position has exactly ONE overnight charge, and it IS the")
print("tom-next swap. Tom-next already prices both currencies' interest — that is what")
print("makes it a differential — so adding a leverage-driven financing charge on top")
print("bills the same money twice.")
print()
for mode, ac in [("spreadbet", "fx"), ("spreadbet", "index"),
                 ("cfd", "fx"), ("cfd", "stocks"), ("equity", None), ("fx_spot", None)]:
    stack = sorted(x.default_cost_stack(mode, ac))
    holding = [c for c in stack if c in ("carry", "financing")]
    print(f"  {mode:10s}/{str(ac):7s} -> {holding}")
print()
print("Combining them raises rather than warns — see Instrument validation.")


# --- 2. What financing is charged ON ---------------------------------------------------
header("2. The financing basis differs by product")

print("This is the distinction that a summarised plan collapsed, and it is worth")
print("stating precisely because both readings look plausible:")
print()
print("  equity (margin share dealing)  -> the DEBIT BALANCE.")
print("      You own the shares and borrow the shortfall, so a position fully covered")
print("      by cash finances nothing.")
print()
print("  spreadbet / cfd                -> the FULL NOTIONAL.")
print("      You own nothing and borrow nothing. The broker funds the entire position")
print("      and holds margin as collateral, so an UNLEVERAGED CFD still pays in full.")
print()
for mode in ("equity", "cfd", "spreadbet"):
    print(f"  financing_basis({mode!r}) = {financing_basis(mode)!r}")
print()
print("Getting this wrong understates derivative financing by 1 - 1/leverage — four")
print("fifths of the true cost at 5x — and zeroes it entirely at 1x.")

print()
print("IG's own worked example, reproduced exactly:")
print("  long 1500 contracts, contract value 1, price 83.90, benchmark 1.89%, fee 3.0%")
notional = 1500 * 1 * 83.90
charge = x.financing_charge(notional, 0.0189, 0.030, True, 1, 360.0)
print(f"  {notional:,.0f} x (3.0% + 1.89%) / 360 = {-float(charge):.2f}")
print("  Note the formula: contracts x value x PRICE. Position value throughout, with")
print("  no reference to account cash.")


# --- 3. The markup is always adverse ---------------------------------------------------
header("3. The broker markup is always adverse")

BENCH, MARKUP = 0.0525, 0.03
print(f"benchmark {BENCH:.2%}, broker admin fee {MARKUP:.2%}")
print(f"  long  pays     {x.financing_rate(BENCH, MARKUP, is_long=True):.2%}  = benchmark + fee")
print(f"  short receives {x.financing_rate(BENCH, MARKUP, is_long=False):.2%}  = benchmark - fee")
print()
print("Brokers express the short side as 'fee minus benchmark', which is the same thing.")
print("A simple addition would credit shorts at 8.25% and flatter every short-heavy")
print("backtest.")
print()
print("When the fee exceeds the benchmark the short is CHARGED, which is correct — in a")
print("low-rate period both sides pay:")
print(f"  benchmark 1.00%, fee 3.00% -> short rate "
      f"{x.financing_rate(0.01, 0.03, is_long=False):.2%}")


# --- 4. Day count follows the market ---------------------------------------------------
header("4. The day-count divisor follows the market's denomination")

print("Not a house convention. Brokers use a 365-day divisor for GBP, SGD and ZAR")
print("denominated markets and 360 for everything else.")
print()
for ccy in ("GBP", "SGD", "ZAR", "USD", "EUR", "JPY"):
    print(f"  {ccy}  /{day_count_for(ccy):.0f}")
print()
args = dict(benchmark_rate=0.05, markup=0.03, is_long=True, day_multiplier=1)
correct = float(x.financing_charge(100_000, day_count=365.0, **args))
wrong = float(x.financing_charge(100_000, day_count=360.0, **args))
print(f"100,000 GBP position, one night at 8%:")
print(f"  correct  (/365) : {-correct:.4f}")
print(f"  uniform  (/360) : {-wrong:.4f}   <- 1.4% too much, every night")
print()
print("Small per night, systematic over years — and it inverts for received credits, so")
print("'overstating is the safe direction' does not hold.")


# --- 5. Which day carries the weekend --------------------------------------------------
header("5. The triple day follows the ASSET CLASS")

print("Two conventions, for two different reasons:")
print()
print("  fx          -> WED   spot settles T+2, so Wednesday's roll reaches Monday.")
print("                       This is a SETTLEMENT rule.")
print("  everything  -> FRI   nothing settles; the triple charge just covers the")
print("  else                 calendar weekend, so it lands on the last trading day.")
print()
for ac in ("fx", "stocks", "index", "commodities"):
    print(f"  triple_day({ac!r}) = {triple_day(ac)!r}")
print()
week = pd.date_range("2024-01-01", periods=7, freq="D")   # Mon .. Sun
for day in ("WED", "FRI"):
    mult = x.day_multipliers(week, triple_day=day)
    print(f"  {day}: {list(mult)}  total {mult.sum():.0f}")
print()
print("Both bill SEVEN days a week. Charging calendar days on top would bill nine.")
print("What changes is per-trade timing: a share position held Thursday to Friday is")
print("charged 1x under the Wednesday convention and 3x under the correct one.")


# --- 6. Financing vs carry, side by side -----------------------------------------------
header("6. Financing and carry are different costs")

print("Financing is charged on borrowed money (equity) or full notional (derivatives).")
print("Carry is charged on the WHOLE position regardless of leverage, because you are")
print("long one currency and short another either way.")
print()
diff = x.tom_next_differential(0.05, 0.02, markup=0.01, is_long=True)
print(f"long EURUSD, EUR 5%, USD 2%, admin fee 1%")
print(f"  differential            : {float(diff):+.4f}  = (5% - 2%) - 1%")
print(f"  carry on 100,000, 1 day : {float(x.carry_charge(100_000, diff, 1)):+.4f}")
print()
short = x.tom_next_differential(0.05, 0.02, markup=0.01, is_long=False)
print(f"  short side differential : {float(short):+.4f}  = -(5% - 2%) - 1%  (adverse both ways)")
print()
thin = x.tom_next_differential(0.030, 0.025, markup=0.01, is_long=True)
print(f"A thin positive differential does not survive the fee: {float(thin):+.4f}")
print("which is why the admin fee defaulting to zero was a real understatement.")


# --- 7. Worked example: long and short across the roll ---------------------------------
header("7. A week held, long and short")

days = pd.date_range("2024-01-01", periods=7, freq="D")
mult = x.day_multipliers(days, triple_day="WED")
total_days = mult.sum()

long_total = float(np.sum(x.financing_series(40_000, 0.0525, 0.03, True, mult)))
short_total = float(np.sum(x.financing_series(40_000, 0.0525, 0.03, False, mult)))
print(f"40,000 borrowed, benchmark 5.25%, fee 3%, {total_days:.0f} funded days:")
print(f"  long  : {long_total:+.4f}")
print(f"  short : {short_total:+.4f}")
print()
print(f"Hand check, long : 40000 x (0.0525 + 0.03) / 360 x 7 = "
      f"{-40_000 * 0.0825 / 360 * 7:.4f}")
print(f"Hand check, short: 40000 x (0.0525 - 0.03) / 360 x 7 = "
      f"{40_000 * 0.0225 / 360 * 7:.4f}")


# --- 8. Rate tables --------------------------------------------------------------------
header("8. Rate tables — real benchmarks, or a static fallback")

print("default benchmark series per currency:")
for ccy, series in DEFAULT_SERIES.items():
    print(f"  {ccy}  {series}")
print()
print("All post-LIBOR overnight benchmarks.")

index = pd.date_range("2024-01-01", periods=5, freq="D")
static = x.static_rate_table(["GBP", "USD"], index, {"GBP": 0.0525, "USD": 0.0533})
print()
print("static fallback (no API key needed):")
print(static.to_string())


# --- 9. How costs reach the equity curve ----------------------------------------------
header("9. How holding costs reach the equity curve")

print("""
Charges are computed from the REALISED positions after the simulation, then folded into
reported cash. That matters because `Portfolio.cash_flow` is rebuilt from ORDER RECORDS,
and a financing charge is not an order — so mutating the simulation's cash alone would
leave the charge invisible to value(), returns() and every metric.

  vbt.x.PortfolioBacktest(..., rates=rates)
      -> simulate normally (signals, weights or hold)
      -> compute charges from the realised positions
      -> fold into cash_flow  -> value, returns, stats all correct

What this does NOT do is feed a charge back into the cash available to the NEXT order
within the same run. That needs the level-3 replay:

  vbt.Portfolio.from_holding_costs(...)   # from_order_func + post_segment_func_nb
                                          # mutates last_cash; needs EXPLICIT sizes,
                                          # so it cannot express entries/exits

Use the replay only when a strategy sizes off live equity while running near fully
invested, where the charges would genuinely have throttled later positions.

Two things that remain true regardless:
  * accrual lands on the next bar (last_value is recomputed BEFORE the callback)
  * end-of-day time is inert on daily bars — one bar per day means the cutoff cannot
    change which bar a charge lands on
""")
