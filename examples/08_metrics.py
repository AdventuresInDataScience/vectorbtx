"""08 — Metrics: the seven returns gaps and the MT4/MT5 trade set.

All of these appear in the relevant `.stats()` output automatically. The kernels are pure
arithmetic, so every number below is reproducible without running a simulation.

Run: python examples/08_metrics.py
"""

import numpy as np
import pandas as pd

from _bootstrap import HAS_VBT, header

from vectorbtx.metrics.returns import nb
from vectorbtx.metrics.trades import calc

rng = np.random.default_rng(42)
RETS = rng.normal(0.0006, 0.011, 504)          # ~2 years of daily returns
BENCH = RETS * 0.7 + rng.normal(0.0002, 0.006, 504)
ANN = 252.0


# --- 1. Ulcer Index -----------------------------------------------------------------
header("1. Ulcer Index — RMS of PERCENTAGE drawdown")

small = np.array([0.01, -0.02, 0.03, 0.005, -0.01, 0.02])
curve = nb.cum_curve_1d_nb(small, 1.0)
peak = np.maximum.accumulate(curve)
dd = curve / peak - 1.0

print("returns :", small)
print("equity  :", curve.round(6))
print("peak    :", peak.round(6))
print("drawdown:", dd.round(6))
print()
manual = np.sqrt(np.sum(dd[1:] ** 2) / (len(curve) - 1))
print(f"hand-computed RMS : {manual:.10f}")
print(f"ulcer_index_1d_nb : {nb.ulcer_index_1d_nb(small):.10f}")
print()
print("PERCENTAGE, not absolute. Squaring absolute drawdowns would make the metric")
print("scale with account size and stop being comparable between portfolios.")


# --- 2. Ulcer Performance Index ---------------------------------------------------------
header("2. Ulcer Performance Index (Martin ratio)")

ui = nb.ulcer_index_1d_nb(RETS)
ann_ret = nb.annualized_return_1d_nb(RETS, ANN)
print(f"annualised return : {ann_ret:.6f}")
print(f"ulcer index       : {ui:.6f}")
print(f"UPI = ret / UI    : {ann_ret / ui:.6f}")
print(f"upi_1d_nb         : {nb.ulcer_performance_index_1d_nb(RETS, ANN):.6f}")


# --- 3. M-squared -------------------------------------------------------------------------
header("3. M-squared — return rescaled to the benchmark's volatility")

r = RETS[~np.isnan(RETS)]
sharpe = np.mean(r) / (np.std(r, ddof=1)) * np.sqrt(ANN)
bm_vol = nb.annualized_volatility_1d_nb(BENCH, ANN)
print(f"portfolio Sharpe     : {sharpe:.6f}")
print(f"benchmark volatility : {bm_vol:.6f}")
print(f"M2 = rf + S x sigma_b: {sharpe * bm_vol:.6f}")
print(f"m2_1d_nb             : {nb.m2_1d_nb(RETS, BENCH, ANN):.6f}")
print()
print("Expressed as a RETURN, not a ratio — which is the whole point of M-squared.")


# --- 4. Treynor -----------------------------------------------------------------------------
header("4. Treynor — excess return per unit of SYSTEMATIC risk")

beta = nb.beta_1d_nb(RETS, BENCH)
print(f"beta            : {beta:.6f}")
print(f"annualised ret  : {ann_ret:.6f}")
print(f"Treynor = ret/b : {ann_ret / beta:.6f}")
print(f"treynor_1d_nb   : {nb.treynor_ratio_1d_nb(RETS, BENCH, ANN):.6f}")
print()
print("Sharpe divides by TOTAL risk; Treynor divides by systematic risk only.")


# --- 5. Geometric vs arithmetic Sharpe ---------------------------------------------------------
header("5. Geometric Sharpe — a genuine gap, not a restatement")

geo = nb.geometric_sharpe_ratio_1d_nb(RETS, ANN)
print(f"arithmetic Sharpe (vectorbt native) : {sharpe:.6f}")
print(f"geometric Sharpe  (vectorbtX)       : {geo:.6f}")
print(f"difference                          : {sharpe - geo:.6f}")
print()
print("Upstream's Sharpe is mean/std x sqrt(ann). This one runs on LOG returns, so it")
print("penalises volatility drag. The gap widens as volatility rises.")

volatile = rng.normal(0.0006, 0.035, 504)
rv = volatile[~np.isnan(volatile)]
s_a = np.mean(rv) / np.std(rv, ddof=1) * np.sqrt(ANN)
s_g = nb.geometric_sharpe_ratio_1d_nb(volatile, ANN)
print(f"at 3.5% daily vol: arithmetic {s_a:.4f} vs geometric {s_g:.4f}  "
      f"(gap {s_a - s_g:.4f})")


# --- 6. SNR / efficiency ratio -------------------------------------------------------------------
header("6. Efficiency ratio — path directness, bounded 0-1")

straight = np.full(50, 0.002)                       # a perfect ramp
choppy = np.tile([0.02, -0.019], 25)                # same-ish endpoint, huge path

print(f"steady climb : {nb.snr_1d_nb(straight):.6f}  (near 1 — no wasted movement)")
print(f"chop         : {nb.snr_1d_nb(choppy):.6f}  (near 0 — all path, no progress)")
print(f"real series  : {nb.snr_1d_nb(RETS):.6f}")
print()
print("|P_end - P_start| / sum(|P_t - P_t-1|). Chosen over the mean/sigma reading, which")
print("just duplicates Sharpe, and over the decibel form, which is unconventional in")
print("finance. It complements the K-ratio: K measures trend CONSISTENCY, this measures")
print("path DIRECTNESS.")


# --- 7. The K-ratio family ---------------------------------------------------------------------------
header("7. K-ratio — three normalisations of one regression")

y = np.cumsum(np.log1p(RETS))
slope, stderr, corr = nb.linreg_1d_nb(y)
n = len(y)
t = slope / stderr

print(f"regression of cumulative log returns on time:")
print(f"  slope       : {slope:.8f}")
print(f"  stderr      : {stderr:.8f}")
print(f"  correlation : {corr:.6f}")
print(f"  t = slope/se: {t:.4f}")
print()
for v in (1, 2, 3):
    print(f"  v{v}: {nb.k_ratio_1d_nb(RETS, ANN, v):12.6f}")
print()
print(f"  v1 = t / n                       = {t / n:.6f}")
print(f"  v2 = v1 x sqrt(ann_factor)       = {t / n * np.sqrt(ANN):.6f}")
print(f"  v3 = v2 / sqrt(n)                = {t / n * np.sqrt(ANN) / np.sqrt(n):.6f}")


# --- 8. Why v3 is the default -------------------------------------------------------------------------
header("8. Only v3 survives a change of backtest length")

base = RETS[:126]
print(f"{'length':>8s} {'v1':>12s} {'v2':>12s} {'v3':>12s}")
for reps in (2, 4, 8, 16):
    series = np.tile(base, reps)
    print(f"{len(series):8d} "
          f"{nb.k_ratio_1d_nb(series, ANN, 1):12.4f} "
          f"{nb.k_ratio_1d_nb(series, ANN, 2):12.4f} "
          f"{nb.k_ratio_1d_nb(series, ANN, 3):12.4f}")
print()
print("The same repeating series, made longer. v1 and v2 climb steadily with length —")
print("they are not comparable across backtests of different durations. v3 settles to a")
print("level, which is exactly why it is the default.")
print()
print("(The raw t-statistic grows roughly as n^1.5, because the slope is stable while")
print("its standard error shrinks as n^-1.5. v1 divides by n, leaving sqrt(n) of growth;")
print("v3's extra sqrt(n) is what finally cancels it.)")


# --- 9. Trade metrics: the balance curve ---------------------------------------------------------------------
header("9. Balance vs Equity drawdown — note the inversion")

PNL = np.array([100.0, -50.0, 30.0, 40.0, -20.0, -10.0, 60.0])
INIT = 1_000.0

fmt = lambda a: "[" + ", ".join(f"{v:g}" for v in np.asarray(a, dtype=float)) + "]"
print("trade PnL   :", fmt(PNL))
print("balance     :", fmt(calc.balance_curve(PNL, INIT)))
print()
print(f"absolute drawdown  : {calc.absolute_drawdown(PNL, INIT):.2f}  "
      f"(this run never dips below its deposit, so zero)")

# A run that DOES dip below the opening deposit.
LOSING = np.array([-80.0, -70.0, 50.0, 40.0, 30.0])
print()
print("a run that starts badly:", fmt(LOSING))
print("balance                :", fmt(calc.balance_curve(LOSING, INIT)))
print(f"absolute drawdown      : {calc.absolute_drawdown(LOSING, INIT):.2f}  "
      f"= 1000 - 850, in CURRENCY not percent")
print()
print(f"balance drawdown   : {calc.balance_drawdown(PNL, INIT):.4%}")
print(f"balance dd (cash)  : {calc.balance_drawdown_abs(PNL, INIT):.2f}")
print(f"recovery factor    : {calc.recovery_factor(PNL, INIT):.4f}  = net profit / max DD")
print()
print("vectorbt's `value` already includes floating open positions, so MT5's EQUITY")
print("drawdown comes free and MT4's BALANCE drawdown — closed positions only — is the")
print("gap. That is the inversion, and it is why this needs a synthetic curve.")


# --- 10. Streaks: longest is not the same as best -----------------------------------------------------------------
header("10. Streak metrics — 'longest' and 'most profitable' are different")

for is_win, length, total in calc.runs(PNL):
    kind = "win " if is_win else "loss"
    print(f"  {kind} run of {length}: {total:+.2f}")
print()
print(f"max consecutive WINS, money (longest run) : {calc.max_consecutive_money(PNL, True):+.2f}")
print(f"maximal consecutive PROFIT (best run)     : {calc.maximal_consecutive(PNL, True):+.2f}")
print()
print("The longest winning run is the 2-trade one worth +70. The most profitable is the")
print("single +100 trade. MT reports both, and conflating them is easy.")
print()
print(f"average consecutive wins   : {calc.avg_consecutive(PNL, True):.4f} trades")
print(f"average consecutive losses : {calc.avg_consecutive(PNL, False):.4f} trades")


# --- 11. Z-score, and what it actually means -------------------------------------------------------------------------
header("11. Z-Score — a DEPENDENCY test, constantly misread")

streaky = np.array([10.0, 20, 30, -10, -20, -30, 15, 25, 35, -15, -25, -35])
alternating = np.array([10.0, -10, 20, -20, 30, -30, 15, -15, 25, -25, 35, -35])

print(f"clustered into streaks : Z = {calc.z_score(streaky):+.4f}")
print(f"strictly alternating   : Z = {calc.z_score(alternating):+.4f}")
print()
print("NEGATIVE Z means outcomes CLUSTER into streaks. POSITIVE means they alternate.")
print("It is a test of dependence between consecutive trades, not a measure of how good")
print("the system is. A great system can have any Z at all.")


# --- 12. Holding-period returns -----------------------------------------------------------------------------------------
header("12. AHPR and GHPR, in MT's convention")

trade_rets = np.array([0.05, -0.02, 0.03, 0.04, -0.01])
print("trade returns:", "[" + ", ".join(f"{v:g}" for v in trade_rets) + "]")
print(f"AHPR = mean(1+r)          : {calc.ahpr(trade_rets):.6f}")
print(f"GHPR = prod(1+r)^(1/n)    : {calc.ghpr(trade_rets):.6f}")
print()
print("Reported as 1.0034-style figures, matching MT, rather than as percentages.")
print("AHPR >= GHPR always; the gap is the volatility drag.")


# --- 13. What is NOT built ------------------------------------------------------------------------------------------------
header("13. Deliberately NOT rebuilt")

print("""
Already native in vectorbt 1.1.0 — duplicating them would only create drift:

  returns : Jensen's alpha (native `alpha` IS annualised Jensen's), Beta, Sharpe,
            Sortino, Calmar, Omega, Skew, Kurtosis, Tail Ratio, Common Sense Ratio,
            VaR, CVaR, up/down capture, and every rolling variant
  trades  : Profit Factor, Expected Payoff, Max Drawdown, Sharpe, Total Trades,
            Win Rate, Best/Worst Trade, Average Win/Loss, streak COUNTS

Not applicable at all: MT's Modelling Quality and Mismatched Charts Errors are
artifacts of tick interpolation and multi-timeframe data handling. vectorbt simulates
the bars it is given.
""")

if HAS_VBT:
    header("14. Live — the metrics in .stats()")
    import vectorbtx as vbt

    rets = pd.Series(RETS, index=pd.date_range("2023-01-01", periods=len(RETS), freq="D"))
    acc = rets.vbt.returns(freq="D", year_freq="365d")
    print(acc.stats())
else:
    print("\n[vectorbt not installed here — .stats() integration not executed]")
