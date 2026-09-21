"""Appendix — strategy 1 written against the explicit layer.

The same backtest as `01_ma_cross_spreadbet_fx.py`, built from the components instead of
through `vbt.x.SingleBacktest`. Kept so the relationship between the two is visible, and
because the explicit layer is what you want when you need to inspect or replace one step:

* to see exactly which `Order` fields an instrument resolves to — `universe.resolve(...)`
* to quantize with a rule the front door does not expose
* to reuse one resolved `Universe` across many backtests
* to build sizes some other way entirely and hand them straight to `from_signals`

For everything else, prefer the front door. This file is roughly three times the length
for the same result, and every extra line is a chance to get the order of operations
wrong.

Original description follows.

Strategy 1 — 50/200 MA cross on EURUSD, traded as a UK spread bet.

A complete, copy-pasteable strategy. Real data from Yahoo, real rates from FRED, £/point
staking, risk-based position sizing, point stops, spread cost, and carry charged on the
position because an FX position is long one currency and short another.

    pip install vectorbtX
    export FRED_API_KEY=...        # free from fred.stlouisfed.org; optional, see below
    python 01_ma_cross_spreadbet_fx.py
"""

import vectorbtx as vbt

# ----------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------
SYMBOL = "EURUSD=X"
START, END = "2019-01-01", "2024-12-31"

FAST, SLOW = 50, 200
STAKE_RISK = 0.01              # risk 1% of equity per trade
STOP_PIPS = 60                 # stop distance, in pips
TARGET_PIPS = 120              # 2:1 reward:risk
INIT_CASH = 10_000.0           # £

# ----------------------------------------------------------------------------------
# 1. The instrument
# ----------------------------------------------------------------------------------
# A spread bet has no contract size — the stake IS the size. point_size is the pip
# (0.0001), even though the quote carries five decimals; the fractional fifth digit is
# part of the spread and must not be truncated.
#
# Spread bets are denominated in the account currency by construction, so there is no
# FX conversion to do here whatever the underlying.
eurusd = vbt.x.Instrument(
    mode="spreadbet",
    asset_class="fx",
    point_size=0.0001,
    min_size=1.0,              # £1/pt minimum stake
    increment=0.01,            # £0.01/pt increments
    spread_points=0.9,         # typical retail EURUSD spread, in pips
    leverage=30.0,
    # commission is 0 for spread bets — the fee lives in the wider spread
)

universe = vbt.x.load_universe({SYMBOL: eurusd})

# ----------------------------------------------------------------------------------
# 2. Data
# ----------------------------------------------------------------------------------
# ⚠️ `.rename(SYMBOL)` is load-bearing. A SINGLE-symbol `Data.get()` returns a Series
# named after the FIELD ("Close"), not the symbol — only a multi-symbol download comes
# back symbol-named. The universe above is keyed by SYMBOL, so without the rename the
# resolver sees a column called "Close" it has no spec for and rejects the whole run.
data = vbt.YFData.download(SYMBOL, start=START, end=END)


def by_date(series):
    """Yahoo returns a tz-aware index; rate tables are tz-naive calendar dates.

    ⚠️ Mixing the two is silent, not loud. Reindexing a tz-naive rate table onto a
    tz-aware price index matches *nothing*, so every rate becomes NaN, every cost
    becomes NaN, and the run completes having placed no orders at all — a 0.00% return
    that looks like a strategy result rather than a data fault.
    """
    series = series.copy()
    series.index = series.index.tz_localize(None).normalize()
    return series


close = by_date(data.get("Close").rename(SYMBOL))
high = by_date(data.get("High").rename(SYMBOL))
low = by_date(data.get("Low").rename(SYMBOL))

# ----------------------------------------------------------------------------------
# 3. Rates — for the carry leg
# ----------------------------------------------------------------------------------
# ESTR for EUR, SOFR for USD. Cached on disk and refreshed on load, so re-running only
# fetches what is new. Without a FRED key, swap in the static line below.
rates = vbt.x.FREDData.download_rates(["EUR", "USD"], start=START, end=END)
# rates = vbt.x.static_rate_table(["EUR", "USD"], close.index, {"EUR": 0.04, "USD": 0.053})

# ----------------------------------------------------------------------------------
# 4. Signals
# ----------------------------------------------------------------------------------
fast_ma = vbt.MA.run(close, FAST)
slow_ma = vbt.MA.run(close, SLOW)

entries = fast_ma.ma_crossed_above(slow_ma)
exits = fast_ma.ma_crossed_below(slow_ma)

# ----------------------------------------------------------------------------------
# 5. Sizing and stops
# ----------------------------------------------------------------------------------
# One formula, every mode: risk / (stop_points x point_size), converted to user units.
# Here that comes back as a £/point stake.
stake = vbt.x.size_for_risk(
    risk_amount=INIT_CASH * STAKE_RISK,
    stop_distance_points=STOP_PIPS,
    spec=eurusd,
    price=close,
)
stake, stake_drift = vbt.x.quantize(stake, eurusd)      # £0.01 increments, £1 minimum

# The engine simulates in engine units, so convert once, here.
size = vbt.x.get_adapter(eurusd).to_engine_units(stake, eurusd, close)

# Native SL/TP takes FRACTIONS of price, so convert the pip distances.
sl_stop = vbt.x.points_to_price_fraction(STOP_PIPS, close, eurusd)
tp_stop = vbt.x.points_to_price_fraction(TARGET_PIPS, close, eurusd)

print(f"stake at the last bar : £{float(stake.iloc[-1]):.2f}/pt")
print(f"stop                  : {STOP_PIPS} pips = {float(sl_stop.iloc[-1]):.5f} of price")
print(f"risk per trade        : £{float(stake.iloc[-1]) * STOP_PIPS:.2f}")

# ----------------------------------------------------------------------------------
# 6. Simulate
# ----------------------------------------------------------------------------------
# universe.resolve() supplies size_granularity, min_size, fees, fixed_fees and slippage
# (spread and slippage summed into the one field the engine has), plus max_size from the
# leverage cap.
#
# ⚠️ **Leverage must be FUNDED, not merely capped — and the explicit route will not do it
# for you.** `resolve()` returns `max_size` so a position cannot exceed `equity x
# leverage`, but nothing in vectorbt *supplies* that leverage: `buy_nb` ends its cash
# calculation with `min(cash_limit, exec_state.cash)`, so a long is bounded by cash, full
# stop. Passing `init_cash=INIT_CASH` here means a 30x spread bet sized at £1.66/pt on a
# £10,000 account tries to commit ~£17,300 of notional against £10,000 and is refused —
# and because `min_size` is 10,000 engine units, it is refused *entirely* rather than
# partially filled. The backtest then completes with **zero trades and a 0.00% return**,
# which reads as a strategy result rather than a funding mistake.
#
# `vbt.x.PortfolioBacktest` does the two steps below for you. This is the single biggest
# reason to prefer the front door. See `vectorbtx/margin/buying_power.py`.
engine_cash, margin_loan = vbt.x.buying_power(universe, INIT_CASH, [SYMBOL])

pf = vbt.Portfolio.from_signals(
    close,
    entries=entries,
    exits=exits,
    size=size,
    high=high,
    low=low,
    sl_stop=sl_stop,
    tp_stop=tp_stop,
    init_cash=engine_cash,      # buying power: equity x leverage
    freq="D",
    **universe.resolve(close, rates=rates, equity=INIT_CASH),
)

# Record the loan, so `get_init_cash` subtracts it back out and every reported figure —
# cash, value, returns, each metric — is stated against the account's real £10,000 rather
# than the £300,000 the engine was handed.
pf = pf.replace(universe=universe, margin_loan=margin_loan)

# Carry, charged on the whole position regardless of leverage. This is the level-3 path,
# so it is opt-in — use it when holding overnight actually matters to the result.
#
# pf = vbt.Portfolio.from_holding_costs(
#     close, size=size, universe=universe, rates=rates, init_cash=INIT_CASH, freq="D",
# )

# ----------------------------------------------------------------------------------
# 7. Results
# ----------------------------------------------------------------------------------
print("\n=== Portfolio ===")
print(pf.stats())

print("\n=== Trades (MT4/MT5 parity set) ===")
print(pf.trades.stats())

print("\n=== Risk-adjusted returns ===")
print(pf.returns_stats())

def scalar(value):
    """Unwrap a one-column metric to a plain float.

    ⚠️ Needed twice over. These accessors return a bare float for a 1-D portfolio but a
    one-element Series for a 1-column 2-D one, and which you get depends on the shape of
    the inputs above. And since **pandas 3.0 removed `Series.__float__`**, `float(series)`
    no longer papers over the difference — it raises `TypeError` even for a single
    element. `.item()` is the replacement.
    """
    return value.item() if hasattr(value, "item") and hasattr(value, "index") else float(value)


print("\n=== vectorbtX extras ===")
print(f"quantization drift, mean : {scalar(stake_drift.abs().mean()):.4f} £/pt")
# `universe=` is optional now that the simulate step attaches it via `.replace(...)`, but
# passing it explicitly documents the dependency — and it IS required for any portfolio
# built the explicit way without that step, since `from_signals(**universe.resolve(...))`
# forwards the resolved Order kwargs and not the Universe itself.
print(f"minimum margin level     : {pf.x.min_margin_level(universe=universe)}")
print(f"Ulcer Index              : {scalar(pf.returns_acc.ulcer_index()):.4f}")
print(f"K-ratio (v3)             : {scalar(pf.returns_acc.k_ratio_v3()):.4f}")
print(f"Efficiency ratio         : {scalar(pf.returns_acc.snr()):.4f}")
print(f"Z-score                  : {scalar(pf.trades.z_score()):.4f}   "
      f"(negative = outcomes cluster into streaks)")

# `column=` is needed because funding the leverage makes this portfolio 2-D: `buying_power`
# returns init_cash per column (here `array([300000.])`), and a per-column init_cash is
# what tells vectorbt to wrap 2-D. One column, but still 2-D — and `pf.plot()` refuses to
# guess, with "Only one column is allowed. Use indexing or column argument."
pf.plot(column=SYMBOL).show()
