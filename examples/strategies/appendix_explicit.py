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
data = vbt.YFData.download(SYMBOL, start=START, end=END)
close = data.get("Close")
high, low = data.get("High"), data.get("Low")

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
pf = vbt.Portfolio.from_signals(
    close,
    entries=entries,
    exits=exits,
    size=size,
    high=high,
    low=low,
    sl_stop=sl_stop,
    tp_stop=tp_stop,
    init_cash=INIT_CASH,
    freq="D",
    **universe.resolve(close, rates=rates, equity=INIT_CASH),
)

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

print("\n=== vectorbtX extras ===")
print(f"quantization drift, mean : {float(stake_drift.abs().mean()):.4f} £/pt")
print(f"minimum margin level     : {pf.x.min_margin_level()}")
print(f"Ulcer Index              : {pf.returns_acc.ulcer_index():.4f}")
print(f"K-ratio (v3)             : {pf.returns_acc.k_ratio_v3():.4f}")
print(f"Efficiency ratio         : {pf.returns_acc.snr():.4f}")
print(f"Z-score                  : {pf.trades.z_score():.4f}   "
      f"(negative = outcomes cluster into streaks)")

pf.plot().show()
