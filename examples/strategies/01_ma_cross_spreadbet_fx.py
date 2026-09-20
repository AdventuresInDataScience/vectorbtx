"""Strategy 1 — 50/200 MA cross on EURUSD, traded as a UK spread bet.

A complete, copy-pasteable strategy: £/point staking, risk-based sizing, point stops,
spread cost, and carry charged on the position because an FX position is long one
currency and short another.

    pip install vectorbtX
    export FRED_API_KEY=...        # free from fred.stlouisfed.org; optional, see below
    python 01_ma_cross_spreadbet_fx.py

Everything instrument-specific is an argument to one call. You never convert to "engine
units", never quantize by hand, and never splat a resolved argument dict. See
`appendix_explicit.py` for the same strategy written against the layer underneath, and
when you would want that instead.
"""

import vectorbtx as vbt

# ----------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------
SYMBOL = "EURUSD=X"
START, END = "2019-01-01", "2024-12-31"

FAST, SLOW = 50, 200
INIT_CASH = 10_000.0           # £

# ----------------------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------------------
data = vbt.YFData.download(SYMBOL, start=START, end=END)
close, open_ = data.get("Close"), data.get("Open")
high, low = data.get("High"), data.get("Low")

# ESTR for EUR, SOFR for USD. Cached on disk and refreshed on load, so re-running only
# fetches what is new. Without a FRED key, swap in the static line below.
rates = vbt.x.FREDData.download_rates(["EUR", "USD"], start=START, end=END)
# rates = vbt.x.static_rate_table(["EUR", "USD"], close.index, {"EUR": 0.04, "USD": 0.053})

# ----------------------------------------------------------------------------------
# 2. Signals
# ----------------------------------------------------------------------------------
fast_ma = vbt.MA.run(close, FAST)
slow_ma = vbt.MA.run(close, SLOW)

entries = fast_ma.ma_crossed_above(slow_ma)
exits = fast_ma.ma_crossed_below(slow_ma)

# ----------------------------------------------------------------------------------
# 3. Backtest — instrument, costs, sizing and stops in one call
# ----------------------------------------------------------------------------------
pf = vbt.x.SingleBacktest(
    close=close, open=open_, high=high, low=low,
    entries=entries,
    exits=exits,

    # --- the instrument ---
    # A spread bet has no contract size: the stake IS the size. `point_size` is the pip,
    # even though the quote carries five decimals — the fractional fifth digit is part of
    # the spread and is never truncated.
    mode="spreadbet",
    asset_class="fx",                 # -> 30x cap, carry, Wednesday triple day (T+2)
    point_size=0.0001,
    base_currency="EUR",              # the pair, so carry can use the differential
    quote_currency="USD",
    min_size=1.0,                     # £1/pt broker minimum
    increment=0.01,                   # £0.01/pt steps
    rounding="floor",

    # --- costs ---
    spread_points=0.9,                # full retail EURUSD gap in pips; half per side
    slippage=0.0002,
    # commission is forced to 0 in spreadbet mode — the fee lives in the wider spread

    # --- money management, in the units you actually trade in ---
    risk_pct=0.01,                    # 1% of init_cash per trade
    stop_points=60,                   # pips, converted internally to price fractions
    target_points=120,                # 2:1 reward:risk
    leverage=30,                      # funded, not just capped: buying power is 30x equity

    # --- holding costs ---
    # Supplying `rates` opts into the level-3 path, which charges carry per bar. Leave it
    # out and the backtest stays on the fast path with no holding costs at all.
    rates=rates,

    # --- account ---
    init_cash=INIT_CASH,
    freq="D",

    # --- straight passthrough to vectorbt ---
    upon_opposite_entry="reverse",
    seed=42,
)

# ----------------------------------------------------------------------------------
# 4. Results
# ----------------------------------------------------------------------------------
print("=== Portfolio ===")
print(pf.stats())

print("\n=== Trades (MT4/MT5 parity set) ===")
print(pf.trades.stats())

print("\n=== Risk-adjusted returns ===")
print(pf.returns_stats())

print("\n=== vectorbtX extras ===")
# Sizes come back in £/point, not engine units — the whole reason to use this layer.
print(f"peak stake            : £{float(pf.x.user_size().abs().max().iloc[0]):.2f}/pt")
print(f"units                 : {pf.x.unit_names()}")
print(f"minimum margin level  : {pf.x.min_margin_level()}")  # % of margin used
print(f"carry paid/received   : {pf.holding_costs.to_numpy().sum():,.2f}")
print(f"Ulcer Index           : {pf.returns_acc.ulcer_index()}")
print(f"K-ratio (v3)          : {pf.returns_acc.k_ratio_v3()}")
print(f"Efficiency ratio      : {pf.returns_acc.snr()}")
