"""Strategy 2 — leveraged multi-stock momentum, per-instrument arguments.

Five stocks across two currencies and two brokers, each with its own commission
structure, leverage cap and minimum fill. This is the case the declarative universe
exists for: hand-building five parallel lists and keeping them aligned in column order is
exactly how CFD costs end up attached to an equity column with nothing raising.

    python 02_leveraged_stock_portfolio.py
"""

import vectorbtX as vbt

START, END = "2019-01-01", "2024-12-31"
SYMBOLS = ["VOD.L", "BP.L", "MSFT", "AAPL", "NVDA"]

LOOKBACK = 90                  # momentum lookback, bars
HOLD_TOP = 3                   # hold the top N by momentum
INIT_CASH = 100_000.0          # £
BASE = "GBP"

# ----------------------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------------------
data = vbt.YFData.download(SYMBOLS, start=START, end=END, missing_index="drop")
price_native = data.get("Close")
open_native = data.get("Open")
high_native, low_native = data.get("High"), data.get("Low")
volume = data.get("Volume")

fx = vbt.YFData.download("GBPUSD=X", start=START, end=END)
rates = vbt.x.FREDData.download_rates(["GBP", "USD"], start=START, end=END)

# ----------------------------------------------------------------------------------
# 2. Convert to base currency BEFORE computing signals
# ----------------------------------------------------------------------------------
# ⚠️ This line stays outside the backtest call on purpose. Ranking prices across
# currencies is meaningless otherwise, and you need to hold the converted series to
# compute signals on it. Converting internally for sizing while signals ran on native
# prices would put two different price series in one backtest.
#
# `fx` here is an ordinary GBPUSD price series — it is coerced to a rate table
# internally, with every reciprocal derived rather than fetched.
QUOTES = {"VOD.L": "GBP", "BP.L": "GBP", "MSFT": "USD", "AAPL": "USD", "NVDA": "USD"}

price = vbt.x.to_base_currency(
    price_native, base=BASE, fx=fx,
    quote_currency=QUOTES,
    fx_markup=0.005,           # broker's conversion spread, adverse on both legs
)
# The open must be converted with the SAME rates, or the fill price and the valuation
# price would sit in different currencies.
open_price = vbt.x.to_base_currency(open_native, base=BASE, fx=fx,
                                    quote_currency=QUOTES, fx_markup=0.005)
high = vbt.x.to_base_currency(high_native, base=BASE, fx=fx,
                              quote_currency=QUOTES, fx_markup=0.005)
low = vbt.x.to_base_currency(low_native, base=BASE, fx=fx,
                             quote_currency=QUOTES, fx_markup=0.005)

# ----------------------------------------------------------------------------------
# 3. Signals — hold the top N by momentum
# ----------------------------------------------------------------------------------
momentum = price.pct_change(LOOKBACK)
in_top = momentum.rank(axis=1, ascending=False) <= HOLD_TOP
entries = in_top & ~in_top.shift(1).fillna(False)
exits = ~in_top & in_top.shift(1).fillna(False)

# ----------------------------------------------------------------------------------
# 4. Backtest — shared terms, with per-symbol overrides
# ----------------------------------------------------------------------------------
# The shared block applies to every column; the `instruments` dict overrides per symbol.
# A per-symbol `mode` overrides the shared one, which is how one CFD sits in a book of
# equities without a second backtest.
pf = vbt.x.PortfolioBacktest(
    close=price, open=open_price, high=high, low=low,
    entries=entries,
    exits=exits,

    # --- shared across every column ---
    mode="equity",
    asset_class="stocks",
    point_size=0.01,
    slippage="auto", slippage_max=0.005, volume=volume,

    # --- per-symbol ---
    instruments={
        "VOD.L": dict(quote_currency="GBP", cost_currency="GBP",
                      commission_flat=10.0, spread_points=0.5, leverage=1.0),
        "BP.L":  dict(quote_currency="GBP", cost_currency="GBP",
                      commission_flat=10.0, spread_points=0.5, leverage=1.0),
        # ⚠️ commission_min is quoted in USD and stays in USD. `cost_currency` is what
        # tells the cost layer to convert it — percentage costs are FX-invariant, fixed
        # ones are not. This is the easiest thing here to get silently wrong.
        "MSFT":  dict(quote_currency="USD", cost_currency="USD",
                      commission_pct=0.001, commission_min=1.0,
                      spread_points=1.0, leverage=2.0, allow_fractional=True),
        "AAPL":  dict(quote_currency="USD", cost_currency="USD",
                      commission_pct=0.001, commission_min=1.0,
                      spread_points=1.0, leverage=2.0, allow_fractional=True),
        # A CFD in a book of equities: lots x contract size, financing on the FULL
        # notional (not the borrowed part — the broker funds the whole position), and a
        # Friday triple day because nothing settles T+2 here.
        "NVDA":  dict(mode="cfd", asset_class="stocks", contract_size=1.0,
                      quote_currency="USD", cost_currency="USD",
                      commission_pct=0.0005, spread_points=2.0, leverage=5.0),
    },

    # --- currency and money management ---
    base_currency=BASE, fx=fx,
    notional=25_000,

    # --- holding costs ---
    rates=rates,

    # --- account ---
    init_cash=INIT_CASH, cash_sharing=True, group_by=True, freq="D",
)

# ----------------------------------------------------------------------------------
# 5. Results
# ----------------------------------------------------------------------------------
print("=== Portfolio ===")
print(pf.stats())

print("\n=== Per instrument ===")
print(pf.trades.stats(group_by=False, agg_func=None))

print("\n=== vectorbtX extras ===")
print(f"units             : {pf.x.unit_names()}")
print("position sizes    :")
print(pf.x.user_size().iloc[-1])
print(f"borrowed (last)   : {pf.x.borrowed_portion().iloc[-1].sum():,.2f}")
print(f"financing total   : {pf.holding_costs.to_numpy().sum():,.2f}")
print(f"min margin level  : {pf.x.min_margin_level()}")
