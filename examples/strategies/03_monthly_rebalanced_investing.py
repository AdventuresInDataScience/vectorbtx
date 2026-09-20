"""Strategy 3 — a monthly-rebalanced global portfolio, in GBP.

Investing rather than trading: hold continuously, adjust weights on a calendar. You never
compute the buy/sell orders — `TargetPercent` derives them, with `call_seq="auto"`
ordering sells before buys so the sells fund the buys.

Four ETFs across two currencies, rebalanced monthly with a drift filter and a
minimum-trade filter so tiny corrections are not paid for.

    python 03_monthly_rebalanced_investing.py
"""

import vectorbtX as vbt

START, END = "2015-01-01", "2024-12-31"
INIT_CASH = 250_000.0
BASE = "GBP"

TARGET = {"VWRL.L": 0.40, "VUSA.L": 0.20, "SPY": 0.25, "AGG": 0.15}

# ----------------------------------------------------------------------------------
# 1. Universe — as a reusable object, since these definitions get shared
# ----------------------------------------------------------------------------------
# Investing is precisely the case where adj_close earns its keep: over a ten-year hold
# dividends are most of the equity return, and equity mode defaults to adj_close for
# exactly that reason. (Spread bets and CFDs default to `close` — an adjusted series
# rescales the whole history on every dividend, so the number of *points* between two
# dates would change with when the data was pulled.)
universe = vbt.x.load_universe({
    "VWRL.L": vbt.x.Instrument(mode="equity", point_size=0.01, quote_currency="GBP",
                               cost_currency="GBP", commission_flat=5.0),
    "VUSA.L": vbt.x.Instrument(mode="equity", point_size=0.01, quote_currency="GBP",
                               cost_currency="GBP", commission_flat=5.0),
    "SPY":    vbt.x.Instrument(mode="equity", point_size=0.01, quote_currency="USD",
                               cost_currency="USD", commission_pct=0.0005,
                               commission_min=1.0, allow_fractional=True),
    "AGG":    vbt.x.Instrument(mode="equity", point_size=0.01, quote_currency="USD",
                               cost_currency="USD", commission_pct=0.0005,
                               commission_min=1.0, allow_fractional=True),
})
print(universe.to_frame())      # eyeball the resolved defaults before trusting them

# ----------------------------------------------------------------------------------
# 2. Data, converted to base currency
# ----------------------------------------------------------------------------------
data = vbt.YFData.download(list(universe.columns), start=START, end=END,
                           missing_index="drop")
fx = vbt.YFData.download("GBPUSD=X", start=START, end=END)

# The universe already knows which columns are USD, so it supplies the currencies.
price = vbt.x.to_base_currency(data.get("Close"), base=BASE, fx=fx,
                               instruments=universe, fx_markup=0.005)
open_price = vbt.x.to_base_currency(data.get("Open"), base=BASE, fx=fx,
                                    instruments=universe, fx_markup=0.005)
high = vbt.x.to_base_currency(data.get("High"), base=BASE, fx=fx,
                              instruments=universe, fx_markup=0.005)
low = vbt.x.to_base_currency(data.get("Low"), base=BASE, fx=fx,
                             instruments=universe, fx_markup=0.005)

# ----------------------------------------------------------------------------------
# 3. Targets and schedule
# ----------------------------------------------------------------------------------
weights = price.copy()
for col, w in TARGET.items():
    weights[col] = w

# "Monthly, but only if something has drifted more than 2%" is just `calendar & threshold`.
# ⚠️ `drifted_weights` needs the schedule: without it drift accumulates from the first bar
# and grows without bound, so the threshold test degenerates to "always true" and the
# composed schedule silently collapses back to plain calendar rebalancing.
calendar = vbt.x.calendar_schedule(price.index, freq="M")
drifted = vbt.x.drifted_weights(weights, price, schedule=calendar)
schedule = calendar & vbt.x.threshold_schedule(drifted, weights, tolerance=0.02)

# ----------------------------------------------------------------------------------
# 4. Backtest
# ----------------------------------------------------------------------------------
# `weights` + `schedule` selects the rebalancing path. Targets are NaN off-schedule —
# never forward-filled, which would rebalance every bar, still run, still look plausible,
# and quietly accrue transaction costs while producing a suspiciously smooth curve.
pf = vbt.x.PortfolioBacktest(
    close=price, open=open_price, high=high, low=low,
    weights=weights,
    schedule=schedule,
    instruments=universe,
    base_currency=BASE, fx=fx,
    min_trade=0.005,               # skip corrections costing more than the drift they fix
    init_cash=INIT_CASH, freq="D",
)

# ----------------------------------------------------------------------------------
# 5. Results
# ----------------------------------------------------------------------------------
print("\n=== Portfolio ===")
print(pf.stats())

print("\n=== Risk-adjusted returns ===")
print(pf.returns_stats())

print("\n=== vectorbtX extras ===")
print(f"rebalances        : {int(schedule.sum())}")
print(f"mean |drift|      : {pf.drift.abs().mean().mean():.6f}")
print("final weights     :")
print((pf.asset_value() / pf.value(group_by=True).iloc[-1]).iloc[-1])
