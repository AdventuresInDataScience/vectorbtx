"""J — a single-country equity portfolio, rebalanced to set weights every 6 months.

Investing, not trading: hold continuously and correct the weights on a calendar. No signals
at all — `weights` + `schedule` selects the rebalancing path, and TargetPercent derives the
orders.

⚠️ `calendar_schedule(index, "6M")` is genuinely six-monthly. Pandas' `to_period` discards a
leading multiplier, so a naive implementation would give you a MONTHLY mask — six times the
rebalances and six times the costs, still running and still looking plausible.
"""
import vectorbtX as vbt

START, END = "2019-01-01", "2024-12-31"

TARGET = {"AZN.L": 0.30, "SHEL.L": 0.25, "HSBA.L": 0.25, "ULVR.L": 0.20}
data = vbt.YFData.download(list(TARGET), start=START, end=END)
close, open_ = data.get("Close"), data.get("Open")
high, low = data.get("High"), data.get("Low")

weights = close.copy()
for col, w in TARGET.items():
    weights[col] = w

schedule = vbt.x.calendar_schedule(close.index, freq="6M")

pf = vbt.x.PortfolioBacktest(
    close=close, open=open_, high=high, low=low,
    weights=weights, schedule=schedule,
    mode="equity", point_size=0.01,
    quote_currency="GBP", cost_currency="GBP",
    commission_flat=6.0, spread_points=1.0,
    min_trade=0.01,                 # skip corrections smaller than the cost of making them
    init_cash=250_000, freq="D",
)
print("\n=== J — UK portfolio, 6-monthly rebalance ===")
print(pf.stats())
print("rebalances:", int(schedule.sum()))
print("mean |drift|:", round(float(pf.drift.abs().mean().mean()), 6))
