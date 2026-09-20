"""G — portfolio of direct equities, SHARED cash, ONE strategy applied to all.

The common case, and the least code: compute the signal on the whole frame at once and pass
it straight in. Every column gets the same rule with the same parameters.
"""
import vectorbtX as vbt

START, END = "2019-01-01", "2024-12-31"

SYMBOLS = ["VOD.L", "BP.L", "HSBA.L", "SHEL.L"]
data = vbt.YFData.download(list(SYMBOLS), start=START, end=END)
close, open_ = data.get("Close"), data.get("Open")
high, low = data.get("High"), data.get("Low")

# One line, applied across every column — `ma_cross` is frame-aware.
fast, slow = vbt.MA.run(close, 50), vbt.MA.run(close, 200)
entries = fast.ma_crossed_above(slow)
exits = fast.ma_crossed_below(slow)

pf = vbt.x.PortfolioBacktest(
    close=close, open=open_, high=high, low=low,
    entries=entries, exits=exits,
    mode="equity", point_size=0.01,
    quote_currency="GBP", cost_currency="GBP",
    commission_flat=6.0, spread_points=1.0,
    notional=25_000,
    init_cash=100_000, cash_sharing=True, group_by=True, freq="D",
)
print("\n=== G — equities, shared cash, one shared strategy ===")
print(pf.stats())
print("trades per column:", dict(pf.trades.count(group_by=False)))
