"""D — single instrument, direct equity, UNLEVERAGED.

The simplest case in the library, and the baseline the others are judged against. Whole
shares, flat commission, no margin loan, no financing.

"""
import vectorbtX as vbt

SYMBOL, START, END = "MSFT", "2019-01-01", "2024-12-31"

data = vbt.YFData.download(SYMBOL, start=START, end=END)
close, open_ = data.get("Close"), data.get("Open")
high, low = data.get("High"), data.get("Low")

fast, slow = vbt.MA.run(close, 50), vbt.MA.run(close, 200)
entries = fast.ma_crossed_above(slow)
exits = fast.ma_crossed_below(slow)

pf = vbt.x.SingleBacktest(
    close=close, open=open_, high=high, low=low,
    entries=entries, exits=exits,
    mode="equity", point_size=0.01,
    quote_currency="USD", cost_currency="USD",
    commission_pct=0.001, commission_min=1.0,
    spread_points=1.0,
    notional=20_000,
    leverage=1.0,                    # no margin: cash only
    init_cash=50_000, freq="D",
)
print("\n=== D — direct equity, unleveraged ===")
print(pf.stats())
print("peak shares:", round(float(pf.x.user_size().abs().max().iloc[0]), 2))
print("margin loan:", pf.margin_loan)
