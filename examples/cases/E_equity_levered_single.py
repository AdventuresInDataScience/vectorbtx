"""E — single instrument, equity, LEVERAGED.

D with a margin account. Two things change and nothing else does: the engine is funded to
`equity x leverage`, and financing is charged on the DEBIT BALANCE — the amount actually
borrowed, not the whole position. That last point is what separates margin share dealing
from a CFD (case A), where the broker funds everything.

"""
import vectorbtx as vbt

SYMBOL, START, END = "MSFT", "2019-01-01", "2024-12-31"

data = vbt.YFData.download(SYMBOL, start=START, end=END)
close, open_ = data.get("Close"), data.get("Open")
high, low = data.get("High"), data.get("Low")

fast, slow = vbt.MA.run(close, 50), vbt.MA.run(close, 200)
entries = fast.ma_crossed_above(slow)
exits = fast.ma_crossed_below(slow)

# Benchmark rates, for the overnight cost. Free key: https://fredaccount.stlouisfed.org/apikeys
# then `export FRED_API_KEY=...`. No key? Comment the first line, uncomment the second.
rates = vbt.x.FREDData.download_rates(["USD"], start=START, end=END)
# rates = vbt.x.static_rate_table(["USD"], close.index, {"USD": 0.0533})

pf = vbt.x.SingleBacktest(
    close=close, open=open_, high=high, low=low,
    entries=entries, exits=exits,
    mode="equity", point_size=0.01,
    quote_currency="USD", cost_currency="USD",
    commission_pct=0.001, commission_min=1.0,
    spread_points=1.0,
    notional=80_000,                 # more than the cash: the loan funds the rest
    leverage=2.0,
    rates=rates,
    init_cash=50_000, freq="D",
)
print("\n=== E — equity, leveraged 2x ===")
print(pf.stats())
print("margin loan:", pf.margin_loan)
print("peak borrowed:", round(float(pf.x.borrowed_portion().max().iloc[0]), 2))
print("financing basis:", vbt.x.financing_basis("equity"))
