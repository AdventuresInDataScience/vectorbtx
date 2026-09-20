"""F — portfolio of direct equities, SHARED cash, SEPARATE strategies per instrument.

One cash pool, but each instrument trades its own signal. Entries and exits are just
DataFrames with one column per instrument, so "separate strategies" needs no special API —
you build the columns however you like and they travel together.

⚠️ Shared cash means the instruments COMPETE for it. `call_seq="auto"` is not in play here
(that is a `from_weights` concern), so within a bar the column order decides who gets funded
first when cash is short. If that matters, give them separate pools — case H.
"""
import pandas as pd
import vectorbtX as vbt

START, END = "2019-01-01", "2024-12-31"

SYMBOLS = ["VOD.L", "BP.L", "HSBA.L"]
data = vbt.YFData.download(list(SYMBOLS), start=START, end=END)
close, open_ = data.get("Close"), data.get("Open")
high, low = data.get("High"), data.get("Low")

# A different strategy per column: fast MA cross, slow MA cross, and mean reversion.
entries, exits = pd.DataFrame(index=close.index), pd.DataFrame(index=close.index)

f, s = close["VOD.L"].rolling(20).mean(), close["VOD.L"].rolling(60).mean()
entries["VOD.L"], exits["VOD.L"] = (f > s) & (f.shift(1) <= s.shift(1)), (f < s) & (f.shift(1) >= s.shift(1))

f, s = close["BP.L"].rolling(50).mean(), close["BP.L"].rolling(200).mean()
entries["BP.L"], exits["BP.L"] = (f > s) & (f.shift(1) <= s.shift(1)), (f < s) & (f.shift(1) >= s.shift(1))

z = (close["HSBA.L"] - close["HSBA.L"].rolling(60).mean()) / close["HSBA.L"].rolling(60).std()
entries["HSBA.L"], exits["HSBA.L"] = z < -1.5, z > 0.0

pf = vbt.x.PortfolioBacktest(
    close=close, open=open_, high=high, low=low,
    entries=entries, exits=exits,
    mode="equity", point_size=0.01,
    quote_currency="GBP", cost_currency="GBP",
    commission_flat=6.0, spread_points=1.0,
    notional=30_000,
    init_cash=100_000, cash_sharing=True, group_by=True, freq="D",
)
print("\n=== F — equities, shared cash, separate strategies ===")
print(pf.stats())
print("trades per column:", dict(pf.trades.count(group_by=False)))
