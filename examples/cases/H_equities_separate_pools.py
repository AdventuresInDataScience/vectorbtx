"""H — portfolio of direct equities, DISTINCT cash pools, distinct strategies and terms.

Each instrument gets its own cash, its own signal and its own broker terms. That is simply
`cash_sharing=False`, which makes `init_cash` per column rather than per group — so the
columns cannot compete for funding, and one blowing up does not starve the others.

⚠️ Per-column `init_cash` also means per-column `leverage` funds each pool separately. Under
shared cash a single pot is funded at the maximum leverage in the group; here each column is
funded at its own.
"""
import numpy as np
import pandas as pd
import vectorbtx as vbt

START, END = "2019-01-01", "2024-12-31"

SYMBOLS = ["VOD.L", "MSFT", "NVDA"]
data = vbt.YFData.download(list(SYMBOLS), start=START, end=END)
close, open_ = data.get("Close"), data.get("Open")
high, low = data.get("High"), data.get("Low")

entries, exits = pd.DataFrame(index=close.index), pd.DataFrame(index=close.index)
for sym, (fast, slow) in {"VOD.L": (20, 60), "MSFT": (50, 200), "NVDA": (10, 40)}.items():
    f, s = close[sym].rolling(fast).mean(), close[sym].rolling(slow).mean()
    entries[sym] = (f > s) & (f.shift(1) <= s.shift(1))
    exits[sym] = (f < s) & (f.shift(1) >= s.shift(1))

# Benchmark rates, for the overnight cost. Free key: https://fredaccount.stlouisfed.org/apikeys
# then `export FRED_API_KEY=...`. No key? Comment the first line, uncomment the second.
rates = vbt.x.FREDData.download_rates(["GBP", "USD"], start=START, end=END)
# rates = vbt.x.static_rate_table(["GBP", "USD"], close.index, {"GBP": 0.0525, "USD": 0.0533})

pf = vbt.x.PortfolioBacktest(
    close=close, open=open_, high=high, low=low,
    entries=entries, exits=exits,
    mode="equity", point_size=0.01,
    instruments={
        "VOD.L": dict(quote_currency="GBP", cost_currency="GBP",
                      commission_flat=6.0, spread_points=1.0, leverage=1.0),
        "MSFT":  dict(quote_currency="USD", cost_currency="USD",
                      commission_pct=0.001, commission_min=1.0,
                      spread_points=1.0, leverage=2.0, allow_fractional=True),
        "NVDA":  dict(quote_currency="USD", cost_currency="USD",
                      commission_pct=0.001, commission_min=1.0,
                      spread_points=2.0, leverage=2.0, allow_fractional=True),
    },
    notional=15_000,
    rates=rates,
    # One pool each. `init_cash` is per COLUMN when cash is not shared.
    init_cash=np.array([40_000.0, 40_000.0, 20_000.0]),
    cash_sharing=False, freq="D",
)
print("\n=== H — equities, separate cash pools ===")
print(pf.stats())
print("init cash / column:", list(np.ravel(np.asarray(pf.init_cash))))
print("margin loan:", list(np.ravel(np.asarray(pf.margin_loan))))
print("final value:", [round(v, 2) for v in np.ravel(np.asarray(pf.value()))[-3:]])
