"""I — portfolio of FX spread bets, SHARED cash, separate strategies.

All FX, so all carry and all Wednesday rolls — but each pair needs its OWN currency pair
declared, because the carry differential is specific to it. That is what the per-symbol
`instruments` dict is for.
"""
import pandas as pd
import vectorbtX as vbt

START, END = "2019-01-01", "2024-12-31"

# A JPY pair alongside dollar pairs on purpose: its pip is 0.01, not 0.0001. You do not
# state that — `point_size` is derived from the QUOTE currency, so one shared block of terms
# covers a mixed book. Setting point_size=0.0001 here by hand would make every USDJPY stop
# 100x too tight, and would warn.
PAIRS = {"EURUSD=X": ("EUR", "USD"), "GBPUSD=X": ("GBP", "USD"), "USDJPY=X": ("USD", "JPY")}
data = vbt.YFData.download(list(PAIRS), start=START, end=END)
close, open_ = data.get("Close"), data.get("Open")
high, low = data.get("High"), data.get("Low")

entries, exits = pd.DataFrame(index=close.index), pd.DataFrame(index=close.index)
for sym, (fast, slow) in {"EURUSD=X": (20, 60), "GBPUSD=X": (50, 200), "USDJPY=X": (10, 40)}.items():
    f, s = close[sym].rolling(fast).mean(), close[sym].rolling(slow).mean()
    entries[sym] = (f > s) & (f.shift(1) <= s.shift(1))
    exits[sym] = (f < s) & (f.shift(1) >= s.shift(1))

# Benchmark rates, for the overnight cost. Free key: https://fredaccount.stlouisfed.org/apikeys
# then `export FRED_API_KEY=...`. No key? Comment the first line, uncomment the second.
rates = vbt.x.FREDData.download_rates(["EUR", "USD", "GBP", "JPY"], start=START, end=END)
# rates = vbt.x.static_rate_table(["EUR", "USD", "GBP", "JPY"], close.index, {"EUR": 0.0400, "USD": 0.0533, "GBP": 0.0525, "JPY": 0.0010})

pf = vbt.x.PortfolioBacktest(
    close=close, open=open_, high=high, low=low,
    entries=entries, exits=exits,
    mode="spreadbet", asset_class="fx",
    min_size=1.0, increment=0.01,     # no point_size: derived per pair
    spread_points=0.9, leverage=30,
    instruments={
        sym: dict(base_currency=base, quote_currency=quote)
        for sym, (base, quote) in PAIRS.items()
    },
    risk_pct=0.005, stop_points=60,
    rates=rates,
    init_cash=25_000, cash_sharing=True, group_by=True, freq="D",
)
print("\n=== I — FX spread bets, shared cash ===")
print(pf.stats())
print("peak stakes:", {c: round(float(v), 2) for c, v in pf.x.user_size().abs().max().items()})
print("derived pip:", {c: pf.x.universe[c].filled().point_size for c in close.columns})
