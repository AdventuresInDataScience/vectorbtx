"""C — single instrument, spread bet on FX.

Same product as B, different underlying — and that changes three things: the leverage tier
(30x), the roll day (WEDNESDAY, because spot settles T+2), and the overnight cost, which is
CARRY rather than financing. Carry needs both sides of the pair.

"""
import vectorbtx as vbt

SYMBOL, START, END = "EURUSD=X", "2019-01-01", "2024-12-31"

data = vbt.YFData.download(SYMBOL, start=START, end=END)
close, open_ = data.get("Close"), data.get("Open")
high, low = data.get("High"), data.get("Low")

fast, slow = vbt.MA.run(close, 50), vbt.MA.run(close, 200)
entries = fast.ma_crossed_above(slow)
exits = fast.ma_crossed_below(slow)

# Benchmark rates, for the overnight cost. Free key: https://fredaccount.stlouisfed.org/apikeys
# then `export FRED_API_KEY=...`. No key? Comment the first line, uncomment the second.
rates = vbt.x.FREDData.download_rates(["EUR", "USD"], start=START, end=END)
# rates = vbt.x.static_rate_table(["EUR", "USD"], close.index, {"EUR": 0.0400, "USD": 0.0533})

pf = vbt.x.SingleBacktest(
    close=close, open=open_, high=high, low=low,
    entries=entries, exits=exits,
    mode="spreadbet", asset_class="fx",      # -> 30x cap, CARRY, WEDNESDAY roll
    point_size=0.0001,                       # the pip
    base_currency="EUR", quote_currency="USD",
    min_size=1.0, increment=0.01,
    spread_points=0.9,
    risk_pct=0.01, stop_points=60, target_points=120,
    leverage=30,
    rates=rates,
    init_cash=10_000, freq="D",
)
print("\n=== C — spread bet on FX ===")
print(pf.stats())
print("peak stake:", f"£{float(pf.x.user_size().abs().max().iloc[0]):.2f}/pt")
print("roll day:", vbt.x.triple_day("fx"))
print("cost stack:", sorted(vbt.x.default_cost_stack("spreadbet", "fx")))
