"""B — single instrument, spread bet on a SHARE.

£ per point, no contract size, no commission (the fee is in the spread). Unlike an FX
spread bet this rolls on FRIDAY and pays financing on the full notional — carry does not
apply, because there is no currency pair underneath.

"""
import vectorbtx as vbt

SYMBOL, START, END = "VOD.L", "2019-01-01", "2024-12-31"

data = vbt.YFData.download(SYMBOL, start=START, end=END)
close, open_ = data.get("Close"), data.get("Open")
high, low = data.get("High"), data.get("Low")

fast, slow = vbt.MA.run(close, 50), vbt.MA.run(close, 200)
entries = fast.ma_crossed_above(slow)
exits = fast.ma_crossed_below(slow)

# Benchmark rates, for the overnight cost. Free key: https://fredaccount.stlouisfed.org/apikeys
# then `export FRED_API_KEY=...`. No key? Comment the first line, uncomment the second.
rates = vbt.x.FREDData.download_rates(["GBP"], start=START, end=END)
# rates = vbt.x.static_rate_table(["GBP"], close.index, {"GBP": 0.0525})

pf = vbt.x.SingleBacktest(
    close=close, open=open_, high=high, low=low,
    entries=entries, exits=exits,
    mode="spreadbet", asset_class="stocks",   # -> 5x cap, financing, FRIDAY roll
    point_size=0.01,                          # a penny of share price
    quote_currency="GBP",
    spread_points=2.0,
    risk_pct=0.01, stop_points=300,           # 300 points = £3.00 of share price
    leverage=5,
    rates=rates,
    # ⚠️ Sized against the £1/pt MINIMUM STAKE. 1% of £20,000 over a 300-point stop is
    # £0.67/pt, which rounds to zero and trades nothing — the library warns loudly rather
    # than silently placing no orders. A wide point stop on a low-priced share needs a
    # bigger account, or the minimum stake becomes the binding constraint.
    init_cash=50_000, freq="D",
)
print("\n=== B — spread bet on a share ===")
print(pf.stats())
print("peak stake:", f"£{float(pf.x.user_size().abs().max().iloc[0]):.2f}/pt")
print("roll day:", vbt.x.triple_day("stocks"))
print("financing basis:", vbt.x.financing_basis("spreadbet"))
