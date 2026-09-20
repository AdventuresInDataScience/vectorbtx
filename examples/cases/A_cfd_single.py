"""A — single instrument, CFD.

Lots x contract size. Financing on the FULL notional (the broker funds the whole position;
you borrow nothing), Friday weekend roll, commission charged.

"""
import vectorbtx as vbt

SYMBOL, START, END = "UKX", "2019-01-01", "2024-12-31"

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
    mode="cfd", asset_class="index",
    point_size=1.0, contract_size=1.0,      # 1 contract = £1 per index point
    quote_currency="GBP",
    commission_pct=0.0005, spread_points=1.0,
    stake=2.0,                              # 2 lots
    stop_points=80, target_points=160,
    leverage=10,                            # index CFD tier
    rates=rates,
    init_cash=25_000, freq="D",
)
print("\n=== A — CFD, single instrument ===")
print(pf.stats())
print("peak lots:", round(float(pf.x.user_size().abs().max().iloc[0]), 2))
print("peak borrowed:", round(float(pf.x.borrowed_portion().max().iloc[0]), 2))
