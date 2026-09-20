"""L — FX and equity spread bets together, POOLED cash, ONE shared strategy, separate terms.

The hardest mix, and the one most likely to be got wrong by hand. Every column is a spread
bet and every column shares the entry rule, but the underlying asset class changes the
economics beneath it:

    FX pairs      -> 30x cap, CARRY, WEDNESDAY roll, pip point size
    share/index   -> 5x/10x cap, FINANCING, FRIDAY roll, penny/point point size

None of that is stated per column below. It follows from `asset_class`, which is exactly the
point: get the asset class right and the cost model, roll day and leverage tier follow.

⚠️ Pooled cash across mixed leverage tiers funds the pot at the HIGHEST tier in the group
(30x here) and warns. Each column's own `max_size` still caps it individually, so the shares
cannot secretly run at 30x — but the group can put all of its capacity into the FX legs.
"""
import vectorbtx as vbt

START, END = "2019-01-01", "2024-12-31"

# One shared `mode`, four different underlyings. Note what is NOT stated per column:
# leverage tier, overnight cost, roll day, and — for the FX legs — the pip. All four follow
# from `asset_class` and the quote currency.
SPEC = {
    "EURUSD=X": dict(asset_class="fx", base_currency="EUR", quote_currency="USD",
                     spread_points=0.9),
    "USDJPY=X": dict(asset_class="fx", base_currency="USD", quote_currency="JPY",
                     spread_points=0.9),          # pip is 0.01 here, derived not stated
    "UKX":      dict(asset_class="index", point_size=1.0, quote_currency="GBP",
                     spread_points=1.0),
    "VOD.L":    dict(asset_class="stocks", point_size=0.01, quote_currency="GBP",
                     spread_points=2.0),
}
data = vbt.YFData.download(list(SPEC), start=START, end=END)
close, open_ = data.get("Close"), data.get("Open")
high, low = data.get("High"), data.get("Low")
fast, slow = vbt.MA.run(close, 50), vbt.MA.run(close, 200)
entries = fast.ma_crossed_above(slow)
exits = fast.ma_crossed_below(slow)      # the SAME rule for every column

# Benchmark rates, for the overnight cost. Free key: https://fredaccount.stlouisfed.org/apikeys
# then `export FRED_API_KEY=...`. No key? Comment the first line, uncomment the second.
rates = vbt.x.FREDData.download_rates(["EUR", "USD", "GBP", "JPY"], start=START, end=END)
# rates = vbt.x.static_rate_table(["EUR", "USD", "GBP", "JPY"], close.index,
#                                 {"EUR": 0.0400, "USD": 0.0533, "GBP": 0.0525, "JPY": 0.0010})

pf = vbt.x.PortfolioBacktest(
    close=close, open=open_, high=high, low=low,
    entries=entries, exits=exits,
    mode="spreadbet",                 # shared: every column is a spread bet
    min_size=1.0, increment=0.01,
    instruments=SPEC,                 # per column: what actually differs
    risk_pct=0.005, stop_points=50,
    rates=rates,
    init_cash=50_000, cash_sharing=True, group_by=True, freq="D",
)

universe = pf.x.universe
print("\n=== L — mixed FX and equity spread bets, pooled cash ===")
print(pf.stats())
print("\nWhat each column resolved to, from asset_class alone:")
print(f"{'':10s} {'pip':>8s} {'lev':>5s} {'roll':>5s}  overnight")
for c in close.columns:
    spec = universe[c].filled()
    overnight = sorted({"carry", "financing"} & spec.active_costs)
    print(f"{c:10s} {spec.point_size:>8g} {spec.leverage:>5g} {spec.triple_day:>5s}  {overnight}")
print("peak stakes:", {c: round(float(v), 2) for c, v in pf.x.user_size().abs().max().items()})
