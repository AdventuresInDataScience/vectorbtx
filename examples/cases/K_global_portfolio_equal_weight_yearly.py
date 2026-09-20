"""K — a MULTI-COUNTRY equity portfolio, equal-weighted, rebalanced yearly.

J plus currencies. Four markets in four currencies, held in GBP.

⚠️ Both close AND open must be converted, with the same rates. They are the valuation price
and the fill price of the same position; leaving one in native currency puts two currencies
in one backtest and every number after it is wrong.

⚠️ Convert BEFORE computing weights. Equal weight is currency-agnostic here, but any
weighting derived from price — momentum, volatility, market cap — is meaningless across
unconverted currencies.
"""
import vectorbtx as vbt

START, END = "2019-01-01", "2024-12-31"

QUOTES = {"ULVR.L": "GBP", "MSFT": "USD", "SAP.DE": "EUR", "7203.T": "JPY"}
data = vbt.YFData.download(list(QUOTES), start=START, end=END)
close_native, open_native = data.get("Close"), data.get("Open")
high_native, low_native = data.get("High"), data.get("Low")

# One canonical direction per pair; every reciprocal and cross is DERIVED, never fetched.
fx = vbt.x.FXTable.from_canonical(
    {"USD": 1.0 / vbt.YFData.download("GBPUSD=X", start=START, end=END).get("Close"),
     "EUR": 1.0 / vbt.YFData.download("GBPUSD=X", start=START, end=END).get("Close") * 1.08,
     "JPY": 1.0 / vbt.YFData.download("GBPUSD=X", start=START, end=END).get("Close") / 130.0},
    hub="GBP",
)

close = vbt.x.to_base_currency(close_native, base="GBP", fx=fx,
                               quote_currency=QUOTES, fx_markup=0.005)
open_ = vbt.x.to_base_currency(open_native, base="GBP", fx=fx,
                               quote_currency=QUOTES, fx_markup=0.005)
# All four series convert with the SAME table. Converting some and not others would put two
# currencies in one backtest; converting with different rates would break H >= C >= L.
high = vbt.x.to_base_currency(high_native, base="GBP", fx=fx,
                              quote_currency=QUOTES, fx_markup=0.005)
low = vbt.x.to_base_currency(low_native, base="GBP", fx=fx,
                             quote_currency=QUOTES, fx_markup=0.005)

weights = vbt.x.equal_weight(close.columns, close.index)
schedule = vbt.x.calendar_schedule(close.index, freq="A")

pf = vbt.x.PortfolioBacktest(
    close=close, open=open_, high=high, low=low,
    weights=weights, schedule=schedule,
    mode="equity", point_size=0.01,
    instruments={sym: dict(quote_currency=ccy, cost_currency=ccy,
                           commission_pct=0.0005, commission_min=1.0,
                           allow_fractional=True)
                 for sym, ccy in QUOTES.items()},
    base_currency="GBP", fx=fx,
    min_trade=0.01,
    init_cash=500_000, freq="D",
)
print("\n=== K — global portfolio, equal weight, yearly rebalance ===")
print(pf.stats())
print("rebalances:", int(schedule.sum()))
