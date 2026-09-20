# Getting your data

A backtest needs four kinds of input. They come from different places, and two of them are
**derived rather than downloaded** — which is the part that most often causes confusion.

| Input | Source | Notes |
|---|---|---|
| Prices — close, open, **high, low** | `vbt.YFData.download(...)` | All four are required |
| Benchmark interest rates | `vbt.x.FREDData.download_rates(...)` | Needs a free API key |
| FX rates | `vbt.YFData.download("GBPUSD=X")` | An ordinary price series |
| Tom-next carry | **derived** | Computed from two currencies' benchmark rates |

---

## Prices

All four series, always:

```python
data = vbt.YFData.download("EURUSD=X", start="2019-01-01", end="2024-12-31")
close, open_ = data.get("Close"), data.get("Open")
high, low = data.get("High"), data.get("Low")
```

Each one does a job, and omitting any of them makes the result optimistic:

- **close** — signals are computed on it, and the equity curve is valued at it.
- **open** — orders fill here, on the bar *after* the signal. Filling at the signal bar's
  close captures an overnight gap you could not have traded.
- **high / low** — stops are checked against them. Without them a stop can only be compared
  to the close, so it fires a bar late or not at all. They also give
  `pf.x.intrabar_drawdown()`, which is routinely a third larger than the close-to-close
  figure, because an excursion that happened and recovered inside a bar is invisible to a
  close-valued equity curve.

All four are required, and all four are **keyword-only**:

```python
pf = vbt.x.PortfolioBacktest(close=close, open=open_, high=high, low=low, ...)
```

There is no default for any of them, because every available default flatters the backtest —
and naming them prevents the silent transposition that four same-shaped frames in a row
would otherwise invite.

For several instruments, pass a list. Use `missing_index="drop"` so a holiday in one market
does not leave NaN rows in the others:

```python
data = vbt.YFData.download(["VOD.L", "MSFT"], start=START, end=END, missing_index="drop")
```

---

## Interest rates

`FREDData` fetches **benchmark interest rates only** — SOFR for USD, ESTR for EUR, SONIA for
GBP, and so on. It does not fetch FX, and it does not fetch carry.

You need these only if you want holding costs. Pass `rates=` to a backtest and financing or
carry is charged; leave it out and no holding costs are applied at all.

### Getting a key

Free, instant, no card: **<https://fredaccount.stlouisfed.org/apikeys>**. Then either

```bash
export FRED_API_KEY=your_key_here          # picked up automatically
```

That works, but an exported variable is easy to forget and easy to leak into a shell history
or a notebook screenshot. For anything you will re-run, keep it in a **`.env` file** that is
never committed:

```bash
# .env  —  add this file to .gitignore
FRED_API_KEY=your_key_here
```

```python
from dotenv import load_dotenv        # pip install python-dotenv

load_dotenv()                         # reads .env into the environment
rates = vbt.x.FREDData.download_rates(["GBP", "USD"], start=START, end=END)
```

Without the extra dependency, the standard library is enough:

```python
import os
from pathlib import Path

for line in Path(".env").read_text().splitlines():
    if line.strip() and not line.startswith("#"):
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())
```

Or keep it alongside other settings in a config file, which is worth doing once you have
more than one secret:

```python
# config.toml
# [data]
# fred_api_key = "your_key_here"
import tomllib

config = tomllib.loads(Path("config.toml").read_text())
rates = vbt.x.FREDData.download_rates(["GBP", "USD"], start=START, end=END,
                                      api_key=config["data"]["fred_api_key"])
```

⚠️ Whichever you choose, **add the file to `.gitignore` before you put a key in it.** A key
committed once is in the history for good, and FRED keys are tied to your account.

Series are cached on disk under `$XDG_CACHE_HOME` (or `~/.cache`) and refreshed
incrementally, so re-running a script only fetches what is new.

### Without a key

Every example ships with this fallback, and it is a legitimate choice — rates move slowly,
and a flat rate over a five-year backtest is usually a smaller error than the uncertainty in
the broker's markup:

```python
rates = vbt.x.static_rate_table(["GBP", "USD"], close.index,
                                {"GBP": 0.0525, "USD": 0.0533})
```

⚠️ A flat rate misses regime changes. A backtest spanning 2021–2024 held at 5.25% charges
2021 as though it were 2023, which materially overstates the cost of holding through the
zero-rate period. Use real rates for anything spanning a rate cycle.

---

## FX rates

An FX rate is just a price series, so it comes from the same place as prices:

```python
fx = vbt.YFData.download("GBPUSD=X", start=START, end=END)
```

Pass it straight to `to_base_currency` or to a backtest as `fx=`; it is coerced to an
`FXTable` internally. For several currencies, build the table from **one canonical direction
per pair** and let the reciprocals be derived:

```python
gbpusd = vbt.YFData.download("GBPUSD=X", start=START, end=END).get("Close")
fx = vbt.x.FXTable.from_canonical({"USD": 1.0 / gbpusd}, hub="GBP")
```

⚠️ **Never fetch both directions of a pair.** Downloading GBPUSD and USDGBP separately gives
you two series captured at different timestamps and different sides of the spread, so
`value(A,B) × value(B,A) ≠ 1` — and a round trip through them manufactures P&L out of
nothing. Deriving the reciprocal makes the identity hold exactly.

### Converting

Convert **both** close and open, with the same table. They are the valuation price and the
fill price of the same position; leaving one in native currency puts two currencies in one
backtest.

```python
close = vbt.x.to_base_currency(close_native, base="GBP", fx=fx,
                               quote_currency={"MSFT": "USD"}, fx_markup=0.005)
open_ = vbt.x.to_base_currency(open_native, base="GBP", fx=fx,
                               quote_currency={"MSFT": "USD"}, fx_markup=0.005)
```

Convert **before** computing signals. Any signal derived from price — momentum, volatility,
ranking across instruments — is meaningless across unconverted currencies.

If you already have a `Universe`, it knows which columns are foreign, so pass it instead of
restating the currencies:

```python
close = vbt.x.to_base_currency(close_native, base="GBP", fx=fx, instruments=universe)
```

---

## Carry

Nothing to fetch. The tom-next differential is **computed** from the two currencies'
benchmark rates:

```python
diff = vbt.x.tom_next_differential(base_rate=0.05, quote_rate=0.02,
                                   markup=0.01, is_long=True)
```

So a carry-bearing backtest — any FX spread bet, FX CFD or FX spot position — needs
`rates=` covering **both** currencies of the pair, and the instrument must name them:

```python
pf = vbt.x.SingleBacktest(
    close=close, open=open_, high=high, low=low,
    mode="spreadbet", asset_class="fx",
    base_currency="EUR", quote_currency="USD",     # both, or carry resolves to zero
    rates=vbt.x.FREDData.download_rates(["EUR", "USD"], start=START, end=END),
    ...
)
```

Omit either currency and carry silently computes as zero. vectorbtX warns when it detects
this, but the warning is easy to miss in a long run.

---

## A note on FX pips

The pip is **derived from the quote currency**, so you do not state `point_size` per pair:

| Quote currency | Pip | Example |
|---|---|---|
| JPY, KRW, HUF, CLP, IDR | `0.01` | USDJPY 150.25 → 150.26 |
| everything else | `0.0001` | EURUSD 1.0925 → 1.0926 |

This is what lets a mixed book share one block of terms:

```python
pf = vbt.x.PortfolioBacktest(
    close=close, open=open_, high=high, low=low,
    mode="spreadbet", asset_class="fx",       # no point_size — derived per pair
    instruments={
        "EURUSD=X": dict(base_currency="EUR", quote_currency="USD"),
        "USDJPY=X": dict(base_currency="USD", quote_currency="JPY"),
    },
    stop_points=30, ...
)
```

⚠️ Setting `point_size=0.0001` by hand on a JPY pair makes a 30-pip stop 0.30 points wide
instead of 30 — a stop **100× tighter** than intended, which still runs, produces *more*
trades rather than fewer, and reports risk that is wrong by two orders of magnitude.
vectorbtX warns if you contradict the convention, but the simplest defence is to omit
`point_size` for FX and let it be derived.

---

## Universes from a file

Once instruments differ from one another, keeping their terms in code means maintaining
parallel dictionaries and remembering which broker charges what. Put them in a CSV or Excel
sheet instead — one row per instrument, `mode` per row, and every `Instrument` field
available as a column:

```csv
name,mode,asset_class,point_size,contract_size,quote_currency,cost_currency,commission_pct,commission_flat,commission_min,spread_points,leverage,allow_fractional,base_currency
VOD.L,equity,stocks,0.01,,GBP,GBP,,6.0,,1.0,1.0,,
BP.L,equity,stocks,0.01,,GBP,GBP,,6.0,,1.0,1.0,,
MSFT,equity,stocks,0.01,,USD,USD,0.001,,1.0,1.0,2.0,true,
NVDA,cfd,stocks,0.01,1.0,USD,USD,0.0005,,,2.0,5.0,,
UKX,spreadbet,index,1.0,,GBP,,,,,1.0,10.0,,
EURUSD=X,spreadbet,fx,0.0001,,USD,,,,,0.9,30.0,,EUR
```

```python
universe = vbt.x.load_universe("universe.csv")     # or .xlsx, needs openpyxl
print(universe.to_frame())                         # what the blanks resolved to
```

Read the rows: four modes, three currencies, two commission structures, and a spread bet
that names its currency pair so carry can be computed. None of that needs a line of Python.

**Blank means "use the mode's default"**, not zero. `UKX` has no commission because spread
bets have none; `MSFT` has no `commission_flat` because it is charged a percentage with a
minimum. `universe.to_frame()` prints the resolved values so you can check the defaults
before trusting them — worth doing once per file.

⚠️ **Column names must match `Instrument` field names exactly**, and unknown columns raise
rather than being ignored. A typo silently dropped would leave an instrument on its default
terms, which is exactly the kind of thing that produces a plausible wrong answer.

Then pass it straight in, and the per-symbol dictionaries disappear:

```python
data = vbt.YFData.download(list(universe.columns), start=START, end=END,
                           missing_index="drop")

pf = vbt.x.PortfolioBacktest(
    close=data.get("Close"), open=data.get("Open"),
    high=data.get("High"), low=data.get("Low"),
    entries=entries, exits=exits,
    instruments=universe,
    notional=25_000, init_cash=250_000, cash_sharing=True, group_by=True, freq="D",
)
```

A `Universe` is reusable and comparable, so the same file can drive several backtests and
`universe_a == universe_b` tells you whether two runs used identical terms.

---

## Prices from a file

Yahoo is convenient, not compulsory. Anything that lands in a DataFrame with a
`DatetimeIndex` works — a broker export, a vendor CSV, a database query, a Parquet file.

```python
import pandas as pd

# One CSV per field, columns = instruments
close = pd.read_csv("prices/close.csv", index_col=0, parse_dates=True)
open_ = pd.read_csv("prices/open.csv", index_col=0, parse_dates=True)
high = pd.read_csv("prices/high.csv", index_col=0, parse_dates=True)
low = pd.read_csv("prices/low.csv", index_col=0, parse_dates=True)
```

A long-format file — one row per date/symbol/field, which is what most vendors ship — needs
a pivot first:

```python
raw = pd.read_csv("prices.csv", parse_dates=["date"])
# columns: date, symbol, open, high, low, close, volume
close = raw.pivot(index="date", columns="symbol", values="close")
open_ = raw.pivot(index="date", columns="symbol", values="open")
high = raw.pivot(index="date", columns="symbol", values="high")
low = raw.pivot(index="date", columns="symbol", values="low")
```

Three things to check before backtesting on your own data, because each produces a
plausible-looking wrong answer rather than an error:

1. **All four frames share one index and one column order.** vectorbtX reindexes to the
   close, so a misaligned open silently becomes NaN rather than raising.
2. **`H >= C >= L` holds on every bar.** vectorbtX warns when it does not, which usually
   means the fields were adjusted on different schedules. Stops would then fire at prices
   that never traded.
3. **Adjusted or unadjusted, consistently.** Equity mode defaults to adjusted close, because
   over a long hold dividends are most of the return. Spread bets and CFDs default to
   unadjusted, because an adjusted series rescales the whole history on every dividend — so
   the number of *points* between two dates would change depending on when you pulled the
   file. Never mix the two in one backtest.

To go through vectorbt's own machinery instead — which gets you its alignment and resampling
helpers — wrap the frames in a `Data` object:

```python
data = vbt.Data.from_data({"Close": close, "Open": open_, "High": high, "Low": low})
```
