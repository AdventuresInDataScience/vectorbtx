# Front-door API — the agreed design

The specification for **Section N** of `PLAN.md`, which is **designed but not implemented**.

`vbt.x.SingleBacktest` and `vbt.x.PortfolioBacktest`: one call, every argument in it,
returns a `Portfolio`. Every code block below is complete and runnable top to bottom —
nothing is assumed to exist from a previous block.

Section N of the plan holds the build requirements and acceptance criteria; this document
holds the worked examples.

---

## 1. `mode` is the switch

It decides which other arguments are legal, required or rejected.

```
mode absent   ──►  plain vectorbt passthrough. Use plain vectorbt instead.
mode="equity"     ──►  shares. commission flat/pct/min, fractional shares, margin
mode="spreadbet"  ──►  £/point. no contract size, no commission, cost in the spread
mode="cfd"        ──►  lots x contract size. commission, financing, carry if fx
mode="fx_spot"    ──►  lots x contract size. pips, carry regardless of leverage
```

| Argument | `equity` | `spreadbet` | `cfd` | `fx_spot` |
|---|---|---|---|---|
| `point_size` | **required** | **required** | **required** | **required** (the pip) |
| `contract_size` | rejected | **rejected** — the stake is the size | **required** | **required** |
| `asset_class` | optional | **drives leverage + carry** | **drives leverage + carry** | fx |
| `commission_pct`/`_flat`/`_min` | yes | **forced to 0** | yes | yes |
| `spread_points` | yes | **primary cost** | yes | **primary cost** |
| `allow_fractional` | **yes** | rejected | rejected | rejected |
| `quote_currency` | yes | n/a | yes | **required** |
| `base_currency` (pair side) | rejected | rejected | rejected | **required** |
| `leverage` default | 2× | 5/10/30× by asset class | 5/10/30× by asset class | 2× |
| `carry` | no | fx asset class only | fx asset class only | **always** |
| `financing` | yes | yes | yes | no |
| `borrow_rate` (a short's borrow fee) | **yes** | rejected — funding covers the short | rejected — funding covers the short | rejected |

The `financing` row is the cost module. The `financing=` *argument* is reserved: financing
is always the benchmark from `rates=` plus `financing_markup=`, for every instrument alike,
and the argument resolves to `"auto"`, which means exactly that. It is kept for a possible
future per-instrument cost structure, which professional desks negotiate and retail
accounts do not see; a number passed to it is not used, and warns.
| `price_field` default | `adj_close` | `close` | `close` | `close` |

Rejected cells raise at construction. A spread bet with a `contract_size` is not a warning
case, it is a misunderstanding of the product.

---

## 2. Single instrument — spread bet on EURUSD

```python
import vectorbtx as vbt

START, END = "2019-01-01", "2024-12-31"

# ---- data ----
data = vbt.YFData.download("EURUSD=X", start=START, end=END)
price = data.get("Close")
high = data.get("High")
low = data.get("Low")

rates = vbt.x.FREDData.download_rates(["EUR", "USD"], start=START, end=END)

# ---- signals ----
fast = vbt.MA.run(price, 50)
slow = vbt.MA.run(price, 200)
entries = fast.ma_crossed_above(slow)
exits = fast.ma_crossed_below(slow)

# ---- backtest ----
pf = vbt.x.SingleBacktest(
    price,
    entries=entries,
    exits=exits,
    high=high, low=low,                      # so stops can fire intrabar

    # instrument
    mode="spreadbet",
    asset_class="fx",                        # -> 30x leverage cap, carry applies
    point_size=0.0001,                       # the pip; quotes keep 5 decimals
    min_size=1.0,                            # £1/pt broker minimum
    increment=0.01,                          # £0.01/pt steps
    rounding="floor",

    # costs
    spread_points=0.9,                       # full bid/ask gap, half per side
    slippage=0.0002,

    # money management
    risk_pct=0.01,                           # 1% of init_cash per trade
    stop_points=60,
    target_points=120,
    leverage=30,

    # holding costs — spreadbet/fx defaults already include carry and financing
    financing="auto",                        # benchmark + 3% markup, adverse both ways
    rates=rates,
    triple_day="WED",

    # account
    init_cash=10_000,
    freq="D",

    # straight passthrough to vectorbt
    accumulate=False,
    upon_opposite_entry="reverse",
    seed=42,
)

# ---- results ----
print(pf.stats())
print(pf.trades.stats())
print(pf.returns_stats())
print(pf.x.user_size().iloc[-1])             # "£1.66/pt", not 16600 engine units
```

---

## 3. Portfolio, form 1 — flat, one set of terms for every column

Five UK equities, same broker, same terms.

```python
import vectorbtx as vbt

START, END = "2019-01-01", "2024-12-31"
SYMBOLS = ["VOD.L", "BP.L", "AZN.L", "SHEL.L", "HSBA.L"]

# ---- data ----
data = vbt.YFData.download(SYMBOLS, start=START, end=END, missing_index="drop")
price = data.get("Close")
volume = data.get("Volume")

# ---- signals: hold the top 3 by 90-day momentum ----
momentum = price.pct_change(90)
in_top = momentum.rank(axis=1, ascending=False) <= 3
entries = in_top & ~in_top.shift(1).fillna(False)
exits = ~in_top & in_top.shift(1).fillna(False)

# ---- backtest ----
pf = vbt.x.PortfolioBacktest(
    price,
    entries=entries,
    exits=exits,

    # instrument — one set, applied to all five columns
    mode="equity",
    asset_class="stocks",
    point_size=0.01,
    quote_currency="GBP",
    cost_currency="GBP",
    commission_flat=10.0,
    spread_points=0.5,
    slippage="auto", slippage_max=0.005, slippage_coef=1.0, volume=volume,
    allow_fractional=False,
    leverage=1.0,

    # money management
    notional=20_000,
    min_trade=0.005,

    # account
    init_cash=100_000,
    cash_sharing=True,
    freq="D",
)

print(pf.stats())
```

---

## 4. Portfolio, form 2 — dict of per-symbol overrides on a shared base

Two currencies, three cost structures. A per-symbol `mode` overrides the shared one, which
is how one CFD sits in a book of equities.

```python
import vectorbtx as vbt

START, END = "2019-01-01", "2024-12-31"
SYMBOLS = ["VOD.L", "BP.L", "MSFT", "AAPL", "NVDA"]

# ---- data ----
data = vbt.YFData.download(SYMBOLS, start=START, end=END, missing_index="drop")
price_native = data.get("Close")
volume = data.get("Volume")

fx = vbt.YFData.download("GBPUSD=X", start=START, end=END)
rates = vbt.x.FREDData.download_rates(["GBP", "USD"], start=START, end=END)

# ---- convert USD names to GBP before computing signals ----
# Ranking prices across currencies is meaningless otherwise.
price = vbt.x.to_base_currency(
    price_native, base="GBP", fx=fx,
    quote_currency={"MSFT": "USD", "AAPL": "USD", "NVDA": "USD"},
    fx_markup=0.005,
)

# ---- signals ----
momentum = price.pct_change(90)
in_top = momentum.rank(axis=1, ascending=False) <= 3
entries = in_top & ~in_top.shift(1).fillna(False)
exits = ~in_top & in_top.shift(1).fillna(False)

# ---- backtest ----
pf = vbt.x.PortfolioBacktest(
    price,
    entries=entries,
    exits=exits,

    # shared defaults for every column
    mode="equity",
    asset_class="stocks",
    point_size=0.01,
    slippage="auto", slippage_max=0.005, volume=volume,

    # per-symbol overrides
    instruments={
        "VOD.L": dict(quote_currency="GBP", cost_currency="GBP",
                      commission_flat=10.0, spread_points=0.5, leverage=1.0),
        "BP.L":  dict(quote_currency="GBP", cost_currency="GBP",
                      commission_flat=10.0, spread_points=0.5, leverage=1.0),
        "MSFT":  dict(quote_currency="USD", cost_currency="USD",
                      commission_pct=0.001, commission_min=1.0,   # $1 min, converted
                      spread_points=1.0, leverage=2.0, allow_fractional=True),
        "AAPL":  dict(quote_currency="USD", cost_currency="USD",
                      commission_pct=0.001, commission_min=1.0,
                      spread_points=1.0, leverage=2.0, allow_fractional=True),
        "NVDA":  dict(mode="cfd",                                 # overrides the base
                      asset_class="stocks", contract_size=1.0,
                      quote_currency="USD", cost_currency="USD",
                      commission_pct=0.0005, spread_points=2.0, leverage=5.0),
    },

    # currency
    base_currency="GBP", fx=fx,

    # money management
    notional=25_000,
    leverage_type="portfolio_level",

    # holding costs
    financing="auto", rates=rates,

    # account
    init_cash=100_000, cash_sharing=True, freq="D",
)

print(pf.stats())
print(pf.trades.stats(group_by=False, agg_func=None))     # per instrument
```

---

## 5. Portfolio, form 3 — a CSV universe

`universe.csv`:

```csv
name,mode,asset_class,point_size,contract_size,quote_currency,cost_currency,commission_pct,commission_flat,commission_min,spread_points,slippage,slippage_max,leverage,allow_fractional,notes
VOD.L,equity,stocks,0.01,,GBP,GBP,,10.0,,0.5,auto,0.005,1.0,FALSE,UK broker A
BP.L,equity,stocks,0.01,,GBP,GBP,,10.0,,0.5,auto,0.005,1.0,FALSE,UK broker A
MSFT,equity,stocks,0.01,,USD,USD,0.001,,1.0,1.0,auto,0.005,2.0,TRUE,US margin
AAPL,equity,stocks,0.01,,USD,USD,0.001,,1.0,1.0,auto,0.005,2.0,TRUE,US margin
NVDA,cfd,stocks,0.01,1.0,USD,USD,0.0005,,,2.0,auto,0.005,5.0,FALSE,CFD broker B
```

`notes` is not a schema field; it is preserved through round-trip export.

```python
import vectorbtx as vbt

START, END = "2019-01-01", "2024-12-31"

# ---- universe first: it is the single source of truth for the symbol list ----
universe = vbt.x.load_universe("universe.csv")
print(universe.to_frame())                   # eyeball resolved defaults before trusting

# ---- data ----
data = vbt.YFData.download(list(universe.columns), start=START, end=END,
                           missing_index="drop")
price_native = data.get("Close")
volume = data.get("Volume")

fx = vbt.YFData.download("GBPUSD=X", start=START, end=END)
rates = vbt.x.FREDData.download_rates(["GBP", "USD"], start=START, end=END)

# ---- convert: the universe already knows which columns are USD ----
price = vbt.x.to_base_currency(price_native, base="GBP", fx=fx,
                               instruments=universe, fx_markup=0.005)

# ---- signals ----
momentum = price.pct_change(90)
in_top = momentum.rank(axis=1, ascending=False) <= 3
entries = in_top & ~in_top.shift(1).fillna(False)
exits = ~in_top & in_top.shift(1).fillna(False)

# ---- backtest ----
pf = vbt.x.PortfolioBacktest(
    price,
    entries=entries,
    exits=exits,
    instruments=universe,                    # mode is per row, in the file
    volume=volume,
    base_currency="GBP", fx=fx,
    notional=25_000,
    financing="auto", rates=rates,
    init_cash=100_000, cash_sharing=True, freq="D",
)

print(pf.stats())
```

---

## 6. Portfolio, form 4 — `Instrument` objects, for reuse

```python
import vectorbtx as vbt

START, END = "2019-01-01", "2024-12-31"

# ---- reusable instrument definitions ----
uk_equity = vbt.x.Instrument(
    mode="equity", asset_class="stocks", point_size=0.01,
    quote_currency="GBP", cost_currency="GBP",
    commission_flat=10.0, spread_points=0.5, leverage=1.0,
)
us_margin = vbt.x.Instrument(
    mode="equity", asset_class="stocks", point_size=0.01,
    quote_currency="USD", cost_currency="USD",
    commission_pct=0.001, commission_min=1.0, spread_points=1.0,
    leverage=2.0, allow_fractional=True,
)
us_cfd = vbt.x.Instrument(
    mode="cfd", asset_class="stocks", point_size=0.01, contract_size=1.0,
    quote_currency="USD", cost_currency="USD",
    commission_pct=0.0005, spread_points=2.0, leverage=5.0,
)

instruments = {"VOD.L": uk_equity, "BP.L": uk_equity,
               "MSFT": us_margin, "AAPL": us_margin, "NVDA": us_cfd}

# ---- data ----
data = vbt.YFData.download(list(instruments), start=START, end=END,
                           missing_index="drop")
price_native = data.get("Close")
volume = data.get("Volume")

fx = vbt.YFData.download("GBPUSD=X", start=START, end=END)

# ---- convert ----
price = vbt.x.to_base_currency(price_native, base="GBP", fx=fx,
                               instruments=instruments, fx_markup=0.005)

# ---- signals ----
momentum = price.pct_change(90)
in_top = momentum.rank(axis=1, ascending=False) <= 3
entries = in_top & ~in_top.shift(1).fillna(False)
exits = ~in_top & in_top.shift(1).fillna(False)

# ---- backtest ----
pf = vbt.x.PortfolioBacktest(
    price,
    entries=entries,
    exits=exits,
    instruments=instruments,
    volume=volume,
    base_currency="GBP", fx=fx,
    notional=25_000,
    init_cash=100_000, cash_sharing=True, freq="D",
)

print(pf.stats())
```

---

## 7. Rebalanced investing — same class, different arguments

```python
import vectorbtx as vbt

START, END = "2015-01-01", "2024-12-31"

universe = vbt.x.load_universe({
    "VWRL.L": vbt.x.Instrument(mode="equity", point_size=0.01, quote_currency="GBP",
                               cost_currency="GBP", commission_flat=5.0),
    "VUSA.L": vbt.x.Instrument(mode="equity", point_size=0.01, quote_currency="GBP",
                               cost_currency="GBP", commission_flat=5.0),
    "SPY":    vbt.x.Instrument(mode="equity", point_size=0.01, quote_currency="USD",
                               cost_currency="USD", commission_pct=0.0005,
                               commission_min=1.0, allow_fractional=True),
    "AGG":    vbt.x.Instrument(mode="equity", point_size=0.01, quote_currency="USD",
                               cost_currency="USD", commission_pct=0.0005,
                               commission_min=1.0, allow_fractional=True),
})

# ---- data ----
data = vbt.YFData.download(list(universe.columns), start=START, end=END,
                           missing_index="drop")
price_native = data.get("Close")
fx = vbt.YFData.download("GBPUSD=X", start=START, end=END)

price = vbt.x.to_base_currency(price_native, base="GBP", fx=fx,
                               instruments=universe, fx_markup=0.005)

# ---- target weights and schedule ----
weights = price.copy()
for col, w in {"VWRL.L": 0.40, "VUSA.L": 0.20, "SPY": 0.25, "AGG": 0.15}.items():
    weights[col] = w

calendar = vbt.x.calendar_schedule(price.index, freq="M")
drifted = vbt.x.drifted_weights(weights, price, schedule=calendar)
schedule = calendar & vbt.x.threshold_schedule(drifted, weights, tolerance=0.02)

# ---- backtest ----
pf = vbt.x.PortfolioBacktest(
    price,
    weights=weights,                         # weights + schedule -> rebalancing
    schedule=schedule,
    instruments=universe,
    base_currency="GBP", fx=fx,
    min_trade=0.005,
    init_cash=250_000, freq="D",
)

print(pf.stats())
print(pf.x.drift().abs().mean())              # what quantization actually cost
```

`entries`/`exits` → signal-driven. `weights`/`schedule` → rebalanced. Neither → hold.
Both → error naming the conflict.

---

## 8. Argument reference

Not examples — the full vocabulary, for lookup.

### Sizing (exactly one)

```python
risk_pct=0.01          # fraction of init_cash risked per trade; requires stop_points
risk_amount=100        # absolute money at risk per trade
stake=2.50             # instrument's own units: £/pt, lots, or shares
notional=25_000        # target exposure per position
size=<array>           # raw passthrough, escape hatch
```

Two given → error naming both. `risk_pct` and `risk_amount` are separate arguments rather
than one `risk=` disambiguated by magnitude.

⚠️ `risk_pct` sizes off `init_cash`, not compounding equity — at level 1 equity is only
known before the simulation runs. Material over a long backtest with real drawdowns.

### Stops (points, not fractions)

```python
stop_points=60
target_points=120
trail_points=40
```

### Slippage

```python
slippage=0.0005        # flat fraction of price
slippage="auto"        # coef x sigma x sqrt(size/ADV), clipped to slippage_max
slippage_max=0.005
slippage_coef=1.0
volume=volume          # required by "auto"
adv=5_000_000          # or a static ADV instead of a volume series
```

⚠️ `"auto"` raises without `volume` or `adv` rather than falling back to zero.

Spread and slippage both land on vectorbt's single `slippage` field and are summed before
assignment. Inspect with `pf.x.order_args()`.

### Exposure

```python
leverage=30
leverage_type="portfolio_level"   # or "instrument_level"
min_trade=0.005
raise_reject=False                # True -> reject rather than partially fill at the cap
```

### Holding costs

```python
financing="auto"       # keyword registry (plan D8): 'auto' pulls from the rate table
active_costs={"commission", "spread", "slippage", "financing", "carry"}
rates=rates
triple_day="WED"
```

`active_costs` is the plan's mechanism (§C, `Instrument.active_costs`). Omitted, it
resolves to the mode's default stack. An earlier draft proposed per-cost booleans
(`carry=True`); that invented a second vocabulary for something already decided.

---

## 9. Decision ledger

Every choice in this document, and where it comes from. **Nothing here is open.**

| Decision | Source |
|---|---|
| Four modes, and what each requires/rejects | plan §C, Part 8 |
| `point_size` as the single unit primitive | plan §C |
| `contract_size` rejected for spread bets | plan §C, §H |
| `commission_*` forced to 0 in spreadbet mode | plan §E default stacks |
| `allow_fractional` equity only | plan §C validation |
| `price_field` defaults per mode | plan D10 |
| Leverage defaults 2× / 5× / 10× / 30× | plan Part 8 |
| `leverage_type` for portfolio-wide exposure | plan §F |
| `active_costs` selects cost modules | plan §C |
| `'auto'` for slippage and financing only | plan D8 keyword registry |
| `slippage="auto"` raises without volume | plan §E |
| Spread and slippage summed into one field | plan §E |
| Four instrument supply forms → `Universe` | plan §C2 |
| Instrument-name mismatches are fatal | plan §C2 |
| Unrecognised keywords are fatal | plan D8 |
| `entries`/`exits` vs `weights`/`schedule` | plan §J |
| Conversion precedes signals | plan §I (D4) |
| **Class names `SingleBacktest` / `PortfolioBacktest`** | **specified by the user** |
| **`risk_pct` / `risk_amount` split, not one `risk=`** | plan D8 ethos: coerce explicitly, never guess |
| **Collision (`fees=` beside `commission_pct=`) raises** | plan §C2 ethos: mismatches are fatal, not warnings |
| **Two names, one implementation** | avoids the divergence the plan warns about for parallel definitions |

The four bold rows are the only decisions this layer adds. The rest is the plan applied to
a new surface.

### Withdrawn from earlier drafts

| Item | Why |
|---|---|
| `max_positions` | Not in the plan. Invented in v5. New capability, not ergonomics. |
| `max_exposure` | Not in the plan, and duplicates `leverage_type="portfolio_level"` (§F). |
| `carry=True` per-cost booleans | Second vocabulary for `active_costs`, already decided in §C. |
| `Risk`/`Stake`/`Notional` sizer classes | Flat arguments do the same job; v0.2 draft only. |
| `Account` object | Subsumed by the backtest call. |
| `bt.trade()` / `from_signals()` verbs | Single-call design has no verbs. |

## 10. Implementation defect to fix

Not a design question — a fault visible in §4, §5 and §6 above.

`to_base_currency` is called three different ways: with a `quote_currency` dict, with a
`Universe`, and with a dict of `Instrument`s. That is one piece of information arriving in
three shapes. It needs one signature that accepts any of them and normalises internally,
resolved during implementation rather than debated.
