# vectorbtX API reference

Every public symbol, with signature, arguments, return value, raises, and an example.

Namespaces:

| Namespace | Contains |
|---|---|
| `vbt.x.*` | constructors, loaders, free functions (64 symbols) |
| `pf.x.*` | portfolio-bound extras |
| `df.vbt.x.*` | data-bound extras |
| `pf.stats()`, `pf.trades.stats()`, `pf.returns_stats()` | metrics, merged with native ones |

Everything upstream remains available unchanged under its usual name.

---

## Contents

0. [**The front door**](#0-the-front-door) — start here
1. [Instruments](#1-instruments)
2. [Universe and resolution](#2-universe-and-resolution)
3. [Quantization](#3-quantization)
4. [Modes](#4-modes)
5. [Multi-currency](#5-multi-currency)
6. [Costs](#6-costs)
7. [Margin and leverage](#7-margin-and-leverage)
8. [Portfolio construction](#8-portfolio-construction)
9. [Rates](#9-rates)
10. [Holding costs](#10-holding-costs)
11. [Metrics](#11-metrics)
12. [Portfolio constructors](#12-portfolio-constructors)
13. [Settings](#13-settings)

---

## 0. The front door

Start here. `SingleBacktest` and `PortfolioBacktest` are a facade over sections 1–11 —
**they add no behaviour**, they just remove the assembly. Both return a
`vectorbtX.Portfolio`, so `pf.stats()`, `pf.trades.stats()`, `pf.returns_stats()` and every
`pf.x.*` extra work with no result wrapper to learn.

### `vbt.x.PortfolioBacktest(*, close, open, high, low, **kwargs)` → `Portfolio`
### `vbt.x.SingleBacktest(*, close, open, high, low, **kwargs)` → `Portfolio`

**All four price series are required, and all four are keyword-only.**

```python
pf = vbt.x.PortfolioBacktest(close=close, open=open_, high=high, low=low, ...)
```

Orders fill at the **next bar's open**: entries, exits and rebalance dates shift one bar, and
`price=open`. Stops are checked against `high` and `low`. There is no `execution=` argument
and no default for the extremes — filling at the signal close is lookahead, checking a stop
against the close fires it late, and an API that offers a broken mode invites its use.

Keyword-only is deliberate. Four same-shaped frames in positional sequence is precisely the
call where transposing two of them — high and low swapped, open where close belongs — runs
cleanly and reports a wrong number. Naming them makes that mistake unspellable.

`pf.x.intrabar_drawdown()` measures the worst peak-to-trough decline against the highs and
lows rather than the closes.

Everything order-time — sizes, stop distances, spread fractions — is priced at the **open
you fill at**, not the close the signal fired on.

`SingleBacktest` validates that the price is 1-D, rejects the portfolio-only arguments with
a message pointing at `PortfolioBacktest`, and forwards. Two names, one implementation — so
there is no divergence risk, and each name shows only the arguments that apply.

**`mode` is the dispatcher.** It decides which other arguments are legal, required or
rejected. Rejected cells raise at construction: a spread bet with a `contract_size` is a
misunderstanding of the product, not a typo.

| Argument | `equity` | `spreadbet` | `cfd` | `fx_spot` |
|---|---|---|---|---|
| `point_size` | **required** | **required** | **required** | derived from quote currency |
| `contract_size` | rejected | **rejected** — the stake *is* the size | **required** | **required** |
| `asset_class` | optional | **drives leverage, carry, roll day** | same | fx |
| `commission_*` | yes | **forced to 0** | yes | yes |
| `spread_points` | yes | **primary cost** | yes | **primary cost** |
| `allow_fractional` | **yes** | rejected | rejected | rejected |
| `base_currency` | rejected | **fx only** (for carry) | **fx only** | **required** |
| `leverage` default | 2× | 5/10/30× by class | 5/10/30× by class | 2× |
| overnight cost | financing | carry if fx, else financing | same | **carry** |
| `price_field` default | `adj_close` | `close` | `close` | `close` |

For `asset_class="fx"`, `point_size` is **derived from the quote currency** — `0.01` for
JPY-quoted pairs, `0.0001` otherwise — so a mixed book needs no per-pair override.
Contradicting the convention warns.

**Strategy type is implied by the arguments.**

| Given | Route |
|---|---|
| `entries` / `exits` | `from_signals` |
| `weights` / `schedule` | `from_weights` |
| neither | `from_holding` |
| both | **error naming the conflict** |

**Sizing — exactly one.** Two given raises, naming both.

```python
risk_pct=0.01          # fraction of init_cash risked per trade; requires stop_points
risk_amount=100        # absolute money at risk per trade
stake=2.50             # the instrument's own units: £/pt, lots, or shares
notional=25_000        # target exposure per position
size=<array>           # raw engine-unit passthrough, the escape hatch
```

⚠️ `risk_pct` sizes off `init_cash`, **not compounding equity** — at level 1 equity is only
known before the simulation runs. Material over a long backtest with real drawdowns.

**Stops in points**, converted internally to the native price fractions:
`stop_points`, `target_points`, `trail_points`. Native `sl_stop` etc. are still accepted.

**Instruments — four supply forms**, all producing an identical `Universe`:

```python
# 1. flat keywords, applied to every column
vbt.x.PortfolioBacktest(close=close, open=open_, high=high, low=low,
                        mode="equity", point_size=0.01, commission_flat=10.0, ...)

# 2. per-symbol overrides on a shared base; a per-symbol `mode` overrides the shared one
vbt.x.PortfolioBacktest(close=close, open=open_, high=high, low=low,
                        mode="equity", point_size=0.01,
                        instruments={"VOD.L": dict(commission_flat=10.0),
                                     "NVDA":  dict(mode="cfd", contract_size=1.0)}, ...)

# 3. a CSV or Excel path — mode is per row, in the file
vbt.x.PortfolioBacktest(close=close, open=open_, high=high, low=low,
                        instruments="universe.csv", ...)

# 4. Instrument objects or an existing Universe, for reuse
vbt.x.PortfolioBacktest(close=close, open=open_, high=high, low=low,
                        instruments=universe, ...)
```

**Unrecognised arguments pass through** to the underlying vectorbt constructor, so nothing
upstream becomes unreachable.

⚠️ **A collision between a passthrough and a resolved value raises.** `fees=0` beside
`commission_pct=0.001` would silently disable a cost model and produce a backtest that
looks excellent, so it is fatal — consistent with §2, where join-key mismatches are fatal
rather than warnings.

⚠️ **FX conversion stays outside the call.** Signals are computed on base-currency prices
and you need to hold that series anyway, so `to_base_currency` remains an explicit line.
Converting internally for sizing while signals ran on native prices would put two different
price series in one backtest.

```python
pf.x.user_size()       # position in £/pt, lots or shares — not engine units
pf.x.unit_names()      # {"EURUSD=X": "£/point"}
pf.margin_loan         # buying power beyond your own equity
pf.holding_costs       # per-bar financing and carry, if rates were supplied
pf.drift               # quantization and suppression drift, if rebalanced
```

---

## 1. Instruments

### `vbt.x.Instrument`

```python
Instrument(
    mode,                        # "spreadbet" | "cfd" | "fx_spot" | "equity"   (required)
    asset_class=None,            # "stocks" | "commodities" | "fx" | "index"
    point_size=None,             # THE unit primitive
    contract_size=None,          # cfd, fx_spot only
    increment=None, min_size=None, allow_fractional=False, rounding=None,
    quote_currency=None, base_currency=None, cost_currency=None,
    price_field=None,            # "close" | "adj_close"
    commission_pct=None, commission_flat=None,
    commission_per_unit=None, commission_min=None,
    spread_points=None,
    slippage=None,               # float or "auto"
    slippage_max=None, slippage_coef=None,
    financing=None,              # float or "auto"
    adv=None,
    active_costs=frozenset(),
    leverage=None, leverage_type="instrument_level",
    triple_day="WED", end_of_day="22:00",
    meta=(),
)
```

Frozen dataclass. **Validates on construction** — an invalid `Instrument` cannot exist.

Key fields:

- **`point_size`** — the price increment worth one point. The single unit primitive: pip
  position derives from it, and there are deliberately no separate `tick_size` or
  `pip_decimal_place` fields. For a 5-decimal FX quote set `point_size = 0.0001` (the
  pip); price precision stays at 5 decimals, because a meaningful part of the spread lives
  in the fractional 5th digit.
- **`contract_size`** — CFD and FX spot only. **Spread bets have none**; the stake is the
  size.
- **`cost_currency`** — separate from `quote_currency` *on purpose*. Converting a US
  equity's prices to GBP does not convert a `$1` minimum commission. Without this field
  the cost layer cannot know which fixed costs need converting. The easiest thing in the
  library to get silently wrong.
- **`allow_fractional`** — equity only. Also drops the fill increment to `1e-6` unless
  `increment` is given explicitly. Without that the flag would be decorative: the spec
  would say fractional and every order would still round to a whole share.
- **`leverage_type`** — `"instrument_level"` is a per-column cap and is exact.
  `"portfolio_level"` is a joint constraint enforced on the weight matrix by
  `from_weights`. Under `group_by` with cash sharing, "portfolio" means **the group**.

**Methods**

| Method | Returns | Notes |
|---|---|---|
| `.filled()` | `Instrument` | Every unspecified field resolved to its mode default. |
| `.replace(**kw)` | `Instrument` | Copy with fields replaced; re-validates. |
| `.to_dict()` | `dict` | Flat, export-ready. `active_costs` joined with `\|`. |
| `.is_nominal_points` | `bool` | True for spread bet and CFD. |
| `.is_derivative` | `bool` | True where there is no ownership. |

**Raises** `SpecError` on: `contract_size` on a spread bet; missing `contract_size` on
CFD/FX spot; `increment <= 0`; `min_size` not a multiple of `increment`;
`allow_fractional` on a mode that cannot fractionalise; unresolvable currency
combinations; leverage above the mode cap.

```python
uk_stock = vbt.x.Instrument(mode="spreadbet", point_size=0.01, asset_class="stocks")
msft = vbt.x.Instrument(mode="equity", point_size=0.01,
                        quote_currency="USD", cost_currency="USD",
                        commission_flat=10.0, commission_min=1.0)
eurusd = vbt.x.Instrument(mode="fx_spot", point_size=0.0001, contract_size=100_000,
                          quote_currency="USD", base_currency="EUR")
```

### `vbt.x.mode_defaults(mode)` → `dict`
Default field values for a mode.

### `vbt.x.max_leverage(mode, asset_class=None)` → `float`
Default maximum leverage. Equity 2×, FX spot 2×; spread bet / CFD 5× stocks, 10×
commodities and indices, 30× FX. Tiered margin is deliberately not modelled — set a
sensible maximum instead.

### `vbt.x.default_cost_stack(mode, asset_class=None)` → `frozenset`

| Mode | Default stack |
|---|---|
| `equity` | commission + spread + slippage + financing |
| `fx_spot` | commission + spread + slippage + carry |
| `cfd` (fx) | commission + spread + slippage + carry + financing |
| `cfd` (other) | commission + spread + slippage + financing |
| `spreadbet` (fx) | spread + slippage + carry + financing |
| `spreadbet` (other) | spread + slippage + financing |

Spread bets carry no commission — the fee is baked into a wider spread.

---

## 2. Universe and resolution

### `vbt.x.load_universe(source)` → `Universe`

`source` may be a `dict[str, Instrument]`, a `DataFrame`, or a path to `.csv` / `.xlsx`
(Excel needs the `excel` extra).

Both front doors normalise to `dict[column, Instrument]` and **converge before
validation** — otherwise the spreadsheet path, precisely where bad data enters, would
bypass every check.

```
dict[str, Instrument]  ─┐
                        ├─► dict[column, Instrument] ─► validate ─► resolve ─► numpy
DataFrame / CSV / XLSX ─┘
```

**Raises**

- `UniverseError` — a price column without a spec, or a spec without a price column.
  **Fatal in both directions**: a warning on a 20-asset backtest scrolls past, and one
  column silently gets defaults.
- `CoercionError` — an ambiguous cell, naming the row and column.
- `KeywordError` — an unrecognised keyword such as `'atuo'`, or a valid keyword on a
  parameter that does not accept it.

**Spreadsheet schema.** Fixed columns matching `Instrument` field names, plus an
instrument-name column (`name`, `symbol`, `ticker` or `instrument`). Extra columns pass
through as metadata and survive round-trip export, so notes and broker names can live in
the sheet.

**Blank cells are not NaN.** vectorbt uses NaN and inf meaningfully in `Order` fields
(`size_granularity=NaN` means *no quantization*, `price=inf` means *use close*), so a
blank becoming NaN could accidentally *mean* something. Blanks become `None` internally,
resolve against mode defaults, and become NaN only at emit.

**Keyword cells.** A cell holds either a literal or a keyword. Currently `slippage='auto'`
and `financing='auto'` only — a large vocabulary invented up front is a liability. Keywords
are per-parameter: `'auto'` slippage means volume-and-volatility scaled, `'auto'` financing
means pull from the rate table. Auto-mode parameters live in **sibling columns**
(`slippage_max`, `slippage_coef`), not compound cell syntax.

### `Universe`

| Member | Returns | Notes |
|---|---|---|
| `.specs` | `dict[str, Instrument]` | |
| `.columns` | `list[str]` | |
| `.subset(cols)` | `Universe` | Used by `Portfolio.indexing_func`. |
| `.order_for(cols)` | `list[Instrument]` | **In the wrapper's column order**, not dict insertion order. |
| `.resolve(data, rates=None, equity=None, fx_table=None, base_currency=None, volume=None, size=None, notional=None)` | `dict` | Per-column `Order` kwargs. |
| `.to_frame()` | `DataFrame` | Fully resolved, including filled defaults. For eyeballing. |
| `.to_csv(path)` | — | Round-trip export. |

`.resolve()` emits `size_granularity`, `min_size`, `max_size`, `fees`, `fixed_fees` and
`slippage`, ready to splat into any vectorbt constructor:

```python
pf = vbt.Portfolio.from_orders(close, size=size, **universe.resolve(close))
```

### `vbt.x.export_universe(universe, path=None)` → `DataFrame`
Round-trip export. If it reads a DataFrame it writes one — the best available defence
against silent misalignment.

---

## 3. Quantization

### `vbt.x.quantize(size, spec, direction=None)` → `(quantized, drift)`

Rounds to `spec.increment`. `direction` falls back to `spec.rounding`, then to the global
default (`floor`).

Sub-minimum sizes **reject to zero**, they do not round to zero silently: a £0.50/point
stake against a £1 minimum returns `(0.0, 0.50)`, not `(0.0, 0.0)`.

Rounding acts on the **magnitude** with the sign preserved, so `floor` always means a
smaller position.

```python
vbt.x.quantize(137.4829, shares)     # (137.0, 0.4829)
vbt.x.quantize(0.437, spread_bet)    # (0.0, 0.437)   below the £1 minimum
```

### `vbt.x.to_order_params(spec, direction="floor")` → `dict`

- `floor` → `{"size_granularity": spec.increment, "min_size": spec.min_size}`. The engine
  already floors (`adj_size // granularity * granularity`), so this is free.
- anything else → `{"size_granularity": np.nan, "min_size": …}` with the caller
  pre-quantizing. **Do not try to make the engine round differently.**

### `vbt.x.drift_report(intended, actual, ...)` → `DataFrame`
Per-order drift. **Kept separate from `max_size` effects** — `max_size` partially fills
rather than rejecting, so realised size can differ from intended for reasons unrelated to
rounding.

### `vbt.x.rebalance_drift(...)` → `DataFrame`
Per-rebalance aggregation. Also reachable as `pf.x.drift()`.

---

## 4. Modes

### `vbt.x.get_adapter(mode_or_spec)` → `ModeAdapter`
Accepts a mode string or an `Instrument`.

| Adapter | `to_engine_units(user_size, spec, price)` |
|---|---|
| `SpreadBetAdapter` | `stake_per_point / point_size` |
| `CFDAdapter` | `lots × contract_size` |
| `FXSpotAdapter` | `lots × contract_size` |
| `EquityAdapter` | `shares` |

All four collapse to the same engine expression, `engine_size × Δprice` — which is what
makes cross-mode equivalence a real test rather than an aspiration:

```python
for spec in (spread_bet, cfd, fx_spot, shares):
    a = vbt.x.get_adapter(spec)
    user = a.from_engine_units(100.0, spec, price)
    assert a.to_engine_units(user, spec, price) == 100.0   # same P&L, four notations
```

The "100 contract size" for a £1/point UK stock bet is *derived* from
`1.0 / 0.01`, not configured.

### `vbt.x.points_to_price_fraction(points, price, spec)` → `float`
`points × point_size / price`. Native SL/TP takes **fractions of price**, so this is the
bridge — do not reimplement stop logic.

```python
pf = vbt.Portfolio.from_signals(
    close, entries, exits,
    sl_stop=vbt.x.points_to_price_fraction(50, close, spec),
)
```

Fractional stops are FX-invariant under uniform rescaling.

### `vbt.x.price_fraction_to_points(fraction, price, spec)` → `float`
The inverse.

### `vbt.x.value_per_point(user_size, spec)` → `float`
Money per point of movement. For a spread bet this is simply the stake.

### `vbt.x.size_for_risk(risk_amount, stop_distance_points, spec, price=None)` → `float | Series`
**One formula, all four modes.** `engine = risk / (stop_points × point_size)`, converted
back to user units. Risking £100 over a 50-point stop gives identical monetary risk
whether expressed as a stake, lots or shares.

Returns a Series aligned to `price` when `price` is a Series. That matters: a bare ndarray
here would strip the index and then silently reindex against a dated frame downstream,
producing a union-shaped result rather than an error.

### `vbt.x.pnl(user_size, entry, exit, spec)` → `float`
P&L in account currency. The assertion target of the cross-mode equivalence test.

---

## 5. Multi-currency

Foreign price series are converted to base currency **before** simulation. This is exact,
not an approximation:

```
shares × (P_usd[t1]·fx[t1] − P_usd[t0]·fx[t0])  ==  shares × (P_gbp[t1] − P_gbp[t0])
```

Cash is base currency throughout, so cash sharing works, cross-currency rebalancing works,
and FX moves correctly affect buying power mid-backtest — without touching the engine.

Applies to **equity and FX spot only**. Spread bets are account-currency-denominated by
construction, as are GBP-quoted CFDs on foreign underlyings.

### `vbt.x.FXTable`

Long form keyed `(date, quote, base) → value`, where `value` multiplies a `quote`-currency
price to give `base`. **Direction lives in the key**, so there is no multiply-or-divide
decision to get wrong.

| Constructor | Notes |
|---|---|
| `FXTable.from_canonical(frame, hub="USD")` | Fetch one canonical direction; reciprocals and triangulations are **derived**. |
| `FXTable.from_long(frame)` | Pre-built long frame. |
| `FXTable.constant(rates)` | Fixed rates — the no-data path. |

Deriving matters. Fetching both directions independently gives non-exact reciprocals
(different timestamps, mid vs bid/ask capture), and a round trip then produces phantom
P&L. That noise is **random in sign**, so it inflates measured volatility rather than
acting as a cost — quietly degrading every risk-adjusted metric with a σ in the
denominator. Deriving guarantees `value(A,B) × value(B,A) = 1` by construction.

| Method | Returns |
|---|---|
| `.get(quote, base, index=None)` | rate series for one pair |
| `.align(index, fill_policy)` | explicit fill policy; warns rather than silently forward-filling |
| `.validate(tolerance=1e-4)` | reciprocal and triangulation checks; warns beyond a basis point |
| `.currencies` | every currency in the grid |

### `vbt.x.to_base_currency(data, base, fx_table, fx_markup=0.005, fill_policy="ffill_with_warning", quote_currency=None)` → `ConvertedFrame`

Converts OHLC using **one rate per bar**. Different rates for open, high, low and close
would break `H ≥ C ≥ L`, and native SL/TP would fire at prices that never existed — the
same failure mode as mixing adjusted and unadjusted fields. One validation check covers
both.

Also reachable as `df.vbt.x.to_base_currency(...)`.

Returns a `ConvertedFrame` carrying:

- `.native` — the original series, retained for decomposition
- `.fx_markup` — picked up by the cost layer and added to `slippage`
- `.decompose()` — splits total return into asset return and FX return

`fx_markup` models the broker's worse-than-market conversion rate. It is applied as
slippage rather than baked into the price series, because a uniform rescaling of a single
series would cancel out and cost nothing — as slippage it is genuinely adverse on entry
*and* on exit. Foreign handling *fees* are separate and go through `cost_currency` as
ordinary `fixed_fees`.

**Convert the benchmark too**, or beta and alpha compare base-currency returns against
foreign-currency returns and every downstream risk metric is wrong.

### `vbt.x.convert_amount(amount, quote, base, table, markup=0.0, direction="to_base")` → `float`
Cash conversion with the markup applied **adversely in both directions**.

### `vbt.x.convert_fixed_costs(amount, cost_currency, base, table, index=None)` → `Series`
Converts a foreign fixed fee. Percentage costs are FX-invariant and need no handling;
fixed and per-unit costs are not.

---

## 6. Costs

### `vbt.x.resolve_costs(universe, price, fx_table=None, base_currency=None, volume=None, size=None, notional=None, rates=None, columns=None)` → `dict`

Composes each instrument's active stack into `Order` kwargs:
`{"fees": …, "fixed_fees": …, "slippage": …}`.

**Mapping**

| Cost | Order field |
|---|---|
| percentage commission | `fees` |
| flat / per-unit commission | `fixed_fees` |
| minimum commission | `fixed_fees`, pre-computed, with `fees` zeroed |
| spread | `slippage` |
| slippage | `slippage` |

**Spread and slippage both land on `slippage`, so they are summed before assignment** —
only one field exists. This is the least obvious constraint in the layer.

A **minimum commission** is pre-computed per order into `fixed_fees`, because
`max(pct × notional, min)` is a conditional and genuinely cannot be expressed as
`fees + fixed_fees`, which are additive. Pass `notional=` or it raises.

**Auto slippage** is `clip(coef × σ × sqrt(size / ADV), 0, slippage_max)`, with σ a 20-bar
realised volatility as a fraction of price. Square-root market impact is the standard
empirical result and beats linear. It **fails at resolve time without a volume column**
rather than silently falling back to zero.

Spread-bet mode sets `fees = fixed_fees = 0`, folding cost entirely into `slippage`.

### `vbt.x.equivalent_pct(flat_fee, notional)` → `float`
£10 on £10,000 → `0.001`. Retail equity commission is often flat, but users often want a
single percentage.

### `vbt.x.equivalent_flat(pct, notional)` → `float`
The inverse.

---

## 7. Margin and leverage

Maximum-leverage enforcement only. Maintenance margin, margin calls and forced liquidation
are deliberately deferred: they need per-bar callbacks and would extend the fragile
dependency on `portfolio/nb.py` well beyond the holding-cost module.

### `vbt.x.notional(user_size, price, spec)` → `float`
Full market value regardless of cash backing. A 5× leveraged £10,000 margin position has
£50,000 notional.

### `vbt.x.margin_required(user_size, price, spec)` → `float`
`notional / leverage`, with mode-specific notional.

### `vbt.x.leverage_cap(spec, equity, price)` → `float`
Maximum engine size at `spec.leverage`.

### `vbt.x.max_size_for(specs, equity, price)` → `list`
Per-column `max_size` for a whole universe (`specs` in resolved column order). Feed to `Order.max_size`, which **partially fills rather than rejecting** — clip semantics
for free. For reject semantics instead, pass `raise_reject=True` to the constructor.

### `vbt.x.borrowed_portion(pf)` → `DataFrame`
Post-simulation split into borrowed vs cash-covered, from `pf.cash(free=True)` and
`pf.asset_value()`. A cash-covered position borrows nothing; £50,000 against £10,000 cash
borrows £40,000. This is Section G's input and a hard requirement.

Also `pf.x.borrowed_portion()`.

### `vbt.x.margin_level(pf, universe)` / `vbt.x.min_margin_level(pf, universe)`
Equity / margin required, per bar and minimum. The minimum is an MT5 parity metric,
satisfied without building liquidation.

Also `pf.x.margin_level()`, `pf.x.min_margin_level()`.

---

## 8. Portfolio construction

### Weights

```python
vbt.x.equal_weight(columns, index)            -> DataFrame
vbt.x.price_weight(prices, index=None)        -> DataFrame
vbt.x.cap_weight(market_caps, index, fx=None) -> DataFrame
vbt.x.inverse_vol_weight(prices, window=20)   -> DataFrame
```

**Cap weights are recomputed at each rebalance.** Fixed at construction is not cap
weighting, it is an initial-cap-weighted buy-and-hold. Market caps need converting with the
same rate table as prices.

### Schedules

```python
vbt.x.calendar_schedule(index, freq="M")                  -> Series[bool]
vbt.x.threshold_schedule(current, target, tolerance=0.05) -> Series[bool]
vbt.x.drifted_weights(weights, prices, schedule=None)     -> DataFrame
vbt.x.always_schedule(index)                              -> Series[bool]
```

⚠️ **Pass `schedule` to `drifted_weights`.** Drift must be measured since the last
rebalance. Without an anchor it accumulates from the first bar of the backtest, grows
without bound, and any threshold test on it degenerates to "always true" — silently
collapsing a composed `calendar & threshold` rule back into plain calendar rebalancing.
The anchor is the last rebalance *strictly before* each bar, so drift on a rebalance bar
reflects the period just elapsed rather than being zero by construction.

Boolean masks, so they **compose**:

```python
schedule = vbt.x.calendar_schedule(index, "M") & vbt.x.threshold_schedule(cur, tgt, 0.05)
```

"Monthly, but only if drift exceeds 5%" is the common real-world rule, and it is just an
`&`.

Threshold rebalancing uses **relative** drift, per asset: a 1% position drifting to 1.5% is
the 50% move it is. Absolute drift under-triggers on small positions.

### `vbt.x.apply_leverage_budget(weights, universe, equity=None)` → `DataFrame`
`portfolio_level` enforcement — scales weights to fit the total notional budget. Exact and
cheap, because weights are known ahead of time.

**Raises** `SpecError` when `portfolio_level` is combined with differing per-ticker
leverage values, rather than silently picking one.

### `vbt.x.to_targets(weights, schedule=None, universe=None, price=None, equity=None, min_trade=None)` → `(targets, drift)`

Emits `TargetPercent`-ready targets: the weight on rebalance bars, **NaN everywhere else**.

> **NaN off-schedule, never forward-fill.** NaN means "no order", and the position drifts
> as intended. Forward-filling rebalances *every bar* — which still runs, still looks
> plausible, and quietly accrues transaction costs while producing a suspiciously smooth
> equity curve. It is the single most likely thing to get written backwards here, so a
> constant non-NaN input across off-schedule bars **warns**.

Quantization happens **before** emission; quantizing afterwards loses the drift
information. The optional minimum-trade filter suppresses trades costing more than the
drift they correct, and routes the suppressed drift into the drift report rather than
letting it vanish as invisible tracking error.

---

## 9. Rates

### `vbt.x.rate_table(currencies, index, source="fred")` → `DataFrame`
One column per currency, aligned to the backtest index.

Defaults: SONIA (GBP), SOFR (USD), ESTR (EUR), TONA (JPY), SARON (CHF) — all post-LIBOR
overnight benchmarks.

### `vbt.x.static_rate_table(currencies, index, rate)` → `DataFrame`
The no-API-key path. **The library must work with a static rate**, and does.

### `vbt.x.annual_to_period(rate, days, day_count=360)` → `float`
Rates are quoted **annualised**; financing accrues daily. This conversion is explicit
rather than assumed. ACT/360 for all currencies — chosen for simplicity and because it
slightly overstates cost, which is the safer direction.

⚠️ Note the inversion for carry: the same arithmetic overstates a *received* credit, so a
positive-carry position looks marginally better than reality. Minor in magnitude, but
"overstates = conservative" is not universal here.

```python
vbt.x.annual_to_period(0.0525, 1)   # 0.00014583  — one day at 5.25%, ACT/360
vbt.x.annual_to_period(0.0525, 3)   # 0.0004375   — a Wednesday triple day
```

### `vbt.x.FREDData`

```python
FREDData.download(series_ids=None, base="GBP", start=None, end=None, cache=True, **kwargs)
```

A `Data` subclass following the `YFData` / `BinanceData` pattern. Caching is keyed by
series and date range and **refreshed on load** for any range extending past the cached
end, so history is not re-downloaded. If FRED is unreachable it uses the cache with a
warning, failing only when there is neither cache nor static fallback.

Needs a free FRED API key (`FRED_API_KEY`). Optional — see `static_rate_table`.

---

## 10. Holding costs

Two conceptually distinct charges, deliberately kept in separate modules with separate
rate tables and separate multipliers.

### `vbt.x.financing_rate(benchmark, markup, is_long)` → `float`

**The markup is always adverse**: `benchmark + markup` financing a long,
`benchmark − markup` when short. A simple addition credits shorts too generously and
flatters short-heavy backtests.

### `vbt.x.financing_charge(borrowed, benchmark_rate, markup, is_long, day_multiplier)` → `float`

Signed cash adjustment (negative = charged). Applies to the **borrowed portion only** — a
£10,000 position fully covered by £10,000 cash finances nothing.

```python
vbt.x.financing_charge(40_000, 0.0525, 0.03, is_long=True,  day_multiplier=1)  # -9.1667
vbt.x.financing_charge(40_000, 0.0525, 0.03, is_long=False, day_multiplier=1)  # +2.50
vbt.x.financing_charge(40_000, 0.0525, 0.03, is_long=True,  day_multiplier=3)  # -27.50
```

Long pays `(5.25 + 3)% × 40,000 / 360`; short receives `(5.25 − 3)% × 40,000 / 360`. If
`benchmark − markup` goes negative, the short is charged — which is correct.

### `vbt.x.carry_charge(notional, tom_next_differential, day_multiplier)` → `float`

Applies to **any** FX position regardless of leverage, because you are long one currency
and short another either way. Must not be merged with financing.

### `vbt.x.tom_next_differential(base_rate, quote_rate)` → `float`
The tom-next differential, not an interpolated one.

### `vbt.x.day_multipliers(index, triple_day="WED", end_of_day="22:00", holidays=None)` → `ndarray`

`1×` per trading day, **`3×` on the triple day** (covering the weekend, since FX spot
settles T+2 and a Wednesday position settles the following Monday), `0` on non-trading
days.

⚠️ Do **not** additionally charge calendar days. Combining the two conventions bills nine
days a week.

The triple day is configurable, not hardcoded to Wednesday. `end_of_day` is **inert on
daily bars** — one bar per day means the cutoff cannot change which bar a charge lands on.
It matters only on intraday data.

---

## 11. Metrics

All metrics appear in the relevant `.stats()` output automatically, with no explicit
selection, and are tagged so `tags="vectorbtx"` or `tags="mt5"` selects them.

### Returns metrics — `pf.returns_stats()`, `rets.vbt.returns.*`

| Method | Metric | Notes |
|---|---|---|
| `.ulcer_index()` | Ulcer Index | Martin (1987). **Percentage** drawdown RMS, not absolute — squaring absolute drawdowns makes the metric scale with account size. |
| `.ulcer_performance_index()` | UPI / Martin ratio | `(annualised return − rf) / UI`. |
| `.m2(benchmark_rets=)` | M² | `rf + Sharpe × σ_benchmark`. Reads as a return. Needs a benchmark. |
| `.treynor_ratio(benchmark_rets=)` | Treynor | Excess return per unit of beta. Needs a benchmark. |
| `.geometric_sharpe_ratio()` | Geometric Sharpe | On **log** returns. Upstream's Sharpe is arithmetic, so this is a genuine gap. |
| `.snr()` | Efficiency ratio | `\|P_end − P_start\| / Σ\|ΔP\|`, bounded 0–1. Path directness. Complements the K-ratio, which measures trend consistency. |
| `.k_ratio(version=3)` | K-ratio | Regression of cumulative log returns on time. v1/v2/v3 differ only in normalisation; **v3 is the default** and is the only length-independent one. |

Rolling variants: `.rolling_ulcer_index()`, `.rolling_geometric_sharpe_ratio()`,
`.rolling_snr()`, `.rolling_k_ratio()`.

**Deliberately not built** — already native, do not duplicate: Jensen's alpha (vectorbt's
`alpha` already *is* annualised Jensen's), Beta, Sharpe, Sortino, Calmar, Omega, Skew,
Kurtosis, Tail Ratio, Common Sense Ratio, VaR, CVaR, up/down capture.

Metrics requiring a benchmark set `check_has_benchmark_rets`; annualised ones set
`check_has_year_freq`. Without those filters they fail confusingly rather than cleanly.

### Trade metrics — `pf.trades.stats()`

**Already native, not rebuilt (13):** Profit Factor, Expected Payoff (`expectancy`),
Maximal Drawdown, Relative Drawdown, Equity Drawdown, Sharpe, Total Trades, Profit
Trades %, Largest Profit/Loss Trade, Average Profit/Loss Trade, Max Consecutive Wins/Losses
counts.

**Added (18):**

| Method | Metric |
|---|---|
| `.total_net_profit()` | Total Net Profit |
| `.gross_profit()` / `.gross_loss()` | Gross Profit / Loss |
| `.loss_rate()` | Loss Trades % |
| `.absolute_drawdown(init_cash=)` | `initial_deposit − minimum_balance`, in **currency** |
| `.balance_drawdown(init_cash=)` | MT4 *Balance* drawdown — closed positions only |
| `.recovery_factor(init_cash=)` | Net profit / max DD |
| `.max_consecutive_wins_money()` | PnL of the **longest** winning run |
| `.max_consecutive_losses_money()` | PnL of the longest losing run |
| `.maximal_consecutive_profit()` | PnL of the **most profitable** run |
| `.maximal_consecutive_loss()` | PnL of the worst run |
| `.avg_consecutive_wins()` / `.avg_consecutive_losses()` | Mean run length |
| `.lr_correlation()` / `.lr_standard_error()` | Regression byproducts |
| `.ahpr()` / `.ghpr()` | Arithmetic / geometric holding-period return |
| `.z_score()` | Wald–Wolfowitz runs test |

Plus `long.win_rate` and `short.win_rate` via the existing filters.

**Note the inversion on drawdowns:** vectorbt's `value` already includes floating open
positions, so MT5's *Equity* drawdown comes free and MT4's *Balance* drawdown is the gap —
which is why it needs a synthetic closed-trade curve.

**Z-Score interpretation**, since it is constantly misread: **negative Z means outcomes
cluster into streaks**, positive means they alternate. It is a dependency test, not a
performance measure.

`absolute_drawdown` needs a **resolved** `init_cash`. Under `InitCashMode.Auto` the raw
field is a mode flag, not a figure, so pass `pf.get_init_cash()`.

**Curated subset on `pf.stats()`:** Total Net Profit, Recovery Factor, Absolute Drawdown,
GHPR. Everything else stays on `pf.trades.stats()`, keeping the default table readable.

### `vbt.x.aggregate(results, how="geometric", weights=None)` → `Series`
Combines several `.stats()` outputs. `"geometric"` compounds rows whose label contains
"Return" and averages the rest — the only sensible default when rows mix returns, ratios
and durations. Also `"mean"`, `"median"`, `"sum"`.

---

## 12. Portfolio constructors

### `vbt.Portfolio.from_weights(close, weights, schedule=None, universe=None, min_trade=None, equity=None, leverage_budget=True, **kwargs)`

Investing as a first-class use case, sitting alongside `from_signals`, `from_orders` and
`from_holding`. Applies the leverage budget, quantizes, emits NaN-off-schedule targets, and
calls `from_orders`.

```python
pf = vbt.Portfolio.from_weights(
    close,
    weights=vbt.x.equal_weight(close.columns, close.index),
    schedule=vbt.x.calendar_schedule(close.index, "M"),
    universe=universe,
    init_cash=100_000,
)
pf.x.drift()      # what quantization and suppression cost you
```

Three defaults are **load-bearing, not niceties**:

- `size_type="targetpercent"` — `Percent` is not a substitute. Upstream explicitly warns it
  misbehaves with `call_seq="auto"` and does not support position reversal.
- `cash_sharing=True` with `group_by=True` — the cash-sharing unit is the only coherent
  budget boundary.
- `call_seq="auto"` — orders sells before buys within a segment so sells fund buys.
  Without it a rebalance can fail for lack of cash despite the weights being perfectly
  fundable.

You never compute the buy/sell orders; `TargetPercent` derives them.

### `vbt.Portfolio.from_holding_costs(close, size, universe, rates=None, **kwargs)`

Routes to `from_order_func`, the only entry point exposing the per-bar callbacks holding
costs require. Slower and more fragile than `from_orders` — use it only when financing or
carry are actually wanted. Everything else stays on the simple path, and keeping that fork
narrow is the central architectural goal.

Sets `call_post_segment=True`, which **defaults to `False`**. Without it the callback never
fires on bars where no order was placed — exactly where financing accrues — producing
output that looks plausible and is wrong.

### `pf.x` — the portfolio namespace

| Member | Returns |
|---|---|
| `.universe` | the `Universe` used, or None |
| `.drift()` | quantization drift recorded at construction |
| `.holding_costs()` | per-bar cash adjustments |
| `.borrowed_portion()` | borrowed vs cash-covered |
| `.margin_level()` / `.min_margin_level()` | margin ratios |

---

## 12b. Margin, holding-cost and FX helpers

### `vbt.x.buying_power(universe, init_cash, columns, cash_sharing=False)` → `(engine_cash, loan)`
Splits account **equity** into the engine's buying power and the broker's margin loan. Called
for you by the front door; useful directly when building a portfolio by hand. See
README → Leverage and margin.

### `vbt.x.leverage_of(universe, columns)` → `np.ndarray`
Per-column maximum leverage, defaulting to 1.

### `vbt.x.financing_basis(mode)` → `"borrowed" | "notional"`
What financing is charged on. `equity` finances its **debit balance** — a position covered by
cash finances nothing. `spreadbet` and `cfd` finance the **full notional**, because the broker
funds the whole position and you borrow nothing, so an unleveraged CFD still pays in full.

### `vbt.x.triple_day(asset_class=None, mode=None)` → `"WED" | "FRI"`
Which weekday carries the weekend roll. `fx` → Wednesday, because spot settles T+2 and the
roll reaches Monday; everything else → Friday, covering the calendar weekend. Both bill seven
days a week; the difference shows in per-trade cost, not annual cost.

### `vbt.x.day_count_for(currency)` → `float`
The ACT divisor for a market's denomination: 365 for GBP, SGD and ZAR markets; 360 elsewhere.

### `vbt.x.as_fx_table(fx, quote=None, base=None, index=None)` → `FXTable`
Coerces whatever a data provider gave you — a `GBPUSD=X` close series, a `Data` object, a
dict of flat rates — into an `FXTable`. Reciprocals and crosses are **derived**, never
fetched, so `value(A,B) × value(B,A) = 1` holds exactly and no round trip can manufacture
phantom P&L.

### `vbt.x.financing_series(...)`, `vbt.x.carry_series(...)`
Vectorised forms of `financing_charge` and `carry_charge` across a whole index.

### `vbt.Portfolio.from_holding_costs(close, size, universe, rates=None, ...)`
The level-3 path: replays orders through `from_order_func` so the simulation's own cash
reflects the charges as it runs. ⚠️ It needs **explicit sizes** and cannot express entries
and exits. For everything else, pass `rates=` to the front door — charges are then computed
from the realised positions and folded into reported cash, which works for any strategy
type and keeps the level-3 dependency off the default path.

---

## 13. Settings

`settings["vectorbtx"]`, registered at import. **`settings["portfolio"]` is never
touched** — mutating it would change behaviour for plain vectorbt code paths and break the
drop-in guarantee.

```python
vbt.settings["vectorbtx"]["rounding"] = "nearest"
vbt.settings["vectorbtx"]["financing_markup"] = 0.025
```

| Key | Default | Meaning |
|---|---|---|
| `rounding` | `"floor"` | Global quantization direction |
| `financing_markup` | `0.03` | Broker markup over benchmark, always adverse |
| `fx_markup` | `0.005` | Broker's worse-than-market FX rate |
| `slippage_max` | `0.005` | Auto-slippage ceiling (50bp) |
| `slippage_coef` | `1.0` | Auto-slippage coefficient |
| `vol_window` | `20` | Realised-volatility window |
| `triple_day` | `"WED"` | Triple accrual day |
| `end_of_day` | `"22:00"` | Broker cutoff (inert on daily bars) |
| `day_count` | `360.0` | ACT/360 |
| `fill_policy` | `"ffill_with_warning"` | Index alignment policy |
| `rate_series` | SONIA/SOFR/ESTR/TONA/SARON | Default benchmark per currency |
| `static_rate` | `None` | Fallback when FRED is unavailable |
| `triangulation_tolerance` | `1e-4` | FX consistency check threshold |

---

## Exceptions

| Exception | Raised by | When |
|---|---|---|
| `SpecError` | `instruments.validate` | Invalid instrument or universe |
| `UniverseError` | `resolve.load` | Join-key mismatch in either direction |
| `CoercionError` | `resolve.coerce` | Ambiguous spreadsheet cell |
| `KeywordError` | `resolve.keywords` | Unrecognised or misapplied keyword |
| `ResolveError` | `resolve.resolver` | `'auto'` cannot be resolved |
| `CostError` | `costs.base` | Missing volume or notional context |
| `FXError` | `fx.table` | Missing pair or failed consistency check |
| `ShadowError` | `core.patch` | vectorbtX would overwrite an upstream attribute |

Every one of these exists because the alternative was a plausible-looking wrong number
rather than a crash.
