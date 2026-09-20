# vectorbtX

A **pure wrapper** over [vectorbt](https://github.com/polakowo/vectorbt) 1.1.0 adding
instrument abstractions, leverage, multi-currency, holding costs, portfolio construction
and the full MT4/MT5 metric set.

```python
import vectorbtx as vbt          # drop-in replacement for `import vectorbt as vbt`
```

Everything upstream still works, unchanged and bit-identical. Extras are additive: new
parameters defaulting to inert values, new methods, and the `.x` namespace. Metrics are
the deliberate exception — they appear next to native ones in `.stats()`.

vectorbtX does **not** fork vectorbt, does not vendor its source, and does not modify its
simulation loop. It subclasses and patches at import time.

---

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Getting your data](docs/DATA.md)
- [Leverage and margin](#leverage-and-margin)
- [Holding costs](#holding-costs)
- [The four instrument modes](#the-four-instrument-modes)
- [Architecture: the simplicity ordering](#architecture-the-simplicity-ordering)
- [**Complete file map**](#complete-file-map) — every file, class and function
- [Call graph](#call-graph)
- [Known limitations](#known-limitations)
- [Testing](#testing)

Further reading: [`docs/API.md`](docs/API.md) for the full reference,
[`docs/DATA.md`](docs/DATA.md) for where prices, rates and FX come from, and
[`docs/EXAMPLES.md`](docs/EXAMPLES.md) for the runnable examples. Design notes, audits and
the build log live in [`docs/dev/`](docs/dev/) and are not needed to use the library.

---

## Install

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"   # or: pip install -e ".[dev]"
uv pip install -e ".[excel,rates]"                    # + openpyxl for .xlsx universes,
                                                      #   requests for FRED
```

Python 3.11–3.14, mirroring upstream. The import name is lowercase — `import vectorbtx as
vbt` — even though the distribution is published as `vectorbtX`.

Two things that will bite you on a first install:

- **`plotly<7` is pinned, and it is not optional.** vectorbt 1.1.0 declares `plotly>=4.12.0`
  with no upper bound but registers a plotly template containing a `scattermapbox` key,
  which plotly 7 removed. Without the cap, `import vectorbt` itself raises `ValueError`
  before any of this library loads. 6.9.0 is the newest that works.
- **Behind a TLS-inspecting proxy, `uv` needs `--native-tls`**, or it fails every fetch with
  `invalid peer certificate: UnknownIssuer`.

---

## Quick start

One call. Every argument in it. Returns a `Portfolio`.

```python
import vectorbtx as vbt

data = vbt.YFData.download("EURUSD=X", start="2019-01-01", end="2024-12-31")
close, open_ = data.get("Close"), data.get("Open")
high, low = data.get("High"), data.get("Low")

fast, slow = vbt.MA.run(close, 50), vbt.MA.run(close, 200)

pf = vbt.x.SingleBacktest(
    close=close, open=open_, high=high, low=low,   # all four required, by keyword
    entries=fast.ma_crossed_above(slow),
    exits=fast.ma_crossed_below(slow),

    mode="spreadbet", asset_class="fx",   # -> 30x cap, carry, Wednesday roll
    point_size=0.0001,                    # the pip
    base_currency="EUR", quote_currency="USD",
    spread_points=0.9,

    risk_pct=0.01, stop_points=60, target_points=120,   # sized in pips, not fractions
    leverage=30,

    init_cash=10_000, freq="D",
)

pf.stats()                      # + Total Net Profit, Recovery Factor, Absolute Drawdown, GHPR
pf.trades.stats()               # + the full MT4/MT5 trade set
pf.returns_stats()              # + Ulcer, UPI, M², Treynor, SNR, K-ratio, geometric Sharpe
pf.x.user_size().iloc[-1]       # "£1.66/pt", not 16600 engine units
```

**All four price series are required, passed by keyword, and orders fill at the next bar's
open.** A signal
from bar `t`'s close is acted on when bar `t+1` opens, so entries, exits and rebalance dates
shift one bar and fill at `open`; sizes, stop distances and spread fractions are priced at
that open. Stops are checked against `high` and `low`, and `pf.x.intrabar_drawdown()`
measures the worst moment against them — routinely a third worse than the close-to-close
figure. None of these has a default, because every available default flatters the result — and they
are keyword-only, because four same-shaped frames in a row is exactly where a silent
transposition produces a plausible wrong answer instead of an error.

You never convert to engine units, never quantize by hand, and never splat a resolved
argument dict. `mode` decides which other arguments are legal, required or rejected —
a spread bet with a `contract_size` raises, because that is a misunderstanding of the
product rather than a typo.

**Multi-instrument** is the same call with `vbt.x.PortfolioBacktest`, and instruments can
be supplied four ways — flat keywords, per-symbol overrides on a shared base, a CSV/Excel
path, or `Instrument` objects. All four produce an identical `Universe`.

**The layer underneath is still public.** `universe.resolve(close)` returns the `Order`
kwargs, and `Portfolio.from_orders`/`from_signals`/`from_weights` take them directly — use
that when you need to inspect or replace one step. See
[`examples/strategies/appendix_explicit.py`](examples/strategies/appendix_explicit.py) for
the same strategy both ways.

---

## The four instrument modes

| Mode | Sized in | P&L | Min size | Increment | Max leverage |
|---|---|---|---|---|---|
| `spreadbet` | £ per point | `points × stake` | £1/pt | £0.01 | 5× / 10× / 30× |
| `cfd` | lots | `Δprice × contract_size × lots` | 0.01 lots | 0.01 | 5× / 10× / 30× |
| `fx_spot` | lots | `Δprice × contract_size × lots` | 0.01 lots | 0.01 | 2× |
| `equity` | shares | `Δprice × shares` | 1 share | 1 | 2× |

Leverage tiers are stocks / commodities-and-indices / FX respectively.

**All four collapse to one engine expression**, `engine_size × Δprice`. That is what makes
cross-mode equivalence testable:

```
spread bet:  engine_size = stake_per_point / point_size
cfd, fx:     engine_size = lots × contract_size
equity:      engine_size = shares
```

A £1/point spread bet on a UK stock quoted in pence (`point_size = 0.01`) is 100 shares —
the "100 contract size" is *derived*, not configured.

---

## Leverage and margin

vectorbtX models a margin account the way a broker does. Leverage is **funded**, not merely
capped: a 30× spread bet is filled at the size its leverage implies, and the account is
reported at its true equity throughout.

| | |
|---|---|
| the engine receives | buying power, `equity × leverage` |
| the portfolio records | the loan, `equity × (leverage − 1)` |
| `get_init_cash` reports | **equity** — the loan subtracted back out |

Reported cash therefore goes negative by exactly the drawn loan, and
`value = cash + asset_value` returns true equity. Everything downstream — `returns`,
`returns_stats`, every metric — derives from those two, so it all follows without touching
the engine.

```python
pf.margin_loan               # buying power beyond your own equity
pf.cash().iloc[-1]           # negative when the loan is drawn
pf.x.borrowed_portion()      # max(0, -cash) — exact by construction
pf.x.min_margin_level()      # closest the account came to a margin call
```

⚠️ Borrowed is not notional: £40,000 held against £10,000 of equity borrows £30,000. That
distinction is what `equity` mode charges financing on. Derivatives work the other way —
see below.

**Not modelled:** maintenance margin, margin calls, forced liquidation. A position can
draw the whole loan and sit there, where a real broker would have closed it.

---

## Holding costs

What you pay to hold a position overnight depends on the **product**, not just the
instrument. The rules below are those of UK retail brokers; the evidence behind each is in
[`docs/dev/AUDIT-G.md`](docs/dev/AUDIT-G.md).

These follow from `mode` and `asset_class`; you do not set them per instrument.

| | financing charged on | overnight cost | day count | weekend roll |
|---|---|---|---|---|
| `equity` (margin share dealing) | the **debit balance** | financing | by currency | Friday |
| `spreadbet` / `cfd`, non-FX | the **full notional** | financing | by currency | Friday |
| `spreadbet` / `cfd`, FX | — | **carry only** | by currency | Wednesday |
| `fx_spot` | — | **carry only** | by currency | Wednesday |

- **Financing and carry are alternatives, never both.** Tom-next already prices both
  currencies' interest, so financing on top would bill the same money twice. Setting both
  raises.
- **An unleveraged CFD still pays in full.** You borrow nothing, but the broker funds the
  entire position.
- **Day count follows the market's denomination:** /365 for GBP, SGD and ZAR; /360
  elsewhere.
- **The markup is always adverse:** `benchmark + fee` long, `benchmark − fee` short. When
  the fee exceeds the benchmark, both sides pay.
- **FX rolls Wednesday** (spot settles T+2, so the roll reaches Monday); everything else
  rolls **Friday** (nothing settles — the charge just covers the calendar weekend). Both
  bill seven days a week.

Supplying `rates=` opts in; leave it out and no holding costs are applied. Charges are
folded into reported cash, so `value`, `returns` and every metric include them:

```python
pf = vbt.x.PortfolioBacktest(close=close, open=open_, high=high, low=low, ..., rates=rates)
pf.holding_costs             # per-bar, per-column signed cash adjustments
```

`Portfolio.from_holding_costs` is the level-3 alternative: it replays orders through
`from_order_func` so the simulation's own cash reflects the charges as it runs. Use it only
when a strategy sizes off live equity while near fully invested — it needs explicit sizes
and cannot express entries and exits.

---

## Architecture: the simplicity ordering

Since targeting the Rust engine was ruled out, **simplicity decides the approach**:

- **Level 1 — parameterise.** Compute array-likes and hand them to existing `Order`
  fields. Nothing in the simulation changes. Covers commission, spread, slippage,
  quantization, unit conversion, leverage caps, SL/TP and all per-asset variation.
  *Most of the library lives here.*
- **Level 2 — pre-compute.** Weight matrices, schedules, FX-converted price series, rate
  tables — built in pandas before simulation.
- **Level 3 — simulation callbacks.** `from_order_func` with `post_segment_func_nb`.
  **Only `holding/` lives here.** That is the entire blast radius of our dependency on
  vectorbt internals.

```
vectorbtx/
  core/         import hook, Portfolio subclass, accessor patching, settings
  instruments/  instrument spec + validation                     (level 1)
  resolve/      dict|DataFrame → validated specs → numpy args     (level 1)
  quantize/     thin wrapper over size_granularity                (level 1)
  costs/        cost modules → Order field array-likes             (level 1)
  modes/        unit adapters                                     (level 1)
  margin/       leverage caps                                     (level 1)
  metrics/      returns metrics (K1) + trade metrics (K2)          (level 1)
  fx/           base-currency pre-conversion                       (level 2)
  rates/        FRED rate data + broker markup                     (level 2)
  construct/    weighting + rebalance scheduling                   (level 2)
  holding/      financing + carry                                  (level 3)
```

---

## Complete file map

Every file, every class, every function: what it does, what it calls, what calls it.

`→` means *calls*. `←` means *called by*. Leading `_` marks a private helper.

---

### `vectorbtx/__init__.py`

The import hook. Star-imports the whole vectorbt namespace, shadows `Portfolio` with the
vectorbtX subclass, applies patches, registers settings, and exposes `vbt.x`.
Mirrors upstream's own 39-line `__init__.py` shape.

→ `core.portfolio.Portfolio`, `core.patch.apply_patches`, `core.settings.register_defaults`, `x`
← the user

### Documentation

| File | Contains |
|---|---|
| `README.md` | This file: what vectorbtX is, the complete file map, the call graph, known limitations, and the log of corrections. |
| `docs/API.md` | API reference — every public symbol with signature, arguments, raises and an example. |
| `docs/EXAMPLES.md` | Guide to the runnable examples: the three strategies and the nine-file reference tour. |
| `docs/dev/PLAN.md` | The design plan, Sections A–N. Part 9 lists everything outstanding. Section N (front-door API) is marked NOT IMPLEMENTED there; it has since been built — see `vectorbtx/frontend/`. |
| `docs/dev/AUDIT.md` | Verification of the plan's 63 acceptance criteria, plus the scope-creep sweep. Now a static record: the `audit_plan.py` that generated it has been removed. |
| `docs/dev/FRONTEND-API.md` | The agreed front-door API design, with complete worked examples. |

### `vectorbtx/x.py`

The `vbt.x` namespace: 49 re-exported constructors, loaders and free functions. Flat by
design, so `vbt.x.<tab>` is a complete inventory of the extras surface.

← `__init__.py`

---

### `core/` — foundations (Section B)

#### `core/_numba.py`
| Symbol | Does | Calls | Called by |
|---|---|---|---|
| `njit` | Real `numba.njit` if available, else a no-op decorator. Lets the pure-arithmetic modules import and be unit-tested without numba. | — | `metrics/returns/nb.py`, `holding/callback.py` |
| `HAS_NUMBA` | Whether numba was importable. | — | `holding/callback.py` |

#### `core/settings.py`
| Symbol | Does | Calls | Called by |
|---|---|---|---|
| `DEFAULTS` | Every vectorbtX default in one dict: rounding, markups, accrual, rate series. | — | `get`, `register_defaults` |
| `register_defaults(force=True)` | Adds the top-level `settings["vectorbtx"]` key. Never touches `settings["portfolio"]` — that would change plain vectorbt behaviour. Idempotent. | vbt `settings.__setitem__` | `__init__.py` |
| `get(key, default)` | Reads one setting, falling back to `DEFAULTS` when vectorbt is absent. | vbt `settings` | `quantize`, `costs`, `holding`, `fx` |

#### `core/patch.py`
| Symbol | Does | Calls | Called by |
|---|---|---|---|
| `ShadowError` | Raised when vectorbtX would overwrite an upstream attribute. | — | `attach`, `merge_metrics`, `accessors` |
| `attach(cls, name, obj, overwrite=False)` | `setattr` with a **collision guard**: raises rather than shadowing. The only defence against silently overwriting a future upstream method. | — | both `register.py` files |
| `merge_metrics(cls, entries)` | Merges `_metrics` entries in place. Upstream explicitly sanctions this. Raises on key collision. | — | both `register.py` files |
| `apply_patches()` | Runs all three registrations. Idempotent. | `register_returns_metrics`, `register_trade_metrics`, `register_accessors` | `__init__.py` |

#### `core/accessors.py`
| Symbol | Does | Calls | Called by |
|---|---|---|---|
| `PortfolioXAccessor` | `pf.x` — portfolio-bound extras. Methods: `universe`, `borrowed_portion`, `margin_level`, `min_margin_level`, `drift`, `holding_costs`. | `margin.exposure.*` | `core.portfolio.Portfolio.x` |
| `_XAccessorBase` | `df.vbt.x` — data-bound extras: `to_base_currency`, `to_returns_metrics`. | `fx.convert.to_base_currency` | registered accessors |
| `register_accessors()` | Registers `x` on `Vbt_SRAccessor` / `Vbt_DFAccessor`, guarding against collision. | vbt `register_*_vbt_accessor`, `ShadowError` | `apply_patches` |

#### `core/portfolio.py`
| Symbol | Does | Calls | Called by |
|---|---|---|---|
| `Portfolio` | Subclass of vbt's. Adds `from_weights`, `from_holding_costs`, `.x`, and carried state (`universe`, `drift`, `holding_costs`). | vbt `Portfolio` | `__init__.py`, the user |
| `.__init__` | Stores extra state and merges it into `.config` so `.replace()` round-trips it. | vbt `Portfolio.__init__` | — |
| `.indexing_func` | Slices vectorbtX state alongside upstream's fields. Without this, per-column state silently desynchronises from its columns. | `Universe.subset` | vbt indexing |
| `.from_weights` | **Section J's entry point.** Applies the leverage budget, quantizes, emits NaN-off-schedule targets, then calls `from_orders` with `TargetPercent` + `cash_sharing` + `call_seq="auto"`. | `construct.rebalance.*`, `Universe.resolve`, vbt `from_orders` | the user |
| `.from_holding_costs` | **Section G's entry point.** Routes to `from_order_func`, the only path with per-bar callbacks. | `holding.callback.simulate_with_holding_costs` | the user |

---

### `instruments/` — the specification (Section C)

#### `instruments/spec.py`
| Symbol | Does | Calls | Called by |
|---|---|---|---|
| `Instrument` | Frozen dataclass, 30 fields, describing one tradable instrument. The vocabulary every later section speaks. | `validate_spec` | everything |
| `.__post_init__` | Validates on construction — bad specs cannot exist. | `validate.validate_spec` | dataclass |
| `.replace(**kw)` | Copy with fields replaced. Re-validates. | `dataclasses.replace` | `resolve`, tests |
| `.filled()` | Copy with every unspecified field resolved to its mode default. | `defaults.fill_defaults` | every consumer |
| `.to_dict()` | Flat dict of all fields, for export and round-tripping. Joins `active_costs` with `"\|"`. | — | `resolve.export` |
| `.is_nominal_points` | True for spread bet and CFD — the modes that cannot use adjusted prices (D10). | — | `validate`, `resolver` |
| `.is_derivative` | True where there is no ownership of the underlying. | — | `margin`, `costs` |

#### `instruments/defaults.py`
| Symbol | Does | Calls | Called by |
|---|---|---|---|
| `MODES`, `ASSET_CLASSES` | The valid enumerations. | — | `validate`, `coerce` |
| `MODE_DEFAULTS` | Per-mode default field values (Part 8 of the plan). | — | `fill_defaults`, `_lookup` |
| `MAX_LEVERAGE` | `(mode, asset_class) → max leverage`. | — | `max_leverage` |
| `DEFAULT_STACKS` | `(mode, asset_class) → active cost modules`. Spread bets carry no commission; anything with a currency pair underneath carries. | — | `default_cost_stack` |
| `_lookup(table, mode, asset_class)` | Resolves an unspecified `asset_class` to the mode's own default before falling back to `(mode, None)`. | — | `max_leverage`, `default_cost_stack` |
| `mode_defaults(mode)` | Default field values for a mode. | — | `fill_defaults` |
| `max_leverage(mode, asset_class)` | Default maximum leverage. | `_lookup` | `validate`, `margin.cap` |
| `default_cost_stack(mode, asset_class)` | Default active cost modules. | `_lookup` | `fill_defaults`, `costs.compose` |
| `FRACTIONAL_INCREMENT` | `1e-6` — the fill step when `allow_fractional` is set, so the flag actually changes behaviour. | — | `fill_defaults` |
| `fill_defaults(spec)` | The workhorse: resolves every `None` to its mode default. `None` means *unspecified* throughout; NaN appears only at emit. | `mode_defaults`, `max_leverage`, `default_cost_stack` | `Instrument.filled` |
| `field_names()` | Ordered field list, for the spreadsheet schema. | — | `resolve.export`, `resolve.load` |

#### `instruments/validate.py`
| Symbol | Does | Calls | Called by |
|---|---|---|---|
| `SpecError` | Raised on an invalid spec or universe. | — | callers |
| `_positive(value, name)` | Shared "must be > 0" check, mirroring `nb.py:414`. | — | `validate_spec` |
| `validate_spec(spec)` | Rejects: `contract_size` on a spread bet; missing `contract_size` on CFD/FX; `increment <= 0`; `min_size` not a multiple of `increment`; `allow_fractional` on a mode that cannot fractionalise; unresolvable currency combinations; leverage above the mode cap. | `_positive`, `max_leverage` | `Instrument.__post_init__` |
| `validate_universe(specs)` | Cross-spec rules — chiefly rejecting `portfolio_level` combined with differing per-ticker leverage values, rather than silently picking one. | — | `Universe.__init__` |

---

### `resolve/` — universe loading and argument resolution (Section C2)

#### `resolve/load.py`
| Symbol | Does | Calls | Called by |
|---|---|---|---|
| `NAME_COLUMNS` | Accepted names for the instrument-name join key. | — | `_from_frame` |
| `UniverseError` | Raised on join-key mismatch or bad input. | — | callers |
| `Universe` | `dict[column, Instrument]` plus resolution. Validated once, on the converged form. | `validate_universe` | the user, `Portfolio.from_weights` |
| `.subset(cols)` | Universe restricted to given columns. | — | `Portfolio.indexing_func` |
| `.order_for(cols)` | Specs **in the wrapper's column order**, not dict insertion order. | — | `resolver`, `costs.compose` |
| `.resolve(data, rates=None)` | The main event: per-column `Order` kwargs. | `resolver.resolve_universe` | `from_weights`, the user |
| `.to_frame()` / `.to_csv()` | Fully-resolved universe including filled defaults, for eyeballing and round-tripping. | `export_universe` | the user |
| `_from_dict(d)` / `_from_frame(df)` | The two front doors. **They converge before validation** — the DataFrame path must not bypass Section C's checks. | `coerce_field`, `coerce_name` | `load_universe` |
| `load_universe(source)` | dict, DataFrame, CSV path or Excel path → `Universe`. | `_from_dict`, `_from_frame` | the user |

#### `resolve/coerce.py`
| Symbol | Does | Calls | Called by |
|---|---|---|---|
| `BOOL/STR/KEYWORD/SET_FIELDS` | Declared type of each spreadsheet column. | — | `coerce_field` |
| `CoercionError` | Raised on an ambiguous cell, naming the cell. | — | callers |
| `is_blank(v)` | **Blank ≠ NaN.** Blanks become `None` and resolve to mode defaults; NaN in an `Order` field *means* something. | — | every coercer |
| `coerce_str/bool/float/name` | Strict per-type coercion. Handles `"FALSE"`, `"no"`, `"0"`, thousands separators, trailing whitespace, and numeric-looking tickers (`1234` must not become `"1234.0"` and fail to join). | `is_blank` | `coerce_field` |
| `coerce_keyword(v, field)` | **Numeric first, then keyword set, then error.** A typo'd `'atuo'` raises instead of falling through to a default. | `check_keyword` | `coerce_field` |
| `coerce_set(v)` | `"spread\|slippage"` → frozenset. Unknown module names raise — a silently dropped cost flatters the backtest. | — | `coerce_field` |
| `coerce_field(field, v, where)` | Dispatches by declared type. | all the above | `_from_frame` |

#### `resolve/keywords.py`
| Symbol | Does | Calls | Called by |
|---|---|---|---|
| `KEYWORDS` | Per-parameter keyword registry. Deliberately tiny: `slippage='auto'` and `financing='auto'` only. | — | `is_keyword`, `check_keyword` |
| `KeywordError` | Raised on an unrecognised keyword, or a valid keyword on a parameter that does not accept it. | — | `check_keyword` |
| `is_keyword(field, v)` | Membership test. | — | `resolver`, `costs` |
| `check_keyword(field, v, where)` | Fatal check, naming the instrument and column. | — | `coerce_keyword` |

#### `resolve/resolver.py`
| Symbol | Does | Calls | Called by |
|---|---|---|---|
| `resolve_universe(universe, data, rates, equity, fx_table, base_currency, volume, size, notional, ...)` | Specs → per-column numpy arrays for every `Order` field, in wrapper column order. Warns when a nominal-point mode meets adjusted data (D10). | `costs.compose.resolve_costs`, `quantize.to_order_params`, `margin.cap.max_size_for` | `Universe.resolve` |
| `_price_frame(data)` | Extracts the price frame from a `Data`, DataFrame or Series. | — | `resolve_universe` |
| `_collapse(values)` | Returns a scalar when every column agrees, else a list — keeps broadcasting cheap. | — | `resolve_universe` |
| `ResolveError` | Raised when `'auto'` cannot be resolved (no volume column, no rate table). | — | callers |

#### `resolve/export.py`
| Symbol | Does | Calls | Called by |
|---|---|---|---|
| `export_universe(universe, path=None)` | Round-trip export of the fully resolved universe, extra metadata columns preserved. The best available defence against silent misalignment. | `Instrument.to_dict`, `field_names` | `Universe.to_frame/to_csv` |

---

### `quantize/` — fill quantization (Section D)

#### `quantize/quantizer.py`
| Symbol | Does | Calls | Called by |
|---|---|---|---|
| `_round_to(size, increment, direction)` | Floor / ceil / nearest on the **magnitude**, sign preserved, so floor always means *smaller position*. | — | `quantize` |
| `quantize(size, spec, direction=None)` | → `(quantized_size, drift)`. Sub-minimum sizes **reject to zero** rather than rounding to zero silently. | `_round_to`, `core.settings.get` | `construct.rebalance`, the user |
| `to_order_params(spec, direction)` | Floor → `{"size_granularity": increment, "min_size": …}` (the engine already floors, so it is free). Any other direction → `size_granularity=NaN` with the caller pre-quantizing. **Do not try to make the engine round differently.** | — | `resolver` |
| `drift_report(intended, actual, ...)` | Per-order drift, **kept separate from `max_size` effects** — `max_size` partially fills, so realised size can differ for reasons unrelated to rounding. | — | the user |
| `rebalance_drift(...)` | Per-rebalance aggregation. What anyone actually reads. | `drift_report` | `construct.rebalance`, `pf.x.drift` |

---

### `modes/` — unit adapters (Section H)

#### `modes/base.py`
| Symbol | Does | Calls | Called by |
|---|---|---|---|
| `ModeAdapter` | Protocol: `to_engine_units`, `from_engine_units`, `unit_name`. Two functions. **Resist making them more** — the moment an adapter makes decisions rather than converting units, the abstraction has failed. | — | the four adapters |
| `register(mode)` | Decorator populating the registry. | — | the four adapters |
| `get_adapter(mode_or_spec)` | Registry lookup; accepts a mode string or an `Instrument`. | — | `helpers`, `margin`, `resolver` |
| `adapters()` | Every registered adapter. | `get_adapter` | tests, docs |

#### `modes/spreadbet.py`, `cfd.py`, `fx_spot.py`, `equity.py`
| Adapter | `to_engine_units` | Notes |
|---|---|---|
| `SpreadBetAdapter` | `stake_per_point / point_size` | **No contract size** — the stake *is* the size. |
| `CFDAdapter` | `lots × contract_size` | `contract_size` alone describes any CFD. |
| `FXSpotAdapter` | `lots × contract_size` | Standard lot 100,000. Fractional pips preserved. |
| `EquityAdapter` | `shares` (identity) | The model vectorbt natively assumes. |

#### `modes/helpers.py` — mode-agnostic, built *above* the adapters
| Symbol | Does | Calls | Called by |
|---|---|---|---|
| `points_to_price_fraction(points, price, spec)` | `points × point_size / price`. Native SL/TP takes **fractions of price**, so this is the bridge. Do not reimplement stop logic. | — | the user |
| `price_fraction_to_points(...)` | The inverse. | — | the user |
| `value_per_point(user_size, spec)` | Money per point of movement. For a spread bet this is just the stake. | `get_adapter` | `margin`, the user |
| `size_for_risk(risk_amount, stop_points, spec, price)` | **One formula for all four modes**: `engine = risk / (stop_points × point_size)`, then convert back to user units. | `get_adapter` | the user |
| `size_for_notional(notional, price, spec)` | User size for a target exposure. Preserves `price`'s pandas index — a bare ndarray would strip it and silently reindex downstream. | `get_adapter`, `_like` | `margin.cap` |
| `_like(reference, values)` | Re-wraps a result in the pandas container of its price input. The guard against index-loss producing union-shaped arrays instead of errors. | — | every helper |
| `pnl(user_size, entry, exit, spec)` | P&L in account currency. The cross-mode equivalence test's assertion target. | `get_adapter` | tests, the user |

---

### `fx/` — multi-currency (Section I)

#### `fx/table.py`
| Symbol | Does | Calls | Called by |
|---|---|---|---|
| `FXError` | Raised on a missing pair or a failed consistency check. | — | callers |
| `FXTable` | Long form keyed `(date, quote, base) → value`, where `value` multiplies a `quote`-currency price to give `base`. **Direction lives in the key**, so there is no multiply-or-divide decision to get wrong. | — | `convert`, `costs` |
| `.from_canonical(frame, hub)` | Fetch one canonical direction and **derive** reciprocals and triangulations. Fetching both directions independently gives non-exact reciprocals, and round-trip conversion then produces phantom P&L that is random in sign — inflating measured volatility and quietly degrading every metric with a σ in the denominator. | `validate` | the user |
| `.from_long(frame)` / `.constant(rates)` | Alternative constructors; `constant` is the no-API-key path. | `validate` | the user, tests |
| `.get(quote, base, index=None)` | Rate series for one pair. | `align` | `convert`, `costs` |
| `.align(index, fill_policy)` | Explicit fill policy. FX trades 24/5, equities have exchange hours, holidays differ by country — mismatches are guaranteed, so this warns rather than silently forward-filling. | — | `.get` |
| `.validate(tolerance)` | Reciprocal consistency and triangulation `v(A,C) ≈ v(A,B) × v(B,C)`. Catches stale rows, bad joins and mixed-source contamination. | — | constructors |

#### `fx/convert.py`
| Symbol | Does | Calls | Called by |
|---|---|---|---|
| `ConvertedFrame` | DataFrame subclass retaining `.native` (the original series) and `.fx_markup`. | — | `to_base_currency` |
| `.decompose()` | Splits total return into asset return and FX return — the thing the converted series conflates. | — | the user |
| `to_base_currency(data, base, fx_table, ...)` | Converts OHLC with **one rate per bar**. Different rates per field would break `H ≥ C ≥ L` and make native SL/TP fire at prices that never existed. | `FXTable.get`, `check_ohlc_ordering` | the user, `df.vbt.x` |
| `check_ohlc_ordering(frame)` | The single validation covering both the FX hazard and D10's mixed-adjustment hazard. | — | `to_base_currency` |
| `convert_amount(amount, quote, base, table, markup, direction)` | Cash conversion with the **adverse** markup applied in both directions. | `FXTable.get` | `convert_fixed_costs` |
| `convert_fixed_costs(amount, cost_currency, base, table, index)` | Converts a foreign fixed fee. Percentage costs are FX-invariant; fixed and per-unit ones are not — this is what `spec.cost_currency` exists for. | `convert_amount` | `costs.commission` |

---

### `costs/` — per-trade costs (Section E)

#### `costs/base.py`
| Symbol | Does | Calls | Called by |
|---|---|---|---|
| `CostModule` | Protocol: `contribute(spec, price, ctx) → partial Order kwargs`. A cost engine returning a single scalar total would be the wrong shape. | — | the three modules |
| `_Base.applies(spec)` | Whether the module is in the spec's active stack. | — | `compose` |
| `CostError` | Raised when a cost cannot be resolved (missing volume, missing notional). | — | callers |

#### `costs/commission.py`
| Symbol | Does | Calls | Called by |
|---|---|---|---|
| `Commission.contribute` | Percentage → `fees`; flat and per-unit → `fixed_fees`. **A minimum is pre-computed per order into `fixed_fees`** with `fees` zeroed: `max(pct × notional, min)` is a conditional and genuinely cannot be expressed as `fees + fixed_fees`. Spread bets return zero for both. | `convert_fixed_costs` | `compose` |

#### `costs/spread.py`
| Symbol | Does | Calls | Called by |
|---|---|---|---|
| `Spread.contribute` | `spread_points` is the **full** bid/ask gap; half is charged per side, so a round trip costs the full spread. Emits into `slippage`. | — | `compose` |

#### `costs/slippage.py`
| Symbol | Does | Calls | Called by |
|---|---|---|---|
| `realised_volatility(price, window)` | 20-bar realised σ as a fraction of price. | — | `auto_slippage` |
| `auto_slippage(price, volume, size, ...)` | `clip(coef × σ × sqrt(size/ADV), 0, slippage_max)`. Square-root market impact is the standard empirical result and beats linear. | `realised_volatility` | `Slippage.contribute` |
| `Slippage.contribute` | Fixed fraction, or `'auto'`. **Fails at resolve time without a volume column** rather than silently falling back to zero. | `auto_slippage` | `compose` |

#### `costs/compose.py`
| Symbol | Does | Calls | Called by |
|---|---|---|---|
| `MODULES` | Name → module instance. | — | `resolve_costs` |
| `resolve_costs(universe, price, ...)` | Composes the active stack into `Order` kwargs. **Spread and slippage both land on `slippage`, so they are summed before assignment** — only one field exists. This is the least obvious constraint in the section. Also folds in the FX markup, which behaves exactly like a spread on price. | every module's `contribute` | `resolver` |
| `_scalarise` / `_collapse` / `_stack` | Return the smallest representation that broadcasts correctly — scalar, per-column list, or full frame. | — | `resolve_costs` |

#### `costs/helpers.py`
| Symbol | Does | Calls | Called by |
|---|---|---|---|
| `equivalent_pct(flat_fee, notional)` | £10 on £10,000 → 0.001. Retail equity commission is often flat; users often want one percentage. | — | the user |
| `equivalent_flat(pct, notional)` | The inverse. | — | the user |

---

### `margin/` — leverage caps (Section F, F1 only)

#### `margin/requirement.py`
| Symbol | Does | Calls | Called by |
|---|---|---|---|
| `notional(user_size, price, spec)` | Full market value regardless of cash backing, mode-specific. | `get_adapter` | `margin_required`, `exposure` |
| `margin_required(user_size, price, spec)` | `notional / leverage`. | `notional` | `min_margin_level`, the user |

#### `margin/cap.py`
| Symbol | Does | Calls | Called by |
|---|---|---|---|
| `leverage_cap(spec, equity, price)` | Maximum engine size at `spec.leverage`. | `max_leverage` | `max_size_for` |
| `max_size_for(specs, equity, price)` | The value to feed `Order.max_size`, which **partially fills rather than rejecting** — clip semantics for free. `raise_reject=True` gives reject semantics instead. | `leverage_cap` | `resolver` |
| `cap_user_size(size, spec, equity, price)` | Same cap expressed in user units. | `leverage_cap` | the user |

#### `margin/exposure.py`
| Symbol | Does | Calls | Called by |
|---|---|---|---|
| `borrowed_portion(pf)` | Post-simulation split into borrowed vs cash-covered, from `pf.cash(free=True)` and `pf.asset_value()`. **A hard requirement for Section G**, since F2 was not built. | vbt `Portfolio` | `holding.callback`, `pf.x` |
| `margin_level(pf, universe)` | Equity / margin required, per bar, as a percentage. | `margin_required` | `min_margin_level` |
| `min_margin_level(pf, universe)` | The lowest ratio over the backtest — an MT5 parity metric, satisfied without F2. | `margin_level` | `pf.x`, K2 |

---

### `construct/` — portfolio construction (Section J)

#### `construct/weights.py`
| Symbol | Does | Calls | Called by |
|---|---|---|---|
| `equal_weight(columns, index)` | `1/n` per column. | — | the user |
| `price_weight(prices)` | Price-proportional. | — | the user |
| `cap_weight(market_caps, index, fx=None)` | **Recomputed at each rebalance** — fixed-at-construction is not cap weighting, it is an initial-cap-weighted buy-and-hold. Market caps need FX conversion with the same table as prices. | `FXTable.get` | the user |
| `inverse_vol_weight(prices, window)` | Risk-parity-ish. Warm-up bars are **NaN, not zero** — a zero weight is a real instruction, NaN means "no estimate yet". | — | the user |

#### `construct/schedule.py`
| Symbol | Does | Calls | Called by |
|---|---|---|---|
| `always_schedule(index)` | Every bar True. | — | defaults |
| `calendar_schedule(index, freq)` | Boolean mask, True on the first bar of each period. A plain `DatetimeIndex` resample. | `_alias` | `to_targets`, the user |
| `_alias(freq)` | Accepts `"M"`, `"ME"`, `"Q"`, `"A"`, `"Y"` etc. across pandas versions. | — | `calendar_schedule` |
| `threshold_schedule(current, target, tolerance)` | **Relative** drift, per asset — absolute under-triggers on small positions (a 1% position drifting to 1.5% is the 50% move it is). | — | the user |
| `drifted_weights(weights, prices, schedule=None)` | What the weights became since the last rebalance. ⚠️ **Pass the schedule.** Unanchored, drift accumulates from bar zero, grows without bound, and any threshold test degenerates to "always true". The anchor is the last rebalance *strictly before* each bar, so drift is not zero by construction on the rebalance bar itself. | — | `threshold_schedule` |

Schedules are boolean masks and **compose with `&` and `\|`**. "Monthly, but only if drift
exceeds 5%" is `calendar & threshold`.

#### `construct/rebalance.py`
| Symbol | Does | Calls | Called by |
|---|---|---|---|
| `validate_weights(weights)` | Warns both ways: under 1 leaves the remainder in cash (fine, often intended); over 1 is implicit leverage that must route through the budget rather than entering by the back door. | — | `apply_leverage_budget` |
| `apply_leverage_budget(weights, universe, equity=None)` | `portfolio_level` enforcement: scales weights to fit the total notional budget. Exact and cheap, because weights are known ahead of time. | `validate_weights`, `max_leverage` | `from_weights` |
| `to_targets(weights, schedule, universe, price, equity, min_trade)` | → `(targets, drift)`. **NaN off-schedule, never forward-filled.** Forward-filling rebalances every bar — which still runs, still looks plausible, and quietly accrues costs while producing a suspiciously smooth equity curve. | `_quantize_weights`, `_apply_min_trade`, `_warn_if_forward_filled` | `from_weights` |
| `_warn_if_forward_filled(weights, mask)` | Detects the classic input error: values that change **only on rebalance bars** and are flat in between. Genuinely constant weights (equal weighting) are exempt, or the check becomes noise. | — | `to_targets` |
| `_quantize_weights(...)` | Quantizes **before** target emission — quantizing afterwards loses the drift information. | `quantize.quantize` | `to_targets` |
| `_apply_min_trade(...)` | Suppresses trades costing more than the drift they correct; **routes the suppressed drift through Section D's reporting** rather than letting it vanish. | — | `to_targets` |

---

### `holding/` — financing and carry (Section G, the only level-3 code)

#### `holding/financing.py`
| Symbol | Does | Calls | Called by |
|---|---|---|---|
| `financing_rate(benchmark, markup, is_long)` | **The markup is always adverse**: `benchmark + markup` financing a long, `benchmark − markup` when short. A simple addition credits shorts too generously and flatters short-heavy backtests. | — | `financing_charge` |
| `financing_charge(borrowed, benchmark, markup, is_long, day_multiplier)` | Signed cash adjustment. Applies to the **borrowed portion only** — a cash-covered position finances nothing. ACT/360. | `financing_rate` | `build_charges`, the user |
| `financing_series(...)` | Vectorised across a whole index. | `financing_charge` | `build_charges` |

#### `holding/carry.py`
| Symbol | Does | Calls | Called by |
|---|---|---|---|
| `tom_next_differential(base_rate, quote_rate)` | The interest differential, tom-next, not an interpolation. | — | `carry_charge` |
| `carry_charge(notional, differential, day_multiplier)` | Applies to **any** FX position regardless of leverage — you are long one currency and short another either way. Conceptually distinct from financing; separate module, separate rates, separate multipliers. | — | `build_charges`, the user |
| `carry_series(...)` | Vectorised. | `carry_charge` | `build_charges` |

#### `holding/accrual.py`
| Symbol | Does | Calls | Called by |
|---|---|---|---|
| `WEEKDAYS` | Name → weekday number. | — | `day_multipliers` |
| `day_multipliers(index, triple_day, end_of_day, holidays)` | 1× per trading day, **3× on the triple day** (covering the weekend, since FX settles T+2), 0 on non-trading days. Do **not** additionally charge calendar days — combining the conventions bills nine days a week. | `_is_intraday`, `_collapse_intraday` | `build_charges`, the user |
| `_is_intraday(index)` | Whether `end_of_day` can matter at all. | — | `day_multipliers` |
| `_collapse_intraday(index, end_of_day)` | Puts the charge on the bar containing the broker's cutoff. **Inert on daily bars** — one bar per day means the cutoff cannot change which bar a charge lands on. | — | `day_multipliers` |
| `trading_day_count(index)` | Convenience count. | — | the user |

#### `holding/callback.py`
| Symbol | Does | Calls | Called by |
|---|---|---|---|
| `post_segment_func_nb(c, charges)` | **The thin level-3 layer.** Mutates `c.last_cash[col]`, a numpy array passed by reference. Deliberately tiny: when upstream breaks the callback signature, only this needs rewriting. | — | vbt `from_order_func` |
| `build_charges(pf_or_state, universe, rates, index)` | Pre-computes the whole charge matrix in pandas, so the callback does no arithmetic. | `financing_series`, `carry_series`, `day_multipliers`, `borrowed_portion` | `simulate_with_holding_costs` |
| `_rate_for(currency, rates, index)` | Currency-specific rate lookup. **Convert the notional, not the rate.** | `rate_table` | `build_charges` |
| `simulate_with_holding_costs(cls, close, size, universe, rates, ...)` | Wires it together and calls `from_order_func` with **`call_post_segment=True`** — which defaults to `False`, and without which the callback never fires on bars where no order was placed, exactly where financing accrues. | `build_charges`, vbt `from_order_func` | `Portfolio.from_holding_costs` |

#### `rates/` — benchmark rate data (Section M)

| File · Symbol | Does | Calls | Called by |
|---|---|---|---|
| `rates/table.py · DEFAULT_SERIES` | SONIA (GBP), SOFR (USD), ESTR (EUR), TONA (JPY), SARON (CHF). | — | `rate_table` |
| `rates/table.py · annual_to_period(rate, days, day_count=360)` | **Rates are quoted annualised; financing accrues daily.** The conversion is explicit rather than assumed. ACT/360 for all currencies. | — | `financing_charge`, `carry_charge` |
| `rates/table.py · static_rate_table(currencies, index, rate)` | The no-API-key path. The library must work without FRED. | — | the user |
| `rates/table.py · rate_table(currencies, index, source)` | One column per currency, aligned to the backtest index. | `FREDData.download_rates`, `align_rates` | `holding.callback` |
| `rates/table.py · align_rates(frame, index, fill_policy)` | Rate series publish on business days; backtests may run on calendar days. Shares Section I's explicit fill-policy discipline. | — | `rate_table` |
| `rates/fred.py · FREDData` | `Data` subclass following the `YFData` / `BinanceData` pattern. Caching keyed by series and date range, **refreshed on load** for any range extending past the cached end. If FRED is unreachable it uses cache with a warning, failing only when there is neither cache nor static fallback. | `_fetch`, `read_cache`, `write_cache` | `rate_table`, the user |

---

### `metrics/` — K1 and K2

#### `metrics/returns/nb.py` (Section K1)
Follows `vectorbt/returns/nb.py` conventions exactly: a `_1d_nb` core, an `_nb` column
loop, a `rolling_*_nb` variant.

| Symbol | Does |
|---|---|
| `cum_curve_1d_nb` | Equity curve from simple returns. Used by Ulcer and SNR. |
| `annualized_return_1d_nb`, `annualized_volatility_1d_nb`, `beta_1d_nb` | Local copies so K1 imports nothing. |
| `linreg_1d_nb(y)` | **The shared primitive** → `(slope, stderr, corr)`. Feeds all three K-ratio variants *and* K2's LR Correlation and LR Standard Error. Five metrics from one regression. |
| `ulcer_index_1d_nb` | Martin (1987): equity → running max → **percentage** drawdown → RMS. Percentage, not absolute — squaring absolute drawdowns would make it scale with account size. |
| `ulcer_performance_index_1d_nb` | `(annualised return − rf) / UI`. |
| `m2_1d_nb` | `rf + Sharpe_p × σ_benchmark`. Reads as a return, not a ratio. |
| `treynor_ratio_1d_nb` | Excess return per unit of beta. |
| `geometric_sharpe_ratio_1d_nb` | Sharpe on **log** returns. Upstream's is arithmetic, so this is a genuine gap. |
| `snr_1d_nb` | Efficiency ratio: `\|P_end − P_start\| / Σ\|ΔP\|`, bounded 0–1. |
| `k_ratio_1d_nb(returns, ann_factor, version)` | Regression of cumulative log returns on time. v1 `t/n`; v2 adds `sqrt(ann_factor)`; v3 adds a further `sqrt(n)` — which is what makes it length-independent. Default v3. |
| `*_nb`, `rolling_*_nb` | Column loops and rolling variants. |

#### `metrics/returns/register.py`
Attaches 14 methods to the base `ReturnsAccessor` and merges 7 `_metrics` entries.
Benchmark-requiring metrics set `check_has_benchmark_rets`; annualised ones set
`check_has_year_freq`. Everything is tagged `vectorbtx`.

#### `metrics/trades/calc.py` (Section K2)
Pure array functions, so all 18 additions are testable against hand-computed numbers
without running a simulation.

| Symbol | Does |
|---|---|
| `total_net_profit`, `gross_profit`, `gross_loss`, `loss_rate`, `win_rate` | The one-line derivations. |
| `balance_curve(pnl, init_cash)` | **Closed-trade** equity. Note the inversion: vbt's `value` already includes floating positions, so MT5's *Equity* drawdown comes free and MT4's *Balance* drawdown is the gap. |
| `absolute_drawdown` | `initial_deposit − minimum_balance`, in **currency, not percent**. Needs a resolved `init_cash` — under `InitCashMode.Auto` the raw field is a mode flag. |
| `balance_drawdown`, `balance_drawdown_abs` | Fractional and currency max drawdown of that curve. |
| `recovery_factor` | Net profit / max DD. Not vbt's `recovery_return`, which is a different thing. |
| `runs(pnl)` | Splits trades into win/loss runs → `(is_win, length, total)`. The basis of every streak metric. |
| `max_consecutive_money` | PnL of the **longest** run. |
| `maximal_consecutive` | PnL of the **most profitable** run. Distinct from the above — the longest streak is not necessarily the best one. |
| `avg_consecutive` | Mean run length. |
| `lr_correlation`, `lr_standard_error` | Byproducts of K1's regression. |
| `ahpr`, `ghpr` | Arithmetic and geometric holding-period return, in MT's `1.0034` convention. |
| `z_score` | Wald–Wolfowitz runs test. **Negative Z means outcomes cluster into streaks**; positive means they alternate. A dependency test, not a performance measure. |

#### `metrics/trades/register.py`
Attaches 20 methods to `Trades`, merges 20 `_metrics` entries, and puts a **curated four**
on `Portfolio._metrics` (Total Net Profit, Recovery Factor, Absolute Drawdown, GHPR) so
the default table stays readable. Everything tagged `mt5`, so `tags="mt5"` selects the
parity set.

#### `metrics/aggregate.py`
`aggregate(results, how="geometric")` — combines several `.stats()` outputs. Compounds
rows whose label contains "Return" and averages the rest, which is the only sensible
default when the rows mix returns, ratios and durations.

---

## Call graph

```
import vectorbtx
  └─ core.patch.apply_patches
       ├─ metrics.returns.register.register_returns_metrics ──► ReturnsAccessor
       ├─ metrics.trades.register.register_trade_metrics ─────► Trades, Portfolio
       └─ core.accessors.register_accessors ──────────────────► pd.Series/DataFrame .vbt.x

load_universe(dict | csv | xlsx)
  └─ resolve.coerce.coerce_field ──► instruments.validate.validate_spec
       └─ Universe  ──► instruments.validate.validate_universe

Universe.resolve(data)
  └─ resolve.resolver.resolve_universe
       ├─ costs.compose.resolve_costs ──► commission | spread | slippage
       │      └─ fx.convert.convert_fixed_costs ──► fx.table.FXTable.get
       ├─ quantize.quantizer.to_order_params
       └─ margin.cap.max_size_for ──► instruments.defaults.max_leverage
                                                    │
                                                    ▼
                                        {size_granularity, min_size, max_size,
                                         fees, fixed_fees, slippage}
                                                    │
Portfolio.from_orders  ◄────────────────────────────┤
Portfolio.from_weights ─► construct.rebalance.to_targets ─► quantize.quantize
Portfolio.from_holding_costs ─► holding.callback.simulate_with_holding_costs
                                   ├─ holding.accrual.day_multipliers
                                   ├─ holding.financing.financing_series
                                   ├─ holding.carry.carry_series
                                   ├─ margin.exposure.borrowed_portion
                                   └─ post_segment_func_nb ─► c.last_cash[col] -= …
```

---

## Known limitations

Documented rather than hidden. Each traces to a recorded design decision.

1. **Symmetric spread only** — no bid/ask asymmetry. vectorbt has one `slippage` field per
   order; asymmetry would need a custom order function for every mode. Strategies
   sensitive to bid/ask asymmetry (HFT, market-making) are out of scope.
2. **Holding costs land on the next bar** — `last_value` is recomputed from `last_cash`
   immediately *before* the callback runs. This shifts *when* a charge appears in the
   equity curve, not the total. For overnight financing it is arguably the more correct
   reading.
3. **Return decomposition needs the retained native series** — the converted series
   conflates asset return with FX return. Correct for total return; use
   `ConvertedFrame.decompose()` to split it.
4. **No dividend treatment for spread bets and CFDs.** Adjusted close is correct for
   equity and wrong for nominal-point modes, which is why `price_field` differs by mode.
   Adjusted histories also change with every new dividend, so a backtest run today will
   not match one run last month.
5. **`portfolio_level` leverage is exact only via `from_weights`.** Signal-driven
   strategies get a per-column backstop, so aggregate exposure can exceed the cap. Exact
   enforcement there needs level 3.
6. **End-of-day time is inert on daily bars.** One bar per day means the cutoff cannot
   change which bar a charge lands on. It matters only on intraday data.
7. **Taxes are out of scope, by decision.** No stamp duty, no PTM levy, no CGT, and none
   of the spread-bet exemptions. Tax treatment is jurisdiction- and wrapper-specific and
   belongs in analysis of a backtest rather than inside its cost model.
   ⚠️ **The cross-mode comparison is therefore pre-tax.** In the UK a spread bet and the
   equivalent share position differ by 0.5% per entry before any P&L is considered.
8. **Maintenance margin, margin calls and forced liquidation are not modelled.** A
   position can draw its whole margin loan and sit there. `pf.x.min_margin_level()`
   reports how close it came.
9. **Short share borrow charge is not modelled.** The rate is instrument-specific,
   time-varying and not available from any free source, so modelling it would mean
   inventing numbers. Short share CFD strategies look cheaper here than in reality.
10. **Futures and forwards have no mode.** They attract no overnight funding at all — the
    cost is in a wider spread — so a future mode would need a stack with neither financing
    nor carry.

Deliberately deferred: maintenance margin, margin calls and forced liquidation (F2);
dividends for nominal-point modes; asymmetric bid/ask; compounding-equity risk sizing
(`risk_pct` sizes off `init_cash`, since at level 1 equity is only known before the run).

---

## Testing

```bash
pytest                                     # full suite
cd examples/cases      && python _harness.py   # all 12 case files, offline
cd examples/strategies && python _harness.py   # all 3 strategies, offline
```

**245 tests pass against a real vectorbt 1.1.0 install** — vectorbt 1.1.0, numba 0.67.0,
numpy 2.5.3, pandas 3.0.6, plotly 6.9.0, on Python 3.12 / Windows. No failures, no skips.
Most are pure NumPy/pandas arithmetic and run without vectorbt at all.

**The numba kernels really are compiled.** All three level-3 kernels in
`holding/callback.py` — `order_func_nb`, `pre_segment_func_nb`, `post_segment_func_nb` —
acquire numba signatures on first call and survive type inference unchanged. A
`from_holding_costs` run costs ~8.4s cold (almost entirely compilation) and ~0.02s warm.

Two test files are worth knowing about:

- `tests/test_holding_broker.py` — Section G fixtures worked from **broker documentation**
  rather than from the plan, including IG's own published example reproduced to the penny.
  Every other test in the suite encodes the plan's assumptions, which is exactly why none
  of them caught the defects in [`docs/dev/AUDIT-G.md`](docs/dev/AUDIT-G.md).
- `tests/test_holding_sim.py` — regressions for the four simulation-path defects found by
  actually executing the engine paths.

Most of the library is pure NumPy/pandas arithmetic and is testable **without vectorbt or
numba installed** — `tests/conftest.py` detects their absence and imports the package's
submodules directly. Tests needing a real simulation are marked and skipped.

⚠️ **`Portfolio.from_weights` has no direct test.** Section J is covered only indirectly,
via `vbt.x.PortfolioBacktest`. See [`docs/dev/AUDIT.md`](docs/dev/AUDIT.md) §1 — it is the
largest known coverage gap.

The worked numerical examples double as the regression suite: they encode not just "does
this run" but "does this still produce the number we agreed was correct", which is what
protects against upstream drift.

Two CI axes are intended: pinned `vectorbt==1.1.0` must pass; a floating-latest job is
informational and **is allowed to be red** — it is the canary for changes to
`portfolio/nb.py` internals and for name collisions from patching. A red canary is
information, not an emergency.

---

## Licence

vectorbt is Apache 2.0 **with Commons Clause**, which removes the right to *Sell* —
including charging for hosting, or for support whose value derives substantially from the
software. vectorbtX derives substantially from vectorbt. Private use, internal use and
free redistribution appear fine; monetisation does not. This is not legal advice.

---

## Examples

**Copy-pasteable strategies** — [`examples/strategies/`](examples/strategies/):

| File | Strategy |
|---|---|
| `01_ma_cross_spreadbet_fx.py` | 50/200 MA cross on EURUSD as a UK spread bet — £/point staking, risk-based sizing, point stops, carry |
| `02_leveraged_stock_portfolio.py` | Top-3 momentum across 5 stocks, 2 currencies, 3 cost structures — per-instrument arguments throughout |
| `03_monthly_rebalanced_investing.py` | Global 60/40 ETF portfolio, monthly with a drift filter — `from_weights` |

Each uses `vbt.YFData.download` for prices, `vbt.x.FREDData` for rates, and ends with
`pf.stats()`, `pf.trades.stats()` and `pf.returns_stats()`.

**Reference tour** — nine numbered files in [`examples/`](examples/), ~90 worked sections
covering one design decision each. Useful when something surprises you and you want to
know why it works that way.

```bash
cd examples/strategies && python 01_ma_cross_spreadbet_fx.py
cd examples && python 09_end_to_end.py
```

See [`docs/EXAMPLES.md`](docs/EXAMPLES.md) for the full index.

---
