# vectorbtX — Implementation Plan (v7, resolved)

## How to use this document

Complete working specification for vectorbtX. Written to be read cold: an implementer who has
never seen the vectorbt source and was not part of the design conversation should be able to read
it top-to-bottom and know what to build, in what order, and why.

**Structure:**

- **Part 1** — what the project is and what it promises. Read first.
- **Part 2** — domain primer. Skip if you know spread betting, CFDs, and FX carry.
- **Part 3** — audit of vectorbt 1.1.0. The factual base everything rests on.
- **Part 4** — design decisions, with reasoning.
- **Part 5** — architecture and runtime flow.
- **Part 6** — build order and dependency graph.
- **Part 7** — the sections. The actual work.
- **Part 8** — settled defaults reference.
- **Part 9** — what remains.

**Status.** Design settled. Domain review complete. All parameter defaults chosen. The only
outstanding work before coding is producing the D7 worked examples (Part 9).

**v7 change:** investing is now a first-class use case. `Portfolio.from_weights` (Section J) sits
alongside vbt's own constructors, wrapping weight construction, scheduling, quantization, and the
leverage budget behind one entry point.

**Line references** (e.g. `nb.py:145`) point into vectorbt 1.1.0 as shipped in
`vectorbt-master.zip`. Pinned to that version; they will drift.

**⚠️ marks a silent-failure hazard** — a case producing wrong numbers rather than an exception.
There are about twenty. They are worth re-reading before writing the section they appear in.

---

## Part 1 — What vectorbtX is

### 1.1 The problem

[vectorbt](https://github.com/polakowo/vectorbt) is the fastest open-source backtesting engine
available. Its core simulation loop is excellent. But the OSS release deliberately omits a tier of
features that live in the commercial PRO version — leverage, portfolio construction helpers,
extended metrics, instrument abstractions — and it is built around a single implicit instrument
model: you own N units of something priced in one currency.

That model does not describe how a lot of retail derivatives trading actually works. A spread bet
has no units and no ownership; you stake £/point. A CFD is priced in lots against a contract size.
FX spot has pip values and carry. None of these fit "N shares at price P" without a translation
layer.

### 1.2 The approach

vectorbtX is a **pure wrapper**. It does not fork vectorbt, does not vendor its source, and does
not modify its simulation loop.

Why not fork:

- Upstream moves fast, and the simulation core is where it changes most.
- Every upstream performance improvement and bug fix should arrive for free.
- The engine is genuinely good. The gap is in what surrounds it.

So: subclass and patch at import time, leave engine internals untouched, and accept that this
constrains what we can build. Where a feature can't be expressed through vectorbt's existing
parameters, use its callback hooks rather than reaching into the loop.

### 1.3 The contract

```python
import vectorbtx as vbt
```

must be a **drop-in replacement** for `import vectorbt as vbt`:

1. **Everything upstream still works, unchanged.** Any script written against vectorbt 1.1.0
   produces identical results. Testable, and tested — Section L.
2. **Extras are additive.** New capability appears as new parameters (defaulting to inert values),
   new methods, or the `.x` namespace.
3. **Namespacing avoids future collisions.** Non-metric extras live under `.x`. Metrics are the
   deliberate exception (D5).
4. **Pinned to one vectorbt version.** `vectorbt==1.1.0`. Upgrades are opt-in and tested.

### 1.4 Non-goals

- **Not a fork.** No vendored source, no patches to the simulation loop.
- **Not a PRO clone.** We build what's needed.
- **Not performance work.** vectorbt is already fast.
- **Not commercial.** See §3.1.
- **No dividends or corporate actions in v1.** Handled via adjusted price series where the mode
  allows; see D10.

### 1.5 Success criteria

A first release is done when a user can:

- describe an instrument once — or a whole universe in a spreadsheet — and have every downstream
  calculation respect it;
- express the same trade as a spread bet, a CFD, an FX spot position, or shares, and get the same
  P&L;
- run a leveraged backtest where financing is charged on the borrowed portion only, at real
  historical rates plus a broker markup;
- hold foreign-currency instruments in a base-currency portfolio and get correct P&L;
- build and rebalance a multi-asset portfolio with realistic fill sizes and per-asset costs;
- run a calendar-rebalanced investment strategy — monthly, quarterly, annual — through a single
  constructor, without hand-deriving the buy/sell orders;
- see the full MT4/MT5 metric set alongside Ulcer, UPI, M², Treynor, SNR, K-ratio, and geometric
  Sharpe in the same `.stats()` output.

---

## Part 2 — Domain primer

The working understanding of the instruments being modelled, as confirmed during domain review.
Broker conventions still vary; these are UK retail defaults and all are configurable.

### 2.1 Instrument modes

**Equity.** Buy N shares at price P. You own the asset. Retail commission is often a **flat fee**
rather than a percentage — e.g. £10 per trade regardless of size — which does not map onto a
percentage `fees` field (§7.E). Leverage possible via margin but not the default. This is the
model vectorbt natively assumes.

**CFD.** A derivative tracking an underlying. Sized in *lots*, where one lot represents a fixed
*contract size* of the underlying. P&L is `price_change × contract_size × lots`. No ownership.

- **Minimum lot typically 0.01**, some brokers 0.10. Per-instrument argument.
- `contract_size` alone is sufficient to describe any CFD — they all trade the same way.
- Margin required is `notional / leverage`.
- Commission common on equity and FX CFDs, less so on index CFDs where cost sits in the spread.
- Overnight positions incur financing.

**Spread bet.** A UK retail product, structurally similar to a CFD but with different mechanics and
tax treatment. Sized in **£ per point**. P&L is `price_change_in_points × stake_per_point`. No
ownership.

- **There is no contract size.** The stake *is* the size. Conversion is
  `engine_size = stake_per_point / point_size` — see §7.H, where this is verified against three
  independent cases.
- **Minimum stake £1/point**, £0.10 at micro brokers. **Stake increment £0.01.**
- **Commodities behave exactly as equities** — same £1/point minimum, no special case.
- **Normally no commission** — the fee is baked into a wider spread.
- Denominated in the account currency by construction, whatever the underlying.

**FX spot.** Currency pairs sized in lots (standard 100,000 units of base, mini 10,000, micro
1,000). Price moves measured in **pips**.

- **Modern quoting is 5 decimal places, not 4** (3 for JPY, not 2). The pip remains the 4th decimal
  (2nd for JPY); the 5th gives *fractional pip* precision. ⚠️ Arithmetically identical, but the
  extra digit must not be truncated — a meaningful part of the spread lives there.
- £1/pip is roughly 0.1 lots (10,000 units) on majors, varying with instrument price.
- Overnight positions earn or pay **carry**.

### 2.2 Cost types

**Per-trade:**

- *Commission* — percentage of notional, flat per trade, or per-unit, often with a minimum.
- *Spread* — the bid/ask gap. You buy at ask and sell at bid, so every trade starts slightly
  underwater. Measured in pips or points.
- *Slippage* — difference between expected and achieved price. Fixed, or modelled from order size
  relative to volume and from volatility.

**Holding:**

- *Leverage financing* — interest on borrowed money, applied **only to the borrowed portion**. A
  £10,000 position fully covered by £10,000 cash finances nothing. £50,000 against £10,000 finances
  £40,000.
  - Rate is a **benchmark plus broker markup**, and the markup is **always adverse**: pay
    `benchmark + markup` financing a long, receive `benchmark − markup` when short. ⚠️ Implementing
    as a simple addition credits shorts too generously.
  - Benchmark rates move substantially over time, so a static rate misstates any multi-year
    backtest. See Section M.
- *Carry (FX)* — the interest differential between the two currencies in a pair, using the
  **tom-next** rate. Applies to **any** FX position regardless of leverage, because you are long one
  currency and short another. Conceptually distinct from financing; must not be merged with it.

**Accrual.** Both holding costs charge at a broker-defined end-of-day, on **trading days only**,
with a **3× multiplier on Wednesday** covering the weekend (FX spot settles T+2, so rolling a
Wednesday position settles the following Monday). Weekends and holidays carry no separate charge.

- Triple day defaults to Wednesday for everything in the UK, and is configurable.
- End-of-day defaults to 22:00 UK.
- ⚠️ **End-of-day time is inert on daily bars** — one bar per day means the cutoff cannot change
  which bar a charge lands on. It matters only on intraday data.

### 2.3 Concepts that recur

- **Notional** — full market value of a position regardless of cash backing. A 5× leveraged £10,000
  margin position has £50,000 notional.
- **Point / pip** — the smallest conventional price increment. Pip risk is entry-to-stop distance
  in these units, and is the basis of position sizing.
- **Quantization** — brokers don't fill arbitrary sizes. Shares are whole numbers (mostly), FX comes
  in lot increments, spread bets have a £1 minimum and £0.01 increment. A backtest assuming
  137.4829 shares is lying, and the lie compounds in small accounts.

---

## Part 3 — Audit of vectorbt 1.1.0

From `vectorbt-master.zip`, `__version__ = "1.1.0"`, dated 2026-08-02.

### 3.1 Environment & constraints

| Item | Value |
|---|---|
| Version | 1.1.0 |
| License | **Apache 2.0 with Commons Clause** |
| Python | `>=3.11,<3.15` |
| Core deps | `numpy>=2.4.6`, `pandas>=3.0.3,<4.0`, `scipy`, `numba>=0.66`, `plotly>=4.12` |
| Rust engine | optional extra: `vectorbt-rust==1.1.0` — **not targeted, D1** |
| Metrics backend | **native numba in `vectorbt/returns/nb.py`** — empyrical is *not* a dependency |
| quantstats | optional only, via `.vbt.returns.qs` |
| Test surface | `tests/test_portfolio.py`, `test_returns.py`, `test_engine.py`, + 7 more |
| CI | {ubuntu, windows, macos} × py{3.11, 3.12, 3.13, 3.14} |

**Licensing.** The Commons Clause removes the right to *Sell*, where "Sell" explicitly includes
charging for hosting, or for consulting and support whose value derives substantially from the
software. vectorbtX derives substantially from vectorbt. Private use, internal use, and free
redistribution appear fine; monetisation does not. Acceptable for a FOSS-for-fun project, and why
§1.4 lists commercial use as a non-goal. Not legal advice.

### 3.2 The Rust engine — why we don't target it

`vectorbt/_engine.py` centralises dispatch; on version mismatch it warns and falls back to numba.
Two facts determined D1:

**It doesn't cover the path we need.** Rust accelerates `simulate_from_orders` and
`simulate_from_signals`. **`Portfolio.from_order_func` has no `engine` parameter and is numba-only**
— and it is the only entry point accepting the per-bar callbacks holding costs require.

**The speedup is small anyway.** From the repo's own benchmark reports (best-of-5 after warmup):

| Function | shape | Rust | Numba | speedup |
|---|---|---|---|---|
| `simulate_from_orders` | 10K×100 | 33.53 ms | 38.27 ms | 1.14× |
| `simulate_from_orders` | 100K×10 | 26.90 ms | 29.66 ms | 1.10× |
| `simulate_from_orders` | 100×1 | 9.75 µs | 7.08 µs | **0.73×** |
| `simulate_from_signals` | 10K×100 | 16.90 ms | 20.87 ms | 1.23× |
| `simulate_from_signals` | 100K×10 | 16.46 ms | 19.74 ms | 1.20× |

~1.1–1.25× on the simulation path, *slower* on small inputs where FFI overhead dominates. Rust's
real wins (~2×) are in generic array kernels, inherited regardless.

### 3.3 What already exists — do NOT rebuild

The most important subsection. Several things the original plan intended to build already exist.

**The `Order` NamedTuple** (`portfolio/enums.py:1502`):

```
size, price, size_type, direction,
fees          — fraction of order value
fixed_fees    — absolute amount per order
slippage      — fraction of price
min_size      — default 1e-8; orders below are rejected
max_size      — orders above are PARTIALLY FILLED, not rejected
size_granularity
reject_prob, lock_cash, allow_partial, raise_reject, log
```

**Every field broadcasts by argument type** (documented with an example at `base.py:78-89`):

```python
size = pd.Series([1, -1, 1, -1])                    # per row
price = pd.DataFrame({'a': [...], 'b': [...]})      # per element
direction = ['longonly', 'shortonly']               # per column
fees = 0.01                                         # per frame
```

So **per-asset costs already work natively** — a plain list is per-column. Nearly free: vbt does not
materialise the broadcast, it keeps the original shape and uses flexible indexing via
`flex_select_auto_nb` (`base.py:268-269`). What's missing is ergonomics, not mechanism — Section C2.

**Keyword-valued numeric arguments are also native idiom.** `map_enum_fields` (`base.py:1898-1901`)
resolves `'longonly'` to an enum inside a numeric field; `price=np.inf` is a sentinel meaning "use
close". Precedent for D8's `'auto'` cells.

- **`size_granularity` is fill quantization, already implemented.** Floor rounding:
  `adj_size // size_granularity * size_granularity` (`nb.py:145`, 170, 294). Validated as NaN or
  finite `> 0` (`nb.py:414`). Missing: configurable direction, intended-vs-actual reporting.
- **`SizeType`**: `Amount, Value, Percent, TargetAmount, TargetValue, TargetPercent`.
- **`Direction`**: `LongOnly, ShortOnly, Both`.
- **SL/TP is fully native** in `from_signals`: `sl_stop`, `sl_trail`, `tp_stop`, `stop_entry_price`,
  `stop_exit_price`, `upon_stop_exit`, `upon_stop_update`, `use_stops`, plus `adjust_sl_func_nb` /
  `adjust_tp_func_nb`. **Stops are fractions of price** — relevant to D4.
- **`InitCashMode.Auto` / `AutoAlign`**; **`cash_sharing` + `group_by` + `call_seq`** (incl. `"auto"`).

**Collateral and debt.** `ExecuteOrderState` carries `(cash, position, debt, free_cash)`. `debt`
tracks short-sale proceeds; `free_cash` is collateral-aware; `lock_cash` stops one column borrowing
another's collateral. Public via `Portfolio.cash(free=True)` and `cash_flow(free=True)`;
`gross_exposure` divides by free cash. `debt` is not exposed on `Portfolio`.

**Simulation callback state — the key enabler for Section G.** `SegmentContext` and siblings expose
mutable numpy arrays:

```
last_cash, last_position, last_debt, last_free_cash,
last_val_price, last_value, second_last_value, last_return
```

Allocated once (`last_cash = init_cash.copy()`, `nb.py:3383`) and passed **by reference** into every
context. Mutating them propagates. This is the cash-injection hook making exact holding costs
possible. Helpers: `sort_call_seq_nb`, `sort_call_seq_out_nb`, `approx_order_value_nb`,
`build_call_seq_nb` (`nb.py:861–1090`).

**Native returns metrics** (`returns/nb.py`; most with `_1d_nb`, `_nb`, `rolling_*_nb`):
total return, annualized return, annualized volatility, drawdown series, max drawdown, Calmar,
Omega, **Sharpe (arithmetic: `mean/std * sqrt(ann_factor)`)**, downside risk, Sortino, Information
Ratio, Beta, **Alpha**, Tail Ratio, Common Sense Ratio, VaR, Conditional VaR, capture / up-capture /
down-capture. Plus `returns/metrics.py`: Deflated Sharpe, `approx_exp_max_sharpe`. Skew/kurtosis via
pandas.

> Native `alpha` **is** annualized Jensen's alpha. `alpha_1d_nb` computes beta internally, then
> `(returns - rf) - beta * (benchmark - rf)`, `nanmean`, annualized as `(mean + 1) ** ann_factor - 1`.
> Do not add it separately.

**Native trade metrics** (`portfolio/trades.py`). Trade records carry `size, entry_idx, entry_price,
entry_fees, exit_idx, exit_price, exit_fees, pnl, return, direction, status, parent_id`. The `Trades`
class provides `winning`/`losing` filters, `winning_streak`/`losing_streak` (as `MappedArray`s giving
per-trade streak position), `win_rate`, `profit_factor`, `expectancy`, `sqn`, `best_trade`,
`worst_trade`, `avg_winning_trade`, `avg_losing_trade`, duration variants, and
`long`/`short`/`open`/`closed` counts. Full audit in K2.

**Native drawdown analysis** (`generic/drawdowns.py`): max/avg drawdown, max/avg duration, recovery
return, recovery duration, active drawdown.

**Extension mechanisms:**

- `StatsBuilderMixin`: `_metrics` is a `Config` dict, deep-copied per instance. Entries take `title`,
  `calc_func` (**callable or dotted string path**), `post_calc_func`, `tags`, `agg_func`,
  `resolve_calc_func`, and filter flags such as `check_has_benchmark_rets`. `stats(metrics=...)`
  defaults to `"all"`, which iterates `_metrics`.
- **`ReturnsAccessor._metrics` and `Trades._metrics` are both directly mutable** — plain `Config`s
  with `copy_kwargs=dict(copy_mode="deep")`, no `readonly`, no `frozen_keys` (`trades.py:807`). The
  docstring at `stats_builder.py:85` explicitly sanctions in-place mutation.
- **The accessor decorator sits on the subclasses, not the base.**
  `@register_series_vbt_accessor("returns")` decorates `ReturnsSRAccessor`; likewise
  `ReturnsDFAccessor`. Base `ReturnsAccessor` is undecorated, so patching it propagates to both
  **without re-registration** — and therefore without a `UserWarning`. No `__slots__`.
- `Configured` (`utils/config.py`): classes accept `**config` and expose `.replace(**kwargs)`
  preserving the concrete class. `from_orders` returns `cls(...)`, so subclass classmethods yield
  subclass instances.
- Settings: nested `Config`, `frozen_keys=True` at top level, but `__setitem__` accepts `force=True`.
- `Portfolio.get_returns_acc` (`base.py:4697`) resolves via `returns.vbt.returns(...)`, so anything
  reaching the accessor reaches `Portfolio.returns_stats()`.
- **`Data` subclass pattern** (`data/custom.py`): `YFData`, `BinanceData`, `CCXTData`, `AlpacaData`
  all subclass `Data`. `data/updater.py` supports scheduled refresh. Template for Section M.
  ⚠️ `YFData` calls `yf.Ticker(...).history(...)` (custom.py:286) passing kwargs through, so
  adjustment behaviour depends on what the caller supplies — see D10.

### 3.4 Confirmed absent — must be built

- **Leverage, margin, maintenance margin, margin call, liquidation, stop-out.** Grep for
  `leverage|margin|liquidat|stop_out` across `vectorbt/` and `rust/src/` returns **only Plotly layout
  margins**.
- **No public cash-injection API.** *But* `last_cash` is mutable from `post_segment_func_nb` (§3.3).
- **No currency, FX rate, or base-currency concept.** Grep for
  `currency|base_ccy|exchange_rate|fx_rate`: zero hits.
- **No bid/ask.** Spread folds into `slippage` (D3).
- **No instrument abstraction** — no contract size, lot, tick value, or point concept.
- **No interest-rate data source.** No FRED or equivalent in `data/custom.py`.
- **No per-asset parameter ergonomics.** Broadcasting works; there is no declarative way to describe
  a universe.
- **Returns metrics missing:** Ulcer Index, UPI (Martin), M², Treynor, SNR, K-ratio (v1/v2/v3),
  Geometric Sharpe.
- **Trade metrics missing:** 12 of the MT4/MT5 set — K2.
- **No portfolio construction helpers**, no rebalance scheduler.
- **No multi-asset aggregate stats wrapper.**

---

## Part 4 — Design decisions

### D1 — Target the numba engine only

**Decision.** Don't treat "does this preserve the Rust fast path" as a design constraint.

**Why.** Rust doesn't cover `from_order_func`, the only route to per-bar callbacks (§3.2). Where it
does apply it's worth 1.1–1.25×, and it's slower on small inputs. Designing around that was
distorting the architecture into two tiers.

**What this is not.** Not removing Rust. A user who installs `vectorbt[rust]` and runs a plain
pass-through path still gets it. We just stop guaranteeing it.

**To reverse:** Rust support for `from_order_func`.

### D2 — Leverage caps in v1; liquidation deferred

**Decision.** Build F1 (pre-trade maximum leverage enforcement). Defer F2 (maintenance margin, margin
calls, forced liquidation).

**Why.** F1 is level 1 — compute a size cap, pass it as `max_size`. F2 needs per-bar callbacks and
therefore depends on `portfolio/nb.py` internals, the most fragile thing available (§5.5). Deferring
confines that risk to Section G. F1 alone is genuinely useful.

**If built later:** liquidate worst position first, then cascade. Matches most broker behaviour.

### D3 — Symmetric spread only

**Decision.** Model spread through `slippage`. No asymmetric bid/ask.

**Why.** vectorbt has no bid/ask concept and one `slippage` field per order. Asymmetric spread would
need a custom order function for *every* mode, dragging all cost modelling to level 3.

**Cost.** Strategies genuinely sensitive to bid/ask asymmetry — high-frequency, market-making,
anything trading the spread itself — are out of scope.

### D4 — Multi-currency by pre-converting price series

**Decision.** Convert foreign-currency price series to base currency *before* simulation, as a
data-preparation step. No intra-simulation currency handling.

**Why.** It is exact. GBP P&L for a GBP investor holding MSFT is
`shares × (P_usd[t₁]·fx[t₁] − P_usd[t₀]·fx[t₀])`. Pre-converting and simulating gives
`shares × (P_gbp[t₁] − P_gbp[t₀])` — the same expression. Cash is base currency throughout, so cash
sharing works, cross-currency rebalancing works, and FX moves correctly affect buying power
mid-backtest, without touching the engine.

**Cost.** The converted series conflates asset return with FX return. Correct for total return, but
decomposition requires retaining the original series. Cheap to keep.

**Scope.** Spread bets are account-currency-denominated by construction, as are GBP-quoted CFDs on
foreign underlyings. Pre-conversion applies only to **equity** and **FX spot**.

### D5 — Patch accessors and record classes; subclass `Portfolio`

**Decision.** Patch `ReturnsAccessor` and `Trades` in place. Subclass `Portfolio`.

**Why patch.** New metrics should appear in `.stats()` next to native ones with the API unchanged.
Both `_metrics` configs are mutable and in-place mutation is explicitly sanctioned (§3.3). The
accessor decorator sits on the subclasses, so patching the base propagates with no re-registration
and no warning. `type()` unchanged. Subclassing and re-registering would achieve the same surface but
replace class identity and emit an override warning — arguably *more* invasive.

**Why subclass `Portfolio`.** There we add constructor parameters and per-column state that must
survive `.replace()` and `indexing_func`.

**Risk.** ⚠️ If upstream later ships its own `ulcer_index`, we silently shadow it. Mitigate with a
`hasattr` guard at patch time that raises rather than overwrites.

### D6 — Financing accrues to the next bar

**Decision.** Accept that a cash mutation inside `post_segment_func_nb` appears in the following
bar's value.

**Why.** `last_value` and `last_return` are recomputed from `last_cash` immediately *before* the
callback runs. Financing debits cash rather than resizing the position, so a one-bar lag shifts
*when* the charge appears in the equity curve, not the total. For overnight financing the lag is
arguably the more correct interpretation.

**To reverse:** update `last_value` and `last_return` in the same callback after mutating
`last_cash`.

### D7 — Worked numerical examples before implementation

**Decision.** Every section that converts units or computes money has a worked numerical example
agreed *before* code is written. That example becomes the test fixture.

**Why.** Catches two failure modes. Methodological: is the formula right? API suitability: does the
formula survive contact with vectorbt's parameter surface?

**Applies to:** C, E, F, G, H, I, M.

**Long-term value.** The examples become the regression suite, which protects against upstream drift.

### D8 — Instrument parameters as a declarative table

**Decision.** Accept a universe description as either a Python dict or a DataFrame (CSV or Excel).
Both normalise to `dict[column, Instrument]`, validate at one site, then resolve to per-column numpy
arrays. Cells may hold a literal or a keyword such as `'auto'`.

**Why.** Per-asset parameters already broadcast natively (§3.3), so the mechanism exists. What
doesn't exist is a safe way to *express* a universe. ⚠️ Hand-building parallel lists across 20 assets
means keeping five lists aligned in column order; one misordered list attaches CFD costs to an equity
column and nothing errors.

**Why keywords in cells.** Unifies static and time-varying parameters: `0.01` is a fixed slippage,
`'auto'` resolves later to a volume-and-volatility-scaled series. Both are "an argument". Native vbt
idiom, not an invention (§3.3).

### D9 — Full MT4/MT5 metric parity

**Decision.** Add all 18 missing MT4/MT5 metrics — 12 genuine gaps plus 6 one-line derivations —
alongside the 7 returns-metric gaps.

**Consequence.** Most MT metrics are **trade-level, not returns-level**, so Section K splits: K1
patches `ReturnsAccessor`, K2 patches `Trades._metrics` and surfaces a curated subset on
`Portfolio._metrics`.

### D10 — Dividends via mode-appropriate price fields

**Decision.** No dividend or corporate-action handling in v1. Instead, `price_field` is a
user-facing per-instrument spec argument: equity mode defaults to adjusted close, spread bet and CFD
default to unadjusted close.

**Why.** Adjusted close is correct for equity mode — shares × price change on an adjusted series
gives true total return. It is **wrong for spread bet and CFD**, which price P&L in *nominal points*:
adjusted close rescales the entire price history on each dividend, so the number of points between
two dates changes depending on when the data was pulled. £1/point on adjusted data does not produce
£1/point of real P&L.

**Two consequences to document loudly:**

- ⚠️ **Adjust the whole bar or none of it.** An adjusted close alongside raw high and low breaks the
  `H ≥ C ≥ L` invariant, and native SL/TP fires at prices that never existed. Same failure mode as
  converting OHLC with mismatched FX rates (Section I gotcha 1). One validation check covers both.
- **Reproducibility.** Adjusted histories change with every new dividend, so a backtest run today
  won't match one run last month. Not wrong, but surprising.

**The remaining hole:** spread bets and CFDs have no dividend treatment at all. Warn at resolve time
when a nominal-point mode is used with unadjusted data.

**Shape of the future fix:** spread bet brokers don't adjust price, they credit or debit the dividend
against the position. So closing this properly means a cash adjustment in Section G's callback —
level 3, alongside financing and carry, where the machinery already exists. A natural extension, not
a new subsystem.

---

## Part 5 — Architecture

### 5.1 Package layout

```
vectorbtx/
  core/         import hook, Portfolio subclass, accessor/record patching, settings
  instruments/  instrument spec + validation                     (level 1)
  resolve/      dict|DataFrame → validated specs → numpy args    (level 1)
  quantize/     thin wrapper over size_granularity               (level 1)
  costs/        cost modules → Order field array-likes           (level 1)
  modes/        unit adapters                                    (level 1)
  fx/           base-currency pre-conversion                     (level 2)
  rates/        FRED rate data + broker markup                   (level 2)
  construct/    weighting + rebalance scheduling                 (level 2)
  margin/       leverage caps                                    (level 1)
  holding/      financing + carry                                (level 3)
  metrics/      returns metrics (K1) + trade metrics (K2)        (level 1)
```

### 5.2 The simplicity ordering

Since D1 removed performance as a design input, **simplicity decides the approach**:

**Level 1 — parameterise.** Compute array-likes and hand them to existing `Order` fields. Nothing in
the simulation changes. Covers commission, spread, slippage, quantization, unit conversion, leverage
caps, position sizing, SL/TP, and all per-asset variation. *Most of the library lands here.*

**Level 2 — pre-compute.** Weight matrices, schedules, FX-converted price series, rate tables — built
in pandas before simulation, or aggregated after.

**Level 3 — simulation callbacks.** `from_order_func` with `pre_segment_func_nb` /
`post_segment_func_nb`. Genuinely path-dependent behaviour only.

After D2 and D4, **level 3 contains only Section G**. That is the entire blast radius of our
dependency on vectorbt internals.

### 5.3 Level-3 mechanics

Read before writing Section G.

**The cash hook.** Mutate `last_cash` in `post_segment_func_nb`. It's a numpy array passed by
reference (§3.3), so the change persists.

⚠️ **`call_post_segment` defaults to `False`.** Must be set `True`, or the callback never fires on
bars where no order was placed — exactly where financing accrues. Produces output that looks
plausible and is wrong.

**Accrual lands on the next bar (D6).** Accepted, not a bug.

### 5.4 Runtime flow

```
1. IMPORT                                                        [Section B]
   import vectorbtx as vbt
   → vectorbt namespace re-exported, Portfolio subclass swapped in
   → ReturnsAccessor and Trades patched with new metrics
   → settings["vectorbtx"] added

2. DESCRIBE THE UNIVERSE                                    [Sections C, C2]
   universe = vbt.x.load_universe("instruments.csv")   # or a dict
   → coerced, validated, keyed by column
   → 'auto' cells recorded for later resolution

3. LOAD RATES (if financing or carry used)                       [Section M]
   rates = vbt.x.FREDData.download(base="GBP")
   → cached, refreshed on load

4. PREPARE DATA                                            [Sections I, B]
   data = vbt.YFData.download([...])          # price_field per spec (D10)
   data = data.vbt.x.to_base_currency("GBP", fx_table, fx_markup=0.005)
   → OHLC converted with ONE rate per bar
   → original retained for decomposition; benchmark converted too

5. BUILD TARGETS (multi-asset only)                        [Sections J, D]
   weights  = vbt.x.equal_weight(assets)
   schedule = vbt.x.calendar_rebalance(index, freq="M")
   targets  = vbt.x.quantize(weights, universe)
   → portfolio-level leverage budget applied here
   → intended-vs-actual drift recorded here
   → NaN off-schedule; never forward-filled

6. RESOLVE ARGS                                     [Sections C2, E, F, H]
   args = universe.resolve(data, rates)
   → per-column arrays for every Order field, in wrapper column order
   → 'auto' slippage → volume/volatility series
   → units converted: £/pt, lots, shares → engine size
   → point stops → price fractions
   → spread + slippage SUMMED (one field)
   → foreign fixed costs FX-converted
   → max_size set from leverage caps

7. SIMULATE — three entry points
   pf = vbt.Portfolio.from_weights(...)          ← investing/rebalancing [Section J]
                                                   wraps steps 5-6 and calls
                                                   from_orders with TargetPercent
   pf = vbt.Portfolio.from_orders(**args)        ← trading, no holding costs
   pf = vbt.Portfolio.from_order_func(...)       ← if financing/carry [Section G]
                                                   post_segment_func_nb
                                                   mutates last_cash

8. ANALYSE                                              [Sections K1, K2]
   pf.stats()          → native + curated MT metrics
   pf.trades.stats()   → full MT4/MT5 trade set
   pf.returns_stats()  → native + Ulcer, UPI, M², Treynor, SNR, K-ratio, geo-Sharpe
   vbt.x.aggregate(results, how="geometric")
```

Step 7's fork is the key structural fact: holding costs force the slower, more fragile path.
Everything else stays on the simple one. Keeping that fork narrow is the central architectural goal.

### 5.5 Version pinning and fragility

Pin `vectorbt==1.1.0`. Pinning converts breakage from something that happens *to* you into something
you opt into on upgrade.

Ranked by fragility:

1. **`portfolio/nb.py` context internals** — `last_cash`, callback signatures, segment ordering.
   **Section G only.** Highest risk: 1.1.0 reshaped these to pair 1:1 with Rust equivalents.
2. **`Order` field names and semantics.**
3. **`settings["portfolio"]` key names.**
4. **`_metrics` schema, `Configured.replace`, accessor registration.** Public, low risk.
5. **Name collisions from D5's patching.**

Mitigation: one informational CI job floating to latest vbt (Section L). Allowed to be red.

---

## Part 6 — Build order

### 6.1 Dependency graph

```
        A (recon) ✅
            │
            ▼
        B (foundations) ─────────────────► K1 (returns metrics)
            │                          └─► K2 (trade metrics)
            ▼                              [parallel, independent]
        C (instrument spec)
            │
            ▼
        C2 (argument resolution)
            │
      ┌─────┼─────┬──────────┐
      ▼     ▼     ▼          ▼
      D     E     H          I (fx)
   (quant)(costs)(modes)     │
      │     │     │          │  I feeds E (foreign fixed costs)
      │     │     │          │  I feeds J (converted market caps)
      ▼     │     │          ▼
      J ◄───┘     │      F (leverage caps)
                  │          │
        M (rates) ┴──────────┤
            │                ▼
            └──────────► G (holding costs)   ← level 3, isolated
                             │
                             ▼
                         L (compatibility)   ← continuous, not final
```

### 6.2 Recommended sequence

1. **A** — done.
2. **B** — nothing works without the import hook and class strategy settled.
3. **K1 + K2** — start in parallel. Touch nothing else, produce visible output early.
4. **C** — the spec is the vocabulary every later section speaks.
5. **C2** — the resolver. Everything downstream consumes its output.
6. **D, H** — quantization and mode adapters. H's cross-mode equivalence test is the most valuable
   test in the project; write it first, failing.
7. **I** — pre-conversion. Needed before E can handle foreign fixed costs.
8. **E** — costs.
9. **F** — leverage caps. Level 1, self-contained.
10. **M** — rate data. Independent; can move earlier.
11. **J** — construction and `from_weights`, consuming D and I. Can move earlier if investing is the
    priority use case: it needs only D and I, not E, F, or M.
12. **G** — holding costs last. Only level-3 section, most fragile.
13. **L** — runs throughout, from B onward.

### 6.3 Viable stopping points

- **After B + C + K1 + K2:** drop-in vectorbt with the full MT4/MT5 metric set plus seven
  risk-adjusted additions.
- **After + C2 + D + H:** multi-mode, multi-asset backtesting with realistic fills, configured from a
  spreadsheet. The core value proposition; a legitimate v0.1.
- **After + I + J:** calendar-rebalanced investing across currencies via `from_weights`. A coherent
  release on its own, and reachable without E, F, or M.
- **After + I + E + F:** foreign instruments, full cost modelling, leverage caps. A complete first
  release.
- **After + M + J + G:** everything.

---

## Part 7 — Sections

> **Build status.** A–M implemented, 168 tests passing, 59/63 "Done when" criteria verified
> by execution. **Section N (front-door API) is designed but not implemented.** See Part 9.


Each carries: **Purpose** → **Deliverable** → **Interface sketch** (illustrative) →
**Depends on / Blocks** → **Notes** → **Worked example (D7)** → **Done when**.

---

### A. Recon & verification — ✅ COMPLETE

**Outcome.** Part 3. Headline findings: fill quantization, SL/TP, per-asset broadcasting, and
rebalancing primitives all already exist; leverage, currency, and rate data don't exist at all;
native `alpha` is already Jensen's; `last_cash` is mutable from callbacks; about a third of the
MT4/MT5 metric set is already native.

---

### B. Package foundations

**Purpose.** Make `import vectorbtx as vbt` a working drop-in replacement, and establish the patterns
every other section follows.

**Deliverable**
- `vectorbtx/__init__.py` — namespace re-export with `Portfolio` substituted
- `vectorbtx/core/portfolio.py` — `Portfolio` subclass, stub
- `vectorbtx/core/patch.py` — `ReturnsAccessor` and `Trades` patching with collision guard
- `vectorbtx/core/settings.py` — `settings["vectorbtx"]` registration
- CI mirroring upstream's matrix
- A test asserting an unmodified vectorbt script produces identical results

**Interface sketch**

```python
# vectorbtx/__init__.py
from vectorbt import *
from vectorbt import settings
from .core.portfolio import Portfolio      # shadows vbt's
from .core.patch import apply_patches
from .core.settings import register_defaults

apply_patches()
register_defaults()
```

```python
# vectorbtx/core/patch.py
def _attach(cls, name, fn):
    if hasattr(cls, name):
        raise RuntimeError(
            f"{cls.__name__}.{name} already exists upstream — "
            f"vectorbtX would shadow it. Rename or remove."
        )
    setattr(cls, name, fn)
```

**Depends on.** A. **Blocks.** Everything.

**Notes**
- Upstream `__init__.py` is 39 lines: star-imports per subpackage, then
  `import_submodules(__name__)`. Mirror that shape.
- **Two strategies, deliberately (D5).** Patch `ReturnsAccessor` and `Trades` in place. Subclass
  `Portfolio`.
- **The `Portfolio` subclass hosts `from_weights`** (Section J), alongside upstream's `from_signals`,
  `from_orders`, and `from_holding`. That is its main user-facing job — the constructor-parameter
  and state argument for subclassing is real but secondary.
- The collision guard is not optional — the only defence against D5's shadowing risk.
- **`.x` appears in two places:** `pf.x.foo()` for anything portfolio-bound, `vbt.x.foo()` for
  constructors and loaders. Matches how vbt already splits accessors from module functions.
- Settings: add a top-level `vectorbtx` key via `__setitem__(..., force=True)`. ⚠️ **Do not mutate
  `settings["portfolio"]`** — that changes behaviour for plain vectorbt paths and breaks §1.3.
- ⚠️ `Portfolio.indexing_func` explicitly slices each field. Any per-column state added to the
  subclass must be sliced there too, or indexing silently desynchronises it.
- **Equivalence test covers numba only.** Per D1 we make no Rust guarantee; testing one would create
  an obligation we've declined.

**Done when**
- An unmodified vectorbt script gives bit-identical results under vectorbtX
- `import vectorbtx` emits no warnings
- The collision guard is proven by a test that deliberately triggers it
- CI green on the full matrix

---

### C. Instrument specification

**Purpose.** One place to describe a tradable instrument. Every later section reads from this.

**Deliverable**
- `vectorbtx/instruments/spec.py` — the dataclass
- `vectorbtx/instruments/defaults.py` — per-mode defaults
- `vectorbtx/instruments/validate.py` — validation
- A worked example spec per mode (D7)

**Interface sketch**

```python
@dataclass(frozen=True)
class Instrument:
    mode: Literal["spreadbet", "cfd", "fx_spot", "equity"]

    # unit definition
    point_size: float | None = None         # THE primitive: price increment = 1 point
    contract_size: float | None = None      # cfd, fx_spot only (spread bets have none)

    # quantization
    increment: float | None = None          # 0.01 lots, £0.01/pt, 1 share
    min_size: float | None = None           # 0.01 lots, £1/pt
    allow_fractional: bool = False
    rounding: Literal["floor","ceil","nearest"] | None = None   # None → global default

    # currency
    quote_currency: str | None = None       # what the instrument is priced in
    base_currency: str | None = None        # fx_spot only: other side of the pair
    cost_currency: str | None = None        # what fixed costs are quoted in

    # data
    price_field: Literal["close","adj_close"] | None = None      # D10; mode default

    # costs and leverage
    active_costs: frozenset[str] = frozenset()
    leverage: float | None = None           # None → per-mode default
    leverage_type: Literal["instrument_level","portfolio_level"] = "instrument_level"

    # accrual
    triple_day: str = "WED"
    end_of_day: str = "22:00"
```

**Depends on.** B. **Blocks.** C2, and through it everything.

**Notes**
- Frozen dataclass rather than vbt's `Config`: fewer moving parts, better IDE support, no
  config-copying semantics, and it's ours.
- **`point_size` is the single unit primitive.** Pip position derives from it; `tick_size` and
  `pip_decimal_place` are not separate fields. For a 5-decimal FX quote, `point_size = 0.0001` (the
  pip) while price precision remains 5 decimals — ⚠️ the fractional 5th digit must not be truncated,
  since a meaningful part of the spread lives there.
- **Spread bets have no `contract_size`.** The stake is the size (§7.H).
- ⚠️ **`cost_currency` is separate from `quote_currency` on purpose.** D4 converts a US equity's
  prices to GBP, but a `$1 minimum commission` is still quoted in dollars. Without this field Section
  E cannot know which fixed costs need converting. Easiest thing in the plan to get silently wrong.
- **`leverage_type`** selects between a per-ticker value and a single portfolio-wide value.
  ⚠️ Validation must reject `portfolio_level` combined with per-ticker leverage values rather than
  silently picking one. ⚠️ With `group_by` and cash sharing, "portfolio" means **the group** — the
  cash-sharing unit is the only coherent budget boundary.
- **`price_field` per D10.** Equity defaults to `adj_close`; spread bet and CFD to `close`. Warn at
  resolve time when a nominal-point mode receives unadjusted data (dividends silently absent).
- `triple_day` and `end_of_day` live here because they are broker-and-instrument properties, not
  global settings.
- **Validation rejects on construction:** lots on a spread bet, `contract_size` on a spread bet,
  `increment <= 0` (mirroring `nb.py:414`), unresolvable currency combinations, `allow_fractional`
  on a mode that doesn't support it, `min_size` not a multiple of `increment`, `portfolio_level` with
  per-ticker leverage.

**Worked example (D7).** One fully-populated spec per mode with the Part 8 defaults, hand-checked.
Becomes fixtures for D, E, F, G, H, I.

**Done when**
- All four modes have a valid default spec
- Every validation rule has a test that triggers it
- The four example specs are committed as fixtures

---

### C2. Argument resolution

**Purpose.** Let a user describe a whole universe declaratively — dict or spreadsheet — and turn it
safely into the per-column arrays vectorbt already broadcasts.

**Deliverable**
- `vectorbtx/resolve/load.py` — dict and DataFrame front doors
- `vectorbtx/resolve/coerce.py` — DataFrame type coercion
- `vectorbtx/resolve/keywords.py` — per-parameter keyword registry
- `vectorbtx/resolve/resolver.py` — specs → per-column numpy arrays
- `vectorbtx/resolve/export.py` — round-trip export

**Interface sketch**

```python
def load_universe(source: dict | pd.DataFrame | str | Path) -> Universe:
    """Both front doors normalise to dict[column, Instrument], then validate."""

class Universe:
    specs: dict[str, Instrument]

    def resolve(self, data, rates=None) -> dict:
        """Order kwargs, per-column, in the wrapper's column order."""

    def to_frame(self) -> pd.DataFrame:
        """Fully-resolved universe including filled defaults. For eyeballing."""
```

Both paths converge *before* validation:

```
dict[str, Instrument]  ─┐
                        ├─► dict[column, Instrument] ─► validate ─► resolve ─► numpy
DataFrame (CSV/Excel) ──┘         (Section C)
```

**Depends on.** C. **Blocks.** D, E, F, H, J.

**Notes**
- **The mechanism already exists; this is ergonomics.** vbt broadcasts a plain list per column at
  almost no memory cost via `flex_select_auto_nb` (§3.3).
- **Converge before validating.** The DataFrame path would otherwise bypass every check in Section C
  — precisely the path where bad data enters.
- ⚠️ **Column order comes from the wrapper's column index**, not dict insertion order or spreadsheet
  row order. `group_by` and cash sharing can change resolved columns, so resolution happens against
  the actual resolved index.
- ⚠️ **The instrument-name column is a join key and mismatches must be fatal.** Every price column
  needs a spec and every spec needs a price column. Error both directions — a warning on a 20-asset
  backtest scrolls past and one column silently gets defaults.
- **Schema:** fixed columns, with extra columns passed through as metadata and preserved on
  round-trip export. Lets users keep notes and broker names in the sheet.
- **CSV first**, Excel behind an optional `openpyxl` extra.

**Type coercion (DataFrame path only).** The dict path arrives correctly typed; CSV and Excel do not.
Expect `increment` as `"0.01"`, `allow_fractional` as `"FALSE"`/`"0"`/`"no"`, `mode` with trailing
whitespace, Excel reformatting numbers or eating leading zeros. Also: the name column goes through
the same reader, so a numeric-looking instrument name needs handling. ⚠️ Every failure here produces
a plausible-looking wrong number rather than an exception — coercion must be strict and loud.

⚠️ **Blank cell ≠ NaN.** vbt uses NaN and inf meaningfully in `Order` fields —
`size_granularity=NaN` means *no quantization*, `price=np.inf` means *use close*. A blank cell
becoming NaN can accidentally *mean* something. Use `None` internally for "unspecified", resolve
against mode defaults, convert to NaN only at emit.

**Keyword cells (D8).** A cell holds either a literal or a keyword.

- **Start with `slippage='auto'` and `financing='auto'` only.** Add on demand; a large keyword
  vocabulary invented up front is a liability.
- **Keywords are per-parameter, not global.** `'auto'` for slippage means volume-and-volatility
  scaled; for financing it means pull from Section M.
- ⚠️ **Unrecognised strings are fatal.** Coerce numeric first, then check the parameter's keyword
  set, then error. A typo'd `'atuo'` falling through to a default is the exact silent-wrong-number
  failure this design exists to prevent. Same for a valid keyword on a parameter that doesn't
  support it.
- **Auto-mode parameters go in sibling columns** (`slippage_max`, `slippage_coef`), not compound cell
  syntax. Parsing a mini-language out of cells looks tidy for two parameters and is unmaintainable at
  six.
- **Resolution failures name the instrument and column.** `'auto'` slippage needs a volume column;
  `'auto'` financing needs Section M loaded and covering the period. Fail at resolve time; don't emit
  NaNs that quietly zero a cost.

**Round-trip export.** If it reads a DataFrame it writes one, showing the fully resolved universe
including filled defaults. The best available defence against silent misalignment.

**Done when**
- dict and DataFrame paths produce identical `Universe` objects from equivalent input
- Validation runs once, on the converged form
- A misordered spreadsheet still resolves correctly
- A missing or extra instrument name raises
- A typo'd keyword raises, naming the cell
- A blank cell resolves to the mode default, not NaN
- Round-trip export reproduces an equivalent universe

---

### D. Fill quantization

**Purpose.** Round intended sizes to what a broker would fill, and report drift.

**Deliverable**
- `vectorbtx/quantize/quantizer.py`
- Drift reporting, per-order and per-rebalance

**Interface sketch**

```python
def quantize(size, spec, direction=None):
    """Round to spec.increment. direction=None → spec.rounding → global default (floor).
    Returns (quantized_size, drift)."""

def to_order_params(spec, direction="floor"):
    """floor:     {"size_granularity": spec.increment, "min_size": spec.min_size}
       otherwise: {"size_granularity": np.nan, "min_size": spec.min_size}
                  with size pre-quantized by the caller."""
```

**Depends on.** C2. **Blocks.** J.

**Notes**
- **A thin wrapper, not a component.** `size_granularity` already floors (§3.3), matching the default.
- **Global default floor, per-instrument override** via `spec.rounding`. Floor is conservative and
  matches vbt's native behaviour, so it's free.
- For non-floor rounding, quantize ourselves *before* handing over, then pass
  `size_granularity=NaN`. Do not try to make the engine round differently.
- `min_size` covers the spread-bet minimum stake (£1) and the CFD minimum lot (0.01).
- ⚠️ **Drift attribution must stay separate from `max_size` effects.** `max_size` partially fills
  rather than rejecting, so realised size can differ from intended for reasons unrelated to rounding.
  Especially once F sets `max_size`.
- **Report both per-order and per-rebalance.** Per-order is the raw record; per-rebalance is what
  anyone actually reads.
- **J's minimum-trade filter routes its suppressed-trade drift through here**, so tracking error
  from skipped rebalancing trades is visible rather than vanishing.
- Mostly unaffected by D4, since increments are in shares, lots, or £/pt. Minimum *notional*
  constraints are the exception.

**Worked example (D7).** Per mode: an intended size off-increment, the quantized result, the drift.
Include a spread-bet stake below £1 and a CFD size below 0.01 lots, confirming both reject rather
than round to zero.

**Done when**
- Each mode's increment rule is tested
- Floor, ceiling, and nearest all correct
- Sub-minimum sizes reject rather than silently rounding to zero
- Rounding drift and `max_size` drift are demonstrably distinguishable

---

### E. Per-trade costs

**Purpose.** Model transaction cost, composably, testable in isolation, enabled per-instrument.

**Deliverable**
- `vectorbtx/costs/base.py` — module interface
- `vectorbtx/costs/commission.py`, `spread.py`, `slippage.py`
- `vectorbtx/costs/compose.py` — resolution into `Order` fields
- `vectorbtx/costs/helpers.py` — flat↔percentage convenience utility

**Interface sketch**

```python
class CostModule(Protocol):
    def contribute(self, spec, price, size, index, ctx) -> dict:
        """Partial Order kwargs, e.g. {"fees": 0.001} or {"slippage": series}."""

def resolve(universe, data, fx=None) -> dict:
    """Compose active modules into Order kwargs.
    Sums slippage contributions; converts fixed costs from spec.cost_currency."""

def equivalent_pct(flat_fee, notional) -> float:
    """£10 on £10,000 notional → 0.001. Helps users pick a percentage when their
    broker charges flat and they want a single number."""
```

Default stacks by mode:

```
equity:          commission + spread + slippage + leverage_financing
fx_spot:         commission + spread + slippage + carry
cfd_fx:          commission + spread + slippage + carry + leverage_financing
cfd_other:       commission + spread + slippage + leverage_financing
spreadbet_fx:    spread + slippage + carry + leverage_financing   (commission = 0)
spreadbet_other: spread + slippage + leverage_financing           (commission = 0)
```

**Depends on.** C2, I. **Blocks.** G.

**Notes**
- Each module emits a partial dict targeting one of `fees` / `fixed_fees` / `slippage`. A cost engine
  returning a single scalar total would be the wrong shape.
- Mapping: percentage commission → `fees`; flat or per-unit → `fixed_fees`; ⚠️ **spread and slippage
  both land on `slippage`, so they must be summed before assignment** — only one field exists. Least
  obvious constraint in the section.
- **Minimum commission is pre-computed per order into `fixed_fees`.** `max(pct × notional, min)` is a
  conditional and genuinely cannot be expressed as `fees` + `fixed_fees`.
- **Retail equity commission is often flat**, which maps to `fixed_fees` cleanly. The convenience
  utility exists because users often want a single percentage: £10 on £10,000 is 0.1%.
- **D3: symmetric spread only.** Document the limitation.
- ⚠️ **D4 interaction, most likely source of silent error:** percentage costs are FX-invariant and
  need no handling. Fixed and per-unit costs quoted in a foreign currency **must** be converted using
  the same rate series as the price conversion. `spec.cost_currency` exists for this.
- Spread-bet mode sets `fees = fixed_fees = 0`, folding cost entirely into `slippage`.
- **Auto slippage:** `slippage = coef × σ × sqrt(size / ADV)`, clipped to `[0, slippage_max]`.
  Square-root market impact is the standard empirical result and beats linear. σ is 20-bar realised
  volatility as a fraction of price. ⚠️ Requires a volume column — fail at resolve time if absent
  rather than silently falling back to zero.
- All target fields broadcast as time series, so variable and volatility-scaled costs need no
  simulation changes.

**Worked example (D7).** One round-trip per mode with every applicable module active, hand-computed to
the penny, including a foreign-currency fixed fee and a flat equity commission.

**Done when**
- Each module unit-tested standalone
- Composition tested, particularly slippage summation
- Every default stack reproduces its hand-computed example
- A foreign-currency fixed fee is demonstrably converted correctly
- Auto slippage produces a sane series and fails cleanly without volume

---

### F. Leverage & margin

**Purpose.** Prevent positions a broker would refuse, and expose the borrowed portion so financing
can be charged on it.

**Deliverable (v1 — F1 only, per D2)**
- `vectorbtx/margin/cap.py` — maximum-leverage enforcement
- `vectorbtx/margin/requirement.py` — margin required, per mode
- `vectorbtx/margin/exposure.py` — borrowed vs cash-covered split

**Interface sketch**

```python
def leverage_cap(spec, equity, price) -> float | np.ndarray:
    """Max position size at spec.leverage. Feed to Order.max_size."""

def margin_required(spec, size, price) -> float:
    """notional / leverage, mode-specific notional calculation."""

def borrowed_portion(pf) -> pd.DataFrame:
    """Post-simulation split into borrowed vs cash-covered.
    From pf.cash(free=True) and pf.asset_value(). Section G's input."""

def min_margin_level(pf, universe) -> pd.Series:
    """Lowest equity/margin ratio over the backtest. MT5 parity metric."""
```

**Depends on.** C2. **Blocks.** G, and K2's Margin Level Min.

**Notes**
- Nothing exists upstream (§3.4). Built on `free_cash`, `debt`, `gross_exposure`.
- **F1 is level 1 and needs no callbacks.** Compute the maximum permissible size and pass it as
  `max_size`, which **partially fills rather than rejecting** — clip semantics for free.
  `raise_reject=True` gives reject semantics instead.
- **`leverage_type` splits the implementation:**
  - **`instrument_level`** — per-column cap, computed independently, straight into `max_size`. Exact.
  - ⚠️ **`portfolio_level`** — a *joint* constraint (total gross notional ≤ leverage × equity), and
    `max_size` is per-column with no knowledge of its siblings. **Enforced on the weight matrix in
    Section J**, before simulation, where weights are known ahead of time and scaling to fit the
    budget is exact and cheap. The per-column cap remains as a backstop.
  - **Known gap:** a signal-driven strategy not going through J gets the backstop only, so aggregate
    exposure can exceed the cap. Exact enforcement there needs level 3. Documented, not built.
- **Tiered margin is deliberately not modelled.** It scales with position size in reality, but the
  complexity outweighs the difference outside extreme-risk strategies. Users set sensible maximum
  leverage instead — see Part 8 for per-mode defaults.
- `lock_cash=True` is the closest native analogue to "don't let one position borrow another's
  collateral" and is likely the right default for margined modes.
- ⚠️ Because F2 isn't built, **borrowed-vs-cash-covered must be derivable post-simulation** from
  `cash(free=True)` and `asset_value`. Section G consumes it — a hard requirement.
- `min_margin_level` is cheap here and satisfies one of K2's gaps without needing F2.

**Worked example (D7).** A 5× leveraged CFD position: notional, margin required, borrowed portion,
and the size cap `max_size` should produce. Plus a two-asset portfolio-level case showing the weight
matrix scaled to fit the budget.

**Done when**
- `max_size` demonstrably clips an over-leveraged order
- `raise_reject=True` demonstrably rejects one
- Margin requirement correct per mode against the worked example
- Borrowed portion correct for cash-covered (zero), partially financed, and fully leveraged
- `portfolio_level` scales a weight matrix to the budget; `instrument_level` caps per column
- `portfolio_level` with per-ticker leverage values raises

---

---

### F1b. Funded margin — added during the build

**The defect.** §F built `max_size` to *cap* leverage and assumed that was enough.
It is not: `buy_nb` ends its cash-limit calculation with `cash_limit = min(cash_limit,
exec_state.cash)` (`nb.py:113`), so a long is bounded by cash and **no `Order` field can
lever it**. A 30x spread bet sized at £1.67/pt on a £10,000 account was filled at about
£1.01/pt. The backtest ran, the equity curve looked plausible, and the leverage was
simply absent — the exact silent-wrong-number failure this plan exists to prevent.

Two knock-ons, both silent:

- `borrowed_portion` derived the borrowed amount from `asset_value` vs `free_cash`. With
  longs unable to exceed cash it stayed near zero, so **Section G's leverage financing
  barely fired for long positions**.
- Every leveraged strategy understated both position size and cost.

**The fix.** Model the margin account as brokers actually do it:

- the **engine** is given buying power, `equity x leverage`, as initial cash;
- the loan, `equity x (leverage - 1)`, is carried on the `Portfolio`;
- `Portfolio.get_init_cash` **reports equity**, subtracting the loan back out.

Reported cash is `equity + cumsum(cash_flow)`, so it goes negative by exactly the drawn
loan, and `value = cash + asset_value` returns true equity. Everything downstream
(`returns`, `returns_stats`, every `_metrics` entry) derives from those, so it all comes
out right with no engine change — still a pure wrapper.

`borrowed_portion` becomes `max(0, -cash)`: exact by construction, and no longer confuses
the notional with the borrowed part (£40,000 against £10,000 borrows £30,000, not
£40,000).

**Still not modelled**, unchanged by this and still deferred by D2: maintenance margin,
margin calls, forced liquidation. A position can draw the whole loan and sit there;
`min_margin_level` reports how close it came.

**Done when**
- A 30x spread bet is filled at the size its leverage implies, not its cash
- Reported initial cash is equity, not buying power
- `cash + asset_value == equity` exactly
- `borrowed_portion` excludes the cash-covered part
- Leverage financing produces a material charge on a levered long
- An unlevered portfolio carries no loan and is bit-identical to before

---

### N2. Next-open execution — added during the build

**The defect.** Nothing in the plan specified an execution price, so every path filled at
the signal bar's close (vectorbt's `price` default of `np.inf`). That is lookahead: a
crossover computed from bar `t`'s close cannot be acted on until bar `t+1` opens.

Two errors, pulling in opposite directions but not cancelling:

- the backtest **captures the overnight gap** it could not have traded, which on a gappy
  instrument is the larger number;
- it **bills a night of financing** on a position not yet held.

**Not a vectorbt limitation.** `from_signals` takes `open` and `price` as first-class
parameters; the idiom is `entries.shift(1)` with `price=open`. The gap was ours.

**The fix, and why there is no opt-in.** `open` is a required argument of both front-door
functions. Entries, exits and rebalance schedules are shifted one bar; `price=open`. All
order-time quantities — sizes, stop distances, spread fractions — resolve against the fill
open rather than the signal close, so no magnitude is derived from a price that was not
knowable when the order was placed.

There is deliberately no `execution=` switch. A close-fill backtest is not a variant worth
supporting, and offering it as an option invites its use; the API is simpler for having one
path rather than two.

**Done when**
- A signal on bar `t` produces an order on bar `t+1`, filled at that bar's open
- A gap between signal close and next open accrues to the market, not the strategy
- Rebalance schedules shift identically
- Omitting `open` raises, with no way to opt out

---

### N3. Mandatory OHLC — added during the build

Signals need the close, fills need the open, and stops need the high and low. All four are
required arguments of the front door, with no default for any of them, because every
available default flatters the backtest:

- no `open` -> fills at the signal bar's close, capturing an untradeable overnight gap;
- no `high`/`low` -> a stop can only be compared to the close, so it fires a bar late or
  not at all, and drawdown is measured close-to-close, missing every intrabar excursion.

`pf.x.intrabar_drawdown()` reports the worst peak-to-trough decline with the peak taken from
the highs and the trough from the lows. On daily data it typically runs a third above the
close-to-close figure. ⚠️ Taking both ends from the lows would depress the running peak by
the same amount as the trough and report a drawdown *smaller* than the close-to-close one.

`H >= C >= L` is checked at construction and warned about, since a violation usually means
the series were adjusted or FX-converted on different schedules — after which stops fire at
prices that never traded.

All four are **keyword-only**. Four same-shaped frames in positional sequence is exactly the
call where transposing two — high and low swapped, open where close belongs — runs cleanly
and reports a wrong number rather than raising. Naming them makes the mistake unspellable,
at the cost of four extra characters per call.

**Done when**
- Omitting any of the four raises
- Passing them positionally raises
- Intrabar drawdown is never below the close-to-close figure
- Inconsistent high/low warns
- Multi-currency examples convert all four series with the same table

---

### G. Holding costs

**Purpose.** Charge for time. Financing on borrowed money, carry on currency positions.

**Deliverable**
- `vectorbtx/holding/financing.py` — pure arithmetic
- `vectorbtx/holding/carry.py` — pure arithmetic
- `vectorbtx/holding/accrual.py` — trading-day and triple-day logic
- `vectorbtx/holding/callback.py` — the thin `post_segment_func_nb`

**Interface sketch**

```python
# pure, testable without a simulation
def financing_charge(borrowed, benchmark_rate, markup, is_long, day_multiplier) -> float:
    """Markup is ALWAYS adverse: benchmark + markup when long, benchmark - markup
    when short. Never a simple addition."""

def carry_charge(notional, tom_next_differential, day_multiplier) -> float: ...

def day_multipliers(index, triple_day="WED", end_of_day="22:00") -> np.ndarray:
    """1x per trading day, 3x on triple_day, 0 on non-trading days."""

@njit
def post_segment_func_nb(c, financing_rates, carry_rates, multipliers, ...):
    for col in range(...):
        c.last_cash[col] -= ...   # see §5.3
```

**Depends on.** E, F, M. **Blocks.** Nothing.

**Notes**
- **The only level-3 section** after D2 and D4, and therefore the only part of the library depending
  on `portfolio/nb.py` internals. Everything about how it's written should serve keeping that
  dependency small.
- ⚠️ **Read §5.3 first.** Two silent failures: `call_post_segment` must be `True` or the callback
  never fires on non-trading bars, which is exactly where financing accrues; and accrual lands on the
  next bar (D6, accepted).
- **Accrual is trading days only, with a 3× multiplier on the triple day.** Weekends and holidays
  carry no separate charge — the Wednesday multiple covers the weekend. ⚠️ Do not additionally charge
  calendar days; combining the two conventions bills nine days a week.
- **Financing applies to the borrowed portion only** — consuming F's output. A cash-covered position
  finances at zero. Carry does **not** work this way: it applies to any FX position regardless of
  leverage, because you are long one currency and short another either way. Separate modules,
  separate rate tables, separate multiplier tables.
- ⚠️ **The broker markup is always adverse.** `benchmark + markup` financing a long,
  `benchmark − markup` when short. A simple addition credits shorts too generously and flatters
  short-heavy backtests.
- **Carry uses the tom-next rate**, not an interpolated differential.
- **Rates come from Section M**, not a static parameter.
- **D4 interaction:** apply the position's *native-currency* rate to the base-currency-equivalent
  notional. Convert the amount, not the rate.
- **Triple-day and end-of-day come from the spec**, not globals. ⚠️ End-of-day is inert on daily bars
  — document it, so nobody debugs a setting that cannot have an effect.
- **Keep the callback body thin.** All arithmetic in pure functions unit-testable without a
  simulation. Good practice, and the mitigation for §5.5 item 1 — when upstream breaks the callback
  signature, only the thin layer needs rewriting.
- **Future dividend support lands here** (D10), as a cash adjustment alongside financing and carry.

**Worked example (D7).** One long and one short held across a Wednesday: financing and carry
hand-computed, including the adverse markup on both sides, the 3× multiplier, and the next-bar
landing (D6).

**Done when**
- Pure arithmetic functions pass hand-computed tests without any simulation
- A cash-covered position incurs zero financing
- A short is charged `benchmark − markup`, not `benchmark + markup`
- Carry applies to an unleveraged FX position
- The triple-day multiplier applies on the configured day, not a hardcoded Wednesday
- No charge accrues on weekends or holidays
- Total charge over a period matches hand computation despite the one-bar lag

---

### H. Instrument modes

**Purpose.** Let a user express a position in the units they actually trade in and translate to what
the engine simulates. This is what makes vectorbtX useful for derivatives.

**Deliverable**
- `vectorbtx/modes/base.py` — adapter protocol
- `vectorbtx/modes/spreadbet.py`, `cfd.py`, `fx_spot.py`, `equity.py`
- `vectorbtx/modes/helpers.py` — mode-agnostic SL/TP, position sizing, point-risk
- **The cross-mode equivalence test** — the section's real deliverable

**The spread-bet conversion, resolved.**

```
engine_size = stake_per_point / point_size
```

Verified against three independent cases:

| Instrument | `point_size` | £1/point → | expected |
|---|---|---|---|
| UK stock (GBP series) | 0.01 | 100 shares | 100 ✓ |
| EURUSD | 0.0001 | 10,000 units = 0.1 lots | ~0.1 lots ✓ |
| Micro FX @ £0.10 | 0.0001 | 1,000 units = 0.01 lots | micro ✓ |

All three fall out of one line — the "100 contract size" for UK stocks is *derived*, not configured.
Commodities use the same £1/point treatment as equities, so there is no special case.

**Interface sketch**

```python
class ModeAdapter(Protocol):
    def to_engine_units(self, user_size, spec, price) -> float: ...
    def from_engine_units(self, engine_size, spec, price) -> float: ...
    def points_to_price_fraction(self, points, price, spec) -> float:
        """For native sl_stop / tp_stop, which are fractions of price."""

def size_for_risk(risk_amount, stop_distance_points, spec, price) -> float: ...
```

**Depends on.** C2. **Blocks.** Nothing directly, but everything user-facing routes through it.

**Notes**
- Level 1. Each adapter is essentially two functions. **Resist making them anything more** — the
  moment an adapter makes decisions rather than converting units, the abstraction has failed.
- **SL/TP is native** (§3.3) and expressed as **fractions of price**. Adapters convert point stops
  into price fractions; do not reimplement stop logic. Useful consequence: fractional stops are
  FX-invariant under D4's uniform scaling.
- ⚠️ **FX fractional pips:** `point_size` is the pip (0.0001), but quotes carry 5 decimals.
  Conversions must not truncate to 4 — a meaningful part of the spread lives in the 5th digit.
- **D4 exemption:** spread bets are account-currency-denominated by construction, as are GBP-quoted
  CFDs on foreign underlyings. Pre-conversion applies only to equity and FX spot.
- Position sizing and point-risk are mode-agnostic — identical formula, different unit conversion.
  Build once above the adapters.

**Worked example (D7), and the section's real deliverable.** An economically identical position —
same underlying, same exposure, same price move — expressed as a spread bet, a CFD, an FX spot
position, and shares. All four must produce identical P&L to floating-point tolerance. **Write the
test first, failing.**

**Done when**
- The cross-mode equivalence test passes
- Each adapter round-trips: `from_engine_units(to_engine_units(x)) == x`
- Point stops convert correctly and fire at the right prices
- A 5-decimal FX quote retains its fractional pip through conversion
- Risk-based position sizing gives identical monetary risk across all four modes

---

### I. Multi-currency

**Purpose.** Let a base-currency portfolio hold foreign-currency instruments with correct P&L,
without touching the engine.

**Deliverable**
- `vectorbtx/fx/convert.py` — the conversion function
- `vectorbtx/fx/table.py` — the rate table, shared with G
- A `Data` subclass method retaining the original series for decomposition

**The rate table.** Long form, keyed `(date, quote, base) → value`, where `value` is what you
multiply a `quote`-currency price by to get `base` currency. **Direction lives in the key**, so
there is no multiply-or-divide decision to get wrong.

- **Fetch one canonical direction and derive the reciprocals.** Providers mostly quote majors against
  USD, so the grid must be filled by inversion and triangulation regardless. ⚠️ Fetching both
  directions independently gives non-exact reciprocals — different timestamps, mid vs bid/ask capture
  — and a round-trip conversion then produces phantom P&L. That noise is **random in sign**, so it
  inflates measured volatility rather than acting as a cost, quietly degrading every risk-adjusted
  metric with a standard deviation in the denominator. Deriving from canonical guarantees
  `value(A,B) × value(B,A) = 1` by construction.
- **Validate on load:** reciprocal consistency, and triangulation `value(A,C) ≈ value(A,B) × value(B,C)`.
  Both are cheap once the full grid is held, and they catch stale rows, bad joins, and mixed-source
  contamination. Warn on divergence beyond a basis point.
- For resolution, pivot to wide per-pair once against the backtest index rather than filtering the
  long frame repeatedly.
- **Cross-pair point value** (neither side is the account currency) is just another lookup — not a
  special case.

**Interface sketch**

```python
def to_base_currency(data, base, fx_table, fx_markup=0.005,
                     fill_policy="ffill_with_warning"):
    """Convert OHLC to base currency using ONE rate per bar.

    fx_markup applies the broker's worse-than-market rate, ADVERSELY in both
    directions — you convert in at a worse rate and out at a worse rate.

    Returns converted data with the original retained as .native.
    """

def convert_fixed_costs(amount, cost_currency, base, fx_table): ...
```

**Depends on.** C2. **Blocks.** E (foreign fixed costs), J (converted market caps).

**Notes**
- **This is exact, not an approximation.** See D4 for the identity.
- Level 2. No callbacks, no engine changes, **no restriction on `cash_sharing`**.
- **`fx_markup` models the broker's worse-than-market conversion rate**, default 0.5%. Not a fudge: a
  spread on the FX rate is exactly analogous to a spread on price, and modelling it here puts the
  cost where it originates. Foreign handling *fees* are separate and go through `cost_currency` as
  ordinary `fixed_fees`.
- **Rate convention:** close-to-close, aligned to the bar's close timestamp.
- **Conversion is explicit, never automatic** — but warn if `quote_currency != base` and no conversion
  happened. Silent auto-conversion of price data is too surprising.

**Four gotchas, all cheap if known up front:**

1. ⚠️ **One FX rate per bar, applied to all of OHLC.** Different rates for open, high, low, and close
   break the `H ≥ C ≥ L` invariant, and native SL/TP fires at prices that never existed. Same failure
   mode as D10's mixed-adjustment hazard — one validation check covers both.
2. ⚠️ **Percentage costs are FX-invariant; fixed and per-unit costs are not.** See E and
   `spec.cost_currency`.
3. ⚠️ **Convert the benchmark too**, or beta and alpha compare base-currency returns against
   foreign-currency returns, and every risk-adjusted metric downstream is wrong.
4. ⚠️ **Financing rates stay currency-specific.** Convert the notional, not the rate. See G.

Also needs converting if used: market caps for cap-weighting (J), and any minimum-notional
constraint (D).

- **What is lost:** the converted series conflates asset return with FX return. Correct for total
  return; decomposition needs the original. Retain it on the `Data` subclass.
- ⚠️ **Index alignment needs an explicit fill policy.** FX trades roughly 24/5, equities have exchange
  hours, holidays differ by country. Mismatches are guaranteed. Do not silently forward-fill — warn,
  or make the policy an explicit argument.

**Worked example (D7).** MSFT held across a period where price and FX move in *opposite* directions,
hand-checked against the D4 identity. Opposite directions matters — same-direction moves can hide
sign errors. Include a non-zero `fx_markup` to confirm it is adverse on both legs.

**Done when**
- The conversion identity is verified numerically
- OHLC ordering survives conversion
- `fx_markup` is adverse in both directions
- Reciprocal and triangulation validation catch a deliberately corrupted table
- A cross-currency rebalance produces correct weights
- The original series is retrievable for decomposition
- A deliberate calendar mismatch warns rather than silently forward-filling

---

### J. Portfolio construction & rebalancing

**Purpose.** Support *investing* strategies as a first-class use case alongside trading. vectorbt's
native constructors all assume signal-driven entry and exit; a rebalanced portfolio instead holds
continuously and periodically adjusts weights. This section provides `Portfolio.from_weights` as a
peer to `from_signals` and `from_orders`, plus the weight and schedule builders behind it.

**Why a constructor and not an instrument mode.** `Instrument.mode` describes *instrument type* —
how user units convert to engine size. Rebalancing is a *strategy style*, orthogonal to that: a
basket of CFDs can be rebalanced perfectly sensibly. Adding `portfolio_investment` to the mode enum
would conflate the two and force rebalancing to imply equities. vbt's own pattern is the right one —
`from_signals`, `from_orders`, `from_holding`, `from_random_signals` are all different ways to
express a strategy, all producing the same `Portfolio`. `from_weights` sits alongside them.

**Deliverable**
- `vectorbtx/construct/weights.py` — equal, price, cap
- `vectorbtx/construct/schedule.py` — calendar and threshold rebalancing, composable
- `vectorbtx/construct/rebalance.py` — drift handling, target emission, leverage budget
- `Portfolio.from_weights` — classmethod on the Section B subclass, implementation here

**Interface sketch**

```python
# weights
def equal_weight(columns, index) -> pd.DataFrame: ...
def price_weight(prices, index) -> pd.DataFrame: ...
def cap_weight(market_caps, index, fx=None) -> pd.DataFrame: ...

# schedules — boolean masks, composable with & and |
def calendar_schedule(index, freq="M") -> pd.Series: ...
def threshold_schedule(current, target, tolerance=0.05) -> pd.Series:
    """Relative drift, per-asset."""

# budget and emission
def apply_leverage_budget(weights, universe, equity) -> pd.DataFrame:
    """portfolio_level enforcement: scale weights to fit total notional budget."""

def to_targets(weights, schedule, universe, min_trade=None)
        -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (quantized targets, drift). NaN off-schedule. Quantizes before emission."""

# the constructor
@classmethod
def from_weights(cls, data, weights, schedule, universe, min_trade=None, **kwargs):
    targets, drift = to_targets(weights, schedule, universe, min_trade)
    return cls.from_orders(
        data, size=targets, size_type="targetpercent",
        cash_sharing=True, group_by=True, call_seq="auto",
        **universe.resolve(data), **kwargs
    )
```

**Depends on.** D, I. **Blocks.** Nothing.

**Notes**
- **You never compute the buy/sells.** `SizeType.TargetPercent` derives orders from target weights;
  combined with `cash_sharing=True` and `group_by`, the messy part is solved upstream (§3.3). This
  section is a weight-matrix builder, not a new simulation path — hence level 2.
- Calendar scheduling is a `DatetimeIndex` resample producing a boolean mask; threshold scheduling
  compares drifted weights against targets. Both pure pandas, pre-simulation.
- **Schedules compose.** "Monthly, but only if drift exceeds 5%" is the common real-world rule and is
  just `calendar & threshold`. Support it explicitly rather than making users build it.
- **Threshold rebalancing uses relative drift, per-asset.** Relative treats a 1% position drifting to
  1.5% as the 50% move it is; absolute under-triggers on small positions.
- **Cap weights are recomputed at each rebalance.** Fixed-at-construction isn't cap weighting, it's an
  initial-cap-weighted buy-and-hold.
- **This is where `leverage_type="portfolio_level"` is enforced** (Section F). Weights are known ahead
  of time, so scaling them to fit the notional budget is exact and cheap.
- **D4 makes cross-currency rebalancing work**, since every column is base currency by the time the
  engine sees it. Cap-weighting needs market caps converted with the same rate table.
- **Minimum-trade filter: available, default off.** Suppresses rebalancing trades costing more in fees
  than the drift they correct. ⚠️ Suppressed trades accumulate as tracking error — route that drift
  through Section D's reporting rather than letting it vanish.
- Quantize before emitting targets. ⚠️ Order matters: quantizing after the engine computes target
  percentages loses the drift information.

**Four hazards specific to this section:**

1. ⚠️ **NaN off-schedule, never forward-fill.** Supply the target only on rebalance bars; NaN
   elsewhere means "no order" and the position drifts as intended. Forward-filling the weights
   rebalances *every bar* — which still runs, still looks plausible, and quietly accrues transaction
   costs while producing a suspiciously smooth equity curve. **The single most likely thing to get
   written backwards in this section.**
2. ⚠️ **`call_seq="auto"` is load-bearing, not a nicety.** It orders sells before buys within a
   segment so the sells fund the buys. Without it a rebalance can fail for lack of cash despite the
   weights being perfectly fundable.
3. ⚠️ **Validate that weights sum sanely.** Under 1 leaves the remainder in cash — fine, and often
   intended. Over 1 is implicit leverage that must route through the `leverage_type` budget rather
   than entering through the back door.
4. **`SizeType.Percent` is not a substitute.** Upstream explicitly warns it misbehaves with
   `call_seq="auto"` and does not support position reversal. `TargetPercent` is correct — document
   why, or someone will "simplify" it later.

- **If combined with Section G** (which requires `from_order_func`), `call_seq="auto"` isn't
  automatic — but `sort_call_seq_nb` and `approx_order_value_nb` (§3.3) are callable from
  `pre_segment_func_nb`. They compose; it takes explicit wiring.
- **Note on D10:** investing is precisely the use case where the `adj_close` default earns its keep.
  Over a long hold, dividends are most of the equity return — so the mode that most needs them gets
  them, while the modes that can't have them (spread bet, CFD) are those with holding periods short
  enough that it matters least.

**Done when**
- Each weighting scheme produces correct weights
- Calendar and threshold schedules both trigger correctly, and compose with `&`
- Off-schedule bars are NaN, and a forward-filled input is rejected or warned
- `from_weights` reproduces a hand-computed two-asset monthly rebalance
- Quantization applied before target emission, with drift reported
- `portfolio_level` leverage scales weights to the budget
- Weights summing above 1 route through the leverage budget rather than silently levering
- Minimum-trade suppression reports its drift
- A cross-currency portfolio rebalances to correct base-currency weights

---

### K1. Returns metrics

**Purpose.** Add the risk-adjusted metrics vectorbt lacks, in the same `.stats()` output as native
ones.

**Deliverable**
- `vectorbtx/metrics/returns/` — one module per metric family
- `_metrics` entries merged into `ReturnsAccessor`

**The seven gaps, with resolved definitions.**

- **Ulcer Index** — Martin (1987), on the equity curve: cumulative returns → running max →
  **percentage** drawdown → RMS. `UI = sqrt(mean(Dᵢ²))`. ⚠️ Percentage, not absolute — squaring
  absolute drawdowns makes the metric scale with account size and non-comparable.
- **Ulcer Performance Index (Martin ratio)** — `(annualised return − rf) / UI`.
- **M² (Modigliani–Modigliani)** — return adjusted to benchmark volatility, expressed as a return.
  Reuses Sharpe and benchmark volatility.
- **Treynor ratio** — excess return per unit of systematic risk. Reuses `beta_1d_nb`.
- **Geometric Sharpe** — Sharpe on log returns. Native Sharpe is arithmetic, so this is a genuine gap.
- **SNR — efficiency ratio.** `|P_end − P_start| / Σ|P_t − P_t−1|`. Net directional movement over
  total path length, bounded 0–1. Chosen over the mean/σ reading (which duplicates Sharpe) and the dB
  form (unconventional in finance). Complements K-ratio: K-ratio measures trend consistency,
  efficiency ratio measures path directness.
- **K-ratio (v1, v2, v3)** — regression of cumulative log returns against time, then
  `slope / standard_error_of_slope`. The three variants differ **only in normalisation**: v1
  (Kestner 1996) divides by n, so isn't comparable across lengths; v2 (2003) adjusts for return
  frequency; v3 (2013) normalises by √n and is frequency- and length-independent. **Build the
  regression as a primitive and expose three thin wrappers. Default v3.**

**Interface sketch**

```python
@njit(cache=True)
def ulcer_index_1d_nb(returns): ...

@njit(cache=True)
def ulcer_index_nb(returns_2d): ...          # column loop

def rolling_ulcer_index_nb(returns, window):  # via generic_nb.rolling_apply_nb
    ...

ReturnsAccessor._metrics["ulcer_index"] = dict(
    title="Ulcer Index",
    calc_func="ulcer_index",
    tags=["vectorbtx", "risk"],
)
```

**Depends on.** B only. **Blocks.** Nothing.

**Notes**
- **Fully independent of C–J.** Parallel track from day one; produces visible output early.
- **Do not build:** Jensen's alpha (native `alpha` already is it — §3.3), Beta, Sharpe, Sortino,
  Calmar, Omega, Skew, Kurtosis, Tail Ratio, Common Sense Ratio, VaR, CVaR, up/down capture, and all
  rolling variants.
- **The K-ratio regression is shared with K2's LR Correlation and LR Standard Error.** Three metrics
  for the price of one — build them together.
- **Follow the file's conventions exactly**: `@njit(cache=True)`, a `_1d_nb` core, an `_nb` column
  loop, a `rolling_*_nb` via `generic_nb.rolling_apply_nb`. Matching this makes the additions
  indistinguishable from native ones.
- Registration per D5: `setattr` methods onto base `ReturnsAccessor`, then merge `_metrics` in place.
  `metrics="all"` picks them up automatically. **Tag every entry.**
- ⚠️ **Metrics needing a benchmark** (Treynor, M²) must set `check_has_benchmark_rets`; those needing
  a year frequency must set the `has_year_freq` filter. Without these they fail confusingly.
- **Implement primary-source definitions**, and add tests documenting where quantstats differs
  (annualisation in `ulcer_index`, risk-free treatment in `treynor_ratio`). Inheriting quantstats'
  deviations means inheriting bugs; but someone migrating will compare numbers, so the deltas should
  be recorded rather than discovered.

**Done when**
- All seven match hand-computed or reference values
- All appear in `.stats()` without explicit selection
- `pf.returns_stats()` includes them
- Rolling variants work
- Benchmark-requiring metrics fail cleanly with no benchmark
- quantstats deltas are documented in tests

---

### K2. Trade metrics — MT4/MT5 parity

**Purpose.** Full MT4/MT5 statistics parity, so output is legible to anyone arriving from those
platforms.

**Deliverable**
- `vectorbtx/metrics/trades/` — the additions
- `_metrics` entries merged into `Trades`, curated subset onto `Portfolio`

**Audit against vectorbt 1.1.0.**

*Already native (13) — do not build:*

| MT metric | vectorbt |
|---|---|
| Profit Factor | `trades.profit_factor` |
| Expected Payoff | `trades.expectancy` |
| Maximal Drawdown | `max_dd`, `drawdowns.max_drawdown` |
| Relative Drawdown | `max_dd` is already % |
| Equity Drawdown (MT5) | native `value` includes floating open positions |
| Sharpe Ratio | `sharpe_ratio` |
| Total Trades | `total_records` |
| Profit Trades % | `win_rate` |
| Largest Profit Trade | `best_trade` |
| Largest Loss Trade | `worst_trade` |
| Average Profit Trade | `avg_winning_trade` |
| Average Loss Trade | `avg_losing_trade` |
| Max Consecutive Wins/Losses (count) | `winning_streak.max()`, `losing_streak.max()` |

*One-line derivations (6):* Total Net Profit (`trades.pnl.sum()`), Gross Profit
(`trades.winning.pnl.sum()`), Gross Loss, Loss Trades % (`1 − win_rate`), Short Trades won % and Long
Trades won % (`trades.short.win_rate` via the existing filter).

*Genuine gaps (12):*

- **Absolute Drawdown** — `initial_deposit − minimum_balance`, in **currency, not percent**. Uses
  `init_cash`. ⚠️ vbt supports `InitCashMode.Auto`, where `init_cash` is derived — resolve to the
  actual figure first, or the metric is meaningless in auto mode.
- **Balance Drawdown** — closed-positions-only equity curve. **Note the inversion:** vbt's `value`
  already includes floating open positions, so MT5's *Equity* drawdown comes free and MT4's *Balance*
  drawdown is the gap. Needs a synthetic curve from closed-trade PnL.
- **Recovery Factor** — net profit / max DD. ⚠️ vbt's `drawdowns` has a "recovery return" that is a
  different thing; don't conflate.
- **Max Consecutive Wins/Losses ($)** — streak *count* is native, the monetary total isn't.
- **Maximal Consecutive Profit / Loss** — the most profitable streak, not necessarily the longest.
  Distinct from the above.
- **Average Consecutive Wins / Losses** — mean streak length.
- **LR Correlation**, **LR Standard Error** — byproducts of K1's K-ratio regression. Build together.
- **GHPR / AHPR** — geometric and arithmetic holding-period return. Overlaps geometric Sharpe.
- **Z-Score** — the Wald–Wolfowitz runs test.
  `Z = (N(R − 0.5) − P) / sqrt(P(P − N) / (N − 1))`, where N = total trades, R = number of runs,
  P = 2 × wins × losses. MT's number is the standard runs test — no variant to choose.
  Document the interpretation: **negative Z means outcomes cluster into streaks**, positive means
  they alternate. It is a dependency test, not a performance measure, and is constantly misread.
- **Margin Level Min** — from F1's `margin_required`; doesn't need F2.

*Not applicable (2):* Modelling Quality and Mismatched Charts Errors are artifacts of MT's
tick-interpolation and multi-timeframe data handling. vbt simulates the bars it's given.

**Depends on.** B; Margin Level Min additionally depends on F. **Blocks.** Nothing.

**Notes**
- `Trades._metrics` is the same mutable `Config` (`trades.py:807`), so D5's patching transfers
  unchanged.
- **Streaks are already `MappedArray`s** giving per-trade streak position — enough to derive every
  consecutive-streak metric without touching the simulation.
- **Curated subset on `Portfolio._metrics`:** Total Net Profit, Recovery Factor, Absolute Drawdown,
  GHPR. Everything else on `pf.trades.stats()`, keeping the default table readable.
- Tag everything, so `tags="mt5"` selects the parity set.

**Done when**
- All 18 additions match hand-computed values
- Full set available on `pf.trades.stats()`
- Curated subset on `pf.stats()`
- Balance and Equity drawdown are demonstrably different on a portfolio with open positions
- Absolute Drawdown resolves correctly under `InitCashMode.Auto`

---

### M. Rate data

**Purpose.** Supply real historical benchmark interest rates, so financing and carry reflect the
period tested rather than a guessed number.

**Deliverable**
- `vectorbtx/rates/fred.py` — `FREDData(Data)` following the `data/custom.py` pattern
- `vectorbtx/rates/table.py` — currency-keyed rate table, shared by G and I
- Caching with refresh-on-load

**Interface sketch**

```python
class FREDData(Data):
    """Follows YFData / BinanceData / AlpacaData pattern in data/custom.py."""

    @classmethod
    def download(cls, series_ids=None, base="GBP", start=None, end=None,
                 cache=True, **kwargs): ...

def rate_table(currencies, index, source="fred") -> pd.DataFrame:
    """Aligned to the backtest index, one column per currency."""
```

**Depends on.** B. **Blocks.** G.

**Notes**
- No FRED support exists in vbt (§3.4), but `data/custom.py` gives a clear template and
  `data/updater.py` supports scheduled refresh, matching the cache-and-update-on-load requirement.
- Needs a free FRED API key. **Optional dependency**: the library must work with a static rate if the
  user doesn't want a key.
- **Default series per currency:** SONIA (GBP), SOFR (USD), ESTR (EUR), TONA (JPY), SARON (CHF). All
  post-LIBOR overnight benchmarks.
- **Day count: ACT/360 for all currencies.** Chosen for simplicity and because it slightly overstates
  cost, which is the safer direction. ⚠️ **Note the inversion for carry:** the same arithmetic
  overstates a *received* credit, so a positive-carry position looks marginally better than reality.
  Minor in magnitude, but "overstates = conservative" is not universal here — document it.
- ⚠️ **Rate series are published on business days**; backtests may run on calendar days. Alignment
  needs the same explicit fill policy discipline as Section I, and the two should share it.
- ⚠️ **Rates are quoted annualised**; financing accrues daily. The conversion must be explicit.
- **Caching:** platform cache dir, keyed by series and date range, refreshed on load for any range
  extending past the cached end.
- **FRED unreachable:** use cache with a warning; fail only if there's no cache and no static
  fallback.
- Keep the markup out of this section. M supplies benchmarks; G applies the adverse markup.

**Worked example (D7).** One daily accrual from an annualised rate under ACT/360, hand-computed, to
confirm the day-count conversion is explicit rather than assumed.

**Done when**
- Rates download and cache
- Refresh-on-load works without re-downloading history
- The library works with a static rate and no API key
- Day-count conversion is explicit and tested
- Business-day rates align to a calendar-day index with a stated policy

---

### L. Compatibility & release

**Purpose.** Keep the wrapper working as upstream moves, and make the failure mode a visible warning
rather than a silent wrong answer.

**Deliverable**
- Two CI axes: pinned and floating
- The drop-in equivalence test from Section B
- Documentation of the extras surface

**Depends on.** B. Runs continuously thereafter.

**Notes**
- **Runs throughout, not at the end.**
- **Two CI axes.** Pinned `vectorbt==1.1.0` must pass. Floating latest vbt is informational and **is
  allowed to be red** — the canary for §5.5 item 1 (nb.py internals) and item 5 (D5's name
  shadowing). A red canary is information, not an emergency.
- Mirror upstream's OS × Python matrix, or explicitly narrow it and say why.
- **No dual-engine testing** (D1). If a user has `vectorbt-rust` installed, plain pass-through paths
  still use it — a bonus, not a contract. Document it that way.
- **The D7 worked examples double as the regression suite.** Their larger long-term value: they
  encode not just "does this run" but "does this still produce the number we agreed was correct."

**Known limitations to document explicitly:**
- D3 — symmetric spread only; no bid/ask asymmetry
- D6 — holding costs land on the next bar
- D4 — return decomposition needs the retained native series
- D10 — no dividend treatment for spread bets and CFDs; adjusted histories are not reproducible
- F — `portfolio_level` leverage is exact only via Section J; signal-driven strategies get a
  per-column backstop
- G — end-of-day time is inert on daily bars
- M — ACT/360 overstates cost (conservative) but also overstates received carry (not conservative)

**Done when**
- Both CI axes running
- Drop-in equivalence test in the pinned axis
- Extras surface documented, including all seven known limitations

---

### N. Front-door API — ✅ IMPLEMENTED

**Status.** Built. `vectorbtx/frontend/` — `prepare.py` (the single resolution pipeline),
`backtest.py` (`PortfolioBacktest`), `single.py` (`SingleBacktest`). Exported on `vbt.x`.
All three strategy examples are rewritten against it and run end to end; the explicit
version of strategy 1 is retained as `examples/strategies/appendix_explicit.py`.

**One design change from the sketch.** Holding costs are applied **after** the simulation
from the realised positions (`apply_holding_costs`), not by replaying orders through
`from_order_func`. The replay cannot express a signal-driven strategy — it has no notion
of entries and exits — so routing every rates-bearing backtest through it silently
discarded the signals. The post-pass works for every strategy type and keeps the only
level-3 dependency off the default path; `Portfolio.from_holding_costs` remains for the
case where in-run cash feedback actually matters.

**Purpose.** v0.1 is correct and exposes its internals as its interface. A single-instrument
spread bet requires a `Universe` of one, a three-call sizing sequence that must happen in
one exact order, manual conversion into "engine units" — a vectorbt implementation detail
no spread bettor should ever type — and an opaque `**universe.resolve(...)` splat. Section N
is a facade over the tested core of C–M. **It adds no new behaviour.**

**Deliverable**
- `vectorbtx/frontend/backtest.py` — `PortfolioBacktest`, the single implementation
- `vectorbtx/frontend/single.py` — `SingleBacktest`, a thin validating forwarder
- `vectorbtx/frontend/prepare.py` — the one resolution pipeline every path funnels through

**Interface sketch**

```python
pf = vbt.x.SingleBacktest(
    price, entries=entries, exits=exits, high=high, low=low,
    mode="spreadbet", asset_class="fx", point_size=0.0001,
    spread_points=0.9, slippage=0.0002,
    risk_pct=0.01, stop_points=60, target_points=120, leverage=30,
    financing="auto", rates=rates,
    init_cash=10_000, freq="D",
)

pf = vbt.x.PortfolioBacktest(
    price, entries=entries, exits=exits,
    instruments="universe.csv",              # or dict, or Instruments, or flat kwargs
    base_currency="GBP", fx=fx,
    notional=25_000, init_cash=100_000, cash_sharing=True, freq="D",
)
```

Returns a `vectorbtx.Portfolio`, so `pf.stats()`, `pf.trades.stats()`,
`pf.returns_stats()` and every `pf.x.*` extra keep working with no result wrapper to learn.

**Depends on.** All of B–M. **Blocks.** Nothing — it is the last layer.

**Notes**
- **`mode` is the dispatcher.** It decides which other arguments are legal, required or
  rejected, and it is always present — shared across columns or given per column. Absent,
  the call is plain vectorbt passthrough, in which case the user should use plain vectorbt.
- **Four instrument supply forms**, all normalising to a `Universe` (§C2): flat kwargs
  across every column; a dict of per-symbol overrides on a shared base; a CSV/Excel path;
  or `Instrument` objects. **`Instrument` becomes optional** — a reuse mechanism rather
  than compulsory scaffolding.
- **Strategy type is implied by the arguments**: `entries`/`exits` → signal-driven,
  `weights`/`schedule` → rebalanced, neither → hold, both → error naming the conflict.
- **Sizing is flat, and exactly one may be given**: `risk_pct`, `risk_amount`, `stake`,
  `notional`, or raw `size`. Two given → error naming both. `risk_pct` and `risk_amount`
  are separate arguments rather than one `risk=` disambiguated by magnitude — a guess
  dressed as a feature is what D8's coercion rules exist to prevent.
- **Stops in points**, `stop_points` / `target_points` / `trail_points`, converted
  internally to the native price fractions (§H). Native `sl_stop` etc. still accepted and
  take precedence.
- **Two names, one implementation.** `SingleBacktest` validates the price is 1-D, rejects
  the portfolio-only arguments with a message pointing at `PortfolioBacktest`, and
  forwards. No divergence risk, and each name shows only the arguments that apply.
- **Unrecognised arguments pass through** to the underlying vectorbt constructor, so
  nothing upstream becomes unreachable.
- ⚠️ **A collision between an explicit passthrough and a resolved value raises** — `fees=0`
  beside `commission_pct=0.001` silently disables a cost model and produces a backtest that
  looks excellent. Consistent with §C2, where join-key mismatches are fatal rather than
  warnings.
- ⚠️ **`risk_pct` sizes off `init_cash`, not compounding equity.** At level 1 equity is only
  known before the simulation. Material over a long backtest with real drawdowns; document
  at the point of use. True compounding risk sizing needs level 3.
- ⚠️ **FX conversion stays outside the call.** Signals are computed on base-currency prices,
  and the user must hold that series, so `to_base_currency` remains an explicit line in
  multi-currency backtests. Converting internally for sizing while signals ran on native
  prices would put two different price series in one backtest — worse than one visible line.
- **Implementation defect to fix while building:** `to_base_currency` currently accepts
  quote-currency information in three shapes (a dict, a `Universe`, a dict of
  `Instrument`s). One signature should normalise all three.

**Done when**
- The EURUSD spread-bet strategy runs in under 15 lines with no `Universe`, no engine
  units, no manual quantization and no splat
- All four instrument supply forms produce identical `Universe` objects
- Every mode's rejected arguments raise, naming mode and argument
- Two sizing arguments raise, naming both
- A colliding passthrough raises, naming the field
- Plain-vectorbt passthrough is unchanged and still bit-identical
- The three strategy examples are rewritten against it, with one appendix retaining the
  explicit layer so the relationship between the two is visible

---

## Part 8 — Settled defaults reference

Everything below is a chosen default, overridable per instrument.

**Sizing and quantization**

| Mode | min size | increment | notes |
|---|---|---|---|
| Spread bet | £1/point (£0.10 micro) | £0.01 | no contract size; `stake / point_size` |
| CFD | 0.01 lots (some 0.10) | 0.01 lots | `contract_size` sufficient for all |
| FX spot | ~0.1 lots @ £1/pip | broker lot step | 5-decimal quotes, pip at 4th |
| Equity | 1 share | 1 share | `allow_fractional=False` |

**Maximum leverage**

| Mode | default |
|---|---|
| Equity | 2× |
| Direct FX | 2× |
| Spread bet / CFD — stocks | 5× |
| Spread bet / CFD — commodities | 10× |
| Spread bet / CFD — FX | 30× |

Tiered margin is not modelled; users set sensible maxima instead.

**Costs**

| Parameter | Default |
|---|---|
| Broker financing markup | 3% over benchmark, always adverse |
| `fx_markup` | 0.5%, adverse both directions |
| `slippage_max` | 0.005 (50bp) |
| Auto slippage | `coef × σ × sqrt(size/ADV)`, σ = 20-bar realised |
| Spread-bet commission | 0 (cost in spread) |
| Rounding direction | floor |

**Accrual**

| Parameter | Default |
|---|---|
| Accrual days | trading days only |
| Triple day | Wednesday, 3× |
| End of day | 22:00 UK (inert on daily bars) |
| Carry rate | tom-next |
| Day count | ACT/360, all currencies |

**Rates**

| Currency | Series |
|---|---|
| GBP | SONIA |
| USD | SOFR |
| EUR | ESTR |
| JPY | TONA |
| CHF | SARON |

**Data and construction**

| Parameter | Default |
|---|---|
| `price_field` — equity | `adj_close` |
| `price_field` — spread bet / CFD | `close` |
| FX rate convention | close-to-close at bar close |
| Currency conversion | explicit, warn on mismatch |
| Threshold rebalancing | relative drift, per-asset |
| Cap weights | recomputed each rebalance |
| Minimum-trade filter | available, off |
| Off-schedule targets | NaN — never forward-filled |
| Rebalance call sequence | `call_seq="auto"` (sells fund buys) |
| Size type for rebalancing | `TargetPercent` (never `Percent`) |
| Weights summing < 1 | remainder held in cash |
| Weights summing > 1 | routed through the leverage budget |

---

## Part 9 — Remaining work

> ⚠️ **Superseded.** The paragraph below describes the v0.1 build and is kept for the
> record. Current status: **245 tests pass** against a real vectorbt 1.1.0 install,
> **60 of 63** criteria are verified, and **Section N is built** (`vectorbtx/frontend/`).
> `audit_plan.py` has been deleted; see [`AUDIT.md`](AUDIT.md) for the live picture.

**Status as of the v0.1 build.** Sections A–M are implemented. 168 tests pass. 59 of the
63 "Done when" criteria in this document are verified by execution (`audit_plan.py`);
the remaining 4 need a running simulation and are listed below. Section N is designed but
unbuilt.

### Incomplete

1. **✅ Section N — the front-door API.** Built and exercised; see Section N above.
2. **⚠️ Four criteria unverified**, because they need a simulation and the build
   environment had no vectorbt installed. Run these first on a machine that does:
   - §B — `import vectorbtx` emits no warnings
   - §J — `from_weights` reproduces a hand-computed monthly rebalance
   - §J — a cross-currency portfolio rebalances to correct base-currency weights
   - §K2 — Balance and Equity drawdown differ on a portfolio with open positions
3. **⚠️ Nothing has been executed against real vectorbt.** The pure arithmetic — roughly
   80% of the library — is genuinely tested. The patching layer, the `Portfolio` subclass
   and the level-3 callback are written against the audited 1.1.0 source but have never
   run. `holding/callback.py` is the most likely first failure and is deliberately thin
   for that reason.
4. **⚠️ CI has never run.** `.github/workflows/ci.yml` defines both axes; neither has
   executed.
5. **Implementation defect** — `to_base_currency` accepts quote-currency information in
   three different shapes. Unify the signature (see Section N).

### Known cost-model limitations (from `docs/dev/AUDIT-G.md`)

- **Taxes are out of scope, by decision.** UK stamp duty (0.5% on share purchases), the
  PTM levy, CGT and the spread-bet exemptions are **not** modelled and will not be. This
  is deliberate, not an oversight: tax treatment is jurisdiction- and
  wrapper-specific, changes with the user rather than the instrument, and belongs in
  post-hoc analysis of a backtest rather than inside its cost model.
  ⚠️ **Consequence to document for users:** the cross-mode comparison (§H) is therefore a
  pre-tax comparison. In the UK a spread bet and the equivalent share position differ by
  0.5% per entry before any P&L is considered, so a user comparing modes must apply that
  themselves.
- **Short borrow charge: direct shorts only, from a user-supplied rate** — G6, resolved
  after v1. It follows how the instrument is traded: a direct (`equity`) short borrows the
  security and pays `Instrument.borrow_rate`; a CFD or spread-bet short pays its funding
  (`benchmark − markup`) and nothing more. The rate is not freely available, so there is
  no default, and a direct short held with none set warns and is charged nothing. A single
  rate per instrument; its variation over time is not modelled.
- **Futures and forwards** carry no overnight funding at all — G7. No dated mode exists
  yet; whoever adds one must give it a stack with neither financing nor carry.
- **Tiered margin is not modelled** (unchanged from §F) — users set a sensible maximum.
- **Maintenance margin, margin calls and liquidation are not modelled** — D2/F1b. A
  position can draw its whole margin loan and sit there; `min_margin_level` reports how
  close it came.

### Deliberately deferred, recorded so they aren't rediscovered

- **F2 (margin calls and liquidation)** — deferred by D2. If built: worst position first,
  then cascade.
- **Dividends and corporate actions for nominal-point modes** — D10. Future fix is a cash
  adjustment in Section G's callback, where the machinery already exists.
- **Exact `portfolio_level` leverage for signal-driven strategies** — needs level 3; the
  Section J route covers weight-matrix strategies exactly.
- **Asymmetric bid/ask spread** — D3; would require a custom order function for every mode.
- **Compounding-equity risk sizing** — Section N sizes off `init_cash`; true compounding
  needs level 3.

### Rejected during the build, recorded so they aren't reintroduced

These were proposed, found to be outside the plan, and removed:

- `max_positions`, `max_exposure` — new capability, and `max_exposure` duplicated
  `leverage_type="portfolio_level"` (§F).
- Per-cost booleans (`carry=True`) — a second vocabulary for `active_costs` (§C).
- `Risk`/`Stake`/`Notional` sizer classes — flat arguments do the same job.
- An `Account` object — subsumed by the backtest call.
- `inverse_vol_weight` — §J specifies exactly three schemes: equal, price, cap.
- `max_consecutive_count` — streak counts are native (§K2, "streak *count* is native, the
  monetary total isn't").
