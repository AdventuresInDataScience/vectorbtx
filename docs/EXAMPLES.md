# Examples guide

Three kinds of example.

- **[`examples/strategies/`](../examples/strategies/)** — complete, copy-pasteable
  strategies, written against the front-door API (`vbt.x.SingleBacktest`,
  `vbt.x.PortfolioBacktest`). Start here if you want to *use* vectorbtX.
  `appendix_explicit.py` is strategy 1 rebuilt from the components, kept so the
  relationship between the two layers is visible — it is about three times the length for
  the same result.
- **[`examples/cases/`](../examples/cases/)** — a case matrix, A–L: every combination of
  instrument mode, single vs portfolio, shared vs separate cash, and shared vs separate
  strategy. Find the row that matches what you are building and copy it.
- **[`examples/`](../examples/)** — a numbered reference tour, one design decision per
  file. Read these when something surprises you and you want to know why it works that
  way.

---

## Strategies

Real data via `vbt.YFData.download`, real rates via `vbt.x.FREDData`, and a full metrics
report at the end of each.

```bash
pip install vectorbtX
export FRED_API_KEY=...        # free from fred.stlouisfed.org; optional
cd examples/strategies && python 01_ma_cross_spreadbet_fx.py
```

| File | Strategy | Shows off |
|---|---|---|
| `01_ma_cross_spreadbet_fx.py` | 50/200 MA cross on EURUSD, as a UK spread bet | £/point staking, risk-based sizing, point stops → native SL/TP, spread cost, carry, FRED rates |
| `02_leveraged_stock_portfolio.py` | Top-3 momentum across 5 stocks, 2 currencies, 3 cost structures | **Per-instrument arguments**, cross-currency conversion, fractional shares, leverage caps, per-column results |
| `03_monthly_rebalanced_investing.py` | Global 60/40 ETF portfolio, monthly with a drift filter | `from_weights`, composed schedules, minimum-trade filter, benchmark conversion, tracking drift |

### The pattern all three follow

```python
import vectorbtx as vbt

universe = vbt.x.load_universe({...})               # 1. describe instruments once
data     = vbt.YFData.download(symbols, ...)        # 2. get data
rates    = vbt.x.FREDData.download_rates([...])     # 3. real benchmark rates
close    = vbt.x.to_base_currency(...)              # 4. convert foreign names
size     = ...                                      # 5. size in YOUR units
pf = vbt.Portfolio.from_signals(                    # 6. simulate
    close, entries, exits, size=size,
    **universe.resolve(close, rates=rates, equity=CASH),
)
pf.stats(); pf.trades.stats(); pf.returns_stats()   # 7. analyse
```

`universe.resolve()` is the load-bearing line. It turns instrument descriptions into
per-column `Order` arguments — `size_granularity`, `min_size`, `max_size`, `fees`,
`fixed_fees`, `slippage`, `lock_cash` — in the wrapper's column order, so five
instruments with five different cost structures need no parallel lists to keep aligned.

### Things worth copying

- **Sizing lives in your units, not the engine's.** Compute a £/point stake or a lot
  count, quantize it, then convert once with `get_adapter(spec).to_engine_units(...)`.
- **Stops are point distances converted to fractions.** `points_to_price_fraction` feeds
  vectorbt's native `sl_stop`/`tp_stop`; no stop logic is reimplemented.
- **Pass the schedule to `drifted_weights`.** Drift must be measured since the last
  rebalance. Without an anchor it accumulates from bar zero and any threshold test
  degenerates to "always true".
- **Check `pf.x.drift()`.** Quantization and suppressed minimum trades are real costs;
  this is where they surface instead of vanishing into tracking error.

### Running them without a network

`strategies/_harness.py` executes each file against deterministic synthetic prices and a
static rate table, substituting stubs for Yahoo and FRED. It verifies everything up to
the `Portfolio.from_*` call — sizing, quantization, FX conversion, schedule construction,
argument resolution — and prints the exact arguments the engine would receive.

```bash
python _harness.py                                  # all three
python _harness.py 02_leveraged_stock_portfolio.py  # just one
```

That is how the strategies were validated in this repo, so treat harness output as
evidence for the *logic* and a live run as evidence for the simulation.

---

## Reference tour

Nine files, ~90 worked sections. Every number printed is computed, not quoted. They run
with or without vectorbt installed — without it the pure arithmetic still executes, and
simulation-dependent sections say they were skipped.

```bash
cd examples && python 09_end_to_end.py
```

| File | Section | Covers |
|---|---|---|
| `01_instruments.py` | C | All four modes, `.filled()`, `price_field` by mode, every validation rule that fires |
| `02_universe.py` | C2 | dict and CSV front doors, column order, fatal join mismatches, strict coercion, blank ≠ NaN, typo'd keywords |
| `03_modes_quantize.py` | H, D | The cross-mode equivalence test, round-tripping, fractional pips, point stops → price fractions, risk sizing, floor/ceil/nearest, sub-minimum rejection |
| `04_multicurrency.py` | I | Rate tables, derived reciprocals, the conversion identity verified to 0.00e+00, OHLC ordering, decomposition, adverse markup both ways, foreign fixed costs |
| `05_costs_margin.py` | E, F | Default stacks, cost→field mapping, minimum commission, spread halving, auto slippage, slippage summation, notional/margin/borrowed, `max_size` |
| `06_construction.py` | J | Weighting schemes, calendar and threshold schedules, composition with `&`, NaN off-schedule, leverage budget, quantize-before-emit |
| `07_holding_rates.py` | G, M | Adverse markup, borrowed-only financing, ACT/360, day multipliers, triple day, carry vs financing, a hand-checked week |
| `08_metrics.py` | K1, K2 | All seven returns gaps derived from first principles, K-ratio length-independence, balance vs equity drawdown, streaks, Z-score |
| `09_end_to_end.py` | all | The complete runtime flow: universe → rates → FX conversion → targets → resolve → simulate → analyse |

### Highlights

**The cross-mode equivalence test** (`03`) is the most valuable single check in the
project. A £1/point spread bet on a pence-quoted stock, 100 CFD lots and 100 shares all
produce engine size 100 and P&L 500.00 on a 5-point move. The "100 contract size" is
derived from `stake / point_size`, never configured.

**The FX identity** (`04`) is verified numerically to a difference of `0.00e+00`, using a
period where price and FX move in *opposite* directions — same-direction moves can hide a
sign error.

**Financing across a week** (`07`) reproduces the hand-computed
`40000 × 0.0825/360 × 7 = −64.1667` for the long and `× 0.0225/360 × 7 = +17.50` for the
short, including the Wednesday 3× multiplier. The asymmetry is the point: the markup is
adverse on both sides.

**K-ratio length-independence** (`08`) is shown by tiling the same returns 2×, 4×, 8× and
16×: v1 climbs 0.0093 → 0.0360 while v3 settles at ~0.0127.


---

## Verification

Every file here is **executed** as part of development, against real vectorbt, not merely
kept up to date by hand:

```bash
for f in examples/0*.py; do python "$f"; done      # the numbered tour
python examples/cases/_harness.py                   # the case matrix, A-L
python examples/strategies/_harness.py              # the full strategies
```

Each harness stubs only the two things that need a network — Yahoo prices and FRED rates —
and uses the real `Portfolio`, the real front door and the real cost, margin and
quantization resolution. Everything else in the files is what you would write for real, so
they can be copied out and pointed at live data unchanged.

Running them in CI is worth the cost: the bugs they have caught are listed in
[`docs/dev/CORRECTIONS.md`](dev/CORRECTIONS.md), and every one was a silent wrong answer
that the test suite passed.

Every example passes the four price series by keyword — `close=`, `open=`, `high=`, `low=` —
because all four are required and positional order would be too easy to transpose.

See [`docs/DATA.md`](DATA.md) for where prices, rates and FX come from, how to keep a FRED
key in a `.env` file, and how to load prices and universes from your own files.
