# Corrections found by executing the examples and cases

A build log, kept because each entry was a **silent-wrong-answer** bug that a passing test
suite did not catch. Running the examples was the thing that caught them, which is the
argument for keeping `examples/cases/_harness.py` and `examples/strategies/_harness.py` in
CI rather than treating the examples as documentation that happens to be code.

Recorded because each was a silent-wrong-answer bug that a passing test suite did not
catch — the examples were the thing that caught them.

| Bug | Symptom | Fix |
|---|---|---|
| `size_for_notional` / `size_for_risk` dropped the pandas index | `close * size` silently reindexed to a `2n`-row union; failure surfaced much later, inside the commission module | `_like()` re-wraps results in the price's container; regression tests assert index preservation and that sizes multiply without reindexing |
| `allow_fractional=True` had no effect | Fill increment stayed at 1 whole share; the flag was decorative | Drops the increment to `FRACTIONAL_INCREMENT` (`1e-6`) unless `increment` is set explicitly |
| `drifted_weights` measured drift from bar zero | Over a long backtest drift grows without bound, so `calendar & threshold` silently collapsed to plain calendar rebalancing | `schedule=` anchors drift to the last rebalance |
| ...and then the anchor included the current bar | Drift became exactly zero on every rebalance bar, so the composed schedule could never fire at all — the opposite failure | Anchor is the last rebalance *strictly before* each bar |
| `resolve_universe` rejected `base_currency` / `notional` | Foreign fixed-cost conversion was unreachable through the public API | Threaded through `Universe.resolve` |
| `FXTable.from_long` stored only supplied pairs | `GBP→USD` raised even with `USD→GBP` present | Derives the full grid; `value(A,B) × value(B,A) == 1` exactly |
| `validate_weights` ignored gross > 1 | Implicit leverage entered through the back door unwarned | Warns, pointing at the `portfolio_level` budget |
| `inverse_vol_weight` returned 0 during warm-up | Silently flattened the portfolio for the first `window` bars | Returns NaN — "no estimate yet" is not "hold nothing" |
| `calendar_schedule` broke on `"QE"` / `"YE"` | pandas renamed offset aliases; `Period` kept the old ones | Alias map accepts both |
| `static_rate_table` rejected a per-currency dict in `rate` | Only worked via `currencies` | Accepts either; raises naming any currency without a rate |

| Rebalanced weights were a fraction of buying power | A 60/40 book on equity mode's *default* 2x leverage silently ran at 120/80 | The rebalancing path is funded at equity, never buying power; weights above 1 warn |
| Orders filled at the signal bar's close | Lookahead: captured the overnight gap and skipped a night of financing | `open` is required; signals and schedules shift one bar; `price=open` |
| Holding costs discarded signal arguments | Any backtest with `rates=` routed to the order-replay path, which has no notion of entries and exits | Costs are computed post-simulation from realised positions |
| FX spread bets could never accrue carry | `base_currency` was rejected outside `fx_spot`, so the pair could not be named | Allowed for any FX underlying, with a warning when the pair is missing |
| Trade metrics needing `init_cash` crashed `pf.trades.stats()` | No back-reference from `Trades` to the portfolio | `get_trades` tags the result |
| Per-column lookups ran off group-length arrays under cash sharing | `IndexError` on the second column | Group-aware fallbacks in charge building and trade metrics |
| `high`/`low` were optional | Stops could only be checked against the close, so they fired a bar late or never; drawdown missed every intrabar excursion. Both flatter the result | Both are required arguments; `pf.x.intrabar_drawdown()` measures the worst moment against them |
| FX `point_size` had to be stated per pair | Using 0.0001 on a JPY pair made every stop 100x too tight, silently, and in the direction that produces more trades | Derived from the quote currency; contradicting the convention warns |
| Price series were positional | Four same-shaped frames in a row; transposing two ran cleanly and reported a wrong number | `close=`, `open=`, `high=`, `low=` are keyword-only |
| `from_weights` re-resolved the universe independently | Discarded the estimated order notional; failed on any minimum commission | Accepts a pre-resolved `order_kwargs`; estimates notional from weights |
