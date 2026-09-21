# Audit — plan compliance, API compliance, and scope creep

Was produced by `audit_plan.py`, which **exercised** the code rather than inspecting it.
No criterion passed on the basis of a symbol merely existing.

> ⚠️ **`audit_plan.py` has been deleted** as a sandbox-era build tool. This file is now the
> record rather than a regenerable report. Its final run was against a real vectorbt 1.1.0
> install; the numbers below are from that run. To revive it, recover it from the initial
> commit — but prefer promoting the four open criteria below into `tests/` instead.

## 1. Plan compliance — 63 criteria from the plan's "Done when" blocks

**63 pass. 0 fail. 0 outstanding.**

Four were long carried as `CANNOT VERIFY` — hardcoded stubs written when vectorbt was
unavailable, never filled in once it was. All four are now covered by real tests:

| Section | Criterion | Covered by |
|---|---|---|
| B | `import vectorbtx` emits no warnings | verified directly — 0 warnings against vectorbt 1.1.0 |
| J | `from_weights` reproduces a hand-computed monthly rebalance | `tests/test_from_weights.py` |
| J | a cross-currency portfolio rebalances to correct base-currency weights | `tests/test_from_weights.py` |
| K2 | Balance and Equity drawdown differ on a portfolio with open positions | `tests/test_metrics_trades.py` |

Until `tests/test_from_weights.py` existed, **`Portfolio.from_weights` had no direct test
at all** — Section J was exercised only indirectly, through `vbt.x.PortfolioBacktest`.

### ⚠️ Writing the K2 test found a real bug

The criterion was there to catch a specific silent-wrong-number failure, and it did.
`_per_col` in `metrics/trades/register.py` read `self.values["pnl"]` for **every** trade
record, open ones included. vectorbt marks an open position's `pnl` to the last price, so
floating P&L was flowing into `balance_curve` — a function whose entire purpose is to be
the *closed-only* counterpart to vectorbt's mark-to-market `value()`.

The effect: on a portfolio holding an open position, Balance and Equity drawdown
converged, which is exactly the distinction MT4/MT5 parity exists to preserve. On the test
case (one position, never closed, priced 100 → 60 → 90) Balance drawdown reported 0.10
where it should report 0. `_per_col` now filters to `TradeStatus.Closed`, matching both
MT's semantics and `balance_curve`'s own docstring.

### Gaps the audit found and fixed

| Section | Gap | Fix |
|---|---|---|
| B | "The collision guard is proven by a test that deliberately triggers it" — no such test existed | `tests/test_foundations.py`: five tests that trigger `ShadowError` on both `attach` and `merge_metrics` |
| B / L | "Drop-in equivalence test in the pinned axis" — missing | `test_unmodified_vectorbt_script_is_bit_identical_under_vectorbtx` compares `order_records`, `total_profit` and `value()` against plain vectorbt |
| F | "`raise_reject=True` demonstrably rejects one" — only ever mentioned in a docstring | test confirming it is a genuine `Order` field, i.e. real passthrough, not an invention |
| F | "Borrowed portion correct for cash-covered, partially financed, fully leveraged" — untestable, buried inside a `pf`-taking function | extracted `borrowed_amount(asset_value, free_cash)` as pure arithmetic; six tests covering all three cases plus shorts and negative cash |

## 2. API compliance — the agreed front door IS built

`docs/dev/FRONTEND-API.md` settles on `vbt.x.SingleBacktest` / `vbt.x.PortfolioBacktest`.

**Both now exist** (`vectorbtx/frontend/`) and are covered by `tests/test_frontend.py`.
This section previously recorded them as unimplemented; that was true of v0.1 and is no
longer.

## 3. Scope creep found and removed

Symbols present in the code but traceable to no plan requirement:

| Removed | Why |
|---|---|
| `inverse_vol_weight` | Plan §J specifies exactly three schemes: "equal, price, cap". This was a fourth. |
| `max_consecutive_count` | Plan line 1617 lists streak **counts** as native (`winning_streak.max()`), and line 1633 states only the *monetary* total is a gap. This rebuilt native functionality. |
| `rolling_ulcer_performance_index_nb` | Dead code: never called, exported or tested. |

Removed earlier in the design rounds, before implementation: `max_positions`,
`max_exposure` (duplicated `leverage_type="portfolio_level"`), per-cost booleans
(`carry=True`, duplicating `active_costs`), `Risk`/`Stake`/`Notional` sizer classes, the
`Account` object, and `bt.trade()` verbs.

### Judgement calls kept, with reasons

| Kept | Reason |
|---|---|
| `equivalent_flat` | Plan §E specifies `equivalent_pct`; this is its inverse. Two lines, tested, symmetric. |
| `always_schedule` | The `schedule=None` default has to be *something*. |
| `borrowed_amount` | Extraction of existing logic to satisfy a plan criterion, not new capability. |
| `rebalance_drift` | Plan §D: "J's minimum-trade filter routes its suppressed-trade drift through here." |
| `trading_day_count`, `cap_user_size` | Small helpers over planned functions; internally used. |
| `FRACTIONAL_INCREMENT` | Needed to make the planned `allow_fractional` field actually change behaviour. |

### Dead-code sweep

1 dead symbol of 171 public symbols, now removed. The remaining 170 are each used
internally, exported through `vbt.x`, or covered by a test.

## 4. Current state

Measured against a real vectorbt 1.1.0 install (numba 0.67.0, numpy 2.5.3, pandas 3.0.6,
plotly 6.9.0, Python 3.12):

- **256 tests pass, 0 fail, 0 skipped** (217 passed / 39 skipped with vectorbt absent)
- **63/63 plan criteria verified**
- All 9 numbered examples and all 12 case files run; all 4 strategies run both through the
  harness **and against live Yahoo + live FRED data**
- All three numba kernels in `holding/callback.py` compile and pass type inference
- The agreed v6 API **is implemented**
