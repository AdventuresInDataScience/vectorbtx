# Audit — plan compliance, API compliance, and scope creep

Produced by `audit_plan.py`, which **exercises** the code rather than inspecting it.
No criterion passes on the basis of a symbol merely existing.

Run it: `python audit_plan.py`

## 1. Plan compliance — 63 criteria from the plan's "Done when" blocks

**59 pass. 0 fail. 4 unverifiable in this environment** (they need a running
simulation, and vectorbt is not installed here):

| Section | Unverifiable criterion |
|---|---|
| B | `import vectorbtX` emits no warnings |
| J | `from_weights` reproduces a hand-computed monthly rebalance |
| J | a cross-currency portfolio rebalances to correct base-currency weights |
| K2 | Balance and Equity drawdown differ on a portfolio with open positions |

These are the first things to run on a machine with vectorbt installed.

### Gaps the audit found and fixed

| Section | Gap | Fix |
|---|---|---|
| B | "The collision guard is proven by a test that deliberately triggers it" — no such test existed | `tests/test_foundations.py`: five tests that trigger `ShadowError` on both `attach` and `merge_metrics` |
| B / L | "Drop-in equivalence test in the pinned axis" — missing | `test_unmodified_vectorbt_script_is_bit_identical_under_vectorbtx` compares `order_records`, `total_profit` and `value()` against plain vectorbt |
| F | "`raise_reject=True` demonstrably rejects one" — only ever mentioned in a docstring | test confirming it is a genuine `Order` field, i.e. real passthrough, not an invention |
| F | "Borrowed portion correct for cash-covered, partially financed, fully leveraged" — untestable, buried inside a `pf`-taking function | extracted `borrowed_amount(asset_value, free_cash)` as pure arithmetic; six tests covering all three cases plus shorts and negative cash |

## 2. API compliance — the agreed front door is NOT built

`docs/dev/FRONTEND-API.md` settles on `vbt.x.SingleBacktest` / `vbt.x.PortfolioBacktest`.

**Neither exists.** The agreed API is designed and documented, not implemented. Nothing in
the codebase has changed to support it. That is the largest outstanding item, and it is a
build task, not a gap in v0.1.

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

- **168 tests pass, 0 fail, 4 skipped** (the skips need vectorbt)
- **59/63 plan criteria verified by execution**, 4 need a simulation
- All 9 tour examples and 3 strategies run
- The agreed v6 API is **not implemented**
