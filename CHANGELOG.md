# Changelog

## Unreleased

### ⚠️ Changes results

- **Holding costs reach `total_profit`, `final_value`, `total_return` and `pf.stats()`**
  (OpenStrategy Studio issue 15). Carry and financing were folded into `cash_flow`, so
  `value()`, `returns()`, `returns_stats()`, Sharpe and drawdown included them. But vectorbt
  computes `total_profit` from the order records, not from `cash_flow`, so `final_value`,
  `total_return` and the End Value and Total Return [%] rows of `pf.stats()` reported the
  result before holding costs, in the same tear sheet. `Portfolio.total_profit` now adds
  the charges, per column and summed per group, so `final_value()` equals `value()[-1]`.
  Any backtest with `rates=` reports a lower End Value and Total Return than before (or a
  higher one, where carry is a net credit). Trade P&L (`pf.trades.pnl`) and the MT5 trade
  metrics built from it (Total Net Profit, Gross Profit and Loss, Recovery Factor, the
  balance drawdowns) stay before holding costs, by design: a charge accrues per bar, not
  per trade. `pf.holding_costs` holds the difference.
- **Holding costs in a shared account are charged per instrument, correctly**
  (OpenStrategy Studio issue 16). A shared account is `cash="shared"`, or
  `cash_sharing=True` with a group. Two faults, in opposite directions:
  - *Carry and full-notional financing were over-charged.* They read position values from
    `pf.asset_value()`, which a shared account reports once for the whole account, and
    that one series was given to every instrument in it. An account of N instruments paid
    up to N times its carry, and an instrument holding nothing paid as much as the others.
    Each instrument is now charged on its own position, so a shared account pays exactly
    what separate accounts placing the same orders pay. Both `PortfolioBacktest` and
    `Portfolio.from_holding_costs` are fixed.
  - *Margin share-dealing financing (`mode="equity"`) was under-charged.* Each
    instrument's borrowing was read as if it had the account's whole equity to itself, so
    two instruments that together borrowed 6,000 against a 10,000 account were charged
    nothing. `borrowed_portion` now takes the account's debit and splits it between its
    instruments in proportion to their long market value (shorts take none; an account
    that owes money while holding no longs splits it equally). Each share is charged at
    that instrument's own currency's rate, as a single instrument is. The account pays
    interest on its whole debit.

  Also: `borrowed_portion(group_by=True)` on separate accounts now sums each account's
  borrowing instead of netting their cash, which let one account's spare cash pay off
  another's debt. `build_charges` now raises when an input has no column for an
  instrument, where it used to share out an account's series or charge nothing.
- **Borrowing is an account's overdraft, `max(0, -cash)`, for every portfolio**
  (OpenStrategy Studio issue 17). A portfolio without a margin loan fell back to
  `|asset value| - free cash` measured after the purchase, which counts money already
  spent as borrowed: 8,000 of shares bought from 10,000 showed 6,000 borrowed. Rebalanced
  portfolios (`weights=`) run without a loan, so every fully invested share portfolio with
  `rates=` paid margin financing on its whole book. Portfolios built directly with
  `from_orders` or `from_signals`, and `Portfolio.from_holding_costs`, were affected the
  same way. A short sale adds cash, so it now borrows nothing; on the borrowed basis
  (`mode="equity"`) a short is no longer credited financing, and interest on an overdraft
  is always charged.
- **`Portfolio.from_holding_costs` funds leverage like `PortfolioBacktest`.** It handed
  `init_cash` to the engine as its cash, so a margin account could never borrow: a 2x
  position sized beyond equity was part-filled at 1x and margin financing never fired.
  `init_cash` is now the account's equity, the engine gets `equity x leverage`, and the
  loan is recorded so equity and returns are reported on the account's real size. Sizes
  within equity fill as before.
- **Without `rates=`, `Portfolio.from_holding_costs` charges no financing or carry.** With
  no benchmark table every rate read as 0%, so financing came to the broker's markup alone
  and carry to the carry markup alone: invented numbers. `PortfolioBacktest` has always
  required `rates=` for both; the replay path now agrees.

### Added

- **The borrow fee on a direct short** (`docs/dev/AUDIT-G.md`, G6, previously a
  documented gap). It follows how the instrument is traded, not what it is: shorting
  something you trade directly (`mode="equity"`: a share, ETF, fund or commodity product)
  means borrowing it, and the lender charges for that. Set `borrow_rate` on the
  instrument, as an annual fraction (`0.005` is 0.5% a year), and a short position pays it
  daily on its market value, accrued like financing (weekend on the triple day, the
  market's day count). Always a charge. CFDs and spread bets have no borrow fee: their
  short side already pays funding at `benchmark − markup`, so `validate_spec` rejects a
  `borrow_rate` outside `equity`. `borrow` is a new cost module in the `equity` default
  stack. It needs no `rates=`, and works on the replay path too. **There is no default
  rate**, because none is published and a made-up one would look authoritative: a direct
  short held with no `borrow_rate` is charged nothing and **warns**, naming the
  instruments. `borrow_rate=0.0` states that the borrow is free. Also
  `vbt.x.borrow_charge` and `vbt.x.borrow_series`.

### Changed

- **The per-instrument `financing` field is reserved.** Nothing ever read a number in it,
  so a rate typed there looked set and was silently ignored. Financing is, and was, the
  benchmark from `rates=` plus `financing_markup=`, the same for every instrument, which
  is how retail brokers charge it. The field now resolves to `"auto"`, which means exactly
  that, and a number in it warns that it is not used. It stays in place for a possible
  future per-instrument cost structure, the kind professional desks negotiate.

## 9f4cf54

### ⚠️ Changes results

- **Stops are checked on the entry bar with stop-adjustment callbacks too** (OpenStrategy
  Studio issue 14). A custom `adjust_sl_func_nb` or `adjust_tp_func_nb` used to rule out
  split bars, so with one the entry bar's range went unchecked (with a warning). Each
  callback is now wrapped in a bar view: it is called exactly as on whole bars — once per
  bar, at the open, with `c.i`, `c.init_i` and `c.curr_i` as bar indices and its own
  arguments untouched — so existing callbacks work unchanged and the entry bar is checked.
  Only a custom `signal_func_nb`, or templates in a callback's arguments, still fall back
  to whole bars, with the warning.

## 16293a2

Fixes for the issues OpenStrategy Studio's phase 0 found in the front door. Most of them
produced **plausible wrong numbers with no error**, so expect saved results to move.

### ⚠️ Changes results

- **Short signals now shift to the next bar, like long ones.** `short_entries` and
  `short_exits` used to reach vectorbt unshifted, so every short filled at the open of the
  bar whose close produced it (lookahead). They are now named `PortfolioBacktest`
  arguments and shift with `entries`/`exits`. **If you shifted your short signals yourself
  to work around this, stop:** they are now delayed two bars.
- **A flat `base_currency` on an FX instrument is the pair's base currency.**
  `SingleBacktest(asset_class="fx", base_currency="EUR", quote_currency="USD", ...)` used to
  set the *account* currency to EUR, leaving the pair one leg short, so a long EURUSD was
  charged the whole USD rate as carry. It now names the pair, so carry is the EUR/USD
  differential. Use the new `account_currency=` for the account. Off FX, a flat
  `base_currency` still means the account currency.
- **A missing FX currency leg skips carry instead of pricing it at 0%.** Without
  `base_currency`, an FX position was charged (or credited) the whole quote rate. Carry is
  now skipped, as the construction warning always said, with a second warning at run time.
- **`carry_markup` defaults to `settings["vectorbtx"]["carry_markup"]`** (1% unless changed)
  in `PortfolioBacktest` and `simulate_with_holding_costs`. The `0.0` default silently
  overrode the setting. Pass `carry_markup=0.0` to charge none.
- **Point stops are anchored at the fill price** (`stop_entry_price="price"`), not at the
  entry bar's close. A 2-point stop on a buy filled at 100 on a bar closing at 103 sat at
  100.94 and fired the next bar. Trailing stops start from the same anchor.
- **Stops and targets are checked on the bar a position is entered.** vectorbt checks a
  position's stops from the bar *after* entry, so a stop the entry bar traded through was
  missed and the position survived — always in the strategy's favour, worst for tight
  stops on daily bars. With any stop or target active, each bar is now simulated as its
  open (signals fill, gap stops fire) plus the rest of the bar (stops checked against the
  real range), and the orders are mapped back onto the original bars. A bar can therefore
  hold an entry and its stop-out; such a position holds nothing overnight. Also: an exit
  signal at the open now beats an intrabar stop on the same bar, and under shared cash a
  stop that gaps through at the open funds that open's entries while an intrabar stop
  does not. Backtests without stops are unchanged. See `vectorbtx/frontend/split_bars.py`.
- **Stop and target exits pay the spread and slippage** (`stop_exit_price="stopmarket"`).
  vectorbt's default exits stops with no slippage, and the spread lives in `slippage`.
  A take-profit therefore also pays slippage, a small conservative bias. Pass either
  argument to override it.
- **An unsized buy-and-hold is funded at equity, not buying power.** On a 30x spread bet
  it held 30x the account. A hold that cannot then reach the instrument's minimum size
  warns. A sized hold (`stake=`, `size=`, ...) is still funded at buying power.
- **Weights asking for more cash than the account holds are scaled down pro rata**, with
  a warning. Targets of 0.75 / 0.75 used to fill 0.75 / 0.25. The constraint is the net
  long weight without `lock_cash` (a short's proceeds are cash) and the gross weight with it.
- **The FX conversion markup is charged.** `pd.DataFrame(open)` dropped
  `ConvertedFrame.fx_markup`, so no markup was ever applied. It is now read off the price
  frames, charged only on columns that were actually converted, and available as an
  explicit `fx_markup=` argument (which used to raise `TypeError`).
  `to_base_currency(fx_markup=None)` now reads `settings["vectorbtx"]["fx_markup"]`.
- **`pf.x.intrabar_drawdown()` counts shared cash once and brackets shorts by the high.**
  Under cash sharing each column was given the whole pot, which made the figure shallower
  than the close-to-close drawdown; a short was valued as a long, so a spike against it
  read as a gain.

### Added

- **Shared cash with an entry priority.** `cash="per_instrument" | "shared"`,
  `entry_priority=None | [columns] | DataFrame | "random"` (with `seed=`) and
  `when_cash_short="skip" | "partial"`. Under a shared account, closing orders go first on
  each bar, then entries in priority order, then the rest. See `vectorbtx/frontend/cash.py`.
- **`pf["A"]` on a shared account** returns A's contribution to the account (the
  account's equity, A's trades only), so the columns add up to the account exactly. It
  used to raise, because vectorbt indexes a grouped portfolio by group. `pf[["A", "B"]]`
  selects several; a selection spanning two cash-sharing groups raises
  `AccountSelectionError`; a group label still selects the group.
- `pf.x.intrabar_drawdown_series()` (the per-bar curve) and `pf.x.intrabar_value()` (the
  best and worst account value inside each bar).
- `account_currency=` on `PortfolioBacktest` and `SingleBacktest`.

### Fixed

- `pf["column"]` on a levered portfolio no longer crashes every metric with a numba
  `TypingError` (the margin loan was sliced to a 0-d array).
- `holding/accrual.py` uses `DatetimeIndex.day_of_week`; pandas 4 removes `.weekday`.
