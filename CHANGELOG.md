# Changelog

## Unreleased

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
