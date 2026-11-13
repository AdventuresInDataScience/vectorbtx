"""Section F — exposure.

Because F2 (maintenance margin, margin calls, forced liquidation) is deferred by D2, the
borrowed-versus-cash-covered split **must be derivable post-simulation**. It is read off
`pf.cash()`: an account borrows exactly what its cash is overdrawn by. Section G consumes
it, so this is a hard requirement rather than a convenience.

Built on vectorbt's existing collateral machinery: `ExecuteOrderState` carries
`(cash, position, debt, free_cash)`, `free_cash` is collateral-aware, and `gross_exposure`
already divides by free cash.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def borrowed_amount(asset_value, free_cash):
    """How much of a position would be borrowed, **before** it is bought, as arithmetic.

    `borrowed = max(0, |asset_value| - max(0, free_cash))`: the part of a purchase the
    cash on hand does not cover. The three cases the plan requires — cash-covered,
    partially financed, fully leveraged — are testable without running a simulation.

    ⚠️ Pre-trade only. After a simulation the cash has already paid for the position, so
    this would count a fully paid £8,000 holding bought from £10,000 as £6,000 borrowed
    (£8,000 less the £2,000 left over). `borrowed_portion`, which reads negative cash, is
    the post-simulation answer, and does not use this.

    Args:
        asset_value: market value of the position(s) to be bought.
        free_cash: cash available to pay for it.

    Returns:
        Borrowed notional, same shape as `asset_value`.

    Examples:
        >>> borrowed_amount(10_000, 10_000)     # fully covered
        0.0
        >>> borrowed_amount(50_000, 10_000)     # 5x leveraged
        40000.0
    """
    av = np.abs(np.asarray(asset_value, dtype=float))
    cash = np.clip(np.asarray(free_cash, dtype=float), 0.0, None)
    return np.clip(av - np.minimum(av, cash), 0.0, None)


def split_by_long_value(debit, long_value, groups):
    """Share each account's borrowing between its instruments, as pure arithmetic.

    Each instrument's share is its long market value over the account's total long market
    value, bar by bar, so the shares add up to the account's debit exactly. Shorts and flat
    instruments get none: a short sale adds cash, which has already reduced the debit. If
    the account owes money but holds no longs, which only happens once losses have
    exceeded the account, the debit is split equally between its instruments.

    Split out from `borrowed_portion` so the split is testable without a simulation.

    Args:
        debit: `(n_bars, n_accounts)` borrowing per account, from negative cash.
        long_value: `(n_bars, n_columns)` market value per instrument, signed; negative
            (short) values count as zero. NaN, an instrument not yet priced, counts as
            holding nothing.
        groups: the account index of each column, `(n_columns,)`.

    Returns:
        `(n_bars, n_columns)` borrowing per instrument.

    Examples:
        >>> split_by_long_value([[6_000.0]], [[12_000.0, 4_000.0]], [0, 0])
        array([[4500., 1500.]])
    """
    debit = np.asarray(debit, dtype=float).reshape(np.shape(debit)[0], -1)
    held = np.nan_to_num(np.clip(np.asarray(long_value, dtype=float), 0.0, None), nan=0.0)
    held = held.reshape(debit.shape[0], -1)
    groups = np.asarray(groups)
    out = np.zeros_like(held)
    for g in range(debit.shape[1]):
        members = np.flatnonzero(groups == g)
        if len(members) == 0:
            continue
        total = held[:, members].sum(axis=1)
        safe = np.where(total > 0, total, 1.0)
        for col in members:
            share = np.where(total > 0, held[:, col] / safe, 1.0 / len(members))
            out[:, col] = debit[:, g] * share
    return out


def borrowed_portion(pf, group_by=False) -> pd.DataFrame:
    """Split exposure into the borrowed portion and the cash-covered portion.

    Args:
        pf: a simulated `Portfolio`.
        group_by: grouping passed through to vectorbt. The default, per column, is what
            Section G charges financing on.

    Returns:
        DataFrame of the borrowed notional per bar and column (or per group) — the
        amount financing is charged on. A position fully covered by cash borrows nothing.

    Notes:
        A £10,000 position fully covered by £10,000 cash finances nothing; £50,000 against
        £10,000 finances £40,000. Before a trade that is `borrowed_amount`; after it, it
        is the account's overdraft, as below.

        **Borrowing belongs to an account, so it is worked out per account first.** A
        broker charges interest on an account's debit balance.

        * **Separate accounts** (no cash sharing): every column is its own account. A
          group of them (`group_by=True`) borrows the **sum** of its accounts' borrowing.
          ⚠️ Taking the group's summed cash instead would let one account's spare cash
          pay off another's debt, which no broker does.
        * **A shared account** (cash sharing): one debit for the whole group. Per column
          it is split between the account's instruments by `split_by_long_value`, in
          proportion to their long market value. ⚠️ Reading each column's own cash
          instead, as this did, measures every instrument as if it had the account's
          whole equity to itself: two instruments that together borrowed 6,000 against
          a 10,000 account showed no borrowing at all, and paid no financing. Each
          instrument's share is then charged at its own currency's rate, as a single
          instrument is.

        **Read straight off negative cash, always.** Under funded margin
        (`vectorbtx.margin.buying_power`) reported cash is `equity + cumsum(cash_flow)`,
        so it goes negative by exactly the amount of the broker's loan that has been
        drawn. `max(0, -cash)` is therefore the borrowed notional by construction. Without
        a margin loan a long is bounded by cash, so cash never goes negative and nothing
        is borrowed, which is also true. A short sale adds cash, so it borrows none.

        ⚠️ The asset-value reading is **not** interchangeable with it. Buying £18,260 of
        asset against £10,000 of equity borrows £8,260, not £18,260. Without a loan this
        used to fall back to `borrowed_amount(asset_value, free_cash)` measured after the
        purchase, which counted a fully paid £8,000 holding bought from £10,000 as £6,000
        borrowed, and a fully invested cash book as borrowed in full. Every rebalanced
        (`weights=`) share portfolio was charged financing on money it never borrowed.

    Examples:
        >>> pf.x.borrowed_portion().iloc[-1]
    """
    grouper = pf.wrapper.grouper
    per_column = not grouper.is_grouped(group_by=group_by)

    if pf.cash_sharing and grouper.is_grouped():
        # One debit per shared account. vectorbt does not let a cash-sharing portfolio
        # be regrouped, so a grouped request is that same grouping.
        accounts = _account_borrowing(pf, group_by=None if per_column else group_by)
        if not per_column:
            return accounts
        long_value = pd.DataFrame(pf.asset_value(group_by=False))
        return pd.DataFrame(
            split_by_long_value(accounts.to_numpy(), long_value.to_numpy(), grouper.get_groups()),
            index=long_value.index,
            columns=long_value.columns,
        )

    # Separate accounts: one per column, summed for a group.
    columns = _account_borrowing(pf, group_by=False)
    if per_column:
        return columns
    groups = np.asarray(grouper.get_groups(group_by=group_by))
    values = columns.to_numpy(dtype=float)
    summed = np.zeros((values.shape[0], groups.max() + 1))
    for col, g in enumerate(groups):
        summed[:, g] += values[:, col]
    return pd.DataFrame(pf.wrapper.wrap(summed, group_by=group_by))


def _account_borrowing(pf, group_by):
    """Borrowing per account (`group_by` makes each reported column one): its overdraft."""
    cash = pd.DataFrame(pf.cash(group_by=group_by))
    return (-cash).clip(lower=0.0)

def margin_level(pf, universe, group_by=False) -> pd.DataFrame:
    """Equity / margin-required ratio per bar, as a percentage (MT5 convention).

    Args:
        pf: a simulated `Portfolio`.
        universe: the `Universe`, supplying per-instrument leverage.

    Returns:
        DataFrame of margin levels in percent. Infinite where no margin is used.
    """
    from .requirement import notional as calc_notional

    # ⚠️ Required, not optional. Without this guard a missing universe surfaced as
    # `TypeError: argument of type 'NoneType' is not iterable` from the `col in universe`
    # test below — which names neither the argument nor what to do about it.
    if universe is None:
        raise ValueError(
            "margin_level needs a Universe to read per-instrument leverage from, and "
            "none was given or attached to this portfolio. Portfolios built through "
            "vbt.x.PortfolioBacktest or Portfolio.from_weights carry theirs "
            "automatically; one built by calling from_signals/from_orders with "
            "**universe.resolve(...) does not, because only the resolved Order kwargs "
            "were passed on and the Universe itself was not. Pass it explicitly, e.g. "
            "pf.x.min_margin_level(universe=universe)."
        )

    asset_value = pd.DataFrame(pf.asset_value(group_by=group_by))
    value = pd.DataFrame(pf.value(group_by=group_by))

    required = pd.DataFrame(0.0, index=asset_value.index, columns=asset_value.columns)
    for col in asset_value.columns:
        spec = universe[col].filled() if col in universe else None
        lev = (spec.leverage if spec is not None else 1.0) or 1.0
        required[col] = asset_value[col].abs() / lev

    total = required.sum(axis=1)
    equity = value.sum(axis=1) if value.shape[1] > 1 else value.iloc[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        level = np.where(total > 0, equity.to_numpy() / total.to_numpy() * 100.0, np.inf)
    return pd.Series(level, index=asset_value.index, name="margin_level")


def min_margin_level(pf, universe, group_by=False) -> float:
    """Lowest equity/margin ratio over the backtest — MT5's *Margin Level Min*.

    Cheap here, and it satisfies one of K2's gaps without needing F2.
    """
    level = margin_level(pf, universe, group_by=group_by)
    finite = level[np.isfinite(level)]
    return float(finite.min()) if len(finite) else np.inf
