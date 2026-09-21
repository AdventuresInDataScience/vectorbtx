"""Section F — exposure.

Because F2 (maintenance margin, margin calls, forced liquidation) is deferred by D2, the
borrowed-versus-cash-covered split **must be derivable post-simulation** from
`pf.cash(free=True)` and `pf.asset_value()`. Section G consumes it, so this is a hard
requirement rather than a convenience.

Built on vectorbt's existing collateral machinery: `ExecuteOrderState` carries
`(cash, position, debt, free_cash)`, `free_cash` is collateral-aware, and `gross_exposure`
already divides by free cash.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def borrowed_amount(asset_value, free_cash):
    """The borrowed portion of an exposure, as pure arithmetic.

    `borrowed = max(0, |asset_value| - max(0, free_cash))`.

    Split out from `borrowed_portion` so the three cases the plan requires — cash-covered,
    partially financed, fully leveraged — are testable without running a simulation.

    Args:
        asset_value: market value of the position(s).
        free_cash: cash available to cover it.

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


def borrowed_portion(pf, group_by=False) -> pd.DataFrame:
    """Split exposure into the borrowed portion and the cash-covered portion.

    Args:
        pf: a simulated `Portfolio`.
        group_by: grouping passed through to vectorbt.

    Returns:
        DataFrame of the borrowed notional per bar and column — the amount financing is
        charged on. A position fully covered by cash borrows nothing.

    Notes:
        A £10,000 position fully covered by £10,000 cash finances nothing; £50,000 against
        £10,000 finances £40,000. The arithmetic lives in `borrowed_amount`.

        **Read straight off negative cash when a margin loan exists.** Under funded
        margin (`vectorbtx.margin.buying_power`) reported cash is
        `equity + cumsum(cash_flow)`, so it goes negative by exactly the amount of the
        broker's loan that has been drawn. `max(0, -cash)` is therefore the borrowed
        notional by construction, with no inference from asset value needed.

        ⚠️ The asset-value reading is **not** interchangeable with it. Buying £18,260 of
        asset against £10,000 of equity borrows £8,260, not £18,260; charging financing on
        the latter would bill the cash-covered part too. The fallback below is only
        equivalent because, without a loan, a long cannot exceed cash at all.

    Examples:
        >>> pf.x.borrowed_portion().iloc[-1]
    """
    if getattr(pf, "margin_loan", None) is not None:
        cash = pd.DataFrame(pf.cash(group_by=group_by))
        return (-cash).clip(lower=0.0)

    asset_value = pd.DataFrame(pf.asset_value(group_by=group_by))
    free_cash = pd.DataFrame(pf.cash(free=True, group_by=group_by))
    if list(free_cash.columns) != list(asset_value.columns):
        # Cash may be reported per group rather than per column under cash sharing.
        free_cash = pd.DataFrame(
            np.broadcast_to(free_cash.to_numpy()[:, :1], asset_value.shape),
            index=asset_value.index,
            columns=asset_value.columns,
        )
    return pd.DataFrame(
        borrowed_amount(asset_value.to_numpy(), free_cash.to_numpy()),
        index=asset_value.index,
        columns=asset_value.columns,
    )

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
