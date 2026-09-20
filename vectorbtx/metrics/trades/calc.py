"""Section K2 — trade-metric arithmetic, on plain arrays.

Every function here takes a 1-D array of trade PnL (or trade returns) ordered by exit and
returns a scalar. Nothing imports vectorbt, so all 18 additions are unit-testable against
hand-computed numbers without running a simulation.

`register.py` is the only place that knows about `Trades`.
"""

from __future__ import annotations

import numpy as np

from ..returns.nb import linreg_1d_nb


def _clean(pnl) -> np.ndarray:
    """Coerce to a finite 1-D float array."""
    a = np.asarray(pnl, dtype=np.float64).ravel()
    return a[~np.isnan(a)]


# ############# One-line derivations ############# #


def total_net_profit(pnl) -> float:
    """Sum of trade PnL. MT's *Total Net Profit*."""
    a = _clean(pnl)
    return float(a.sum()) if a.size else np.nan


def gross_profit(pnl) -> float:
    """Sum of winning-trade PnL."""
    a = _clean(pnl)
    return float(a[a > 0].sum()) if a.size else np.nan


def gross_loss(pnl) -> float:
    """Sum of losing-trade PnL — **negative**, as MT reports it."""
    a = _clean(pnl)
    return float(a[a < 0].sum()) if a.size else np.nan


def loss_rate(pnl) -> float:
    """`1 - win_rate`."""
    a = _clean(pnl)
    if a.size == 0:
        return np.nan
    return float((a < 0).sum() / a.size)


def win_rate(pnl) -> float:
    """Fraction of trades with positive PnL."""
    a = _clean(pnl)
    if a.size == 0:
        return np.nan
    return float((a > 0).sum() / a.size)


# ############# Balance curve and drawdowns ############# #


def balance_curve(pnl, init_cash: float) -> np.ndarray:
    """Closed-trade *balance* curve: `init_cash + cumsum(pnl)`.

    Note the inversion relative to MT: vectorbt's `value` already includes floating open
    positions, so MT5's *Equity* drawdown comes free and MT4's *Balance* drawdown — this
    curve — is the gap.
    """
    a = np.asarray(pnl, dtype=np.float64).ravel()
    a = np.nan_to_num(a, nan=0.0)
    return np.concatenate(([init_cash], init_cash + np.cumsum(a)))


def absolute_drawdown(pnl, init_cash: float) -> float:
    """`initial_deposit - minimum_balance`, in **currency, not percent**.

    Pass a resolved `init_cash`: under `InitCashMode.Auto` the raw field is a mode flag,
    and the metric is meaningless without resolving it first.
    """
    curve = balance_curve(pnl, init_cash)
    return float(init_cash - curve.min())


def balance_drawdown(pnl, init_cash: float) -> float:
    """Maximum drawdown of the closed-trade balance curve, as a **fraction**."""
    curve = balance_curve(pnl, init_cash)
    peak = np.maximum.accumulate(curve)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, curve / peak - 1.0, 0.0)
    return float(-dd.min())


def balance_drawdown_abs(pnl, init_cash: float) -> float:
    """Maximum drawdown of the balance curve, in currency."""
    curve = balance_curve(pnl, init_cash)
    peak = np.maximum.accumulate(curve)
    return float((peak - curve).max())


def recovery_factor(pnl, init_cash: float) -> float:
    """`net profit / max balance drawdown` (currency).

    Not to be confused with vectorbt's `drawdowns.recovery_return`, which is a different
    quantity entirely.
    """
    dd = balance_drawdown_abs(pnl, init_cash)
    if dd == 0:
        return np.nan
    return total_net_profit(pnl) / dd


# ############# Streaks ############# #


def runs(pnl):
    """Split trades into consecutive win/loss runs.

    Returns a list of `(is_win, length, total_pnl)`. Zero-PnL trades break a run without
    joining either side, matching MT's treatment of scratch trades.
    """
    a = np.asarray(pnl, dtype=np.float64).ravel()
    out = []
    cur_sign, cur_len, cur_sum = 0, 0, 0.0
    for v in a:
        if np.isnan(v):
            continue
        s = 1 if v > 0 else (-1 if v < 0 else 0)
        if s == cur_sign and s != 0:
            cur_len += 1
            cur_sum += v
        else:
            if cur_sign != 0:
                out.append((cur_sign > 0, cur_len, cur_sum))
            cur_sign, cur_len, cur_sum = s, (1 if s != 0 else 0), (v if s != 0 else 0.0)
    if cur_sign != 0:
        out.append((cur_sign > 0, cur_len, cur_sum))
    return out


def _runs_of(pnl, winning: bool):
    return [r for r in runs(pnl) if r[0] == winning]




def max_consecutive_money(pnl, winning: bool = True) -> float:
    """PnL of the **longest** run of wins (or losses).

    Distinct from `maximal_consecutive`: the longest streak is not necessarily the most
    profitable one.
    """
    rs = _runs_of(pnl, winning)
    if not rs:
        return np.nan
    best = max(rs, key=lambda r: (r[1], abs(r[2])))
    return float(best[2])


def maximal_consecutive(pnl, winning: bool = True) -> float:
    """PnL of the **most profitable** (or most damaging) run."""
    rs = _runs_of(pnl, winning)
    if not rs:
        return np.nan
    return float(max(r[2] for r in rs) if winning else min(r[2] for r in rs))


def avg_consecutive(pnl, winning: bool = True) -> float:
    """Mean run length for wins (or losses)."""
    rs = _runs_of(pnl, winning)
    if not rs:
        return np.nan
    return float(np.mean([r[1] for r in rs]))


# ############# Regression byproducts ############# #


def lr_correlation(pnl, init_cash: float) -> float:
    """Correlation of the balance curve with a straight line. K1's regression, reused."""
    return float(linreg_1d_nb(balance_curve(pnl, init_cash))[2])


def lr_standard_error(pnl, init_cash: float) -> float:
    """Standard error of the balance-curve regression slope."""
    return float(linreg_1d_nb(balance_curve(pnl, init_cash))[1])


# ############# Holding-period returns ############# #


def ahpr(trade_returns) -> float:
    """Arithmetic holding-period return — `mean(1 + r)`, MT's convention (e.g. 1.0034)."""
    a = _clean(trade_returns)
    if a.size == 0:
        return np.nan
    return float(np.mean(1.0 + a))


def ghpr(trade_returns) -> float:
    """Geometric holding-period return — `prod(1 + r) ** (1/n)`."""
    a = _clean(trade_returns)
    a = a[a > -1.0]
    if a.size == 0:
        return np.nan
    return float(np.exp(np.mean(np.log(1.0 + a))))


# ############# Runs test ############# #


def z_score(pnl) -> float:
    """Wald–Wolfowitz runs test.

    `Z = (N(R - 0.5) - P) / sqrt(P(P - N) / (N - 1))` with `N` total trades, `R` number of
    runs and `P = 2 * wins * losses`.

    Interpretation, since it is constantly misread: **negative Z means outcomes cluster
    into streaks**, positive means they alternate. It is a dependency test, not a
    performance measure.
    """
    a = _clean(pnl)
    a = a[a != 0]
    n = a.size
    if n < 3:
        return np.nan
    wins = int((a > 0).sum())
    losses = n - wins
    if wins == 0 or losses == 0:
        return np.nan
    r = 1 + int(np.sum(np.sign(a[1:]) != np.sign(a[:-1])))
    p = 2.0 * wins * losses
    denom = p * (p - n) / (n - 1)
    if denom <= 0:
        return np.nan
    return float((n * (r - 0.5) - p) / np.sqrt(denom))
