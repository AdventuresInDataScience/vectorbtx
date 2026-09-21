"""Section K2 — attach MT4/MT5 trade metrics to `Trades`, plus a curated subset to
`Portfolio` (D5).

`Trades._metrics` is the same mutable `Config` as `ReturnsAccessor._metrics`, so the
patching pattern transfers unchanged. Everything is tagged `mt5`, so `tags="mt5"` selects
the whole parity set.
"""

from __future__ import annotations

import numpy as np

from ...core.patch import attach, merge_metrics
from . import calc


#: `TradeStatus.Closed`. Hardcoded rather than imported so this module keeps importing
#: without vectorbt; asserted against the real enum in `tests/test_metrics_trades.py`.
_STATUS_CLOSED = 1


def _per_col(self, func, group_by=None, needs_init_cash=False, init_cash=None, field="pnl"):
    """Apply a `calc` function per column (or per group) and wrap the result.

    ⚠️ **Open trades are excluded.** Every metric here is MT4/MT5 parity, and MT computes
    its report over *closed* deals: an open position is not yet a win, a loss, or a point
    on the balance curve. vectorbt's trade records carry open positions too, with `pnl`
    marked to the last price, so taking `self.values["pnl"]` wholesale silently mixes
    floating P&L into figures that are supposed to exclude it.

    That mattered most for `balance_curve`, whose whole reason to exist is being the
    closed-only counterpart to vectorbt's mark-to-market `value()` — MT5's *Equity*
    drawdown comes free, MT4's *Balance* drawdown is the gap. Feeding it open trades
    collapsed that distinction and made the two drawdowns converge.
    """
    from vectorbt.utils.config import merge_dicts

    col_map = self.col_mapper.get_col_map(group_by=group_by)
    col_idxs, col_lens = col_map
    values = self.values[field]
    status = self.values["status"]
    out = np.full(col_lens.shape[0], np.nan, dtype=np.float64)
    start = 0
    for c in range(col_lens.shape[0]):
        n = col_lens[c]
        idxs = col_idxs[start : start + n]
        start += n
        idxs = idxs[status[idxs] == _STATUS_CLOSED]
        sel = values[idxs]
        if needs_init_cash:
            # Under cash sharing `init_cash` is reported per GROUP while this loop runs
            # per column, so a positional lookup runs off the end. One shared value is
            # the group's and applies to every member.
            if np.isscalar(init_cash):
                ic = init_cash
            else:
                flat = np.asarray(init_cash).ravel()
                ic = flat[c] if c < len(flat) else flat[0]
            out[c] = func(sel, float(ic))
        else:
            out[c] = func(sel)
    return out


def _resolve_init_cash(self, init_cash):
    """Resolve `init_cash`, honouring `InitCashMode.Auto`."""
    if init_cash is not None:
        return init_cash
    pf = getattr(self, "_x_portfolio", None)
    if pf is not None:
        return np.asarray(pf.get_init_cash()).ravel()
    raise ValueError(
        "init_cash is required for this metric. Pass init_cash=..., or read it from "
        "pf.get_init_cash() so InitCashMode.Auto is resolved to a real figure first."
    )


def _wrap(self, out, name, group_by, wrap_kwargs):
    from vectorbt.utils.config import merge_dicts

    wrap_kwargs = merge_dicts(dict(name_or_index=name), wrap_kwargs)
    return self.wrapper.wrap_reduced(out, group_by=group_by, **wrap_kwargs)


def _simple(name, func, field="pnl"):
    def method(self, group_by=None, wrap_kwargs=None):
        out = _per_col(self, func, group_by=group_by, field=field)
        return _wrap(self, out, name, group_by, wrap_kwargs)

    method.__name__ = name
    method.__doc__ = f"See `vectorbtx.metrics.trades.calc.{func.__name__}`."
    return method


def _with_cash(name, func):
    def method(self, init_cash=None, group_by=None, wrap_kwargs=None):
        ic = _resolve_init_cash(self, init_cash)
        out = _per_col(self, func, group_by=group_by, needs_init_cash=True, init_cash=ic)
        return _wrap(self, out, name, group_by, wrap_kwargs)

    method.__name__ = name
    method.__doc__ = f"See `vectorbtx.metrics.trades.calc.{func.__name__}`."
    return method


def _streak(name, func, winning):
    def method(self, group_by=None, wrap_kwargs=None):
        out = _per_col(self, lambda a: func(a, winning), group_by=group_by)
        return _wrap(self, out, name, group_by, wrap_kwargs)

    method.__name__ = name
    method.__doc__ = f"See `vectorbtx.metrics.trades.calc.{func.__name__}`."
    return method


METHODS = dict(
    total_net_profit=_simple("total_net_profit", calc.total_net_profit),
    gross_profit=_simple("gross_profit", calc.gross_profit),
    gross_loss=_simple("gross_loss", calc.gross_loss),
    loss_rate=_simple("loss_rate", calc.loss_rate),
    z_score=_simple("z_score", calc.z_score),
    ahpr=_simple("ahpr", calc.ahpr, field="return"),
    ghpr=_simple("ghpr", calc.ghpr, field="return"),
    absolute_drawdown=_with_cash("absolute_drawdown", calc.absolute_drawdown),
    balance_drawdown=_with_cash("balance_drawdown", calc.balance_drawdown),
    balance_drawdown_abs=_with_cash("balance_drawdown_abs", calc.balance_drawdown_abs),
    recovery_factor=_with_cash("recovery_factor", calc.recovery_factor),
    lr_correlation=_with_cash("lr_correlation", calc.lr_correlation),
    lr_standard_error=_with_cash("lr_standard_error", calc.lr_standard_error),
    max_consecutive_wins_money=_streak(
        "max_consecutive_wins_money", calc.max_consecutive_money, True
    ),
    max_consecutive_losses_money=_streak(
        "max_consecutive_losses_money", calc.max_consecutive_money, False
    ),
    maximal_consecutive_profit=_streak(
        "maximal_consecutive_profit", calc.maximal_consecutive, True
    ),
    maximal_consecutive_loss=_streak("maximal_consecutive_loss", calc.maximal_consecutive, False),
    avg_consecutive_wins=_streak("avg_consecutive_wins", calc.avg_consecutive, True),
    avg_consecutive_losses=_streak("avg_consecutive_losses", calc.avg_consecutive, False),
)


def _m(title, calc_func, extra_tags=()):
    return dict(
        title=title,
        calc_func=calc_func,
        tags=["vectorbtx", "trades", "mt5", *extra_tags],
    )


METRICS = dict(
    total_net_profit=_m("Total Net Profit", "total_net_profit"),
    gross_profit=_m("Gross Profit", "gross_profit"),
    gross_loss=_m("Gross Loss", "gross_loss"),
    loss_rate=dict(
        title="Loss Trades [%]",
        calc_func="loss_rate",
        post_calc_func=lambda self, out, settings: out * 100,
        tags=["vectorbtx", "trades", "mt5"],
    ),
    long_win_rate=dict(
        title="Long Trades Won [%]",
        calc_func="long.win_rate",
        post_calc_func=lambda self, out, settings: out * 100,
        tags=["vectorbtx", "trades", "mt5"],
    ),
    short_win_rate=dict(
        title="Short Trades Won [%]",
        calc_func="short.win_rate",
        post_calc_func=lambda self, out, settings: out * 100,
        tags=["vectorbtx", "trades", "mt5"],
    ),
    absolute_drawdown=_m("Absolute Drawdown", "absolute_drawdown", ("drawdowns",)),
    balance_drawdown=dict(
        title="Balance Drawdown [%]",
        calc_func="balance_drawdown",
        post_calc_func=lambda self, out, settings: out * 100,
        tags=["vectorbtx", "trades", "mt5", "drawdowns"],
    ),
    recovery_factor=_m("Recovery Factor", "recovery_factor", ("drawdowns",)),
    max_consecutive_wins_money=_m("Max Consecutive Wins ($)", "max_consecutive_wins_money"),
    max_consecutive_losses_money=_m("Max Consecutive Losses ($)", "max_consecutive_losses_money"),
    maximal_consecutive_profit=_m("Maximal Consecutive Profit", "maximal_consecutive_profit"),
    maximal_consecutive_loss=_m("Maximal Consecutive Loss", "maximal_consecutive_loss"),
    avg_consecutive_wins=_m("Average Consecutive Wins", "avg_consecutive_wins"),
    avg_consecutive_losses=_m("Average Consecutive Losses", "avg_consecutive_losses"),
    lr_correlation=_m("LR Correlation", "lr_correlation"),
    lr_standard_error=_m("LR Standard Error", "lr_standard_error"),
    ahpr=_m("AHPR", "ahpr"),
    ghpr=_m("GHPR", "ghpr"),
    z_score=_m("Z-Score", "z_score"),
)

#: The four MT metrics worth surfacing on the default `pf.stats()` table. Everything else
#: stays on `pf.trades.stats()` so the default output remains readable.
PORTFOLIO_METRICS = dict(
    total_net_profit=dict(
        title="Total Net Profit",
        calc_func="trades.total_net_profit",
        tags=["vectorbtx", "portfolio", "mt5"],
    ),
    recovery_factor=dict(
        title="Recovery Factor",
        calc_func=lambda self: self.trades.recovery_factor(init_cash=np.asarray(self.get_init_cash()).ravel()),
        tags=["vectorbtx", "portfolio", "mt5"],
    ),
    absolute_drawdown=dict(
        title="Absolute Drawdown",
        calc_func=lambda self: self.trades.absolute_drawdown(init_cash=np.asarray(self.get_init_cash()).ravel()),
        tags=["vectorbtx", "portfolio", "mt5"],
    ),
    ghpr=dict(
        title="GHPR",
        calc_func="trades.ghpr",
        tags=["vectorbtx", "portfolio", "mt5"],
    ),
)

_DONE = False


def register_trade_metrics() -> None:
    """Attach K2's methods and `_metrics` entries. Idempotent."""
    global _DONE
    if _DONE:
        return
    from vectorbt.portfolio.trades import Trades
    from vectorbt.portfolio.base import Portfolio

    for name, fn in METHODS.items():
        fn.__qualname__ = f"Trades.{name}"
        attach(Trades, name, fn)
    merge_metrics(Trades, METRICS)
    merge_metrics(Portfolio, PORTFOLIO_METRICS)
    _DONE = True
