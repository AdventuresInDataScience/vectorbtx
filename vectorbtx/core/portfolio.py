"""Section B — the `Portfolio` subclass.

Subclassed rather than patched (D5) because it carries extra constructor parameters and
per-column state that must survive `.replace()` and `indexing_func`.

Its main user-facing job is hosting `from_weights` (Section J) alongside upstream's
`from_signals`, `from_orders` and `from_holding`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from vectorbt.portfolio.base import Portfolio as _Portfolio

from .accessors import PortfolioXAccessor


class AccountSelectionError(ValueError):
    """Raised when a column selection spans more than one cash-sharing account."""


class Portfolio(_Portfolio):
    """Drop-in replacement for `vectorbt.portfolio.base.Portfolio`.

    Adds:

    * `Portfolio.from_weights` — calendar/threshold rebalanced investing (Section J)
    * `Portfolio.from_holding_costs` — financing and carry (Section G)
    * `pf.x` — the vectorbtX namespace
    * carried state: `universe`, `drift`, `holding_costs`
    """

    def __init__(self, *args, universe=None, drift=None, holding_costs=None,
                 margin_loan=None, high=None, low=None, **kwargs):
        _Portfolio.__init__(self, *args, **kwargs)
        self._universe = universe
        self._drift = drift
        self._holding_costs = holding_costs
        self._margin_loan = margin_loan
        self._high = high
        self._low = low
        # Keep them in the config so `.replace()` round-trips them.
        #
        # ⚠️ Must go through `update_config`, which is `Config.update(..., force=True)`
        # and mutates in place. `self.config.merge_with(...)` looks like the obvious
        # alternative and is wrong: `merge_with` calls `merge_dicts(..., to_dict=True)`,
        # which returns a **plain dict**. Assigning that to `_config` leaves the instance
        # with a dict where a `Config` is expected, and the very next `.replace()` —
        # which every constructor here ends with — dies on `dict.merge_with`.
        self.update_config(
            dict(universe=universe, drift=drift, holding_costs=holding_costs,
                 margin_loan=margin_loan, high=high, low=low)
        )

    # ############# State ############# #

    @property
    def universe(self):
        """The `Universe` used to build this portfolio, or None."""
        return self._universe

    @property
    def drift(self):
        """Quantization drift frame recorded at construction, or None."""
        return self._drift

    @property
    def holding_costs(self):
        """Per-bar holding-cost cash adjustments, or None."""
        return self._holding_costs

    @property
    def high(self):
        """Intrabar highs, if supplied. Used for stops and intrabar drawdown."""
        return self._high

    @property
    def low(self):
        """Intrabar lows, if supplied."""
        return self._low

    @property
    def margin_loan(self):
        """The broker's margin loan — buying power beyond the account's own equity.

        In the same basis vectorbt uses for `init_cash`: per group under cash sharing,
        per column otherwise. None for an unlevered portfolio.
        """
        return self._margin_loan

    @property
    def x(self) -> PortfolioXAccessor:
        """The vectorbtX namespace."""
        return PortfolioXAccessor(self)

    # ############# Reporting ############# #

    def get_init_cash(self, group_by=None, wrap_kwargs=None, engine=None):
        """Initial cash **as equity**, with the broker's margin loan removed.

        ⚠️ **This is what makes funded leverage report correctly.** The engine is given
        buying power (`equity x leverage`) because `buy_nb` bounds every long by cash and
        there is no other way to fund a levered position (see
        `vectorbtx.margin.buying_power`). Left alone, the portfolio would then report an
        initial balance of £300,000 on a £10,000 account and understate every return by
        the leverage factor.

        Subtracting the loan here fixes all of it at once: `cash` is
        `init_cash + cumsum(cash_flow)`, and `value`, `returns`, `returns_stats` and every
        `_metrics` entry are derived from those two. Reported cash goes **negative** when
        the loan is drawn, which is correct — that is the borrowed amount, and
        `borrowed_portion` reads it straight off.

        The loan is stored in the same basis vectorbt uses for `init_cash` — per group
        under cash sharing, per column otherwise — so it is mapped to the requested
        grouping with upstream's own two functions rather than a parallel reimplementation
        that could drift out of step with them.
        """
        out = _Portfolio.get_init_cash(
            self, group_by=group_by, wrap_kwargs=wrap_kwargs, engine=engine
        )
        loan = self._margin_loan
        if loan is None:
            return out

        from vectorbt.portfolio import dispatch
        from vectorbt.utils.config import merge_dicts

        loan = np.atleast_1d(np.asarray(loan, dtype=float))
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            group_lens = self.wrapper.grouper.get_group_lens(group_by=group_by)
            adj = dispatch.init_cash_grouped(loan, group_lens, self.cash_sharing, engine=engine)
        else:
            group_lens = self.wrapper.grouper.get_group_lens()
            adj = dispatch.init_cash(loan, group_lens, self.cash_sharing, engine=engine)

        values = np.asarray(out, dtype=float) - np.asarray(adj, dtype=float)
        wrap_kwargs = merge_dicts(dict(name_or_index="init_cash"), wrap_kwargs)
        return self.wrapper.wrap_reduced(values, group_by=group_by, **wrap_kwargs)

    def get_trades(self, group_by=None, **kwargs):
        """Trades, tagged with a back-reference to this portfolio.

        Several MT4/MT5 trade metrics — Absolute Drawdown and Recovery Factor among them —
        need `init_cash`, which a `Trades` object has no way to reach on its own. Without
        the tag they raise inside `pf.trades.stats()`, taking the whole table down with
        them.

        The back-reference also means `InitCashMode.Auto` is resolved to a real figure
        first, via `get_init_cash` — which additionally makes these metrics correct under
        funded margin, since that is where the margin loan is subtracted back out.
        """
        trades = _Portfolio.get_trades(self, group_by=group_by, **kwargs)
        try:
            trades._x_portfolio = self
        except AttributeError:  # pragma: no cover - defensive, Trades has no __slots__
            pass
        return trades

    def cash_flow(self, group_by=None, free: bool = False, wrap_kwargs=None, engine=None):
        """Cash flow, with Section G's holding costs folded in.

        ⚠️ **This override is what makes holding costs visible at all.** Plan §5.3 treats
        mutating `last_cash` from `post_segment_func_nb` as sufficient, and it is not.
        That mutation changes the cash the *simulation* sees, so buying power and every
        sizing decision downstream of it are correct — but `Portfolio.cash_flow`
        reconstructs cash purely from **order records** (`base.py`, `dispatch.cash_flow`
        over `self.orders.values`), and `cash`, `value`, `returns` and every metric are
        derived from that. A financing charge is not an order, so it leaves no record,
        and the reported equity curve would show none of it: the simulation would know
        it had been charged while `pf.stats()` reported the gross return.

        Folding the charges in here rather than at each call site means `cash`, `value`,
        `returns`, `returns_stats` and the whole `_metrics` surface pick them up with no
        further work, because all of them route through this method.

        Only the ungrouped branch adds them. Upstream's grouped branch re-enters this
        method with `group_by=False` and then aggregates, so adding in both would
        double-count.
        """
        out = _Portfolio.cash_flow(
            self, group_by=group_by, free=free, wrap_kwargs=wrap_kwargs, engine=engine
        )
        costs = self._holding_costs
        if costs is None or self.wrapper.grouper.is_grouped(group_by=group_by):
            return out

        frame = pd.DataFrame(costs)
        values = np.array(np.asarray(out, dtype=float), copy=True)
        flat = values.reshape(len(self.wrapper.index), -1)
        for i, col in enumerate(self.wrapper.columns):
            if col in frame.columns:
                flat[:, i] += frame[col].to_numpy(dtype=float)
        return self.wrapper.wrap(flat, group_by=False, **(wrap_kwargs or {}))

    def indexing_func(self, pd_indexing_func, **kwargs):
        """Slice per-column vectorbtX state alongside upstream's fields.

        Upstream slices each of its own fields explicitly; anything we add must be sliced
        here too or it silently desynchronises from the columns it describes.

        On a grouped, cash-sharing portfolio upstream selects **groups**, so `pf["A"]`
        failed for an instrument inside a shared account. A key that names no group but
        does name columns now selects those columns' contribution to their account — see
        `_select_within_account`. A key naming a group selects the group, as upstream.
        """
        try:
            new_pf = _Portfolio.indexing_func(self, pd_indexing_func, **kwargs)
        except Exception as upstream_error:
            if not (self.cash_sharing and self.wrapper.grouper.is_grouped()):
                raise
            try:
                return self._select_within_account(pd_indexing_func, **kwargs)
            except AccountSelectionError:
                raise
            except Exception as column_error:
                # Neither a group nor a column. The column lookup's error (a KeyError
                # naming the key) says more than upstream's "already contains one column".
                raise column_error from upstream_error

        # The loan shares `init_cash`'s basis, so it must be sliced by the same indices —
        # groups under cash sharing, columns otherwise. Slicing it by the wrong axis would
        # leave a portfolio reporting equity for columns it no longer holds.
        margin_loan = self._margin_loan
        if margin_loan is not None:
            _, _, group_idxs, col_idxs = self.wrapper.indexing_func_meta(
                pd_indexing_func, column_only_select=True, **kwargs
            )
            idxs = group_idxs if self.cash_sharing else col_idxs
            # ⚠️ `pf["A"]` selects with a scalar index, which leaves a 0-d array here, and
            # upstream's `init_cash_grouped_nb` needs 1-D: every metric on a levered
            # portfolio then died with a numba TypingError while `pf[["A"]]` worked.
            margin_loan = np.atleast_1d(np.asarray(margin_loan, dtype=float)[idxs])
        return self._slice_extras(new_pf, margin_loan)

    def _select_within_account(self, pd_indexing_func, **kwargs):
        """Columns of a shared account, as their contribution to that account.

        A column inside a shared account has no cash of its own, so "its" portfolio needs
        a definition. This one is the column's **contribution**: the account's starting
        equity, only the selected columns' orders and holding costs, and therefore
        `total_profit` equal to their P&L and `total_return` equal to that P&L over the
        account's equity. Contributions add up: summed over every column they give the
        account's profit and return exactly. It is also what upstream's own ungrouped
        figures mean under cash sharing (`pf.total_return(group_by=False)`), so the two
        agree.

        What it is not: a re-simulation. The orders are the ones the shared account
        actually placed, constrained by its cash — not what the column would have done
        with an account to itself.

        Raises:
            AccountSelectionError: if the selection spans more than one cash-sharing
                account.
        """
        from vectorbt.base.reshape_fns import to_2d_array

        ungrouped = self.wrapper.regroup(False)
        new_wrapper, _, _, col_idxs = ungrouped.indexing_func_meta(
            pd_indexing_func, column_only_select=True, **kwargs
        )
        col_idxs = np.atleast_1d(np.asarray(col_idxs))
        groups = np.asarray(self.wrapper.grouper.get_groups())[col_idxs]
        if len(np.unique(groups)) > 1:
            raise AccountSelectionError(
                "This selection spans more than one cash-sharing account, so there is no "
                "single account for it to be a contribution to. Select columns of one "
                "account at a time, or select the groups themselves."
            )
        equity = np.atleast_1d(np.asarray(self.get_init_cash(), dtype=float))[groups[0]]

        # Several columns of one account still share its cash; one column is its own.
        shared = len(col_idxs) > 1
        if shared:
            # `replace`, not `regroup`: the ungrouped wrapper refuses to enable grouping.
            new_wrapper = new_wrapper.replace(group_by=True)
        new_pf = self.replace(
            wrapper=new_wrapper,
            close=new_wrapper.wrap(to_2d_array(self.close)[:, col_idxs], group_by=False),
            order_records=self.orders.get_by_col_idxs(col_idxs),
            log_records=self.logs.get_by_col_idxs(col_idxs),
            init_cash=np.array([equity], dtype=float),
            cash_sharing=shared,
            # The account's call order ranks every column; it means nothing for a subset.
            call_seq=None,
            # `equity` already has the loan taken out, so the slice carries none.
            margin_loan=None,
        )
        return self._slice_extras(new_pf, margin_loan=None)

    def _slice_extras(self, new_pf, margin_loan):
        """Carry the vectorbtX state, and the already-sliced loan, to a column selection."""
        cols = list(new_pf.wrapper.columns)

        universe = self._universe
        if universe is not None:
            universe = universe.subset(cols)

        def _slice(obj):
            if isinstance(obj, pd.DataFrame):
                keep = [c for c in cols if c in obj.columns]
                return obj[keep]
            return obj

        return new_pf.replace(
            universe=universe,
            drift=_slice(self._drift),
            holding_costs=_slice(self._holding_costs),
            margin_loan=margin_loan,
            high=_slice(self._high),
            low=_slice(self._low),
        )

    # ############# Constructors ############# #

    @classmethod
    def from_weights(
        cls,
        close,
        weights,
        schedule=None,
        universe=None,
        min_trade=None,
        equity=None,
        leverage_budget=True,
        order_kwargs=None,
        **kwargs,
    ):
        """Simulate a rebalanced portfolio from a matrix of target weights (Section J).

        Args:
            close: price frame, already base-currency converted if multi-currency.
            weights: DataFrame of target weights, one column per asset.
            schedule: boolean Series — True on rebalance bars. None → every bar.
            universe: `Universe` supplying per-column order arguments and quantization.
            min_trade: suppress rebalance trades below this absolute weight change.
            equity: notional account equity used for quantization (defaults to `init_cash`).
            leverage_budget: apply `leverage_type="portfolio_level"` scaling (Section F).

        Returns:
            `Portfolio`, with `pf.drift` carrying quantization and suppression drift.

        Notes:
            Uses `SizeType.TargetPercent` with `cash_sharing=True`, `group_by=True` and
            `call_seq="auto"` — see Section J for why none of those three is optional.
        """
        from ..construct.rebalance import apply_leverage_budget, to_targets

        weights = pd.DataFrame(weights)
        if leverage_budget and universe is not None:
            weights = apply_leverage_budget(weights, universe)

        if equity is None:
            equity = kwargs.get("init_cash", 100.0)
            if not np.isscalar(equity):
                equity = float(np.asarray(equity).sum())

        targets, drift = to_targets(
            weights,
            schedule,
            universe=universe,
            price=close,
            equity=equity,
            min_trade=min_trade,
        )

        # Prefer an already-resolved set. The front door resolves once, with the order
        # notional it estimated from the weights — re-resolving here would discard that
        # and fail on any instrument with a minimum commission.
        if order_kwargs is None:
            if universe is None:
                order_kwargs = {}
            else:
                # A rebalanced portfolio has no explicit size — the engine derives orders
                # from target percentages — so any instrument with a minimum commission
                # would fail here for want of an order notional. Estimate it as
                # `weight x equity`: exact on a rebalance bar where the portfolio is at
                # target, and close enough between them. The alternative is refusing to
                # model a minimum commission on any rebalanced portfolio.
                est_notional = (
                    pd.DataFrame(weights).abs().reindex(index=pd.DataFrame(close).index)
                    .ffill().fillna(0.0) * float(equity)
                )
                order_kwargs = universe.resolve(close, notional=est_notional)
        else:
            order_kwargs = dict(order_kwargs)
        order_kwargs.update(kwargs)
        order_kwargs.setdefault("cash_sharing", True)
        order_kwargs.setdefault("group_by", True)
        order_kwargs.setdefault("call_seq", "auto")

        pf = cls.from_orders(
            close,
            size=targets,
            size_type="targetpercent",
            **order_kwargs,
        )
        return pf.replace(universe=universe, drift=drift)

    @classmethod
    def from_holding_costs(cls, close, size, universe, rates=None, **kwargs):
        """Simulate with financing and carry charged per bar (Section G).

        Routes through `from_order_func` because that is the only entry point exposing the
        per-bar callbacks holding costs need. Slower and more fragile than `from_orders`;
        use it only when holding costs are actually wanted.
        """
        from ..holding.callback import simulate_with_holding_costs

        return simulate_with_holding_costs(cls, close, size, universe, rates=rates, **kwargs)
