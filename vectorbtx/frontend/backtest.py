"""Section N — `PortfolioBacktest`, the single implementation.

One call, every argument in it, returns a `Portfolio`. `SingleBacktest` is a thin
forwarder onto this, so there is no second code path to keep in step.

The dispatch, in one line each:

* `mode` decides which instrument arguments are legal, required or rejected.
* `entries`/`exits` → `from_signals`; `weights`/`schedule` → `from_weights`;
  neither → `from_holding`.
* Holding costs (financing or carry in the active stack, with rates supplied) route
  through `from_holding_costs` instead, which is the only path using `from_order_func`.

Unrecognised arguments pass through to the underlying vectorbt constructor, so nothing
upstream becomes unreachable from here.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .prepare import BacktestError, SIZING_ARGS, _SPEC_FIELDS, prepare

#: Arguments this layer consumes itself. Anything else is forwarded to vectorbt, so the
#: full upstream surface stays reachable through the front door.
_FRONTEND_ARGS = frozenset(
    {
        "instruments", "entries", "exits", "weights", "schedule",
        "stop_points", "target_points", "trail_points",
        "volume", "rates", "fx", "base_currency", "fx_markup",
        "min_trade", "leverage_budget", "holidays", "financing_markup", "carry_markup",
        *SIZING_ARGS,
        *_SPEC_FIELDS,
    }
)

#: `Order` fields a user may pass straight through; collisions with resolved values raise.
_ORDER_PASSTHROUGH = frozenset(
    {"fees", "fixed_fees", "slippage", "min_size", "max_size", "size_granularity", "lock_cash"}
)


def PortfolioBacktest(
    *,
    close=None,
    open=None,
    high=None,
    low=None,
    # --- instruments (four supply forms; see prepare.to_universe) ---
    instruments=None,
    # --- strategy ---
    entries=None,
    exits=None,
    weights=None,
    schedule=None,
    # --- sizing (exactly one) ---
    risk_pct=None,
    risk_amount=None,
    stake=None,
    notional=None,
    size=None,
    # --- stops, in points ---
    stop_points=None,
    target_points=None,
    trail_points=None,
    # --- data ---
    volume=None,
    rates=None,
    fx=None,
    base_currency=None,
    # --- construction ---
    min_trade=None,
    leverage_budget=True,
    # --- holding costs ---
    holidays=None,
    financing_markup=None,
    carry_markup=0.0,
    # --- account ---
    init_cash=None,
    **kwargs,
):
    """Run a backtest from one call and return a `vectorbtx.Portfolio`.

    Args:
        close: close price, one column per instrument. Signals are computed on it and the
            equity curve is valued at it. Already base-currency converted if the portfolio
            is multi-currency — see Notes.
        open: open price, same shape. **Required** — orders fill here, on the bar after
            the signal.
        high, low: intrabar extremes, same shape. **Required.** Stops are checked against
            them, and drawdown is measured against them. See Notes.
        instruments: the universe. A dict of per-symbol overrides, a CSV/Excel path, a
            `Universe`, `Instrument` objects, or None to use the flat keyword form.
        entries, exits: boolean signal frames → signal-driven strategy.
        weights, schedule: target weights and rebalance mask → rebalanced strategy.
        risk_pct, risk_amount, stake, notional, size: **exactly one.** Two raises.
        stop_points, target_points, trail_points: stops in points, converted internally
            to the native price fractions.
        volume: volume frame, required by `slippage="auto"`.
        rates: benchmark rate table (Section M), required by `financing="auto"`.
        fx: an `FXTable` for converting foreign-currency fixed costs.
        base_currency: the account currency.
        min_trade: suppress rebalance trades below this weight change.
        leverage_budget: apply `leverage_type="portfolio_level"` scaling.
        holidays: dates carrying no holding-cost accrual.
        financing_markup: broker markup over the benchmark, always adverse.
        carry_markup: markup on the carry differential, also adverse.
        init_cash: account size.
        **kwargs: anything else, forwarded to the underlying vectorbt constructor.

    Returns:
        `vectorbtx.Portfolio`.

    Raises:
        BacktestError: on a missing or unknown `mode`, an argument a mode rejects, two
            sizing arguments, both signals and weights, or a passthrough colliding with a
            resolved cost.

    Notes:
        **Orders fill at the next bar's open, always.** A signal computed from bar `t`'s
        close cannot be acted on until bar `t+1` opens, so entries and exits are shifted
        one bar and filled at `open`. There is no option to fill at the signal bar's close,
        because that is lookahead: it captures the overnight gap you could not have traded,
        and bills a night of financing you did not owe. A backtest that does it is not
        worth running, so the API does not offer it.

        Everything order-time — sizes, stop distances, spread fractions — is therefore
        priced at the **open you fill at**, not the close the signal fired on.

        ⚠️ **FX conversion stays outside this call.** Signals are computed on
        base-currency prices and the user needs to hold that series anyway, so
        `to_base_currency` remains an explicit line in multi-currency backtests.
        Converting internally for sizing while signals ran on native prices would put two
        different price series in one backtest — worse than one visible line.

        ⚠️ **`risk_pct` sizes off `init_cash`, not compounding equity.** Material over a
        long backtest with real drawdowns.

    Examples:
        >>> pf = vbt.x.PortfolioBacktest(
        ...     close=close, open=open_, high=high, low=low,
        ...     entries=entries, exits=exits,
        ...     mode="equity", point_size=0.01, commission_flat=10.0,
        ...     notional=20_000, init_cash=100_000, freq="D",
        ... )
    """
    spec_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in _SPEC_FIELDS}
    passthrough = {k: kwargs.pop(k) for k in list(kwargs) if k in _ORDER_PASSTHROUGH}

    if close is None:
        raise BacktestError(
            "`close` is required. All four price series are passed by keyword:\n"
            "  vbt.x.PortfolioBacktest(close=..., open=..., high=..., low=..., ...)"
        )
    close = pd.DataFrame(close)
    if open is None:
        raise BacktestError(
            "`open` is required. Orders fill at the next bar's open, because a signal from "
            "this bar's close cannot be acted on until the next bar opens. Filling at the "
            "signal bar's close is lookahead — it captures the overnight gap and skips a "
            "night of financing — so there is no option to do it.\n"
            "  vbt.x.PortfolioBacktest(data.get('Close'), data.get('Open'), ...)"
        )
    open = pd.DataFrame(open).reindex(index=close.index, columns=close.columns)

    if high is None or low is None:
        raise BacktestError(
            "`high` and `low` are required. Without them a stop can only be compared "
            "against the close, so it fires late or not at all, and drawdown is measured "
            "close-to-close — which misses every intrabar excursion. Both errors flatter "
            "the result, so neither is offered as a default.\n"
            "  d = vbt.YFData.download(SYMBOL, ...)\n"
            "  vbt.x.PortfolioBacktest(d.get('Close'), d.get('Open'), d.get('High'), d.get('Low'), ...)"
        )
    high = pd.DataFrame(high).reindex(index=close.index, columns=close.columns)
    low = pd.DataFrame(low).reindex(index=close.index, columns=close.columns)

    invalid = ((high < low) | (high < close) | (low > close)).to_numpy().any()
    if invalid:
        warnings.warn(
            "high/low do not bracket close on every bar (H >= C >= L is violated "
            "somewhere). This usually means the series were adjusted or FX-converted "
            "inconsistently. Stops will fire at prices that never traded.",
            stacklevel=2,
        )

    ctx = prepare(
        close,
        fill_price=open,
        instruments=instruments,
        entries=entries, exits=exits, weights=weights, schedule=schedule,
        risk_pct=risk_pct, risk_amount=risk_amount, stake=stake,
        notional=notional, size=size,
        stop_points=stop_points, target_points=target_points, trail_points=trail_points,
        volume=volume, rates=rates, fx=fx, base_currency=base_currency,
        init_cash=init_cash,
        spec_kwargs=spec_kwargs,
        passthrough=passthrough,
    )

    from ..core.portfolio import Portfolio

    universe = ctx["universe"]
    close = ctx["price"]
    fill = ctx["fill_price"]
    order_kwargs = ctx["order_kwargs"]

    sim_kwargs = dict(kwargs)
    margin_loan = None
    if init_cash is not None:
        # Fund the leverage rather than only capping it. `buy_nb` bounds every long by
        # cash, so a levered position is unreachable unless the engine is handed the
        # buying power; `Portfolio.get_init_cash` subtracts the loan back out so reported
        # equity, returns and every metric stay on the account's real size.
        from ..margin.buying_power import buying_power

        # `from_weights` turns cash sharing on itself (Section J: it is load-bearing for
        # `call_seq="auto"`), so the basis has to be decided here rather than read off
        # whatever the caller happened to pass.
        shares_cash = bool(
            sim_kwargs.get("cash_sharing", ctx["strategy"] == "weights")
        )
        engine_cash, margin_loan = buying_power(
            universe, init_cash, ctx["columns"], cash_sharing=shares_cash,
        )
        sim_kwargs["init_cash"] = engine_cash

    strategy = ctx["strategy"]

    if strategy == "weights":
        if weights is None:
            raise BacktestError("`schedule` was given without `weights`; there is nothing to target.")
        # ⚠️ The rebalancing path is funded at EQUITY, never at buying power.
        #
        # `TargetPercent` is a fraction of the account value the ENGINE sees. Hand the
        # engine `equity x leverage` and a 60/40 book on equity mode's default 2x silently
        # becomes 120/80 — fully invested by the user's reckoning, 2x levered in fact.
        # Rescaling the weights by 1/leverage does not fix it either: the loan is a
        # constant while equity moves, so the ratio drifts and the book slowly
        # under-invests.
        #
        # So weights mean "fraction of your equity", exactly, at every point in the run.
        # Express leverage by making them sum above 1 — which is what `leverage_type=
        # "portfolio_level"` enforces on the weight matrix, per §J.
        if init_cash is not None:
            sim_kwargs["init_cash"] = float(np.sum(np.asarray(init_cash, dtype=float)))
            margin_loan = None
        _warn_if_weights_exceed_cash(weights, close)
        pf = Portfolio.from_weights(
            close,
            weights,
            schedule=_next_bar(schedule, close) if schedule is not None else None,
            price=fill,
            universe=universe,
            min_trade=min_trade,
            leverage_budget=leverage_budget,
            order_kwargs=order_kwargs,
            **sim_kwargs,
        ).replace(margin_loan=margin_loan)
    elif strategy == "signals":
        pf = Portfolio.from_signals(
            close,
            entries=_next_bar(entries, close),
            exits=_next_bar(exits, close),
            size=ctx["size"],
            open=fill,
            high=high, low=low,       # so stops are checked intrabar
            price=fill,               # fill at the open, not the close
            **order_kwargs,
            **ctx["stops"],
            **sim_kwargs,
        )
    elif ctx["size"] is not None:
        pf = Portfolio.from_orders(
            close, size=_entry_only(ctx["size"]), price=fill,
            **order_kwargs, **sim_kwargs
        )
    else:
        pf = Portfolio.from_holding(close, price=fill, **order_kwargs, **sim_kwargs)

    # Carry the universe so `pf.x.user_size()` can report in £/pt, lots or shares rather
    # than engine units — the whole reason a spread bettor uses this layer — and the loan
    # so reported equity stays the account's, not the buying power's.
    pf = pf.replace(universe=universe, margin_loan=margin_loan, high=high, low=low)

    # Holding costs are applied AFTER the simulation, from the realised positions, rather
    # than by replaying orders through `from_order_func`. That keeps every strategy type
    # supported — the replay path has no notion of entries and exits — and keeps the
    # level-3 dependency out of the default path entirely. See `apply_holding_costs` for
    # the trade-off, and use `Portfolio.from_holding_costs` when the in-run feedback
    # matters.
    if _wants_holding_costs(universe, rates):
        from ..holding.callback import apply_holding_costs

        pf = apply_holding_costs(
            pf, universe, rates=rates, markup=financing_markup,
            carry_markup=carry_markup, holidays=holidays,
        )
    return pf


def _wants_holding_costs(universe, rates) -> bool:
    """True when any instrument has financing or carry active and rates were supplied.

    Deliberately requires `rates`. Routing to `from_order_func` is the slowest and most
    fragile path in the library (it is the only level-3 one), so it is opt-in by the user
    actually supplying the data it needs rather than implied by a mode default — every
    mode's default stack contains financing or carry.
    """
    if rates is None:
        return False
    return any(
        {"financing", "carry"} & set(spec.filled().active_costs)
        for spec in universe.specs.values()
    )


def _warn_if_weights_exceed_cash(weights, close):
    """Warn when target weights imply leverage the rebalancing path cannot fund.

    Weights summing above 1 ask for more exposure than the account holds. Since this path
    is funded at equity, the excess is simply not filled — the orders are clipped, the
    backtest runs, and the book quietly sits below target.
    """
    total = pd.DataFrame(weights).abs().sum(axis=1).max()
    if float(total) > 1.0 + 1e-9:
        warnings.warn(
            f"Target weights sum to {float(total):.2f}, which is more exposure than the "
            f"account holds. Rebalancing is funded at equity, not at buying power, so the "
            f"excess will not fill and the book will sit below target. Scale the weights "
            f"to sum to 1, or use a signal-driven backtest, where leverage is funded.",
            stacklevel=3,
        )


def _next_bar(signal, close):
    """Shift a boolean signal one bar forward so it acts on the next open.

    A crossover seen at bar `t`'s close is only tradeable once bar `t+1` opens. Shifting
    here rather than asking the user to remember `.shift(1)` means the rule cannot be
    forgotten — and it is applied to the rebalance schedule for the same reason, since a
    target computed from today's close is executed tomorrow morning too.
    """
    if signal is None:
        return None
    if isinstance(signal, pd.Series) or np.ndim(signal) == 1:
        # A rebalance schedule is one mask for the whole portfolio, not one per column.
        series = pd.Series(np.asarray(signal).ravel(), index=close.index)
        return series.shift(1).fillna(False).astype(bool)
    frame = pd.DataFrame(signal).reindex(index=close.index, columns=close.columns)
    return frame.shift(1).fillna(False).astype(bool)


def _entry_only(size):
    """Place a hold's size on the first bar and NaN thereafter.

    A size repeated on every bar would re-order every bar. NaN means "no order", which is
    the same convention Section J relies on for off-schedule rebalance targets.
    """
    frame = pd.DataFrame(size).copy()
    out = pd.DataFrame(np.nan, index=frame.index, columns=frame.columns)
    out.iloc[0] = frame.iloc[0]
    return out
