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

from .prepare import BacktestError, SIZING_ARGS, _SPEC_FIELDS, prepare, strategy_of

#: Arguments this layer consumes itself. Anything else is forwarded to vectorbt, so the
#: full upstream surface stays reachable through the front door.
_FRONTEND_ARGS = frozenset(
    {
        "instruments", "entries", "exits", "short_entries", "short_exits",
        "weights", "schedule",
        "stop_points", "target_points", "trail_points",
        "stop_price", "target_price", "trail_price",
        "volume", "rates", "fx", "account_currency", "base_currency", "fx_markup",
        "min_trade", "leverage_budget", "holidays", "financing_markup", "carry_markup",
        "cash", "entry_priority", "when_cash_short", "risk_of",
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
    short_entries=None,
    short_exits=None,
    weights=None,
    schedule=None,
    # --- sizing (exactly one) ---
    risk_pct=None,
    risk_amount=None,
    stake=None,
    notional=None,
    size=None,
    notional_pct=None,
    risk_of=None,
    # --- stops, in points ---
    stop_points=None,
    target_points=None,
    trail_points=None,
    # --- stops, as price levels ---
    stop_price=None,
    target_price=None,
    trail_price=None,
    # --- data ---
    volume=None,
    rates=None,
    fx=None,
    account_currency=None,
    base_currency=None,
    fx_markup=None,
    # --- construction ---
    min_trade=None,
    leverage_budget=True,
    # --- holding costs ---
    holidays=None,
    financing_markup=None,
    carry_markup=None,
    # --- account ---
    init_cash=None,
    cash=None,
    entry_priority=None,
    when_cash_short=None,
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
        short_entries, short_exits: boolean short signal frames, alongside or instead of
            `entries`/`exits`. Shifted to the next bar exactly like them.
        weights, schedule: target weights and rebalance mask → rebalanced strategy.
        risk_pct, risk_amount, stake, notional, size, notional_pct: **exactly one.** Two
            raises. `risk_pct` and `risk_amount` size each entry to lose that risk at its
            stop, from whichever stop is given (`stop_points`, `sl_stop`, `stop_price`, or
            a trailing stop's first distance). With no stop, a position is worth the risk
            times its leverage: its margin is the risk, an implied stop 1 / leverage of the
            price away, and no stop is placed (a warning says so). `notional_pct` sizes
            each position to be worth that fraction of the account (`0.5` is half of it;
            above 1 is levered, funded up to the leverage); a number, or one per column.
        risk_of: what `risk_pct` or `notional_pct` is a fraction of: `"init_cash"` (the
            default, the same amount on every trade), `"balance"` (cash after closed
            trades, costs and holding costs so far) or `"equity"` (the balance plus open
            positions marked at the close). Read at the close of the signal bar, the last
            value known when the order is sent. Separate accounts read their own column; a
            shared account reads the account. See Notes.
        stop_points, target_points, trail_points: stops in points, converted internally
            to the native price fractions. Each takes a number, one value per column (a
            dict, or a list in column order), or a DataFrame per bar, aligned with the
            signals and shifted with them. A trailing stop in points stays that many
            points behind the best price.
        stop_price, target_price, trail_price: the stop or target as a price **level**
            (an EMA, a swing low), the same shapes, aligned with the signals and shifted
            with them: the level at the signal bar's close is where the stop or target
            sits from the fill on, whatever the gap to the next open. A level on the wrong
            side of the fill (a long's stop at or above it) is an order a broker rejects:
            the entry is not placed, and a warning counts them. `trail_price` starts at
            the signal bar's level and moves the stop to each completed bar's level,
            never loosening it (a chandelier exit). Each leg is given once: a level
            cannot sit beside the same leg in points or as `sl_stop` / `tp_stop`.
        volume: volume frame, required by `slippage="auto"`.
        rates: benchmark rate table (Section M), required by `financing="auto"`.
        fx: an `FXTable` for converting foreign-currency fixed costs.
        account_currency: the account currency.
        base_currency: with flat instrument terms describing an FX underlying
            (`asset_class="fx"` or `mode="fx_spot"`), the **pair's** base currency — the
            one carry needs. Otherwise the account currency, as before; prefer
            `account_currency=` for that, which is never ambiguous.
        fx_markup: the broker's FX conversion markup, a fraction charged as slippage on
            every fill of a converted column. None reads it off the price frames when
            they came from `to_base_currency`. Columns already quoted in the account
            currency pay none.
        min_trade: suppress rebalance trades below this weight change.
        leverage_budget: apply `leverage_type="portfolio_level"` scaling.
        holidays: dates carrying no holding-cost accrual.
        financing_markup: broker markup over the benchmark, always adverse. None reads
            `settings["vectorbtx"]["financing_markup"]`.
        carry_markup: markup on the carry differential, also adverse. None reads
            `settings["vectorbtx"]["carry_markup"]`; pass 0.0 to charge none.
        init_cash: account size.
        cash: `"per_instrument"` (each column trades its own copy of `init_cash`; the
            default for signals) or `"shared"` (one account across every column). See
            `vectorbtx.frontend.cash`.
        entry_priority: under `cash="shared"`, who is filled first when entries compete
            for cash: None (column order), a list of columns, a DataFrame of per-bar
            scores (higher first, aligned with the signals and shifted with them), or
            `"random"` (seeded by `seed=`).
        when_cash_short: `"skip"` (the default under shared cash) or `"partial"` — what an
            entry bigger than its leverage allows (a position worth more than `leverage x` the
            equity at its signal bar), or than the cash can cover, does: cut down to what is
            funded, rounded down to the size step, or not placed. Maps to vectorbt's
            `allow_partial` for the cash. An exit always closes the whole position.
        **kwargs: anything else, forwarded to the underlying vectorbt constructor.

    Returns:
        `vectorbtx.Portfolio`.

    Raises:
        BacktestError: on a missing or unknown `mode`, an argument a mode rejects, two
            sizing arguments, both signals and weights, or a passthrough colliding with a
            resolved cost.

    Notes:
        **Orders fill at the next bar's open, always.** A signal computed from bar `t`'s
        close cannot be acted on until bar `t+1` opens, so entries, exits, short entries
        and short exits are all shifted one bar and filled at `open`. There is no option
        to fill at the signal bar's close, because that is lookahead: it captures the
        overnight gap you could not have traded, and bills a night of financing you did
        not owe. A backtest that does it is not worth running, so the API does not offer
        it. Pass signals unshifted; shifting them yourself as well delays them two bars.

        Everything order-time — sizes, stop distances, spread fractions — is therefore
        priced at the **open you fill at**, not the close the signal fired on. Stops are
        anchored there too (`stop_entry_price="price"`, the order price before
        slippage, which is what point stops are converted against), and a stop or target
        exit pays the spread and slippage like any other fill
        (`stop_exit_price="stopmarket"`). Either can be overridden by passing it.

        **Stops are checked on the entry bar.** With a stop or target active, each bar is
        simulated as its open (where signals fill) plus the rest of the bar (where stops
        are checked against the high and low), so a stop the entry bar trades through is
        hit on that bar. See `vectorbtx.frontend.split_bars`.

        **A buy-and-hold with no size is funded at equity**, not buying power: holding
        "everything" means holding the account's own money once, not `leverage` times
        over. A sized hold (`stake=`, `size=`, ...) is still funded at buying power.

        **Weights asking for more cash than the account holds are scaled down pro rata**,
        with a warning, so every position gets the same fraction of its target.

        ⚠️ **FX conversion stays outside this call.** Signals are computed on
        base-currency prices and the user needs to hold that series anyway, so
        `to_base_currency` remains an explicit line in multi-currency backtests.
        Converting internally for sizing while signals ran on native prices would put two
        different price series in one backtest — worse than one visible line.

        **Stops given per bar are shifted with the signals**, `stop_points`, `target_points`
        and `trail_points` frames and vectorbt's own `sl_stop`, `tp_stop` and `sl_trail`
        frames alike: a stop worked out at the signal bar's close (`3 * atr`) is the one
        used from the fill. vectorbt alone reads a per-bar `sl_stop` at the fill bar, a bar
        late — lookahead. A scalar or per-column `sl_stop` passes through unchanged.

        **`risk_pct` and `notional_pct` size off `init_cash` by default**, the same amount
        on every trade. `risk_of="balance"` or `"equity"` sizes each entry from the account
        at its signal bar instead. Those depend on every fill, cost and stop before the entry, so the
        backtest is simulated again, sized from the previous run's account, until no
        entry's size changes. Each size depends only on bars before it, so this ends, and
        the result is exactly what sizing inside the simulation would give; it typically
        takes two to four runs, and one more for each entry vectorbt has to cut to the cash
        (`_settle`). Holding costs are charged after each run, so the balance and equity
        include them.

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

    account_currency = _resolve_currencies(
        account_currency, base_currency, spec_kwargs, instruments
    )

    if close is None:
        raise BacktestError(
            "`close` is required. All four price series are passed by keyword:\n"
            "  vbt.x.PortfolioBacktest(close=..., open=..., high=..., low=..., ...)"
        )
    raw_close = close
    close = pd.DataFrame(close)
    if open is None:
        raise BacktestError(
            "`open` is required. Orders fill at the next bar's open, because a signal from "
            "this bar's close cannot be acted on until the next bar opens. Filling at the "
            "signal bar's close is lookahead — it captures the overnight gap and skips a "
            "night of financing — so there is no option to do it.\n"
            "  vbt.x.PortfolioBacktest(data.get('Close'), data.get('Open'), ...)"
        )
    # ⚠️ Read the FX markup BEFORE coercing: `pd.DataFrame(...)` drops ConvertedFrame's
    # attributes, which is how the markup used to vanish and never be charged. Orders fill
    # at the open, so the open's markup is the one that applies.
    from ..fx.convert import converted_currency_of, fx_markup_of, fx_rate_of

    if fx_markup is None:
        fx_markup = fx_markup_of(open, raw_close)
    converted_to = converted_currency_of(open, raw_close)
    # ⚠️ And the conversion rate, so a point (a unit of the native price) is priced in the
    # account currency: stops, targets, trails and the spread (issue 29).
    point_fx = fx_rate_of(open, raw_close, index=close.index, columns=close.columns)
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

    # The cash model is settled before `prepare`, because whether a cash-short order is
    # skipped or partially filled changes what the cash-constraint warning has to say.
    from .cash import resolve_cash_model, shared_call_seq

    sim_kwargs = dict(kwargs)
    strategy = strategy_of(entries, exits, weights, schedule, short_entries, short_exits)
    cash = resolve_cash_model(
        cash,
        strategy,
        sized=any(v is not None for v in (risk_pct, risk_amount, stake, notional, size,
                                          notional_pct)),
        sim_kwargs=sim_kwargs,
        entry_priority=entry_priority,
        when_cash_short=when_cash_short,
    )
    risk_of = _check_risk_of(risk_of, risk_pct, notional_pct, init_cash)

    # ⚠️ ALL FOUR signal arrays shift, not just the long pair. Short signals used to arrive
    # in `**kwargs` and reach vectorbt unshifted, so every short filled at the open of the
    # bar whose close produced it: lookahead on every short. They are shifted once, here,
    # before `prepare`, so the sizing warnings can measure the bars orders are placed on.
    signals = None
    sides = None
    order_bars = None
    native_stops = {}
    if strategy == "signals":
        signals = dict(
            entries=_next_bar(entries, close),
            exits=_next_bar(exits, close),
            short_entries=_next_bar(short_entries, close),
            short_exits=_next_bar(short_exits, close),
        )
        sides = _opening_sides(signals, close, sim_kwargs.get("direction"))
        order_bars = sides["long"] | sides["short"]
        # vectorbt's own stops are resolved with the front door's, so a per-bar one is
        # shifted with the signals and risk sizing can read it (issue 26).
        native_stops = {name: sim_kwargs.pop(name) for name in ("sl_stop", "tp_stop", "sl_trail")
                        if name in sim_kwargs}
    elif strategy == "hold":
        # A sized hold places one order, on the first bar (`_entry_only`).
        order_bars = pd.DataFrame(False, index=close.index, columns=close.columns)
        order_bars.iloc[0] = True

    def prepared(risk_basis=None, leverage_basis=None, ceilings=None):
        """The whole resolution pipeline, for one account basis of `risk_pct`, one equity the
        leverage limit is measured against, and the entries' funded ceilings (see `_settle`)."""
        return prepare(
            close,
            fill_price=open,
            instruments=instruments,
            entries=entries, exits=exits, weights=weights, schedule=schedule,
            short_entries=short_entries, short_exits=short_exits,
            risk_pct=risk_pct, risk_amount=risk_amount, stake=stake,
            notional=notional, size=size, notional_pct=notional_pct,
            stop_points=stop_points, target_points=target_points, trail_points=trail_points,
            sl_stop=native_stops.get("sl_stop"), tp_stop=native_stops.get("tp_stop"),
            sl_trail=native_stops.get("sl_trail"),
            stop_price=stop_price, target_price=target_price, trail_price=trail_price,
            long_open=sides["long"] if sides else None,
            short_open=sides["short"] if sides else None,
            volume=volume, rates=rates, fx=fx, base_currency=account_currency,
            fx_markup=fx_markup, fx_markup_currency=converted_to, point_fx=point_fx,
            init_cash=init_cash,
            allow_partial=bool(sim_kwargs.get("allow_partial", True)),
            risk_basis=risk_basis,
            order_bars=order_bars,
            leverage_basis=leverage_basis,
            ceilings=ceilings,
            spec_kwargs=spec_kwargs,
            passthrough=passthrough,
        )

    def simulated(ctx):
        """Simulate one resolved context and charge its holding costs."""
        from ..core.portfolio import Portfolio

        kw = dict(sim_kwargs)
        universe = ctx["universe"]
        price = ctx["price"]
        fill = ctx["fill_price"]
        order_kwargs = ctx["order_kwargs"]

        margin_loan = None
        engine_cash = None
        if init_cash is not None:
            # Fund the leverage rather than only capping it. `buy_nb` bounds every long by
            # cash, so a levered position is unreachable unless the engine is handed the
            # buying power; `Portfolio.get_init_cash` subtracts the loan back out so
            # reported equity, returns and every metric stay on the account's real size.
            from ..margin.buying_power import buying_power

            # `from_weights` turns cash sharing on itself (Section J: it is load-bearing
            # for `call_seq="auto"`), so the basis has to be decided here rather than read
            # off whatever the caller happened to pass.
            shares_cash = bool(kw.get("cash_sharing", strategy == "weights"))
            engine_cash, margin_loan = buying_power(
                universe, init_cash, ctx["columns"], cash_sharing=shares_cash,
            )
            kw["init_cash"] = engine_cash

        if strategy == "weights":
            if weights is None:
                raise BacktestError("`schedule` was given without `weights`; there is nothing to target.")
            # ⚠️ The rebalancing path is funded at EQUITY, never at buying power.
            #
            # `TargetPercent` is a fraction of the account value the ENGINE sees. Hand the
            # engine `equity x leverage` and a 60/40 book on a 2x margin account silently
            # becomes 120/80 — fully invested by the user's reckoning, 2x levered in fact.
            # Rescaling the weights by 1/leverage does not fix it either: the loan is a
            # constant while equity moves, so the ratio drifts and the book slowly
            # under-invests.
            #
            # So weights mean "fraction of your equity", exactly, at every point in the
            # run, and a book whose longs need more cash than that equity cannot fill.
            # Rather than let vectorbt starve the last columns, `_fit_weights_to_cash`
            # scales those rows pro rata (with a warning). Leverage belongs to the signal
            # path, where it is funded.
            if init_cash is not None:
                kw["init_cash"] = float(np.sum(np.asarray(init_cash, dtype=float)))
                margin_loan = None
            fitted = _fit_weights_to_cash(weights, order_kwargs.get("lock_cash"))
            pf = Portfolio.from_weights(
                price,
                fitted,
                schedule=_next_bar(schedule, price) if schedule is not None else None,
                price=fill,
                universe=universe,
                min_trade=min_trade,
                leverage_budget=leverage_budget,
                order_kwargs=order_kwargs,
                **kw,
            ).replace(margin_loan=margin_loan)
        elif strategy == "signals":
            # Point stops were converted to fractions of the fill price (the open), so they
            # must be anchored there. vectorbt's default anchor is the entry bar's CLOSE: a
            # buy filled at 100 on a bar closing at 103 had its 2-point stop at 100.94,
            # above the entry. "price" is the order price before slippage — exactly what
            # the points were converted against ("fillprice" would anchor after the spread).
            kw.setdefault("stop_entry_price", "price")
            # vectorbt's default "stoplimit" applies NO slippage to a stop fill, and the
            # spread lives in `slippage`, so every stop and target exit paid no spread.
            # "stopmarket" charges spread plus slippage like any other fill. A take-profit
            # is a limit order and arguably should pay the spread but not slippage; one
            # `slippage` field cannot separate the two, so it pays both — the conservative
            # side of the error.
            kw.setdefault("stop_exit_price", "stopmarket")

            # ⚠️ With a stop active, every bar is simulated as its open plus the rest of
            # the bar, so a stop is checked on the bar it was entered on. Whole bars leave
            # the entry bar's range unchecked — always in the strategy's favour, and worst
            # for tight stops on daily bars. See `split_bars`.
            from .split_bars import can_split, simulate_split_bars, stops_active

            split = stops_active(ctx["stops"], kw)
            if split and not can_split(kw):
                split = False
                warnings.warn(
                    "A custom signal_func_nb, or templates in a stop-adjustment callback's "
                    "arguments, were given, so bars cannot be split into the open and the "
                    "rest of the bar, and stops are NOT checked on the bar a position is "
                    "entered: a stop the entry bar trades through is missed. That flatters "
                    "the result, most for tight stops on daily bars. Pass the arrays a "
                    "callback needs as plain per-bar arrays instead of templates to avoid it.",
                    stacklevel=3,
                )
            # An entry whose stop or target level is on the wrong side of its fill is an
            # order a broker rejects, so it is not placed.
            placing = _drop_rejected(signals, sides, ctx["rejected"], price)
            trailing = [name for name, key in (("trail_points", "trail_distance"),
                                               ("trail_price", "trail_level"))
                        if ctx[key] is not None]
            if trailing:
                # A trailing stop in points keeps its distance behind the best price
                # rather than becoming a fraction of the fill (issue 27), and one at a level
                # ratchets to each completed bar's level. Each is a stop-adjustment callback
                # of its own, so it cannot share the slot with one.
                from .split_bars import _custom
                from .trail import trail_level_nb, trail_points_nb

                if _custom(kw.get("adjust_sl_func_nb")):
                    raise BacktestError(
                        f"`{trailing[0]}` moves the trailing stop with a stop-adjustment "
                        f"callback of its own, so it cannot be combined with "
                        f"`adjust_sl_func_nb`. Trail inside your callback, or use a trailing "
                        f"stop in % of the price (`sl_stop=` with `sl_trail=True`)."
                    )
                steps = 2 if split else 1
                if ctx["trail_distance"] is not None:
                    kw["adjust_sl_func_nb"] = trail_points_nb
                    kw["adjust_sl_args"] = (
                        np.ascontiguousarray(ctx["trail_distance"].to_numpy(dtype=np.float64)),
                        steps,
                    )
                else:
                    n_cols = len(ctx["columns"])
                    kw["adjust_sl_func_nb"] = trail_level_nb
                    kw["adjust_sl_args"] = (
                        np.ascontiguousarray(ctx["trail_level"].to_numpy(dtype=np.float64)),
                        steps,
                        np.full(n_cols, np.nan),                 # each position's stop so far
                        np.full(n_cols, -1, dtype=np.int64),     # the entry it belongs to
                    )
            if cash == "shared" and "call_seq" not in kw:
                kw["call_seq"] = shared_call_seq(
                    price.index, ctx["columns"], **placing,
                    entry_priority=_next_bar_scores(entry_priority, price),
                    seed=kw.get("seed"),
                    direction=kw.get("direction"),
                    split_bars=split,
                )
            if split:
                pf = simulate_split_bars(
                    Portfolio, price, fill, high, low, price=fill, signals=placing,
                    call_seq=kw.pop("call_seq", None),
                    size=ctx["size"],
                    **order_kwargs,
                    **ctx["stops"],
                    **kw,
                )
            else:
                pf = Portfolio.from_signals(
                    price,
                    **placing,
                    size=ctx["size"],
                    open=fill,
                    high=high, low=low,       # so stops are checked intrabar
                    price=fill,               # fill at the open, not the close
                    **order_kwargs,
                    **ctx["stops"],
                    **kw,
                )
        elif ctx["size"] is not None:
            pf = Portfolio.from_orders(
                price, size=_entry_only(ctx["size"]), price=fill,
                **order_kwargs, **kw
            )
        else:
            # ⚠️ An unsized hold is funded at EQUITY, like the weights branch. It buys with
            # all the cash it has, so handing it buying power held `leverage` times the
            # account: a "buy and hold" on a 30x spread bet carried 30x its equity and lost
            # 247% while the market fell 6.5%.
            if engine_cash is not None:
                kw["init_cash"] = np.asarray(engine_cash, dtype=float) - np.asarray(
                    margin_loan, dtype=float
                )
                margin_loan = None
            pf = Portfolio.from_holding(price, price=fill, **order_kwargs, **kw)
            if init_cash is not None:
                _warn_if_hold_did_not_trade(pf, ctx["columns"])

        # Carry the universe so `pf.x.user_size()` can report in £/pt, lots or shares
        # rather than engine units — the whole reason a spread bettor uses this layer — and
        # the loan so reported equity stays the account's, not the buying power's.
        pf = pf.replace(universe=universe, margin_loan=margin_loan, high=high, low=low)

        # Holding costs are applied AFTER the simulation, from the realised positions,
        # rather than by replaying orders through `from_order_func`. That keeps every
        # strategy type supported — the replay path has no notion of entries and exits —
        # and keeps the level-3 dependency out of the default path entirely. See
        # `apply_holding_costs` for the trade-off, and use `Portfolio.from_holding_costs`
        # when the in-run feedback matters.
        if _wants_holding_costs(universe, rates):
            from ..holding.callback import apply_holding_costs

            pf = apply_holding_costs(
                pf, universe, rates=rates, markup=financing_markup,
                carry_markup=carry_markup, holidays=holidays,
            )
        else:
            # No holding cost is computed, so nothing else would notice a short share held
            # with no borrow rate. It must not borrow for free in silence (G6).
            from ..holding.borrow import warn_unpriced_shorts

            warn_unpriced_shorts(universe, pd.DataFrame(pf.assets()), list(pf.wrapper.columns),
                                 stacklevel=5)
        return pf

    sized = any(v is not None for v in (risk_pct, risk_amount, stake, notional, size, notional_pct))
    if strategy != "signals" or not sized or init_cash is None:
        return simulated(prepared())
    # Every sized rule strategy settles its entries against the account as the run goes: the
    # leverage limit against the equity at each signal bar (issue 33), a share of the balance or
    # equity (`risk_of`), and an entry vectorbt had to cut to the cash re-asked at what was
    # funded, on the size step (issue 34). A run where none of that binds is simulated once.
    return _settle(prepared, simulated, risk_of, order_bars, bool(sim_kwargs.get("allow_partial", True)))


def _wants_holding_costs(universe, rates) -> bool:
    """True when any holding cost can be charged: financing or carry with `rates`
    supplied, or a stock-borrow fee with a rate set.

    Financing and carry deliberately require `rates`: they are priced off the benchmark
    table, and every mode's default stack contains one of them, so they are opt-in by the
    user supplying the data rather than implied by a mode default. The borrow fee needs
    no benchmark; its rate is the instrument's own `borrow_rate`, so setting one is the
    opt-in.
    """
    for spec in universe.specs.values():
        filled = spec.filled()
        if rates is not None and {"financing", "carry"} & set(filled.active_costs):
            return True
        if "borrow" in filled.active_costs and filled.borrow_rate:
            return True
    return False


#: What `risk_pct` may be a fraction of.
RISK_BASES = ("init_cash", "balance", "equity")


def _check_risk_of(risk_of, risk_pct, notional_pct, init_cash) -> str:
    """Validate `risk_of`; None means `"init_cash"`."""
    if risk_of is None:
        return "init_cash"
    if risk_of not in RISK_BASES:
        raise BacktestError(f"`risk_of` must be one of {list(RISK_BASES)}, got {risk_of!r}.")
    if risk_of != "init_cash":
        if risk_pct is None and notional_pct is None:
            raise BacktestError(
                "`risk_of` says what `risk_pct` or `notional_pct` is a fraction of, so it "
                "needs one of them. `risk_amount`, `stake`, `notional` and `size` are fixed "
                "on every trade and have no basis."
            )
        if init_cash is None:
            raise BacktestError("`risk_of` needs `init_cash=`, the account the run starts from.")
    return risk_of


def _signal_mask(signal, close) -> np.ndarray:
    """A shifted signal as an `(n_bars, n_cols)` boolean array; None is all False."""
    if signal is None:
        return np.zeros(close.shape, dtype=bool)
    if isinstance(signal, pd.Series) or np.ndim(signal) == 1:
        row = np.asarray(signal, dtype=bool).ravel()
        return np.repeat(row[:, None], close.shape[1], axis=1)
    frame = pd.DataFrame(signal).reindex(index=close.index, columns=close.columns)
    return frame.fillna(False).to_numpy(dtype=bool)


def _opening_sides(signals, close, direction) -> dict:
    """The bars long and short entries fill on, and which signal array opens each side.

    With the short arrays, `entries` open longs and `short_entries` shorts. Without them,
    `direction` decides: `entries` open longs (`"longonly"`) or shorts (`"shortonly"`), and
    under `"both"` `exits` open the shorts. An unknown or per-column `direction` is read as
    `"both"`: a bar measured too many is the safe side for a warning.

    Returns:
        dict with boolean frames `long` and `short`, and `long_key` / `short_key`, the
        signal array opening each side (None for a side nothing opens).
    """
    from .cash import _direction_mode

    if signals["short_entries"] is not None or signals["short_exits"] is not None:
        long_key, short_key = "entries", "short_entries"
    else:
        try:
            mode = _direction_mode(direction)
        except BacktestError:
            mode = "both"
        long_key, short_key = {"longonly": ("entries", None), "shortonly": (None, "entries"),
                               "both": ("entries", "exits")}[mode]
    out = dict(long_key=long_key, short_key=short_key)
    for side, key in (("long", long_key), ("short", short_key)):
        mask = _signal_mask(signals[key], close) if key else np.zeros(close.shape, dtype=bool)
        out[side] = pd.DataFrame(mask, index=close.index, columns=close.columns)
    return out


def _drop_rejected(signals, sides, rejected, close) -> dict:
    """The shifted signals without the entries whose stop or target level was rejected."""
    if rejected is None or not bool(rejected.to_numpy().any()):
        return signals
    out = dict(signals)
    for side, key in (("long", sides["long_key"]), ("short", sides["short_key"])):
        if key is None:
            continue
        drop = rejected.to_numpy(dtype=bool) & sides[side].to_numpy(dtype=bool)
        kept = _signal_mask(out[key], close) & ~drop
        out[key] = pd.DataFrame(kept, index=close.index, columns=close.columns)
    return out


def _settle(prepared, simulated, risk_of, order_bars, partial):
    """Settle every entry against the account as the run goes, then return the backtest.

    Three things about an entry depend on every fill, cost and stop before it, so none can be
    worked out before the simulation, and vectorbt's signal callback cannot see the cash:

    * its size, when it is a share of the balance or equity (`risk_of`);
    * its leverage limit, a position worth at most `leverage x` the equity at the signal bar
      (OpenStrategy Studio issue 33; `cap_to_leverage`);
    * what an account can pay for, when vectorbt had to cut an entry to the cash (a levered
      account's loan is fixed at the start, a shared account's cash is spoken for), which it
      does off the size step: such an entry is asked again at what was funded, rounded down to
      the size step, and kept there while the account at its signal bar stays the same
      (issue 34; `_cash_ceilings`).

    So the backtest is run, each entry is settled from that run's account at its signal bar,
    and it is run again, until no entry's size changes. A run where nothing binds (a size from
    the starting cash, far inside its leverage, never short of cash) is simulated once.

    It ends, and exactly: each of the three depends only on the account at the bar before the
    entry, so once the first k entries are settled, the account entry k + 1 sees is final, and
    it settles within two runs: one that asks it afresh on that account (vectorbt cuts it, if
    the cash is short, to what is funded) and one that asks it at that ceiling. A settled entry
    is never undone. So there are at most two runs per entry bar, plus two; one per entry cut
    to the cash in practice, and a single run when nothing binds.

    ⚠️ Only the ceilings up to each account's first change are applied on the next run
    (`_current_ceilings`). An entry after it will see another account, and asked at its old
    ceiling (rounded down) it is mostly filled in full, which says nothing of what the new
    account funds; asking it afresh there and cutting it a run later took two runs for every
    entry cut to the cash, and a long levered run that made money gave up after one run per
    entry bar (OpenStrategy Studio issue 36).

    Warnings are recorded on each step and said once, about the backtest returned: the
    simulation's from the run returned, and the sizing's from the last preparation, the one
    that read the run returned and found its sizes settled. ⚠️ Not the preparation that made
    those sizes: it read the run before, whose account can be far from the one returned (the
    first run sizes every entry from the starting cash, and after heavy losses its account can
    be below zero), so its words named an account the backtest never had and could leave out
    the real reason an entry was not placed (OpenStrategy Studio issue 35).
    """
    placed = order_bars.to_numpy(dtype=bool)
    # A bound, by the argument above, not a guess: two runs per entry bar, plus two.
    limit = 2 * int(placed.any(axis=1).sum()) + 2
    kept = {}

    def recorded(step, *args):
        """One step, and the warnings it raised, kept apart from every other step's."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = step(*args)
        return out, list(caught)

    ctx, sizing_said = recorded(prepared)
    pf, simulation_said = recorded(simulated, ctx)

    for _ in range(limit):
        equity = account_basis(pf, "equity")
        basis = None if risk_of == "init_cash" else (equity if risk_of == "equity" else account_basis(pf, risk_of))
        ceilings = _cash_ceilings(pf, ctx["size"], placed, equity, kept) if partial else None
        # Read from `pf`'s own account: when the sizes are settled, these are the words about it.
        following, sizing_said = recorded(prepared, basis, equity, ceilings)
        if _same_sizes(following["size"], ctx["size"], placed):
            break
        if ceilings is not None:
            current = _current_ceilings(ceilings, following["size"], ctx["size"], placed, pf)
            if not np.array_equal(np.isnan(current.to_numpy()), np.isnan(ceilings.to_numpy())):
                # Some entries after the first change are asked afresh instead (see above).
                following, sizing_said = recorded(prepared, basis, equity, current)
        ctx = following
        pf, simulation_said = recorded(simulated, ctx)
    else:  # pragma: no cover - impossible by the argument above
        # A BacktestError, so a caller that records a run it cannot make (an optimiser's trial)
        # records this one too rather than stopping.
        raise BacktestError(
            f"The entries did not settle after {limit} runs. Please report this: each one "
            f"depends only on earlier bars, so they should."
        )

    for item in sizing_said + simulation_said:
        warnings.warn_explicit(item.message, item.category, item.filename, item.lineno)
    return pf


def _opened_sizes(pf) -> np.ndarray:
    """The size each order opened, per bar and column, in engine units; NaN where none opened.

    An order from flat opens all of itself; one that reverses a position opens what is left
    after the close. Read from the order records in each column's own sequence, so an entry
    stopped out later in its own bar (split bars) still counts what it opened.
    """
    records = pf.orders.values
    n, m = len(pf.wrapper.index), len(pf.wrapper.columns)
    out = np.full((n, m), np.nan)
    position = np.zeros(m)
    for r in np.lexsort((np.arange(len(records)), records["idx"], records["col"])):
        col, i = int(records["col"][r]), int(records["idx"][r])
        q = records["size"][r] if records["side"][r] == 0 else -records["size"][r]
        before, after = position[col], position[col] + q
        if abs(after) < 1e-12:
            after = 0.0
        if after != 0.0 and (before == 0.0 or np.sign(after) != np.sign(before)):
            out[i, col] = abs(after)
        position[col] = after
    return out


def _cash_ceilings(pf, asked, placed, equity, kept) -> pd.DataFrame:
    """The most each entry may be asked for on the next run, as this run found it funded.

    An entry that opened less than it asked was cut by vectorbt to the cash, off the size step:
    its ceiling is what it opened (`prepare` rounds it down to the step). A ceiling found on an
    earlier run is kept while the equity at that entry's signal bar is unchanged, because the
    entry then sees the same account; when the account before it has changed, it is asked
    afresh. `kept` carries the ceilings from run to run: {(bar, column): (ceiling, equity)}.
    """
    opened = _opened_sizes(pf)
    wanted = np.abs(pd.DataFrame(asked).to_numpy(dtype=float))
    account = pd.DataFrame(equity).to_numpy(dtype=float)
    out = np.full(wanted.shape, np.nan)
    for (i, k), (roof, seen) in list(kept.items()):
        if np.isclose(account[i, k], seen, rtol=1e-12, atol=0.0):
            out[i, k] = roof
        else:
            del kept[(i, k)]
    with np.errstate(invalid="ignore"):
        cut = placed & np.isfinite(opened) & np.isfinite(wanted) & (opened < wanted * (1.0 - 1e-9))
    for i, k in zip(*np.nonzero(cut)):
        out[i, k] = opened[i, k] if np.isnan(out[i, k]) else min(out[i, k], opened[i, k])
        kept[(int(i), int(k))] = (out[i, k], account[i, k])
    return pd.DataFrame(out, index=pf.wrapper.index, columns=pf.wrapper.columns)


def _current_ceilings(ceilings, new, old, placed, pf) -> pd.DataFrame:
    """The ceilings to ask at on the next run: each account's up to its first entry that changes.

    A ceiling is what an entry was funded on this run's account. On the next run the entry sees
    that account again only if no entry before it, in its account, changes size; after the
    first one that does (`new` against `old`, on the bars an order may be placed), it sees
    another, so its ceiling is left out and it is asked afresh (OpenStrategy Studio issue 36;
    `_settle`). Separate accounts are columns; a shared account (`cash_sharing` with a group)
    is its group, whose first change leaves out the ceilings of every column after it. Entries
    on the bar of the first change keep theirs.
    """
    with np.errstate(invalid="ignore"):
        changed = placed & ~np.isclose(pd.DataFrame(new).to_numpy(dtype=float),
                                       pd.DataFrame(old).to_numpy(dtype=float),
                                       rtol=1e-12, atol=0.0, equal_nan=True)
    roof = ceilings.to_numpy(dtype=float).copy()
    columns = len(pf.wrapper.columns)
    accounts = np.asarray(pf.wrapper.grouper.get_groups()) if pf.cash_sharing else np.arange(columns)
    for account in np.unique(accounts):
        members = np.nonzero(accounts == account)[0]
        rows = np.nonzero(changed[:, members].any(axis=1))[0]
        if len(rows):
            roof[rows[0] + 1:, members] = np.nan
    return pd.DataFrame(roof, index=ceilings.index, columns=ceilings.columns)


def _same_sizes(new, old, placed) -> bool:
    """Whether two size frames agree on every bar an order may be placed on."""
    a = pd.DataFrame(new).to_numpy(dtype=float)[placed]
    b = pd.DataFrame(old).to_numpy(dtype=float)[placed]
    return bool(np.all(np.isclose(a, b, rtol=1e-12, atol=0.0, equal_nan=True)))


def account_basis(pf, risk_of) -> pd.DataFrame:
    """The balance or equity each column's orders are sized from, per fill bar.

    Args:
        pf: a front-door `Portfolio`, holding costs charged.
        risk_of: `"balance"` or `"equity"`.

    Returns:
        DataFrame on the portfolio's index and columns. Row `t` is the account at bar
        `t - 1`'s close, the last value known when an order filling at bar `t`'s open was
        sent; row 0 is the starting equity.

    Notes:
        **Equity** is the account's value at the close: cash plus open positions marked
        at the close, holding costs included (`pf.value()`, which nets out a margin loan).
        **Balance** is the equity less the open positions' profit or loss, which is the
        cash plus what the open positions cost: starting cash, plus closed trades' P&L,
        less every cost and holding cost charged so far. A commission is out of the
        balance when it is paid; a spread sits in the fill price, so it is in the open
        P&L until the position closes.

        Separate accounts read their own column; a shared account (`cash_sharing` with a
        group) reads the account, so every column in it sizes from the same figure.
    """
    columns = list(pf.wrapper.columns)
    index = pf.wrapper.index
    assets = pd.DataFrame(pf.assets(), index=index, columns=columns)
    close = pd.DataFrame(pf.close, index=index, columns=columns)
    open_pnl = assets * close - _cost_basis(pf, index, columns)

    if pf.cash_sharing:
        groups = np.asarray(pf.wrapper.grouper.get_groups())
        value = pd.DataFrame(pf.value())
        start = np.atleast_1d(np.asarray(pf.get_init_cash(), dtype=float))
        out = {}
        for k, col in enumerate(columns):
            members = [columns[j] for j in range(len(columns)) if groups[j] == groups[k]]
            account = value.iloc[:, groups[k]].to_numpy(dtype=float)
            if risk_of == "balance":
                account = account - open_pnl[members].sum(axis=1).to_numpy(dtype=float)
            out[col] = np.concatenate([[start[groups[k]]], account[:-1]])
        return pd.DataFrame(out, index=index)

    value = pd.DataFrame(pf.value(group_by=False), index=index, columns=columns)
    if risk_of == "balance":
        value = value - open_pnl
    start = np.broadcast_to(np.asarray(pf.get_init_cash(group_by=False), dtype=float),
                            (len(columns),))
    shifted = value.shift(1)
    shifted.iloc[0] = start
    return shifted


def _cost_basis(pf, index, columns) -> pd.DataFrame:
    """What each column's open position cost, at each bar's close: size x average entry.

    Zero when flat. Fees are not in it (they left the cash when paid), so cash plus the
    cost basis is the balance.
    """
    records = pf.orders.values
    signed = np.where(records["side"] == 0, records["size"], -records["size"])
    order = np.lexsort((np.arange(len(records)), records["idx"], records["col"]))
    out = np.full((len(index), len(columns)), np.nan)
    position = np.zeros(len(columns))
    basis = np.zeros(len(columns))
    for r in order:
        col, i, q, p = int(records["col"][r]), int(records["idx"][r]), signed[r], records["price"][r]
        before = position[col]
        after = before + q
        if abs(after) < 1e-12:
            after, cost = 0.0, 0.0
        elif before == 0 or np.sign(q) == np.sign(before):
            cost = basis[col] + q * p                  # adding: the new units at their price
        elif np.sign(after) == np.sign(before):
            cost = basis[col] * after / before         # reducing: the average stays
        else:
            cost = after * p                           # reversed: the new side at this price
        position[col], basis[col] = after, cost
        out[i, col] = cost
    frame = pd.DataFrame(out, index=index, columns=columns).ffill().fillna(0.0)
    return frame


def _resolve_currencies(account_currency, base_currency, spec_kwargs, instruments):
    """Decide what a flat `base_currency` means, and return the account currency.

    ⚠️ `base_currency` names two different things. On an FX instrument it is the pair's
    base — the leg carry needs. On the front door it was the account currency. A flat
    `SingleBacktest(asset_class="fx", base_currency="EUR", quote_currency="USD", ...)`
    therefore set the ACCOUNT to EUR and left the pair with one leg, so a long EURUSD was
    charged the whole USD rate as carry rather than the EUR/USD differential.

    So a flat `base_currency` goes to the instrument whenever the flat terms describe an
    FX underlying (`asset_class="fx"` or `mode="fx_spot"`), which is the only reading under
    which writing it next to `quote_currency` makes sense. Anywhere else it is still the
    account currency, as before. `account_currency=` names the account unambiguously.
    """
    flat_terms_apply = instruments is None or isinstance(instruments, dict)
    describes_fx = spec_kwargs.get("asset_class") == "fx" or spec_kwargs.get("mode") == "fx_spot"
    if base_currency is not None and flat_terms_apply and describes_fx:
        spec_kwargs["base_currency"] = base_currency
        return account_currency

    if (account_currency is not None and base_currency is not None
            and account_currency != base_currency):
        raise BacktestError(
            f"`account_currency={account_currency!r}` and `base_currency={base_currency!r}` "
            f"both name the account currency here (the instrument terms do not describe an "
            f"FX pair). Give one: `account_currency=`."
        )
    return account_currency if account_currency is not None else base_currency


def _fit_weights_to_cash(weights, lock_cash):
    """Scale target weights pro rata where they ask for more cash than the account holds.

    The rebalancing path is funded at equity, so a book whose longs need more than the
    account's cash cannot fill. vectorbt fills orders in column order until the cash runs
    out, so targets of 0.75 / 0.75 became a book of 0.75 / 0.25: the first column in full
    and the last starved, while the old warning implied an even shortfall. Scaling the
    row instead gives every position the same fraction of its target: 0.50 / 0.50.

    What a row needs from the cash depends on `lock_cash`. Without it a short's proceeds
    are cash the longs can spend, so the constraint is the NET long weight; with it a
    short ties up cash too, so it is the GROSS weight. Rows within the constraint, and
    every row of a book that fits, are left exactly as given.
    """
    w = pd.DataFrame(weights)
    numeric = w.astype(float)
    longs = numeric.clip(lower=0.0).sum(axis=1)
    shorts = -numeric.clip(upper=0.0).sum(axis=1)
    locks = bool(np.any(np.asarray(lock_cash if lock_cash is not None else False)))
    need = longs + shorts if locks else longs - shorts

    over = need > 1.0 + 1e-9
    if not bool(over.any()):
        return weights
    worst = float(need[over].max())
    warnings.warn(
        f"Target weights ask for up to {worst:.2f}x the account's equity in cash on "
        f"{int(over.sum())} bar(s), which is more exposure than the account holds. "
        f"Rebalancing is funded at equity, not at buying power, so those bars' weights "
        f"were scaled down pro rata to fit (targets of 0.75 / 0.75 become 0.50 / 0.50): "
        f"every position gets the same fraction of its target, rather than the first "
        f"columns filling in full and the last being starved. Scale the weights yourself "
        f"to silence this, or use a signal-driven backtest, where leverage is funded.",
        stacklevel=3,
    )
    scale = (1.0 / need).where(over, 1.0)
    return numeric.mul(scale, axis=0)


def _warn_if_hold_did_not_trade(pf, columns):
    """Warn when an unsized hold placed no order at all.

    A hold is funded at equity, so a small account may not reach an instrument's minimum
    size — £10,000 cannot buy £1/pt of EURUSD (about £11,000 of exposure) at 1x. The
    backtest still runs and reports a flat line, which reads as a market that went nowhere.
    """
    traded = set(np.asarray(pf.orders.values["col"]).tolist())
    idle = [col for i, col in enumerate(columns) if i not in traded]
    if idle:
        warnings.warn(
            f"The buy-and-hold placed no order for {idle}. A hold with no size is funded at "
            f"the account's equity (1x), not its buying power, and that equity does not "
            f"reach the instrument's minimum size, so these columns sit in cash. Give a "
            f"size (`stake=`, `notional=`, ...), which is funded at buying power, or raise "
            f"`init_cash`.",
            stacklevel=3,
        )


def _next_bar_scores(entry_priority, close):
    """Shift a per-bar priority score with the signals it ranks.

    A score computed at bar `t`'s close ranks the entry signal from that bar, which fills
    at bar `t+1`'s open, so the score moves with it. Anything other than a DataFrame is
    not per-bar and passes through untouched.
    """
    if not isinstance(entry_priority, pd.DataFrame):
        return entry_priority
    frame = entry_priority.reindex(index=close.index)
    return frame.shift(1)


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
