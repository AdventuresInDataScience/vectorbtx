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
        "volume", "rates", "fx", "account_currency", "base_currency", "fx_markup",
        "min_trade", "leverage_budget", "holidays", "financing_markup", "carry_markup",
        "cash", "entry_priority", "when_cash_short",
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
    # --- stops, in points ---
    stop_points=None,
    target_points=None,
    trail_points=None,
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
        risk_pct, risk_amount, stake, notional, size: **exactly one.** Two raises.
        stop_points, target_points, trail_points: stops in points, converted internally
            to the native price fractions.
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
            order the cash cannot cover does. Maps to vectorbt's `allow_partial`.
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
    from ..fx.convert import converted_currency_of, fx_markup_of

    if fx_markup is None:
        fx_markup = fx_markup_of(open, raw_close)
    converted_to = converted_currency_of(open, raw_close)
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
    cash = resolve_cash_model(
        cash,
        strategy_of(entries, exits, weights, schedule, short_entries, short_exits),
        sized=any(v is not None for v in (risk_pct, risk_amount, stake, notional, size)),
        sim_kwargs=sim_kwargs,
        entry_priority=entry_priority,
        when_cash_short=when_cash_short,
    )

    ctx = prepare(
        close,
        fill_price=open,
        instruments=instruments,
        entries=entries, exits=exits, weights=weights, schedule=schedule,
        short_entries=short_entries, short_exits=short_exits,
        risk_pct=risk_pct, risk_amount=risk_amount, stake=stake,
        notional=notional, size=size,
        stop_points=stop_points, target_points=target_points, trail_points=trail_points,
        volume=volume, rates=rates, fx=fx, base_currency=account_currency,
        fx_markup=fx_markup, fx_markup_currency=converted_to,
        init_cash=init_cash,
        allow_partial=bool(sim_kwargs.get("allow_partial", True)),
        spec_kwargs=spec_kwargs,
        passthrough=passthrough,
    )

    from ..core.portfolio import Portfolio

    universe = ctx["universe"]
    close = ctx["price"]
    fill = ctx["fill_price"]
    order_kwargs = ctx["order_kwargs"]

    margin_loan = None
    engine_cash = None
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
        # So weights mean "fraction of your equity", exactly, at every point in the run,
        # and a book whose longs need more cash than that equity cannot fill. Rather than
        # let vectorbt starve the last columns, `_fit_weights_to_cash` scales those rows
        # pro rata (with a warning). Leverage belongs to the signal path, where it is funded.
        if init_cash is not None:
            sim_kwargs["init_cash"] = float(np.sum(np.asarray(init_cash, dtype=float)))
            margin_loan = None
        weights = _fit_weights_to_cash(weights, order_kwargs.get("lock_cash"))
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
        # ⚠️ ALL FOUR signal arrays shift, not just the long pair. Short signals used to
        # arrive in `**kwargs` and reach vectorbt unshifted, so every short filled at the
        # open of the bar whose close produced it: lookahead on every short.
        signals = dict(
            entries=_next_bar(entries, close),
            exits=_next_bar(exits, close),
            short_entries=_next_bar(short_entries, close),
            short_exits=_next_bar(short_exits, close),
        )
        # Point stops were converted to fractions of the fill price (the open), so they
        # must be anchored there. vectorbt's default anchor is the entry bar's CLOSE: a buy
        # filled at 100 on a bar closing at 103 had its 2-point stop at 100.94, above the
        # entry. "price" is the order price before slippage — exactly what the points were
        # converted against ("fillprice" would anchor after the spread).
        sim_kwargs.setdefault("stop_entry_price", "price")
        # vectorbt's default "stoplimit" applies NO slippage to a stop fill, and the spread
        # lives in `slippage`, so every stop and target exit paid no spread. "stopmarket"
        # charges spread plus slippage like any other fill. A take-profit is a limit order
        # and arguably should pay the spread but not slippage; one `slippage` field cannot
        # separate the two, so it pays both — the conservative side of the error.
        sim_kwargs.setdefault("stop_exit_price", "stopmarket")

        # ⚠️ With a stop active, every bar is simulated as its open plus the rest of the
        # bar, so a stop is checked on the bar it was entered on. Whole bars leave the
        # entry bar's range unchecked — always in the strategy's favour, and worst for
        # tight stops on daily bars. See `split_bars`.
        from .split_bars import can_split, simulate_split_bars, stops_active

        split = stops_active(ctx["stops"], sim_kwargs)
        if split and not can_split(sim_kwargs):
            split = False
            warnings.warn(
                "A custom signal_func_nb, or templates in a stop-adjustment callback's "
                "arguments, were given, so bars cannot be split into the open and the rest "
                "of the bar, and stops are NOT checked on the bar a position is entered: a "
                "stop the entry bar trades through is missed. That flatters the result, "
                "most for tight stops on daily bars. Pass the arrays a callback needs as "
                "plain per-bar arrays instead of templates to avoid it.",
                stacklevel=2,
            )
        if cash == "shared" and "call_seq" not in sim_kwargs:
            sim_kwargs["call_seq"] = shared_call_seq(
                close.index, ctx["columns"], **signals,
                entry_priority=_next_bar_scores(entry_priority, close),
                seed=sim_kwargs.get("seed"),
                direction=sim_kwargs.get("direction"),
                split_bars=split,
            )
        if split:
            pf = simulate_split_bars(
                Portfolio, close, fill, high, low, price=fill, signals=signals,
                call_seq=sim_kwargs.pop("call_seq", None),
                size=ctx["size"],
                **order_kwargs,
                **ctx["stops"],
                **sim_kwargs,
            )
        else:
            pf = Portfolio.from_signals(
                close,
                **signals,
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
        # ⚠️ An unsized hold is funded at EQUITY, like the weights branch. It buys with all
        # the cash it has, so handing it buying power held `leverage` times the account:
        # a "buy and hold" on a 30x spread bet carried 30x its equity and lost 247% while
        # the market fell 6.5%.
        if engine_cash is not None:
            sim_kwargs["init_cash"] = np.asarray(engine_cash, dtype=float) - np.asarray(
                margin_loan, dtype=float
            )
            margin_loan = None
        pf = Portfolio.from_holding(close, price=fill, **order_kwargs, **sim_kwargs)
        if init_cash is not None:
            _warn_if_hold_did_not_trade(pf, ctx["columns"])

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
    else:
        # No holding cost is computed, so nothing else would notice a short share held
        # with no borrow rate. It must not borrow for free in silence (G6).
        from ..holding.borrow import warn_unpriced_shorts

        warn_unpriced_shorts(universe, pd.DataFrame(pf.assets()), list(pf.wrapper.columns),
                             stacklevel=4)
    return pf


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
