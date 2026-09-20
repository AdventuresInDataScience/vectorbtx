"""Section N — the one resolution pipeline every front-door path funnels through.

`PortfolioBacktest` and `SingleBacktest` both call `prepare`. Nothing else in this package
resolves arguments, so there is exactly one place where a user-facing argument becomes an
`Order` field — which is the whole point of the layer.

The pipeline, in order:

1. **Normalise the instrument description** to a `Universe` (four accepted forms).
2. **Check mode legality** — every mode declares which arguments it requires and which it
   rejects.
3. **Pick the strategy type** from which arguments are present.
4. **Resolve sizing** into engine-unit sizes.
5. **Resolve `Order` fields** via the existing `Universe.resolve`.
6. **Detect collisions** between an explicit passthrough and a resolved value.

Step 6 is the one that earns its keep. `fees=0` sitting beside `commission_pct=0.001`
silently disables a cost model and produces a backtest that looks excellent, so it raises
— consistent with §C2, where join-key mismatches are fatal rather than warnings.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..instruments.spec import Instrument
from ..resolve.load import Universe, load_universe


class BacktestError(ValueError):
    """Raised when a front-door call describes something that cannot be simulated."""


# ############# Mode legality (FRONTEND-API §1) ############# #

#: Per mode: arguments that must be present, and arguments that are refused outright.
#:
#: Rejected cells raise at construction. A spread bet with a `contract_size` is not a
#: warning case, it is a misunderstanding of the product — so it is refused rather than
#: quietly ignored.
MODE_RULES = {
    "equity": dict(
        required=("point_size",),
        rejected=("contract_size", "base_currency"),
    ),
    "spreadbet": dict(
        required=("point_size",),
        rejected=("contract_size", "base_currency", "allow_fractional"),
    ),
    "cfd": dict(
        required=("point_size", "contract_size"),
        rejected=("base_currency", "allow_fractional"),
    ),
    "fx_spot": dict(
        required=("point_size", "contract_size"),
        rejected=("allow_fractional",),
    ),
}

#: Reasons, so the error says *why* rather than only *what*.
_REJECT_REASONS = {
    ("spreadbet", "contract_size"): (
        "spread bets have no contract size — the stake IS the size "
        "(engine_size = stake_per_point / point_size)"
    ),
    ("spreadbet", "allow_fractional"): (
        "spread-bet sizing is already fractional; use `increment` (default £0.01/pt)"
    ),
    ("cfd", "allow_fractional"): "CFD sizing is already fractional; use `increment` (lot step)",
    ("fx_spot", "allow_fractional"): "FX sizing is already fractional; use `increment` (lot step)",
    ("equity", "contract_size"): "shares have no contract size",
}

#: Spread bets fold their cost entirely into the spread, so commission is forced to zero
#: rather than merely defaulted — a spread bet with a commission is a misdescription.
_FORCED_ZERO = {"spreadbet": ("commission_pct", "commission_flat", "commission_per_unit",
                              "commission_min")}

#: Front-door argument -> `Instrument` field. Everything not listed is a simulation
#: argument, a sizing argument, or a passthrough.
_SPEC_FIELDS = frozenset(
    f for f in Instrument.__dataclass_fields__ if f != "meta"
)

#: `Order` fields the user may also pass directly. Colliding with a resolved value raises.
_PASSTHROUGH_ORDER_FIELDS = (
    "fees", "fixed_fees", "slippage", "min_size", "max_size", "size_granularity",
    "lock_cash",
)

#: Exactly one of these may be given.
SIZING_ARGS = ("risk_pct", "risk_amount", "stake", "notional", "size")


# ############# 1. Instrument normalisation ############# #


def to_universe(instruments=None, columns=None, **spec_kwargs) -> Universe:
    """Normalise any of the four supported instrument forms into one `Universe`.

    The four forms, all producing an identical `Universe` for equivalent input:

    1. **Flat kwargs** — one set of terms applied to every column.
    2. **A dict of per-symbol overrides** on a shared base, where a per-symbol `mode`
       overrides the shared one (this is how one CFD sits in a book of equities).
    3. **A CSV/Excel path** — mode is per row, in the file.
    4. **`Instrument` objects**, or an existing `Universe`.

    Args:
        instruments: form 2, 3 or 4, or None to use form 1.
        columns: the price frame's columns. Required for forms 1 and 2, which describe
            terms without naming the instruments they apply to.
        **spec_kwargs: shared `Instrument` fields.

    Returns:
        `Universe`.

    Raises:
        BacktestError: if no mode can be determined, or the form is unrecognised.

    Notes:
        `Instrument` becomes **optional** here — a reuse mechanism rather than compulsory
        scaffolding. Form 1 is the common case and needs no object at all.
    """
    # --- forms 3 and 4: already a universe, a path, or a frame ---
    if isinstance(instruments, Universe):
        return instruments
    if isinstance(instruments, (str, Path)) or isinstance(instruments, pd.DataFrame):
        return load_universe(instruments)

    shared = {k: v for k, v in spec_kwargs.items() if v is not None}

    # --- form 2: dict of per-symbol overrides on the shared base ---
    if isinstance(instruments, dict):
        specs = {}
        for col, override in instruments.items():
            if isinstance(override, Instrument):
                specs[col] = override
            elif isinstance(override, dict):
                merged = {**shared, **override}
                if "mode" not in merged:
                    raise BacktestError(
                        f"{col!r}: no mode. Give `mode=` on the call for every column, or "
                        f"`mode` inside this instrument's overrides."
                    )
                specs[col] = _build_spec(merged, col)
            else:
                raise BacktestError(
                    f"{col!r}: expected an Instrument or a dict of overrides, got "
                    f"{type(override).__name__}."
                )
        return Universe(specs)

    if instruments is not None:
        raise BacktestError(
            f"Cannot build instruments from {type(instruments).__name__}. Pass a dict of "
            f"per-symbol overrides, a CSV/Excel path, a Universe, Instrument objects, or "
            f"flat keyword arguments."
        )

    # --- form 1: flat kwargs across every column ---
    if "mode" not in shared:
        raise BacktestError(
            "`mode` is required — it decides which other arguments are legal. Use one of "
            f"{sorted(MODE_RULES)}. Without a mode there is nothing this layer adds over "
            "plain vectorbt, so use plain vectorbt instead."
        )
    if columns is None:
        raise BacktestError("Cannot apply flat instrument terms without the price columns.")
    return Universe({col: _build_spec(shared, col) for col in columns})


def _build_spec(kwargs: dict, column) -> Instrument:
    """Check mode legality, then construct. Legality is checked on what the *user* gave."""
    mode = kwargs["mode"]
    if mode not in MODE_RULES:
        raise BacktestError(f"Unknown mode {mode!r} for {column!r}. Expected one of {sorted(MODE_RULES)}.")

    rules = MODE_RULES[mode]
    for name in rules["rejected"]:
        value = kwargs.get(name)
        if value is None or value is False:
            continue
        # `base_currency` is legal on ANY fx-underlying instrument, not just fx_spot.
        # Carry is the differential between the two currencies of the pair, so an FX
        # spread bet or CFD must be able to name them — without this it silently accrues
        # zero carry.
        if name == "base_currency" and kwargs.get("asset_class") == "fx":
            continue
        reason = _REJECT_REASONS.get((mode, name))
        detail = f" — {reason}" if reason else ""
        raise BacktestError(f"{column!r}: `{name}` is not valid in {mode!r} mode{detail}.")

    for name in rules["required"]:
        if kwargs.get(name) is None:
            from ..instruments.defaults import mode_defaults

            default = mode_defaults(mode).get(name)
            if default is None:
                raise BacktestError(
                    f"{column!r}: {mode!r} mode requires `{name}` and has no default for it."
                )

    fields = {k: v for k, v in kwargs.items() if k in _SPEC_FIELDS}
    for name in _FORCED_ZERO.get(mode, ()):
        fields[name] = 0.0
    return Instrument(**fields)


# ############# 2. Strategy type ############# #


def strategy_of(entries=None, exits=None, weights=None, schedule=None) -> str:
    """Infer the strategy type from which arguments are present.

    `entries`/`exits` → signal-driven. `weights`/`schedule` → rebalanced. Neither → hold.
    Both → an error naming the conflict, because there is no sensible merge of the two.
    """
    has_signals = entries is not None or exits is not None
    has_weights = weights is not None or schedule is not None
    if has_signals and has_weights:
        raise BacktestError(
            "Both signal arguments (entries/exits) and rebalancing arguments "
            "(weights/schedule) were given. A backtest is one or the other: signals "
            "enter and exit positions, weights hold continuously and periodically "
            "adjust. Drop one set."
        )
    if has_signals:
        return "signals"
    if has_weights:
        return "weights"
    return "hold"


# ############# 3. Sizing ############# #


def resolve_size(
    universe,
    price,
    columns,
    risk_pct=None,
    risk_amount=None,
    stake=None,
    notional=None,
    size=None,
    stop_points=None,
    init_cash=None,
):
    """Turn exactly one sizing argument into engine-unit sizes.

    Args:
        universe: the `Universe`, supplying each column's mode adapter.
        price: price frame aligned to `columns`.
        columns: resolved column order.
        risk_pct: fraction of `init_cash` risked per trade. Requires `stop_points`.
        risk_amount: absolute money at risk per trade. Requires `stop_points`.
        stake: size in the instrument's own units — £/pt, lots or shares.
        notional: target exposure per position.
        size: raw engine-unit passthrough; the escape hatch.
        stop_points: entry-to-stop distance, required by the risk sizers.
        init_cash: account size, which `risk_pct` is a fraction of.

    Returns:
        DataFrame of **engine-unit** sizes, or None when no sizing was given.

    Raises:
        BacktestError: if two sizing arguments are given, or a risk sizer has no stop.

    Notes:
        ⚠️ **`risk_pct` sizes off `init_cash`, not compounding equity.** At level 1 equity
        is only known before the simulation runs. Material over a long backtest with real
        drawdowns; true compounding risk sizing needs level 3.

        `risk_pct` and `risk_amount` are separate arguments rather than one `risk=`
        disambiguated by magnitude. A guess dressed as a feature is exactly what D8's
        coercion rules exist to prevent.
    """
    given = {
        name: value
        for name, value in zip(SIZING_ARGS, (risk_pct, risk_amount, stake, notional, size))
        if value is not None
    }
    if len(given) > 1:
        raise BacktestError(
            f"Give exactly one sizing argument; got {sorted(given)}. They mean different "
            f"things and there is no sensible way to combine them."
        )
    if not given:
        return None

    from ..modes.base import get_adapter
    from ..modes.helpers import size_for_risk

    price = pd.DataFrame(price)[list(columns)]

    if size is not None:
        return _spread(size, price, columns)

    if risk_pct is not None or risk_amount is not None:
        if stop_points is None:
            raise BacktestError(
                "Risk-based sizing needs `stop_points` — the entry-to-stop distance is "
                "what turns a money-at-risk figure into a position size. Give "
                "`stop_points=`, or size with `stake`/`notional` instead."
            )
        if risk_pct is not None and init_cash is None:
            raise BacktestError("`risk_pct` is a fraction of `init_cash`; give `init_cash=`.")

        # `init_cash` may be per column — separate cash pools size their own risk. Keeping
        # it per column matters: a £30,000 sleeve risking 1% is not the same trade as a
        # £20,000 one, and collapsing to a scalar would silently equalise them.
        def _amount_for(i):
            if risk_pct is None:
                return float(risk_amount)
            cash = np.asarray(init_cash, dtype=float)
            per = float(cash) if cash.ndim == 0 else float(cash.ravel()[i])
            return float(risk_pct) * per

        out = {}
        for i, col in enumerate(columns):
            spec = universe[col].filled()
            user = size_for_risk(_amount_for(i), stop_points, spec, price[col])
            out[col] = get_adapter(spec.mode).to_engine_units(
                np.asarray(user, dtype=float), spec, price[col]
            )
        return pd.DataFrame(out, index=price.index)

    if stake is not None:
        staked = _spread(stake, price, columns)
        out = {}
        for col in columns:
            spec = universe[col].filled()
            out[col] = get_adapter(spec.mode).to_engine_units(
                staked[col].to_numpy(dtype=float), spec, price[col]
            )
        return pd.DataFrame(out, index=price.index)

    # notional: engine size is exposure / price, by definition of engine units.
    target = _spread(notional, price, columns)
    with np.errstate(divide="ignore", invalid="ignore"):
        return pd.DataFrame(
            target.to_numpy(dtype=float) / price.to_numpy(dtype=float),
            index=price.index,
            columns=list(columns),
        )


def quantize_engine_size(universe, engine_size, columns):
    """Round sizes to fillable multiples **here**, not in the engine.

    ⚠️ **vectorbt's granularity floor is not float-safe.** `buy_nb` applies
    `adj_size // size_granularity * size_granularity` (`nb.py:145`) with no epsilon, and
    binary floating point makes that land an increment low on values that are exact
    multiples of the increment:

        1.00 // 0.01 * 0.01  ->  0.99
        1.66 // 0.01 * 0.01  ->  1.65
        0.30 // 0.10 * 0.10  ->  0.20

    Section D assumed the floor path "costs nothing" because the engine already floors
    the way we want. It does not. Two consequences, both silent:

    * every order comes back one increment small, and
    * an order sized at exactly the broker minimum — the £1/pt minimum stake, the 0.01
      minimum lot, the single share — floors to just below `min_size` and is **rejected
      outright**. A spread bet at the minimum stake never traded at all.

    `vectorbtx.quantize.quantizer.quantize` guards this with an epsilon, so quantizing
    here and passing `size_granularity=NaN` makes our quantizer the single source of
    truth and removes the dependence on upstream float behaviour entirely.
    """
    if engine_size is None:
        return None

    from ..modes.base import get_adapter
    from ..quantize.quantizer import quantize

    frame = pd.DataFrame(engine_size)
    out = {}
    for col in columns:
        spec = universe[col].filled()
        adapter = get_adapter(spec.mode)
        user = adapter.from_engine_units(frame[col].to_numpy(dtype=float), spec)
        q_user, _ = quantize(pd.Series(user, index=frame.index), spec)
        q_arr = np.asarray(q_user, dtype=float)

        # ⚠️ A size below the broker minimum quantizes to zero, and a zero-size order is
        # simply never placed — so the backtest runs, reports a flat equity curve and
        # zero trades, and looks like a strategy that never triggered rather than a
        # sizing mistake. Risk sizing hits this easily: 1% of £10,000 over a 400-point
        # stop is £0.25/pt against a £1/pt minimum.
        wanted = np.abs(np.asarray(user, dtype=float))
        killed = np.isfinite(wanted) & (wanted > 0) & (np.abs(q_arr) == 0)
        if killed.any():
            unit = adapter.unit_name(spec)
            warnings.warn(
                f"{col}: {killed.sum()} of {int((wanted > 0).sum())} sized bars fall below "
                f"the minimum tradeable size ({spec.min_size} {unit}) and were rounded to "
                f"zero, so NO ORDER is placed on those bars. Largest suppressed size was "
                f"{np.nanmax(wanted[killed]):.4g} {unit}. Increase risk_pct/stake, tighten "
                f"the stop, or lower min_size to match your broker.",
                stacklevel=3,
            )
        out[col] = adapter.to_engine_units(q_arr, spec)
    return pd.DataFrame(out, index=frame.index)


def _spread(value, price, columns) -> pd.DataFrame:
    """Broadcast a scalar, per-column list or frame across the price frame."""
    if isinstance(value, pd.DataFrame):
        return value.reindex(index=price.index, columns=list(columns))
    if isinstance(value, pd.Series):
        return pd.DataFrame(
            np.repeat(value.to_numpy(dtype=float)[:, None], len(columns), axis=1),
            index=price.index,
            columns=list(columns),
        )
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return pd.DataFrame(float(arr), index=price.index, columns=list(columns))
    if arr.ndim == 1 and len(arr) == len(columns):
        return pd.DataFrame(
            np.broadcast_to(arr[None, :], (len(price.index), len(columns))),
            index=price.index,
            columns=list(columns),
        )
    raise BacktestError(f"Cannot broadcast a sizing value of shape {arr.shape} to {len(columns)} columns.")


# ############# 4. Stops ############# #


def resolve_stops(universe, price, columns, stop_points=None, target_points=None,
                  trail_points=None):
    """Convert point stops into the price fractions vectorbt's native SL/TP expects.

    Native `sl_stop` / `tp_stop` are **fractions of price**, not distances, so the
    conversion has to happen somewhere. Doing it here means the user types the units they
    actually trade in. Native arguments still work and take precedence — see
    `_collision_check`.

    A useful consequence of staying on the native fields: fractional stops are
    FX-invariant under D4's uniform scaling.
    """
    from ..modes.helpers import points_to_price_fraction

    price = pd.DataFrame(price)[list(columns)]
    out = {}
    for name, points in (("sl_stop", stop_points), ("tp_stop", target_points),
                         ("sl_trail_points", trail_points)):
        if points is None:
            continue
        frame = {}
        for col in columns:
            spec = universe[col].filled()
            frame[col] = np.asarray(
                points_to_price_fraction(points, price[col], spec), dtype=float
            )
        out[name] = pd.DataFrame(frame, index=price.index)

    if "sl_trail_points" in out:
        # A trailing stop is an ordinary stop with `sl_trail=True`; there is no separate
        # distance field upstream, so a trail overrides a fixed stop of the same size.
        out["sl_stop"] = out.pop("sl_trail_points")
        out["sl_trail"] = True
    return out


# ############# 5. Collisions ############# #


def collision_check(resolved: dict, explicit: dict) -> dict:
    """Merge resolved `Order` fields with explicit passthroughs, refusing to overwrite.

    ⚠️ A collision **raises**. `fees=0` beside `commission_pct=0.001` silently disables a
    cost model and produces a backtest that looks excellent, which is the single most
    expensive way for this layer to fail. Consistent with §C2, where join-key mismatches
    are fatal rather than warnings.

    Args:
        resolved: fields produced by the cost, quantization and leverage layers.
        explicit: fields the user passed directly.

    Returns:
        The merged dict.

    Raises:
        BacktestError: naming the field and both sources.
    """
    clashes = [
        name for name in _PASSTHROUGH_ORDER_FIELDS
        if name in explicit and name in resolved
    ]
    if clashes:
        detail = ", ".join(
            f"`{n}` (you passed {explicit[n]!r}; the instrument description resolved one too)"
            for n in clashes
        )
        raise BacktestError(
            f"Conflicting values for {detail}. Passing an Order field directly while also "
            f"describing it through the instrument would silently override a cost model. "
            f"Drop one: remove the explicit argument, or remove the instrument fields that "
            f"produce it."
        )
    return {**resolved, **explicit}


# ############# The pipeline ############# #


def prepare(
    price,
    fill_price=None,
    instruments=None,
    entries=None,
    exits=None,
    weights=None,
    schedule=None,
    # sizing
    risk_pct=None, risk_amount=None, stake=None, notional=None, size=None,
    # stops
    stop_points=None, target_points=None, trail_points=None,
    # data
    volume=None, rates=None, fx=None, base_currency=None,
    # account
    init_cash=None,
    # everything else
    spec_kwargs=None,
    passthrough=None,
):
    """Run the whole front-door pipeline and return everything a constructor needs.

    Returns:
        dict with `universe`, `strategy`, `columns`, `price`, `size`, `order_kwargs`,
        `stops` and `passthrough`.
    """
    spec_kwargs = dict(spec_kwargs or {})
    passthrough = dict(passthrough or {})

    price = pd.DataFrame(price)
    columns = list(price.columns)

    # ⚠️ Everything order-time is priced at the bar you actually TRADE at, not the bar the
    # signal fired on. Sizes, stop distances and spread fractions all resolve against
    # `fill_price` (the next open), never against the signal bar's close. Using the close
    # would size a position from a price that was not yet knowable when the order was
    # placed — the same lookahead as filling at it.
    fill = pd.DataFrame(fill_price if fill_price is not None else price)[columns]

    # Coerce `fx` once, here. The cost layer needs a real FXTable to convert
    # foreign-currency fixed costs, and users pass whatever their data provider gave them
    # — a GBPUSD close series, a Data object, a dict of flat rates.
    from ..fx.convert import as_fx_table

    fx = as_fx_table(fx, index=price.index)

    universe = to_universe(instruments, columns=columns, **spec_kwargs)
    strategy = strategy_of(entries, exits, weights, schedule)

    # Fail on a spec/price mismatch before doing any work, in both directions.
    universe.order_for(columns)

    engine_size = resolve_size(
        universe, fill, columns,
        risk_pct=risk_pct, risk_amount=risk_amount, stake=stake,
        notional=notional, size=size, stop_points=stop_points, init_cash=init_cash,
    )
    engine_size = quantize_engine_size(universe, engine_size, columns)

    equity = init_cash if init_cash is not None else None
    resolved = universe.resolve(
        fill,
        rates=rates,
        equity=equity,
        fx_table=fx,
        base_currency=base_currency,
        volume=volume,
        size=engine_size,
        notional=_notional_of(engine_size, fill, weights, init_cash, columns),
    )
    if engine_size is not None:
        # Sizes are already quantized above, so the engine must not floor them again —
        # doing so would knock a clean multiple down one further increment.
        resolved["size_granularity"] = np.nan
    order_kwargs = collision_check(resolved, passthrough)

    stops = resolve_stops(
        universe, fill, columns,
        stop_points=stop_points, target_points=target_points, trail_points=trail_points,
    )

    warn_if_cash_constrained(universe, engine_size, fill, columns, init_cash)

    return dict(
        universe=universe,
        strategy=strategy,
        columns=columns,
        price=price,
        fill_price=fill,
        size=engine_size,
        order_kwargs=order_kwargs,
        stops=stops,
    )


def warn_if_cash_constrained(universe, engine_size, price, columns, init_cash) -> None:
    """Warn when a position exceeds even the **levered** buying power.

    Leverage is funded, not merely capped — `vectorbtx.margin.buying_power` hands the
    engine `equity x leverage` and `Portfolio.get_init_cash` subtracts the loan back out
    — so a position within its leverage limit is filled in full. What remains possible is
    asking for more than the leverage itself allows.

    ⚠️ That case is still a **partial fill, not a rejection**: `max_size` clips rather
    than refusing (`allow_partial` defaults to True), and `buy_nb` then bounds what is
    left by the buying power. Either way the order comes back smaller than requested, the
    backtest runs, and the equity curve looks plausible. Hence the warning, naming the
    shortfall. Pass `raise_reject=True` for reject semantics instead.
    """
    if init_cash is None or engine_size is None:
        return
    frame = pd.DataFrame(engine_size)
    px = pd.DataFrame(price)[list(columns)]
    notional = (frame.abs() * px).max(axis=0)

    cash = np.asarray(init_cash, dtype=float)
    offenders = {}
    for i, col in enumerate(columns):
        spec = universe[col].filled()
        want = float(notional.get(col, 0.0))
        # init_cash may be per column when the pools are separate.
        own = float(cash) if cash.ndim == 0 else float(cash.ravel()[i])
        power = own * float(spec.leverage or 1.0)
        if not np.isfinite(want) or want <= power * (1.0 + 1e-9):
            continue
        offenders[col] = (want, power)

    if not offenders:
        return
    detail = ", ".join(
        f"{c} needs {want:,.0f} but its leverage funds only {power:,.0f}"
        for c, (want, power) in offenders.items()
    )
    warnings.warn(
        f"Sized positions exceed their funded buying power ({detail}; equity="
        f"{np.asarray(init_cash, dtype=float).sum():,.0f} total). These orders will be "
        f"PARTIALLY FILLED at a smaller "
        f"size than requested rather than rejected, so the backtest will run and quietly "
        f"understate the position. Raise `leverage`, reduce the size, or pass "
        f"`raise_reject=True` to reject instead of partially filling.",
        stacklevel=3,
    )


def _notional_of(engine_size, price, weights=None, init_cash=None, columns=None):
    """Order notional, which minimum commission needs to pre-compute `max(pct x N, min)`.

    A rebalanced strategy has no explicit size — the engine derives orders from target
    percentages — so the notional is estimated as `weight x equity`. That is exact on a
    rebalance bar where the portfolio is at target, and close enough between them; the
    alternative is refusing to model a minimum commission on any rebalanced portfolio.
    """
    if engine_size is None:
        if weights is None or init_cash is None:
            return None
        w = pd.DataFrame(weights).reindex(columns=list(columns))
        cash = np.asarray(init_cash, dtype=float)
        per = cash if cash.ndim else np.full(len(columns), float(cash))
        return (w.abs() * per).reindex(index=price.index).ffill().fillna(0.0)
    return pd.DataFrame(
        np.abs(np.asarray(engine_size, dtype=float)) * price.to_numpy(dtype=float),
        index=price.index,
        columns=price.columns,
    )
