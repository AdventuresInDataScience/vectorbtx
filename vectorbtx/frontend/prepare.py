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
SIZING_ARGS = ("risk_pct", "risk_amount", "stake", "notional", "size", "notional_pct")


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


def strategy_of(entries=None, exits=None, weights=None, schedule=None,
                short_entries=None, short_exits=None) -> str:
    """Infer the strategy type from which arguments are present.

    `entries`/`exits`/`short_entries`/`short_exits` → signal-driven. `weights`/`schedule` →
    rebalanced. Neither → hold. Both → an error naming the conflict, because there is no
    sensible merge of the two.
    """
    has_signals = any(
        s is not None for s in (entries, exits, short_entries, short_exits)
    )
    has_weights = weights is not None or schedule is not None
    if has_signals and has_weights:
        raise BacktestError(
            "Both signal arguments (entries/exits/short_entries/short_exits) and rebalancing arguments "
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
    *,
    stop_distance=None,
    stop_columns=None,
    risk_basis=None,
    order_bars=None,
    notional_pct=None,
    keep=None,
):
    """Turn exactly one sizing argument into engine-unit sizes.

    Args:
        universe: the `Universe`, supplying each column's mode adapter.
        price: price frame aligned to `columns` (the fill price).
        columns: resolved column order.
        risk_pct: fraction of the account risked per trade: of `init_cash`, or of
            `risk_basis` when given.
        risk_amount: absolute money at risk per trade.
        stake: size in the instrument's own units — £/pt, lots or shares.
        notional: target exposure per position.
        size: raw engine-unit passthrough; the escape hatch.
        notional_pct: each position's value as a fraction of the account (`0.5` is half
            of it): of `init_cash`, or of `risk_basis` when given. A number, or one per
            column. Above 1 is a levered position, funded up to the leverage.
        stop_points: entry-to-stop distance in points — a number, or one per column. Used
            when `stop_distance` is not given; a per-bar frame here is read as it stands,
            on the fill bars (the front door shifts per-bar stops before calling).
        init_cash: account size, which `risk_pct` is a fraction of.
        stop_distance: the stop-loss distance **in price** at each fill bar, from
            `stop_plan` — whichever stop was given (points, a fraction of the price, a
            trailing stop's first distance). Overrides `stop_points`.
        stop_columns: the columns a stop was given for. A column with none is sized from
            its leverage (see Notes). Default: every column with a distance anywhere.
        risk_basis: the money `risk_pct` or `notional_pct` is a fraction of, per fill bar
            and column — the balance or equity at the signal bar's close (`risk_of`).
            Default `init_cash`.
        order_bars: boolean frame of the bars an order is placed on, for the warnings.
        keep: a dict that, for `risk_pct` and `notional_pct`, is given what the sizes are worked
            out from besides the basis (`"share"`, the arrays), so the front door's re-sizing loop
            can size again from another basis with the same arithmetic (`size_from_basis`).

    Returns:
        DataFrame of **engine-unit** sizes, or None when no sizing was given.

    Raises:
        BacktestError: if two sizing arguments are given.

    Notes:
        **Risk sizing from a stop.** The position loses the risk `R` if the stop is hit:
        `engine size = R / distance`, the distance in price per engine unit (P&L is
        `engine_size x price change` in every mode). A distance missing on a fill bar where
        the column has a stop (an ATR still warming up) gives no size, so no entry there,
        and a warning counts the entries skipped.

        **Risk sizing with no stop** (OpenStrategy Studio issue 28). A column with no stop
        is sized as if the trade's margin were the risk: at leverage `L` a position worth
        `V` posts `V / L` and loses all of it on a move of `1 / L` of the price, so
        `V = R x L` — the price going to zero bought outright, a 20% move at 5x, 3.33% at
        30x. It sizes only: no stop is placed, so a gap past `1 / L` loses more than `R`
        (and from 1x a short can lose more too), and a real account would meet a margin
        call first, at whatever close-out level the broker sets. That is left out, and a
        warning says so.

        `risk_pct` and `notional_pct` size off `init_cash` unless `risk_basis` is given;
        the front door's `risk_of="balance" | "equity"` passes one, worked out from the
        simulation.

        `risk_pct` and `risk_amount` are separate arguments rather than one `risk=`
        disambiguated by magnitude. A guess dressed as a feature is exactly what D8's
        coercion rules exist to prevent.
    """
    given = {
        name: value
        for name, value in zip(SIZING_ARGS, (risk_pct, risk_amount, stake, notional, size,
                                             notional_pct))
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

    price = pd.DataFrame(price)[list(columns)]

    if size is not None:
        return _spread(size, price, columns)

    if risk_pct is not None or risk_amount is not None:
        if risk_pct is not None and init_cash is None and risk_basis is None:
            raise BacktestError("`risk_pct` is a fraction of `init_cash`; give `init_cash=`.")
        return _risk_sizes(
            universe, price, columns, risk_pct=risk_pct, risk_amount=risk_amount,
            init_cash=init_cash, stop_points=stop_points, stop_distance=stop_distance,
            stop_columns=stop_columns, risk_basis=risk_basis, order_bars=order_bars, keep=keep,
        )

    if notional_pct is not None:
        if init_cash is None and risk_basis is None:
            raise BacktestError("`notional_pct` is a fraction of `init_cash`; give `init_cash=`.")
        if isinstance(notional_pct, pd.DataFrame) or (
            isinstance(notional_pct, pd.Series) and isinstance(notional_pct.index, pd.DatetimeIndex)
        ):
            raise BacktestError(
                "`notional_pct` takes one value, or one per column (a dict or a list in "
                "column order), not a value per bar."
            )
        share, named = stop_values(notional_pct, price.index, columns, "notional_pct")
        missing = [c for c in columns if c not in named]
        if missing:
            raise BacktestError(f"`notional_pct` gives no value for {missing}.")
        # An account at or below zero gives no order (issue 40).
        money = _money_above_zero(_account_money(price.index, columns, init_cash, risk_basis),
                                  price.index, columns, order_bars)
        shares, prices = share.to_numpy(dtype=float), price.to_numpy(dtype=float)
        if keep is not None:
            keep.update(share="notional_pct", shares=shares, prices=prices)
        return pd.DataFrame(_notional_pct_core(shares, money, prices), index=price.index, columns=list(columns))

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


def _notional_pct_core(shares, money, prices) -> np.ndarray:
    """A position worth that share of the account: engine size is value / price."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return shares * money / prices


def _risk_core(amount, distance, fill, has_stop, leverage) -> np.ndarray:
    """Engine sizes losing `amount` at the stop `distance`, or worth `amount x leverage` with no stop."""
    out = np.full(fill.shape, np.nan)
    for k in range(fill.shape[1]):
        if has_stop[k]:
            d = distance[:, k]
            with np.errstate(divide="ignore", invalid="ignore"):
                out[:, k] = np.where(np.isfinite(d) & (d > 0), amount[:, k] / d, np.nan)
        else:
            # V = R x L: the trade's margin is the risk (issue 28).
            with np.errstate(divide="ignore", invalid="ignore"):
                out[:, k] = amount[:, k] * leverage[k] / fill[:, k]
    return out


def _money_above_zero(money, index=None, columns=None, order_bars=None) -> np.ndarray:
    """The account a share is taken of, NaN (no order) where it is at or below zero.

    OpenStrategy Studio issue 40. A share of a negative balance was a negative size. The leverage
    cap leaves it as it is when the equity, which it is measured on, is above zero (an open trade
    in profit), and vectorbt raises `ValueError: Negative size is not allowed` on any bar with a
    signal, an exit included. The re-sizing loop reaches such an account on its way to a run
    that settles: a pass sizes from the run before it, which can hold a trade its own account
    could not have opened. So `risk_pct` or `notional_pct` of the balance raised on runs that
    have an answer. A share of nothing, or of a debt, is no position: NaN, the sizing's "no
    order", as an entry with no stop distance is.

    With `order_bars` (a whole preparation, not one of the loop's passes) the entries left out
    are said. The loop says the warnings of the preparation that read the run returned, so the
    words describe that run (issue 35).
    """
    money = np.asarray(money, dtype=float)
    out = np.where(money > 0, money, np.nan)              # NaN stays NaN
    if order_bars is None:
        return out
    placed = pd.DataFrame(order_bars).reindex(index=index, columns=list(columns)).fillna(False)
    placed = placed.to_numpy(dtype=bool)
    hit = placed & np.isfinite(money) & (money <= 0)
    if hit.any():
        details = []
        for k, col in enumerate(columns):
            if not hit[:, k].any():
                continue
            first = int(np.argmax(hit[:, k]))
            details.append(f"{col}: {int(hit[:, k].sum())} of {int(placed[:, k].sum())}, the first on "
                           f"{_when(index[first])} at {money[first, k]:,.0f}")
        warnings.warn(
            f"Entries on an account at or below zero ({', '.join(details)}) cannot be sized as a "
            f"share of it, so NO ORDER is placed on them. The account had lost everything before "
            f"them: a smaller share, a wider stop or less leverage keeps it above zero.",
            stacklevel=4,
        )
    return out


def size_from_basis(keep, basis, risk_pct) -> np.ndarray:
    """Sizes from another basis, with the arithmetic `resolve_size` used (`keep`, as it left it).

    The front door's re-sizing loop (OpenStrategy Studio issue 37): `risk_pct` and `notional_pct`
    sized again from each pass's balance or equity, without the frames around them. An account
    at or below zero gives no order, as in `resolve_size` (issue 40), without its warning: a
    pass's words are never said.
    """
    money = _money_above_zero(basis)
    if keep["share"] == "notional_pct":
        return _notional_pct_core(keep["shares"], money, keep["prices"])
    return _risk_core(float(risk_pct) * money, keep["distance"], keep["fill"], keep["has_stop"], keep["leverage"])


def _risk_sizes(universe, price, columns, risk_pct, risk_amount, init_cash, stop_points,
                stop_distance, stop_columns, risk_basis, order_bars, keep=None):
    """Engine-unit sizes that lose the risk at the stop, or from the leverage with none.

    See `resolve_size` for the rules; this is its risk branch.
    """
    columns = list(columns)
    index = price.index
    fill = price.to_numpy(dtype=float)

    # --- the money at risk, per fill bar and column ---
    if risk_pct is None:
        amount = np.full(fill.shape, float(risk_amount))
    else:
        # An account at or below zero gives no order (issue 40).
        money = _money_above_zero(_account_money(index, columns, init_cash, risk_basis), index, columns, order_bars)
        amount = float(risk_pct) * money

    # --- the stop's distance in price, per fill bar and column ---
    if stop_distance is None and stop_points is not None:
        point_size = np.array([universe[c].filled().point_size for c in columns], dtype=float)
        if isinstance(stop_points, pd.DataFrame) or (
            isinstance(stop_points, pd.Series) and isinstance(stop_points.index, pd.DatetimeIndex)
        ):
            frame = pd.DataFrame(stop_points)
            if frame.shape[1] == 1 and len(columns) > 1:
                frame = pd.DataFrame(np.repeat(frame.to_numpy(), len(columns), axis=1),
                                     index=frame.index, columns=columns)
            points = frame.reindex(index=index, columns=columns).astype(float)
            named = [c for c in columns if points[c].notna().any()]
        else:
            points, named = stop_values(stop_points, index, columns, "stop_points")
        stop_distance = points * point_size[None, :]
        stop_columns = named if stop_columns is None else stop_columns
    if stop_distance is not None:
        distance = pd.DataFrame(stop_distance).reindex(index=index, columns=columns).to_numpy(dtype=float)
        if stop_columns is None:
            stop_columns = [c for k, c in enumerate(columns) if np.isfinite(distance[:, k]).any()]
    else:
        distance = np.full(fill.shape, np.nan)
        stop_columns = []

    if order_bars is not None:
        placed = pd.DataFrame(order_bars).reindex(index=index, columns=columns).fillna(False)
        placed = placed.to_numpy(dtype=bool)
    else:
        placed = np.ones(fill.shape, dtype=bool)

    has_stop = np.array([col in stop_columns for col in columns], dtype=bool)
    leverage = np.array([float(universe[col].filled().leverage or 1.0) for col in columns])
    out = _risk_core(amount, distance, fill, has_stop, leverage)
    if keep is not None:
        keep.update(share="risk", distance=distance, fill=fill, has_stop=has_stop, leverage=leverage)
    from_leverage = {}
    skipped = {}
    for k, col in enumerate(columns):
        if has_stop[k]:
            d = distance[:, k]
            missing = placed[:, k] & ~(np.isfinite(d) & (d > 0))
            if missing.any():
                skipped[col] = (int(missing.sum()), int(placed[:, k].sum()))
        else:
            from_leverage[col] = float(leverage[k])

    if from_leverage:
        detail = ", ".join(
            f"{col} at {lev:g}x: worth {lev:g} x the risk, an implied stop "
            f"{100.0 / lev:.4g}% of the price away" for col, lev in from_leverage.items()
        )
        warnings.warn(
            f"Risk sizing with no stop ({detail}). Each position is sized as if its margin "
            f"were the risk, so it loses the risk on a move of 1 / leverage of the price. "
            f"NO STOP IS PLACED: a move past that point loses more than the risk, and a real "
            f"account would meet a margin call first, which is not modelled. Give a stop "
            f"(stop_points, sl_stop or trail_points) to size from it instead.",
            stacklevel=4,
        )
    if skipped:
        detail = ", ".join(f"{col}: {n} of {m}" for col, (n, m) in skipped.items())
        warnings.warn(
            f"Entries with no stop distance on their bar ({detail}), for example an "
            f"indicator still warming up, cannot be risk-sized, so NO ORDER is placed on "
            f"them. Fill the stop's missing values, or start the backtest later.",
            stacklevel=4,
        )
    return pd.DataFrame(out, index=index, columns=columns)


def _account_money(index, columns, init_cash, risk_basis) -> np.ndarray:
    """The account a fractional size is a share of, per fill bar and column.

    `risk_basis` (the balance or equity at each signal bar) when given, else `init_cash`.
    `init_cash` may be per column — separate cash pools size their own trades. Keeping it
    per column matters: a £30,000 sleeve risking 1% is not the same trade as a £20,000 one,
    and collapsing to a scalar would silently equalise them.
    """
    shape = (len(index), len(columns))
    if risk_basis is not None:
        if (isinstance(risk_basis, pd.DataFrame) and list(risk_basis.columns) == list(columns)
                and risk_basis.index.equals(index)):
            # Already on these bars and columns (the front door's own basis): reindexing would copy
            # the same values, on every pass of the re-sizing loop.
            return risk_basis.to_numpy(dtype=float, copy=True)
        basis = pd.DataFrame(risk_basis).reindex(index=index, columns=list(columns))
        return basis.to_numpy(dtype=float)
    cash = np.asarray(init_cash, dtype=float)
    per = np.full(len(columns), float(cash)) if cash.ndim == 0 else cash.ravel()[: len(columns)]
    return np.broadcast_to(per, shape).astype(float)


def quantize_engine_size(universe, engine_size, columns, order_bars=None):
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

    `order_bars`, a boolean frame of the bars an order is placed on, limits the
    minimum-size warning to those bars. A size is resolved for every bar, but only the
    ones an order uses can fall below the minimum and matter (OpenStrategy Studio issue
    22: "5 of 10 sized bars" was counted on bars that traded nothing).
    """
    if engine_size is None:
        return None

    from ..modes.base import get_adapter
    from ..quantize.quantizer import quantize

    frame = pd.DataFrame(engine_size)
    if order_bars is not None:
        placed = pd.DataFrame(order_bars).reindex(index=frame.index, columns=list(columns))
        placed = placed.fillna(False).astype(bool)
        noun = "entries"
    else:
        placed = pd.DataFrame(True, index=frame.index, columns=list(columns))
        noun = "sized bars"
    out = {}
    for col in columns:
        spec = universe[col].filled()
        adapter = get_adapter(spec.mode)
        user = adapter.from_engine_units(frame[col].to_numpy(dtype=float), spec)
        q_arr = _quantized_user(user, spec)

        # ⚠️ A size below the broker minimum quantizes to zero, and a zero-size order is
        # simply never placed — so the backtest runs, reports a flat equity curve and
        # zero trades, and looks like a strategy that never triggered rather than a
        # sizing mistake. Risk sizing hits this easily: 1% of £10,000 over a 400-point
        # stop is £0.25/pt against a £1/pt minimum.
        wanted = np.abs(np.asarray(user, dtype=float))
        on = placed[col].to_numpy(dtype=bool)
        sized = np.isfinite(wanted) & (wanted > 0) & on
        killed = sized & (np.abs(q_arr) == 0)
        if killed.any():
            unit = adapter.unit_name(spec)
            warnings.warn(
                f"{col}: {killed.sum()} of {int(sized.sum())} {noun} fall below "
                f"the minimum tradeable size ({spec.min_size} {unit}) and were rounded to "
                f"zero, so NO ORDER is placed on those bars. Largest suppressed size was "
                f"{np.nanmax(wanted[killed]):.4g} {unit}. Increase risk_pct/stake, tighten "
                f"the stop, or lower min_size to match your broker.",
                stacklevel=3,
            )
        out[col] = adapter.to_engine_units(q_arr, spec)
    return pd.DataFrame(out, index=frame.index)


def _quantized_user(user, spec) -> np.ndarray:
    """User-unit sizes rounded as `quantize` rounds them, for a filled spec."""
    from ..quantize.quantizer import quantize_filled

    return np.asarray(quantize_filled(np.asarray(user, dtype=np.float64), spec), dtype=float)


def column_terms(universe, columns) -> list:
    """Each column's filled spec and mode adapter, for the cores below; worked out once per run."""
    from ..modes.base import get_adapter

    out = []
    for col in columns:
        spec = universe[col].filled()
        out.append((spec, get_adapter(spec.mode)))
    return out


def quantize_core(engine_size, terms) -> np.ndarray:
    """`quantize_engine_size`'s rounding without its frames or warnings (the re-sizing loop's)."""
    sizes = np.asarray(engine_size, dtype=float)
    out = np.empty_like(sizes)
    for k, (spec, adapter) in enumerate(terms):
        user = adapter.from_engine_units(sizes[:, k], spec)
        out[:, k] = adapter.to_engine_units(_quantized_user(user, spec), spec)
    return out


def _cells(value, shape) -> np.ndarray:
    """A resolved order field (a scalar, one value per column, or a frame) as one array per cell."""
    if value is None:
        return np.zeros(shape)
    if isinstance(value, (pd.DataFrame, pd.Series)):
        value = value.to_numpy()
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1 and arr.shape[0] == shape[0] and shape[0] != shape[1]:
        arr = arr[:, None]                      # one value per bar, for every column
    return np.broadcast_to(arr, shape)


def _floor_engine(universe, col, engine_values) -> np.ndarray:
    """Engine-unit magnitudes rounded DOWN to the column's size step (`quantize`, float-safe).

    Always down, whatever the instrument's own rounding: a size that rounded up past a limit
    would not fit it. Below the minimum size it is zero, so no order is placed.
    """
    from ..modes.base import get_adapter

    spec = universe[col].filled()
    return _floor_core(engine_values, spec, get_adapter(spec.mode))


def _floor_core(engine_values, spec, adapter) -> np.ndarray:
    """`_floor_engine` for a filled spec and its adapter."""
    from ..quantize.quantizer import quantize_filled

    user = adapter.from_engine_units(np.abs(np.asarray(engine_values, dtype=float)), spec)
    floored = quantize_filled(np.asarray(user, dtype=np.float64), spec, "floor")
    return np.asarray(adapter.to_engine_units(np.asarray(floored, dtype=float), spec), dtype=float)


def cap_core(sizes, money, px, slip, fees, fixed, terms, partial) -> np.ndarray:
    """`cap_to_leverage` on arrays, each column's leverage and size step from `terms`."""
    out = np.array(sizes, dtype=float, copy=True)
    for k, (spec, adapter) in enumerate(terms):
        leverage = float(spec.leverage or 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            limit = (leverage * money[:, k] - fixed[:, k]) / (px[:, k] * (1.0 + slip[:, k]) * (1.0 + fees[:, k]))
            over = np.isfinite(out[:, k]) & np.isfinite(limit) & (np.abs(out[:, k]) > limit * (1.0 + 1e-9))
        if not over.any():
            continue
        if partial:
            floored = _floor_core(np.maximum(limit[over], 0.0), spec, adapter)
            out[over, k] = np.sign(out[over, k]) * floored
        else:
            out[over, k] = np.nan
    return out


def ceiling_core(sizes, roof, terms) -> np.ndarray:
    """`_apply_ceilings` on arrays."""
    out = np.array(sizes, dtype=float, copy=True)
    for k, (spec, adapter) in enumerate(terms):
        has = np.isfinite(roof[:, k]) & np.isfinite(out[:, k])
        if not has.any():
            continue
        floored = _floor_core(roof[has, k], spec, adapter)
        out[has, k] = np.sign(out[has, k]) * np.minimum(np.abs(out[has, k]), floored)
    return out


def cap_to_leverage(universe, engine_size, price, columns, account, costs, partial=True) -> pd.DataFrame:
    """Every entry cut down to what its leverage allows: a position worth at most `leverage x account`.

    OpenStrategy Studio issues 32 to 34. The limit is on the position, so it is applied here, to
    the entry sizes, and never reaches vectorbt as `max_size`, which vectorbt applies to every
    order (closes included). `account` is the equity each entry is measured against, per fill
    bar and column: the equity at the signal bar's close when the front door knows it (issue 33:
    a position may grow with the account), else the starting cash.

    An entry over the limit is cut to the limit rounded DOWN to the size step, allowing for the
    spread (`slippage`) and commission (`fees`, `fixed_fees`) it pays, so an unlevered account
    can pay for it in full and vectorbt never cuts it again to the cash, off the size step
    (issue 34). With `partial=False` (`when_cash_short="skip"`) it is not placed at all (NaN).
    Sizes within the limit are returned as they are.
    """
    frame = pd.DataFrame(engine_size)[list(columns)]
    arrays = cap_arrays(price, frame.index, columns, costs, frame.shape)
    out = cap_core(frame.to_numpy(dtype=float), np.asarray(account, dtype=float), *arrays,
                   column_terms(universe, columns), partial)
    return pd.DataFrame(out, index=frame.index, columns=list(columns))


def cap_arrays(price, index, columns, costs, shape) -> tuple:
    """`cap_to_leverage`'s price, slippage, fee and fixed-fee arrays, per bar and column."""
    px = pd.DataFrame(price)[list(columns)].reindex(index=index).to_numpy(dtype=float)
    return (px, _cells(costs.get("slippage"), shape), _cells(costs.get("fees"), shape),
            _cells(costs.get("fixed_fees"), shape))


def _apply_ceilings(universe, engine_size, ceilings, columns) -> pd.DataFrame:
    """Each entry no bigger than its ceiling (what an earlier run found funded), rounded down."""
    frame = pd.DataFrame(engine_size)[list(columns)]
    roof = pd.DataFrame(ceilings).reindex(index=frame.index, columns=list(columns)).to_numpy(dtype=float)
    out = ceiling_core(frame.to_numpy(dtype=float), roof, column_terms(universe, columns))
    return pd.DataFrame(out, index=frame.index, columns=list(columns))


def _same_frame(a, b) -> bool:
    """Two size frames equal, NaN with NaN."""
    return bool(np.array_equal(pd.DataFrame(a).to_numpy(dtype=float), pd.DataFrame(b).to_numpy(dtype=float),
                               equal_nan=True))


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


def stop_values(value, index, columns, name):
    """A stop or target, in whatever shape it was given, as a frame on `index` x `columns`.

    Accepted shapes:

    * a scalar — one value for every column and bar;
    * a dict `{column: value}`, a list in column order, or a Series indexed by columns —
      one value per column (columns a dict leaves out have none);
    * a DataFrame, or a Series on the bar index — one value per bar.

    ⚠️ **A per-bar value is aligned with the signals and shifted with them.** It is worked
    out at the signal bar's close (`3 * atr`), and the order fills at the next bar's open,
    so the value used at the fill is the one from the bar before it. Read at the fill bar
    instead, the stop would include that bar's high, low and close: lookahead (OpenStrategy
    Studio issue 26).

    Returns:
        `(frame, named)`: the values on the fill bars, NaN where there are none, and the
        columns the argument named at all. None when `value` is None.

    Raises:
        BacktestError: on a column the backtest does not have, or a shape that fits neither
            the columns nor the bars.
    """
    if value is None:
        return None
    columns = list(columns)

    def per_column(values: dict):
        unknown = [c for c in values if c not in columns]
        if unknown:
            raise BacktestError(f"`{name}` names columns not in the backtest: {unknown}.")
        row = np.array([float(values[c]) if c in values and values[c] is not None else np.nan
                        for c in columns])
        frame = pd.DataFrame(np.broadcast_to(row, (len(index), len(columns))).copy(),
                             index=index, columns=columns)
        return frame, [c for c in columns if c in values and values[c] is not None]

    def per_bar(frame: pd.DataFrame):
        unknown = [c for c in frame.columns if c not in columns]
        if unknown:
            raise BacktestError(f"`{name}` names columns not in the backtest: {unknown}.")
        frame = frame.reindex(index=index, columns=columns).astype(float)
        named = [c for c in columns if c in frame.columns and frame[c].notna().any()]
        return frame.shift(1), named

    if isinstance(value, dict):
        return per_column(value)
    if isinstance(value, pd.DataFrame):
        return per_bar(value)
    if isinstance(value, pd.Series):
        on_columns = not isinstance(value.index, pd.DatetimeIndex) and all(
            label in columns for label in value.index
        )
        if on_columns:
            return per_column(value.to_dict())
        series = value.reindex(index).astype(float)
        frame = pd.DataFrame(np.repeat(series.to_numpy()[:, None], len(columns), axis=1),
                             index=index, columns=columns)
        return per_bar(frame)
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return per_column({c: float(arr) for c in columns})
    if arr.ndim == 1 and len(arr) == len(columns):
        return per_column(dict(zip(columns, arr.tolist())))
    if arr.ndim == 2 and arr.shape == (len(index), len(columns)):
        return per_bar(pd.DataFrame(arr, index=index, columns=columns))
    raise BacktestError(
        f"Cannot read `{name}` of shape {arr.shape}: give a number, one value per column "
        f"(a dict, or a list in column order) or a DataFrame of values per bar."
    )


def stop_plan(universe, price, columns, stop_points=None, target_points=None,
              trail_points=None, sl_stop=None, tp_stop=None, sl_trail=None, point_fx=None,
              stop_price=None, target_price=None, trail_price=None, long_open=None,
              short_open=None):
    """Every stop and target, resolved: what vectorbt gets, and the distances sizing reads.

    Points are converted to the price fractions vectorbt's native `sl_stop` / `tp_stop`
    expect, against the fill price (the open the order fills at). Native fractions given
    per bar are shifted with the signals like points; a scalar or per-column native value
    passes through unchanged.

    `point_fx` is the rate each column's price was converted into the account currency
    at, per bar (`vectorbtx.fx.convert.fx_rate_of`). A point is a unit of the native price,
    so on a converted price it is `point_size x rate`; without the rate a 50-cent stop on a
    USD share converted to GBP sat 50 pence away (OpenStrategy Studio issue 29).

    **Price levels** (`stop_price`, `target_price`, `trail_price`) are where the stop or
    target sits, not how far away: an EMA, a swing low. Each is a level per bar (or a
    number, or one per column), aligned with the signals and shifted with them, so the
    level at the signal bar's close is the one used from the fill. A distance worked out at
    that close (`close - ema`) would miss the gap to the next open and put the stop off the
    level by it; a level does not. At the fill it becomes vectorbt's fraction of the fill
    price on the right side for the entry — below the fill for a long's stop and a short's
    target, above for a short's stop and a long's target — which needs `long_open` and
    `short_open`, the bars each side's entries fill on. A level on the wrong side (a long's
    stop at or above its fill) is an invalid order, and a broker rejects it: the entry is
    returned in `rejected`, and the front door does not place it. Without the sides (a
    direct call) each level is taken to be on the right side. `trail_price` starts the stop
    at the signal bar's level and moves it to each completed bar's level after that,
    never loosening it (a chandelier exit; see `vectorbtx.frontend.trail`).

    Returns:
        dict with
        * `stops` — keyword arguments for `from_signals` (`sl_stop`, `tp_stop`, `sl_trail`);
        * `risk_distance` — the stop-loss distance in price at each fill bar, NaN where
          there is none, or None with no stop at all;
        * `stop_columns` — the columns a stop-loss argument named;
        * `trail_distance` — a trailing stop's distance in price at each fill bar, when it
          was given in points (see `vectorbtx.frontend.trail`), else None;
        * `trail_level` — `trail_price` per fill bar (the signal bar's level), else None;
        * `rejected` — boolean frame of entries whose level is on the wrong side of the
          fill, or None when no level was given.

    Raises:
        BacktestError: when a stop or target is given more than once.
    """
    losses = [name for name, value in (("stop_points", stop_points), ("trail_points", trail_points),
                                       ("sl_stop", sl_stop), ("stop_price", stop_price),
                                       ("trail_price", trail_price)) if value is not None]
    if ("stop_price" in losses or "trail_price" in losses) and len(losses) > 1:
        raise BacktestError(
            f"Give the stop-loss once; got {losses}. vectorbt has one stop-loss slot, so a "
            f"level (`stop_price`, `trail_price`) cannot sit beside a distance or another level."
        )
    if target_price is not None and (target_points is not None or tp_stop is not None):
        raise BacktestError(
            "Give the take-profit once: `target_price=` (a level), `target_points=` (in "
            "points) or `tp_stop=` (a fraction of the price)."
        )
    if trail_price is not None and sl_trail is not None:
        raise BacktestError("`trail_price` trails by itself; drop `sl_trail`.")
    if sl_stop is not None and (stop_points is not None or trail_points is not None):
        raise BacktestError(
            "Give the stop-loss once: `stop_points=` / `trail_points=` (in points) or "
            "`sl_stop=` (a fraction of the price), not both. vectorbt has one stop-loss slot."
        )
    if tp_stop is not None and target_points is not None:
        raise BacktestError(
            "Give the take-profit once: `target_points=` (in points) or `tp_stop=` (a "
            "fraction of the price), not both."
        )

    price = pd.DataFrame(price)[list(columns)]
    index = price.index
    fill = price.to_numpy(dtype=float)
    point_size = np.array([universe[c].filled().point_size for c in columns], dtype=float)
    # A point in the price the engine sees: point size x the conversion rate on the fill bar.
    point_value = np.broadcast_to(point_size[None, :], fill.shape)
    if point_fx is not None:
        rate = pd.DataFrame(point_fx).reindex(index=index, columns=list(columns))
        point_value = point_value * rate.fillna(1.0).to_numpy(dtype=float)

    stops = {}
    risk_distance = None
    stop_columns = []
    trail_distance = None

    # In points: a distance in price is points x point size, and vectorbt's fraction is that
    # distance over the fill price, the anchor the front door sets (`stop_entry_price`).
    for name, points in (("target_points", target_points), ("stop_points", stop_points),
                         ("trail_points", trail_points)):
        resolved = stop_values(points, index, columns, name)
        if resolved is None:
            continue
        frame, named = resolved
        distance = frame * point_value
        fraction = distance / fill
        if name == "target_points":
            stops["tp_stop"] = fraction
            continue
        # A trailing stop is an ordinary stop with `sl_trail=True`; there is no separate
        # distance field upstream, so a trail overrides a fixed stop.
        stops["sl_stop"] = fraction
        risk_distance, stop_columns = distance, named
        if name == "trail_points":
            stops["sl_trail"] = True
            trail_distance = distance

    # Natively, as fractions of the price (vectorbt's own `sl_stop` / `tp_stop`).
    for name, native in (("sl_stop", sl_stop), ("tp_stop", tp_stop)):
        if native is None:
            continue
        frame, named = stop_values(native, index, columns, name)
        # ⚠️ vectorbt reads a per-bar `sl_stop` at the FILL bar, so a stop worked out at the
        # signal bar's close and aligned with the signals was read a bar late, lookahead.
        # A per-bar value is replaced by its shifted frame. A scalar passes through as
        # given; a per-column value goes as its frame, since vectorbt would broadcast a
        # Series keyed by column along the bars.
        stops[name] = native if np.ndim(native) == 0 else frame
        if name == "sl_stop":
            risk_distance, stop_columns = frame * fill, named

    # As price levels, read at the signal bar and placed on the entry's side of the fill.
    rejected = None
    trail_level = None
    if any(v is not None for v in (stop_price, target_price, trail_price)):
        def side(signal):
            if signal is None:
                return np.zeros(fill.shape, dtype=bool)
            frame = pd.DataFrame(signal).reindex(index=index, columns=list(columns))
            return frame.fillna(False).to_numpy(dtype=bool)

        known = long_open is not None or short_open is not None
        longs, shorts = side(long_open), side(short_open)
        one = longs ^ shorts                       # both at once is a conflict: no entry
        rejected = np.zeros(fill.shape, dtype=bool)
        for name, level_value in (("target_price", target_price), ("stop_price", stop_price),
                                  ("trail_price", trail_price)):
            if level_value is None:
                continue
            frame, named = stop_values(level_value, index, columns, name)
            level = frame.to_numpy(dtype=float)
            with np.errstate(invalid="ignore", divide="ignore"):
                below = (fill - level) / fill      # a long's stop, a short's target
                above = (level - fill) / fill      # a short's stop, a long's target
            for_long, for_short = (above, below) if name == "target_price" else (below, above)
            if known:
                fraction = np.where(longs & one, for_long, np.where(shorts & one, for_short, np.nan))
                bad = one & np.isfinite(level) & np.isfinite(fill) & ~(fraction > 0)
                fraction = np.where(bad, np.nan, fraction)
                rejected |= bad
            else:
                fraction = np.abs(below)
            fraction = pd.DataFrame(fraction, index=index, columns=list(columns))
            if name == "target_price":
                stops["tp_stop"] = fraction
                continue
            stops["sl_stop"] = fraction
            risk_distance, stop_columns = fraction * fill, named
            if name == "trail_price":
                stops["sl_trail"] = True
                trail_level = frame
        rejected = pd.DataFrame(rejected, index=index, columns=list(columns))

    if sl_trail is not None:
        if isinstance(sl_trail, pd.DataFrame):
            stops["sl_trail"] = (sl_trail.reindex(index=index, columns=list(columns))
                                 .shift(1).fillna(False).astype(bool))
        else:
            stops["sl_trail"] = sl_trail

    return dict(stops=stops, risk_distance=risk_distance, stop_columns=stop_columns,
                trail_distance=trail_distance, trail_level=trail_level, rejected=rejected)


def resolve_stops(universe, price, columns, stop_points=None, target_points=None,
                  trail_points=None):
    """Convert point stops into the price fractions vectorbt's native SL/TP expects.

    Native `sl_stop` / `tp_stop` are **fractions of price**, not distances, so the
    conversion has to happen somewhere. Doing it here means the user types the units they
    actually trade in. See `stop_plan` for the shapes a stop may take (a number, one per
    column, or one per bar shifted with the signals) and for native fractions.

    A useful consequence of staying on the native fields: fractional stops are
    FX-invariant under D4's uniform scaling.
    """
    return stop_plan(universe, price, columns, stop_points=stop_points,
                     target_points=target_points, trail_points=trail_points)["stops"]


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
    short_entries=None,
    short_exits=None,
    # sizing
    risk_pct=None, risk_amount=None, stake=None, notional=None, size=None, notional_pct=None,
    # stops
    stop_points=None, target_points=None, trail_points=None,
    sl_stop=None, tp_stop=None, sl_trail=None,
    stop_price=None, target_price=None, trail_price=None,
    long_open=None, short_open=None,
    # data
    volume=None, rates=None, fx=None, base_currency=None,
    fx_markup=None, fx_markup_currency=None, point_fx=None,
    # account
    init_cash=None,
    allow_partial=True,
    risk_basis=None,
    order_bars=None,
    leverage_basis=None,
    ceilings=None,
    # everything else
    spec_kwargs=None,
    passthrough=None,
):
    """Run the whole front-door pipeline and return everything a constructor needs.

    Args:
        sl_stop, tp_stop, sl_trail: vectorbt's own stop arguments, when given. A per-bar
            value is shifted with the signals, and risk sizing reads `sl_stop`.
        stop_price, target_price, trail_price: the stop or target as a price level, per
            bar and shifted with the signals (see `stop_plan`).
        long_open, short_open: boolean frames of the bars long and short entries fill on,
            which put a level on the right side of the fill. An entry whose level is on
            the wrong side is not placed (`rejected`), and the warnings leave it out.
        base_currency: the ACCOUNT currency (the front door's `account_currency`).
        fx_markup: the FX conversion markup, a fraction. Charged as slippage on every
            column except those already quoted in the account currency.
        fx_markup_currency: the currency the prices were converted into, used to tell
            which columns were converted when `base_currency` is not given.
        point_fx: the rate each column was converted at, per bar, so that everything in
            points (stops, targets, trails, the spread) is in the account currency.
        allow_partial: whether an entry bigger than its leverage allows is cut down to
            the limit (True) or not placed (False); the same for vectorbt's cash check.
        risk_basis: the money `risk_pct` is a fraction of, per fill bar and column, when
            it is not `init_cash` (see `resolve_size`).
        order_bars: boolean frame of the bars an order is placed on (the shifted entries).
            The sizing warnings measure only those. None measures every bar.
        leverage_basis: the equity each entry's leverage limit is measured against, per
            fill bar and column (the equity at the signal bar's close). None measures it
            against `init_cash` (see `cap_to_leverage`).
        ceilings: the most each entry may be, per fill bar and column, in engine units, as
            the front door found it funded on an earlier run (`_cash_ceilings`); NaN for
            none. Rounded down to the size step here.

    Returns:
        dict with `universe`, `strategy`, `columns`, `price`, `fill_price`, `size`,
        `order_kwargs`, `stops`, `trail_distance`, `trail_level`, `rejected` and
        `risk_distance`.
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
    strategy = strategy_of(entries, exits, weights, schedule, short_entries, short_exits)

    # Fail on a spec/price mismatch before doing any work, in both directions.
    universe.order_for(columns)

    # Stops first: risk sizing reads their distance.
    plan = stop_plan(
        universe, fill, columns,
        stop_points=stop_points, target_points=target_points, trail_points=trail_points,
        sl_stop=sl_stop, tp_stop=tp_stop, sl_trail=sl_trail, point_fx=point_fx,
        stop_price=stop_price, target_price=target_price, trail_price=trail_price,
        long_open=long_open, short_open=short_open,
    )
    if plan["rejected"] is not None:
        _warn_if_levels_rejected(plan["rejected"], order_bars)
        if order_bars is not None:
            # A rejected entry places no order, so it is not measured by the warnings.
            order_bars = pd.DataFrame(order_bars).reindex(index=fill.index, columns=columns) \
                .fillna(False).astype(bool) & ~plan["rejected"]

    def resolved_for(sizes):
        # ⚠️ No `equity=`, so no `max_size` (OpenStrategy Studio issue 32). vectorbt applies
        # `max_size` to EVERY order, the one that closes a position included: an exit worth more
        # than leverage x the starting cash at its price was cut short, and a remainder under the
        # minimum size was never sold, which blocked every later entry. The leverage limit is a
        # limit on the position, so it is applied to the entries, below (`cap_to_leverage`).
        return universe.resolve(
            fill,
            rates=rates,
            equity=None,
            fx_table=fx,
            base_currency=base_currency,
            volume=volume,
            size=sizes,
            notional=_notional_of(sizes, fill, weights, init_cash, columns),
            fx_markup=per_column_fx_markup(
                universe, columns, fx_markup,
                base_currency if base_currency is not None else fx_markup_currency,
            ),
            point_fx=point_fx if point_fx is not None else pd.DataFrame(1.0, index=fill.index,
                                                                        columns=columns),
        )

    # Whether any cost depends on the size ordered: a commission per unit or with a minimum, or
    # automatic slippage. Without one, the costs are the same whatever the sizes, so the re-sizing loop
    # resolves them once (`kept`, below) and can replay the simulation with new sizes alone.
    costs_follow_size = _costs_follow_size(universe, columns)
    kept = {}

    def sized(risk_basis, leverage_basis, ceilings):
        """The entry sizes for one account basis, leverage basis and set of ceilings, the sizes asked
        before the leverage limit, and the costs the sizes pay: `(size, asked, resolved)`.

        `prepare` calls it once, and the front door's re-sizing loop again for each pass
        (`ctx["resize"]`, OpenStrategy Studio issue 37), so the loop sizes exactly as a whole
        preparation does. The first call goes through the whole functions and keeps their inputs; a
        pass goes straight to the same arithmetic on them (the `*_core` functions, which the whole
        functions call too): the share of the basis (`size_from_basis`), the rounding (`quantize_core`),
        the leverage limit (`cap_core`) and the ceilings (`ceiling_core`). The sizes asked, when they
        are not a share of a basis that moves, and the costs, when they do not follow the size, are
        kept as they are.
        """
        moves = risk_basis is not None and (risk_pct is not None or notional_pct is not None)
        again = "asked" in kept and not costs_follow_size          # a pass of the re-sizing loop
        index = fill.index
        if not again:
            share = {}
            engine_size = resolve_size(
                universe, fill, columns,
                risk_pct=risk_pct, risk_amount=risk_amount, stake=stake,
                notional=notional, size=size, notional_pct=notional_pct, init_cash=init_cash,
                stop_distance=plan["risk_distance"], stop_columns=plan["stop_columns"],
                risk_basis=risk_basis, order_bars=order_bars, keep=share,
            )
            engine_size = quantize_engine_size(universe, engine_size, columns, order_bars=order_bars)
            kept.update(asked=engine_size, share=share, terms=column_terms(universe, columns))
        elif moves:
            raw = size_from_basis(kept["share"], _account_money(index, columns, init_cash, risk_basis), risk_pct)
            engine_size = quantize_core(raw, kept["terms"])          # an array, framed once, below
        else:
            engine_size = kept["asked"]
        asked = engine_size

        def costs(sizes):
            if costs_follow_size:
                return resolved_for(sizes)
            if "costs" not in kept:
                kept["costs"] = resolved_for(sizes)
            return dict(kept["costs"])

        resolved = costs(engine_size)
        if engine_size is not None and (init_cash is not None or leverage_basis is not None):
            # The leverage limit, on the entries only, measured against the equity at the signal
            # bar when the front door knows it (issue 33), rounded down to the size step and
            # allowing for the spread and commission, so vectorbt never has to cut the order to
            # the cash itself (issue 34).
            account = _account_money(index, columns, init_cash, leverage_basis)
            if again:
                capped = cap_core(np.asarray(asked, dtype=float), np.asarray(account, dtype=float), *kept["cap"],
                                  kept["terms"], allow_partial)
                if ceilings is not None:
                    # The front door's ceilings are on these bars and columns already (`_account_money`).
                    roof = _account_money(index, columns, None, ceilings)
                    capped = ceiling_core(capped, roof, kept["terms"])
                engine_size = pd.DataFrame(capped, index=index, columns=columns)
            else:
                engine_size = cap_to_leverage(universe, engine_size, fill, columns, account, resolved,
                                              partial=allow_partial)
                if ceilings is not None:
                    engine_size = _apply_ceilings(universe, engine_size, ceilings, columns)
                kept["cap"] = cap_arrays(fill, index, columns, resolved, engine_size.shape)
            if costs_follow_size and not _same_frame(engine_size, asked):
                # Size-dependent costs (a minimum commission, automatic slippage) follow the sizes
                # actually ordered.
                resolved = resolved_for(engine_size)
        if engine_size is not None and not isinstance(engine_size, pd.DataFrame):
            engine_size = pd.DataFrame(engine_size, index=index, columns=columns)
        if engine_size is not None:
            # Sizes are already quantized above, so the engine must not floor them again —
            # doing so would knock a clean multiple down one further increment, and its floor
            # would apply to the orders that close a position too (`sell_nb` floors a close).
            resolved["size_granularity"] = np.nan
        return engine_size, asked, resolved

    engine_size, asked, resolved = sized(risk_basis, leverage_basis, ceilings)
    order_kwargs = collision_check(resolved, passthrough)

    warn_if_cash_constrained(universe, asked, fill, columns, init_cash,
                             partial=allow_partial, order_bars=order_bars,
                             basis=leverage_basis)

    return dict(
        universe=universe,
        strategy=strategy,
        columns=columns,
        price=price,
        fill_price=fill,
        size=engine_size,
        order_kwargs=order_kwargs,
        stops=plan["stops"],
        trail_distance=plan["trail_distance"],
        trail_level=plan["trail_level"],
        rejected=plan["rejected"],
        risk_distance=plan["risk_distance"],
        # The re-sizing loop's passes (OpenStrategy Studio issue 37): the sizes for another account,
        # sized as above. Its warnings are the caller's to keep or drop.
        resize=lambda risk_basis, leverage_basis, ceilings: sized(risk_basis, leverage_basis, ceilings)[0],
        costs_follow_size=costs_follow_size,
    )


def _costs_follow_size(universe, columns) -> bool:
    """Whether any column's costs depend on the size ordered (`vectorbtx.costs`).

    A commission per unit, or one with a minimum, is a fixed fee worked out from each order's size or
    notional; automatic slippage scales with the order against the volume. A spread bet pays no
    commission. Every other cost is a fraction of the price or a fixed fee per order.
    """
    for col in columns:
        spec = universe[col].filled()
        if spec.slippage == "auto":
            return True
        if spec.mode == "spreadbet":
            continue
        # A minimum of zero is no minimum (`costs.commission`, issue 38).
        if spec.commission_min or np.any(np.asarray(spec.commission_per_unit or 0.0) != 0.0):
            return True
    return False


def _warn_if_levels_rejected(rejected, order_bars) -> None:
    """Say how many entries were not placed because a level was on the wrong side."""
    counts = rejected.sum(axis=0)
    counts = counts[counts > 0]
    if counts.empty:
        return
    if order_bars is not None:
        totals = pd.DataFrame(order_bars).reindex(index=rejected.index, columns=rejected.columns)
        totals = totals.fillna(False).astype(bool).sum(axis=0)
        detail = ", ".join(f"{col}: {int(n)} of {int(totals[col])}" for col, n in counts.items())
    else:
        detail = ", ".join(f"{col}: {int(n)}" for col, n in counts.items())
    warnings.warn(
        f"Entries NOT PLACED because a stop or target level was on the wrong side of the "
        f"price they would fill at ({detail}): a long's stop at or above its fill, or its "
        f"target at or below it, and the reverse for a short. A broker rejects such an "
        f"order. The level is read at the signal bar's close and the order fills at the "
        f"next open, so a gap through the level rejects it too.",
        stacklevel=4,
    )


def per_column_fx_markup(universe, columns, fx_markup, account_currency=None):
    """The FX conversion markup for each column, or None when there is none.

    Only a column that was actually converted pays it. One already quoted in the account
    currency was multiplied by a rate of 1 and crossed no exchange, so charging it the
    markup would over-cost every domestic holding in a mixed book. When either currency is
    unknown the column is charged: over-costing is the safer error.
    """
    if fx_markup is None:
        return None
    from ..instruments.defaults import major_currency

    markup = float(fx_markup)
    out = {}
    for col in columns:
        # Pence to pounds crosses no exchange: a `GBp` share in a GBP account pays none.
        quote = major_currency(universe[col].quote_currency)
        domestic = account_currency is not None and quote is not None and quote == account_currency
        out[col] = 0.0 if domestic else markup
    return out


def warn_if_cash_constrained(universe, engine_size, price, columns, init_cash,
                             partial=True, order_bars=None, basis=None) -> None:
    """Warn when a position exceeds even the **levered** buying power.

    Leverage is funded, not merely capped — `vectorbtx.margin.buying_power` hands the
    engine `equity x leverage` and `Portfolio.get_init_cash` subtracts the loan back out
    — so a position within its leverage limit is filled in full. What remains possible is
    asking for more than the leverage itself allows.

    ⚠️ That case is a **cut, not a rejection**, by default: `cap_to_leverage` cuts the entry
    down to the limit, rounded down to the size step (`when_cash_short="partial"`), or does
    not place it (`"skip"`). Either way the backtest runs with smaller positions than asked,
    and the equity curve looks plausible. Hence the warning, naming the shortfall. Only the
    entries are cut: an exit always closes the whole position (OpenStrategy Studio issue 32).

    ⚠️ **Only the bars an order is placed on are measured** (`order_bars`). A size is
    resolved for every bar, and a stake or risk size is the same on each, so taking the
    largest value over the whole history named a figure no trade ever reached — the stake
    times the highest price in ten years — and reported an early, correctly sized trade as
    cut down once the price rose (OpenStrategy Studio issue 22). The warning names the
    largest order that does not fit, and its date. With `basis` (the equity at each signal
    bar, which the front door works out from the simulation) each order is measured against
    that equity times the leverage, as `cap_to_leverage` measures it (issue 33); without it,
    against the starting cash.
    """
    if engine_size is None or (init_cash is None and basis is None):
        return
    columns = list(columns)
    frame = pd.DataFrame(engine_size).reindex(columns=columns)
    px = pd.DataFrame(price)[columns].reindex(index=frame.index)
    notional = (frame.abs() * px).to_numpy(dtype=float)
    if order_bars is not None:
        placed = pd.DataFrame(order_bars).reindex(index=frame.index, columns=columns)
        placed = placed.fillna(False).to_numpy(dtype=bool)
        notional = np.where(placed, notional, np.nan)

    if basis is not None:
        own = pd.DataFrame(basis).reindex(index=frame.index, columns=columns)
        own = own.to_numpy(dtype=float)
    else:
        # init_cash may be per column when the pools are separate.
        cash = np.asarray(init_cash, dtype=float)
        per = np.full(len(columns), float(cash)) if cash.ndim == 0 else cash.ravel()[: len(columns)]
        own = np.broadcast_to(per, notional.shape)

    offenders = {}
    for k, col in enumerate(columns):
        power = own[:, k] * float(universe[col].filled().leverage or 1.0)
        with np.errstate(invalid="ignore"):
            short = np.isfinite(notional[:, k]) & (notional[:, k] > power * (1.0 + 1e-9))
        if not short.any():
            continue
        worst = int(np.nanargmax(np.where(short, notional[:, k], -np.inf)))
        offenders[col] = (notional[worst, k], power[worst], frame.index[worst], int(short.sum()))

    if not offenders:
        return
    detail = ", ".join(
        f"{c}: {n} order(s), the largest needs {want:,.0f} on {_when(when)} but its "
        f"leverage funds only {power:,.0f}"
        for c, (want, power, when, n) in offenders.items()
    )
    if not partial:
        warnings.warn(
            f"Sized positions exceed their funded buying power ({detail}). With "
            f"when_cash_short='skip' these orders are NOT PLACED AT ALL, so the backtest "
            f"will run with fewer trades than signals. Raise `leverage` or reduce the size.",
            stacklevel=3,
        )
        return
    warnings.warn(
        f"Sized positions exceed their funded buying power ({detail}). These entries are "
        f"CUT DOWN to what the leverage funds (rounded down to the size step) rather than "
        f"rejected, so the backtest will run and quietly understate the position. Raise "
        f"`leverage`, reduce the size, or pass when_cash_short='skip' to place nothing instead.",
        stacklevel=3,
    )


def _when(label) -> str:
    """A bar's label for a message: the date alone for a midnight timestamp."""
    if isinstance(label, pd.Timestamp):
        return str(label.date()) if label == label.normalize() else str(label)
    return str(label)


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
