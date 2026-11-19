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
        stop_points: entry-to-stop distance in points — a number, or one per column. Used
            when `stop_distance` is not given; a per-bar frame here is read as it stands,
            on the fill bars (the front door shifts per-bar stops before calling).
        init_cash: account size, which `risk_pct` is a fraction of.
        stop_distance: the stop-loss distance **in price** at each fill bar, from
            `stop_plan` — whichever stop was given (points, a fraction of the price, a
            trailing stop's first distance). Overrides `stop_points`.
        stop_columns: the columns a stop was given for. A column with none is sized from
            its leverage (see Notes). Default: every column with a distance anywhere.
        risk_basis: the money `risk_pct` is a fraction of, per fill bar and column — the
            balance or equity at the signal bar's close (`risk_of`). Default `init_cash`.
        order_bars: boolean frame of the bars an order is placed on, for the warnings.

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

        `risk_pct` sizes off `init_cash` unless `risk_basis` is given; the front door's
        `risk_of="balance" | "equity"` passes one, worked out from the simulation.

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

    price = pd.DataFrame(price)[list(columns)]

    if size is not None:
        return _spread(size, price, columns)

    if risk_pct is not None or risk_amount is not None:
        if risk_pct is not None and init_cash is None and risk_basis is None:
            raise BacktestError("`risk_pct` is a fraction of `init_cash`; give `init_cash=`.")
        return _risk_sizes(
            universe, price, columns, risk_pct=risk_pct, risk_amount=risk_amount,
            init_cash=init_cash, stop_points=stop_points, stop_distance=stop_distance,
            stop_columns=stop_columns, risk_basis=risk_basis, order_bars=order_bars,
        )

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


def _risk_sizes(universe, price, columns, risk_pct, risk_amount, init_cash, stop_points,
                stop_distance, stop_columns, risk_basis, order_bars):
    """Engine-unit sizes that lose the risk at the stop, or from the leverage with none.

    See `resolve_size` for the rules; this is its risk branch.
    """
    columns = list(columns)
    index = price.index
    fill = price.to_numpy(dtype=float)

    # --- the money at risk, per fill bar and column ---
    if risk_pct is None:
        amount = np.full(fill.shape, float(risk_amount))
    elif risk_basis is not None:
        basis = pd.DataFrame(risk_basis).reindex(index=index, columns=columns)
        amount = float(risk_pct) * basis.to_numpy(dtype=float)
    else:
        # `init_cash` may be per column — separate cash pools size their own risk. Keeping
        # it per column matters: a £30,000 sleeve risking 1% is not the same trade as a
        # £20,000 one, and collapsing to a scalar would silently equalise them.
        cash = np.asarray(init_cash, dtype=float)
        per = np.full(len(columns), float(cash)) if cash.ndim == 0 else cash.ravel()[: len(columns)]
        amount = float(risk_pct) * np.broadcast_to(per, fill.shape)

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

    out = np.full(fill.shape, np.nan)
    from_leverage = {}
    skipped = {}
    for k, col in enumerate(columns):
        if col in stop_columns:
            d = distance[:, k]
            with np.errstate(divide="ignore", invalid="ignore"):
                out[:, k] = np.where(np.isfinite(d) & (d > 0), amount[:, k] / d, np.nan)
            missing = placed[:, k] & ~(np.isfinite(d) & (d > 0))
            if missing.any():
                skipped[col] = (int(missing.sum()), int(placed[:, k].sum()))
        else:
            # V = R x L: the trade's margin is the risk (issue 28).
            leverage = float(universe[col].filled().leverage or 1.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                out[:, k] = amount[:, k] * leverage / fill[:, k]
            from_leverage[col] = leverage

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
        q_user, _ = quantize(pd.Series(user, index=frame.index), spec)
        q_arr = np.asarray(q_user, dtype=float)

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
              trail_points=None, sl_stop=None, tp_stop=None, sl_trail=None, point_fx=None):
    """Every stop and target, resolved: what vectorbt gets, and the distances sizing reads.

    Points are converted to the price fractions vectorbt's native `sl_stop` / `tp_stop`
    expect, against the fill price (the open the order fills at). Native fractions given
    per bar are shifted with the signals like points; a scalar or per-column native value
    passes through unchanged.

    `point_fx` is the rate each column's price was converted into the account currency
    at, per bar (`vectorbtx.fx.convert.fx_rate_of`). A point is a unit of the native price,
    so on a converted price it is `point_size x rate`; without the rate a 50-cent stop on a
    USD share converted to GBP sat 50 pence away (OpenStrategy Studio issue 29).

    Returns:
        dict with
        * `stops` — keyword arguments for `from_signals` (`sl_stop`, `tp_stop`, `sl_trail`);
        * `risk_distance` — the stop-loss distance in price at each fill bar, NaN where
          there is none, or None with no stop at all;
        * `stop_columns` — the columns a stop-loss argument named;
        * `trail_distance` — a trailing stop's distance in price at each fill bar, when it
          was given in points (see `vectorbtx.frontend.trail`), else None.

    Raises:
        BacktestError: when a stop or target is given both in points and natively.
    """
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
    if sl_trail is not None:
        if isinstance(sl_trail, pd.DataFrame):
            stops["sl_trail"] = (sl_trail.reindex(index=index, columns=list(columns))
                                 .shift(1).fillna(False).astype(bool))
        else:
            stops["sl_trail"] = sl_trail

    return dict(stops=stops, risk_distance=risk_distance, stop_columns=stop_columns,
                trail_distance=trail_distance)


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
    risk_pct=None, risk_amount=None, stake=None, notional=None, size=None,
    # stops
    stop_points=None, target_points=None, trail_points=None,
    sl_stop=None, tp_stop=None, sl_trail=None,
    # data
    volume=None, rates=None, fx=None, base_currency=None,
    fx_markup=None, fx_markup_currency=None, point_fx=None,
    # account
    init_cash=None,
    allow_partial=True,
    risk_basis=None,
    order_bars=None,
    # everything else
    spec_kwargs=None,
    passthrough=None,
):
    """Run the whole front-door pipeline and return everything a constructor needs.

    Args:
        sl_stop, tp_stop, sl_trail: vectorbt's own stop arguments, when given. A per-bar
            value is shifted with the signals, and risk sizing reads `sl_stop`.
        base_currency: the ACCOUNT currency (the front door's `account_currency`).
        fx_markup: the FX conversion markup, a fraction. Charged as slippage on every
            column except those already quoted in the account currency.
        fx_markup_currency: the currency the prices were converted into, used to tell
            which columns were converted when `base_currency` is not given.
        point_fx: the rate each column was converted at, per bar, so that everything in
            points (stops, targets, trails, the spread) is in the account currency.
        allow_partial: whether a cash-short order is partially filled (True) or skipped
            (False). Only changes what the cash-constraint warning says.
        risk_basis: the money `risk_pct` is a fraction of, per fill bar and column, when
            it is not `init_cash` (see `resolve_size`).
        order_bars: boolean frame of the bars an order is placed on (the shifted entries).
            The sizing warnings measure only those. None measures every bar.

    Returns:
        dict with `universe`, `strategy`, `columns`, `price`, `fill_price`, `size`,
        `order_kwargs`, `stops`, `trail_distance` and `risk_distance`.
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
    )

    engine_size = resolve_size(
        universe, fill, columns,
        risk_pct=risk_pct, risk_amount=risk_amount, stake=stake,
        notional=notional, size=size, init_cash=init_cash,
        stop_distance=plan["risk_distance"], stop_columns=plan["stop_columns"],
        risk_basis=risk_basis, order_bars=order_bars,
    )
    engine_size = quantize_engine_size(universe, engine_size, columns, order_bars=order_bars)

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
        fx_markup=per_column_fx_markup(
            universe, columns, fx_markup,
            base_currency if base_currency is not None else fx_markup_currency,
        ),
        point_fx=point_fx if point_fx is not None else pd.DataFrame(1.0, index=fill.index,
                                                                    columns=columns),
    )
    if engine_size is not None:
        # Sizes are already quantized above, so the engine must not floor them again —
        # doing so would knock a clean multiple down one further increment.
        resolved["size_granularity"] = np.nan
    order_kwargs = collision_check(resolved, passthrough)

    warn_if_cash_constrained(universe, engine_size, fill, columns, init_cash,
                             partial=allow_partial, order_bars=order_bars,
                             risk_basis=risk_basis if risk_pct is not None else None)

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
        risk_distance=plan["risk_distance"],
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
                             partial=True, order_bars=None, risk_basis=None) -> None:
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

    ⚠️ **Only the bars an order is placed on are measured** (`order_bars`). A size is
    resolved for every bar, and a stake or risk size is the same on each, so taking the
    largest value over the whole history named a figure no trade ever reached — the stake
    times the highest price in ten years — and reported an early, correctly sized trade as
    cut down once the price rose (OpenStrategy Studio issue 22). The warning names the
    largest order that does not fit, and its date. With `risk_basis` (sizing from the
    balance or equity) each order is measured against that account size times the
    leverage, the funds it was sized from.
    """
    if engine_size is None or (init_cash is None and risk_basis is None):
        return
    columns = list(columns)
    frame = pd.DataFrame(engine_size).reindex(columns=columns)
    px = pd.DataFrame(price)[columns].reindex(index=frame.index)
    notional = (frame.abs() * px).to_numpy(dtype=float)
    if order_bars is not None:
        placed = pd.DataFrame(order_bars).reindex(index=frame.index, columns=columns)
        placed = placed.fillna(False).to_numpy(dtype=bool)
        notional = np.where(placed, notional, np.nan)

    if risk_basis is not None:
        own = pd.DataFrame(risk_basis).reindex(index=frame.index, columns=columns)
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
        f"Sized positions exceed their funded buying power ({detail}). These orders will "
        f"be PARTIALLY FILLED at a smaller "
        f"size than requested rather than rejected, so the backtest will run and quietly "
        f"understate the position. Raise `leverage`, reduce the size, or pass "
        f"`raise_reject=True` to reject instead of partially filling.",
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
