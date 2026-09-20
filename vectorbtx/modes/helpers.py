"""Section H — mode-agnostic helpers.

Position sizing and point-risk are identical across modes; only the unit conversion
differs. So they are built once, above the adapters, rather than four times inside them.

SL/TP is fully native in vectorbt's `from_signals` and expressed as **fractions of
price**. These helpers convert point stops into those fractions; they do not reimplement
stop logic. A useful consequence: fractional stops are FX-invariant under D4's uniform
scaling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _like(reference, values):
    """Re-wrap `values` in the pandas container of `reference`, if it has one.

    Every helper that takes a price series must return something aligned to it. A bare
    ndarray looks fine in isolation and then reindexes against a dated frame downstream,
    silently producing a union-shaped result instead of raising.
    """
    if isinstance(reference, pd.Series):
        return pd.Series(values, index=reference.index, name=reference.name)
    if isinstance(reference, pd.DataFrame):
        return pd.DataFrame(values, index=reference.index, columns=reference.columns)
    return values


def points_to_price_fraction(points, price, spec) -> float:
    """Convert a stop distance in points to the price fraction `sl_stop` expects.

    Args:
        points: stop distance in points.
        price: entry price, in the same series the engine sees.
        spec: the `Instrument`.

    Returns:
        Fraction of price, ready for `sl_stop` / `tp_stop`.

    Examples:
        >>> points_to_price_fraction(50, 1.2345, eurusd)   # 50 pips at 1.2345
        0.00405...
    """
    spec = spec.filled()
    return _like(
        price,
        np.asarray(points, dtype=np.float64) * spec.point_size / np.asarray(price, dtype=np.float64),
    )


def price_fraction_to_points(fraction, price, spec) -> float:
    """Inverse of `points_to_price_fraction`."""
    spec = spec.filled()
    return _like(
        price,
        np.asarray(fraction, dtype=np.float64) * np.asarray(price, dtype=np.float64) / spec.point_size,
    )


def value_per_point(user_size, spec, price=None) -> float:
    """Money value of a one-point move, for a position of `user_size` user units.

    Args:
        user_size: position size in user units.
        spec: the `Instrument`.
        price: unused for every current mode; accepted for adapter symmetry.

    Returns:
        Value of one point, in the instrument's quote currency.

    Notes:
        For a spread bet this is the stake itself, which is the whole point of the mode.
    """
    from .base import get_adapter

    spec = spec.filled()
    engine = get_adapter(spec.mode).to_engine_units(user_size, spec, price)
    return engine * spec.point_size


def size_for_risk(risk_amount, stop_distance_points, spec, price=None) -> float:
    """Position size such that hitting the stop loses exactly `risk_amount`.

    Args:
        risk_amount: money at risk.
        stop_distance_points: entry-to-stop distance, in points.
        spec: the `Instrument`.
        price: entry price. Unused by the arithmetic, accepted for symmetry.

    Returns:
        Size in **user units** for this instrument's mode.

    Notes:
        One formula for every mode. P&L is `engine_size x price_change` and a stop is
        `stop_points x point_size` of price change, so

            engine_size = risk_amount / (stop_points x point_size)

        and the mode adapter converts that to user units. Identical monetary risk across
        all four modes falls straight out.

    Examples:
        >>> size_for_risk(100, 50, uk_stock_spreadbet)   # £100 at risk over 50 points
        2.0     # £2/point
    """
    from .base import get_adapter

    spec = spec.filled()
    denom = np.asarray(stop_distance_points, dtype=np.float64) * spec.point_size
    engine = np.asarray(risk_amount, dtype=np.float64) / denom
    if np.ndim(engine) == 0 and isinstance(price, (pd.Series, pd.DataFrame)):
        # A constant risk budget against a price series still wants to come back aligned,
        # so it can be multiplied by other series without silently reindexing.
        engine = _like(price, np.broadcast_to(engine, np.shape(price)))
    else:
        engine = _like(price, engine)
    return get_adapter(spec.mode).from_engine_units(engine, spec, price)


def size_for_notional(notional, price, spec):
    """User-unit size giving a target notional exposure at `price`.

    Preserves the pandas container of `price`. Returning a bare ndarray here would strip
    the index, and the result then silently reindexes against a dated frame downstream —
    producing a wrong-shaped array rather than an error.
    """
    from .base import get_adapter

    spec = spec.filled()
    engine = _like(price, np.asarray(notional, dtype=np.float64) / np.asarray(price, dtype=np.float64))
    return get_adapter(spec.mode).from_engine_units(engine, spec, price)


def pnl(user_size, entry_price, exit_price, spec) -> float:
    """P&L of a position, in the instrument's quote currency.

    The cross-mode identity in one function: every mode reduces to
    `engine_size x (exit - entry)`.
    """
    from .base import get_adapter

    spec = spec.filled()
    engine = get_adapter(spec.mode).to_engine_units(user_size, spec, entry_price)
    return engine * (np.asarray(exit_price, dtype=np.float64) - np.asarray(entry_price, dtype=np.float64))
