"""Section D — fill quantization.

A thin wrapper, not a component. `size_granularity` already floors in vectorbt
(`adj_size // granularity * granularity`, nb.py:145), which matches our default, so the
floor path costs nothing and simply hands parameters over. Only non-floor rounding needs
us to quantize ourselves first.

A backtest assuming 137.4829 shares is lying, and the lie compounds in small accounts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def _round_to(magnitude, increment, direction):
    """Round a positive magnitude to a multiple of `increment`."""
    if increment is None or increment <= 0:
        return magnitude
    q = magnitude / increment
    if direction == "floor":
        n = np.floor(q + EPS)
    elif direction == "ceil":
        n = np.ceil(q - EPS)
    elif direction == "nearest":
        n = np.round(q)
    else:
        raise ValueError(f"Unknown rounding direction {direction!r}")
    return n * increment


def quantize(size, spec, direction=None):
    """Round an intended size to what a broker would actually fill.

    Args:
        size: intended size in **user units** (£/point, lots, shares). Scalar or array.
        spec: the `Instrument`.
        direction: `"floor"`, `"ceil"` or `"nearest"`. None uses `spec.rounding`, which in
            turn falls back to the global default (floor).

    Returns:
        `(quantized_size, drift)` where `drift = size - quantized_size`, in user units.

    Notes:
        Magnitude is quantized and the sign preserved, so "floor" always means *smaller
        position*, which is the conservative reading in both directions.

        A magnitude below `spec.min_size` quantizes to **zero**, i.e. the order is
        rejected. It does not round up to the minimum: a broker refuses a sub-minimum
        order, it does not silently enlarge it.

    Examples:
        >>> spec = Instrument(mode="spreadbet", point_size=1.0, increment=0.01, min_size=1.0)
        >>> quantize(2.3456, spec)
        (2.34, 0.0056...)
        >>> quantize(0.5, spec)          # below the £1/point minimum
        (0.0, 0.5)
    """
    spec = spec.filled()
    direction = direction or spec.rounding or "floor"
    arr = np.asarray(size, dtype=np.float64)
    sign = np.sign(arr)
    mag = np.abs(arr)

    q = _round_to(mag, spec.increment, direction)
    q = np.where(q < spec.min_size - EPS, 0.0, q)
    out = sign * q
    drift = arr - out

    if np.isscalar(size) or np.ndim(size) == 0:
        return float(out), float(drift)
    if isinstance(size, pd.Series):
        return pd.Series(out, index=size.index, name=size.name), pd.Series(
            drift, index=size.index, name=size.name
        )
    if isinstance(size, pd.DataFrame):
        return (
            pd.DataFrame(out, index=size.index, columns=size.columns),
            pd.DataFrame(drift, index=size.index, columns=size.columns),
        )
    return out, drift


def to_order_params(spec, direction=None) -> dict:
    """Translate a spec's quantization rules into `Order` fields.

    Args:
        spec: the `Instrument`.
        direction: rounding direction override.

    Returns:
        `{"size_granularity": ..., "min_size": ...}` in **engine units**.

    Notes:
        For floor rounding the engine does the work: `size_granularity` already floors.
        For ceil or nearest, `size_granularity` is NaN (no engine quantization) and the
        caller must pre-quantize with `quantize`. Do not try to make the engine round
        differently — it has one behaviour and it is floor.
    """
    from ..modes.base import get_adapter

    spec = spec.filled()
    direction = direction or spec.rounding or "floor"
    adapter = get_adapter(spec.mode)

    increment = adapter.to_engine_units(spec.increment, spec)
    min_size = adapter.to_engine_units(spec.min_size, spec)

    if direction == "floor":
        return {"size_granularity": float(increment), "min_size": float(min_size)}
    return {"size_granularity": np.nan, "min_size": float(min_size)}


def drift_report(intended, actual, index=None, columns=None) -> pd.DataFrame:
    """Per-order quantization drift.

    Args:
        intended: intended sizes, user units.
        actual: quantized sizes, user units.
        index, columns: optional labels.

    Returns:
        DataFrame of `intended - actual`.

    Notes:
        Rounding drift must stay separate from `max_size` effects. `max_size` partially
        fills rather than rejecting, so realised size can differ from intended for reasons
        that have nothing to do with rounding — especially once Section F sets `max_size`.
        Compare against the *pre-cap* intended size to keep the two attributions distinct.
    """
    a = np.asarray(intended, dtype=np.float64)
    b = np.asarray(actual, dtype=np.float64)
    d = a - b
    if d.ndim == 1:
        d = d[:, None]
    return pd.DataFrame(d, index=index, columns=columns)


def rebalance_drift(drift) -> pd.Series:
    """Aggregate per-order drift to per-rebalance totals.

    Per-order is the raw record; per-rebalance is what anyone actually reads.
    """
    frame = pd.DataFrame(drift)
    total = frame.abs().sum(axis=1)
    return total[total > 0]
