"""Section F — maximum-leverage enforcement (F1).

Level 1: compute the maximum permissible size and pass it as `max_size`. vectorbt
**partially fills rather than rejecting** when `max_size` is exceeded, so we get clip
semantics for free; `raise_reject=True` gives reject semantics instead.

`leverage_type` splits the implementation:

* `instrument_level` — a per-column cap, computed independently, straight into
  `max_size`. Exact.
* `portfolio_level` — a *joint* constraint (total gross notional <= leverage x equity),
  and `max_size` is per-column with no knowledge of its siblings. It is enforced on the
  weight matrix in Section J, before simulation, where weights are known ahead of time
  and scaling to fit the budget is exact and cheap. The per-column cap stays as a
  backstop.

**Known gap:** a signal-driven strategy not going through Section J gets the backstop
only, so aggregate exposure can exceed the cap. Exact enforcement there needs level 3.
Documented, not built.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def leverage_cap(spec, equity, price):
    """Maximum position size at `spec.leverage`, in **engine units**.

    Args:
        spec: the `Instrument`.
        equity: account equity backing the position.
        price: price per engine unit.

    Returns:
        Maximum engine size. Feed straight to `Order.max_size`.

    Examples:
        >>> leverage_cap(cfd_5x, 10_000, 500.0)
        100.0     # £50,000 notional / £500
    """
    spec = spec.filled()
    lev = spec.leverage or 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        cap = lev * np.asarray(equity, dtype=float) / np.asarray(price, dtype=float)
    return np.where(np.isfinite(cap), cap, np.inf)


def max_size_for(specs, equity, price):
    """Per-column `max_size` array for a whole universe.

    Args:
        specs: list of `Instrument`, in resolved column order.
        equity: scalar equity, or per-column equity.
        price: price frame in the same column order.

    Returns:
        A DataFrame of engine-unit caps, aligned to `price`.
    """
    price = pd.DataFrame(price)
    # ⚠️ Per-column equity must be indexed by COLUMN, not broadcast against the price
    # series. Separate cash pools give one equity figure per sleeve, and handing the whole
    # length-`n_cols` array to a length-`n_bars` price column raises — or, worse, silently
    # aligns if the two happen to match.
    arr = None if np.isscalar(equity) else np.asarray(equity, dtype=float).ravel()
    out = {}
    for i, (spec, col) in enumerate(zip(specs, price.columns)):
        if arr is None:
            eq = equity
        elif arr.size == len(price.columns):
            eq = float(arr[i])
        else:
            eq = arr                       # per-bar equity, one series for every column
        out[col] = np.asarray(leverage_cap(spec, eq, price[col]), dtype=float)
    return pd.DataFrame(out, index=price.index)


def cap_user_size(user_size, spec, equity, price):
    """Clip a user-unit size to the leverage cap, returning `(capped, was_capped)`.

    Useful for reporting: `max_size` drift must stay attributable separately from
    quantization drift (Section D), because partial fills and rounding are different
    causes of the same symptom.
    """
    from ..modes.base import get_adapter

    spec = spec.filled()
    adapter = get_adapter(spec.mode)
    engine = np.asarray(adapter.to_engine_units(user_size, spec, price), dtype=float)
    cap = np.asarray(leverage_cap(spec, equity, price), dtype=float)
    clipped = np.clip(np.abs(engine), None, cap) * np.sign(engine)
    return adapter.from_engine_units(clipped, spec, price), bool(np.any(np.abs(engine) > cap))
