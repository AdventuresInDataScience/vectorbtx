"""Section F — margin requirement.

`notional / leverage`, with the notional computed per mode. Nothing about leverage or
margin exists upstream: grepping `leverage|margin|liquidat|stop_out` across `vectorbt/`
returns only Plotly layout margins.

Tiered margin is deliberately not modelled. It scales with position size in reality, but
the complexity outweighs the difference outside extreme-risk strategies; users set a
sensible maximum leverage instead (Part 8 gives per-mode defaults).
"""

from __future__ import annotations

import numpy as np


def notional(user_size, price, spec) -> float:
    """Full market value of a position, regardless of cash backing.

    A 5x leveraged £10,000 margin position has £50,000 notional.

    Args:
        user_size: size in user units (£/point, lots, shares).
        price: price per engine unit.
        spec: the `Instrument`.

    Returns:
        Notional value in the quote currency.

    Examples:
        >>> notional(1.0, 500.0, uk_spreadbet)   # £1/pt at 500, point_size 0.01
        50000.0
    """
    from ..modes.base import get_adapter

    spec = spec.filled()
    engine = get_adapter(spec.mode).to_engine_units(user_size, spec, price)
    return np.abs(np.asarray(engine, dtype=float) * np.asarray(price, dtype=float))


def margin_required(user_size, price, spec) -> float:
    """Margin a broker would require for this position.

    `notional / leverage`, using `spec.leverage` (which defaults per mode and asset
    class).

    Examples:
        >>> margin_required(1.0, 500.0, uk_spreadbet_5x)
        10000.0
    """
    spec = spec.filled()
    lev = spec.leverage or 1.0
    return notional(user_size, price, spec) / lev
