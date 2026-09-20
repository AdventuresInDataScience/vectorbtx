"""Section E — spread.

The bid/ask gap. You buy at ask and sell at bid, so every trade starts slightly
underwater. Modelled through `slippage` (D3) — vectorbt has no bid/ask concept and one
`slippage` field per order, and asymmetric spread would need a custom order function for
every mode, dragging all cost modelling to level 3.

`spec.spread_points` is the **full** gap. Half is charged on each side, so a round trip
pays the whole spread — which is what actually happens.

Limitation, documented loudly: symmetric only. Strategies genuinely sensitive to bid/ask
asymmetry — high-frequency, market-making, anything trading the spread itself — are out
of scope.
"""

from __future__ import annotations

import numpy as np

from .base import _Base


class Spread(_Base):
    """Bid/ask gap as a fraction of price."""

    name = "spread"

    def contribute(self, spec, price, ctx) -> dict:
        spec = spec.filled()
        points = spec.spread_points or 0.0
        if points == 0:
            return {"slippage": 0.0}
        half_gap = 0.5 * points * spec.point_size
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = half_gap / np.asarray(price, dtype=float)
        return {"slippage": np.nan_to_num(frac, nan=0.0, posinf=0.0)}


SPREAD = Spread()
