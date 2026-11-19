"""Section N — a trailing stop in points stays that many points behind the best price.

⚠️ **The fault this removes** (OpenStrategy Studio issue 27). vectorbt's trailing stop is a
FRACTION of the best price since entry: a long's stop sits at `peak x (1 - f)`. The front
door used to convert `trail_points` into that fraction once, against the fill price
(`points x point_size / fill`), so the distance in points grew in proportion to the price.
A 2-point trail on a long filled at 120 sat 3.33 points behind a peak of 200 (2 x 200 /
120). The error grows with the move since entry, so it flatters exactly the trades that ran
furthest. In every broker's terms a trailing stop in points is a fixed distance behind the
best price: a spread bet's 20-point trailing stop stays 20 points behind.

**The fix.** The front door installs `trail_points_nb` as vectorbt's `adjust_sl_func_nb`.
vectorbt calls it before every stop check with the current best price (`curr_price`, which
it ratchets after each check), and it returns `distance / curr_price` as the trailing
fraction, so the stop sits at `peak - distance` for a long and `trough + distance` for a
short. The distance is read on the entry bar, from the trail per fill bar the front door
resolved (shifted with the signals when given per bar).

It runs on every step of the split bars (`vectorbtx.frontend.split_bars`), unlike a user's
callback, which sees bars: it reads nothing per bar except the distance at the entry, which
it indexes by bar itself (`steps_per_bar`). So after the open step raises the best price to
the open, the rest of the bar is checked against the new peak minus the distance.

A trailing stop in % of the price (`sl_stop` with `sl_trail=True`) is a fraction of the
best price by definition and needs none of this.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def trail_points_nb(c, distance, steps_per_bar):
    """vectorbt `adjust_sl_func_nb`: keep a trailing stop `distance` in price behind the peak.

    Args:
        c: vectorbt's `AdjustSLContext`.
        distance: `(n_bars, n_cols)` trailing distance in price per fill bar, NaN for none.
        steps_per_bar: 2 on split bars (the context's indices are steps), else 1.

    Returns:
        `(stop fraction, trailing)` for vectorbt.
    """
    if c.position_now == 0 or c.init_i < 0 or not c.curr_trail or np.isnan(c.curr_stop):
        return c.curr_stop, c.curr_trail
    d = distance[c.init_i // steps_per_bar, c.col]
    if np.isnan(d) or not (c.curr_price > 0):
        return c.curr_stop, c.curr_trail
    return d / c.curr_price, True
