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

`trail_level_nb` does the same for `trail_price`, a level rather than a distance: see the
note above it.
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


# ############# A trailing stop at a price level (`trail_price`) ############# #
#
# A chandelier exit, or a stop under each completed bar's EMA: the stop starts at the signal
# bar's level and, on every later bar, moves to the level of the bar just completed, never
# loosening (up only for a long, down only for a short). The level at bar t's close is
# known only once bar t is complete, so bar t + 1 is the first to use it: the front door
# passes the levels shifted by one bar, as it does every per-bar input, and bar t reads
# row t. vectorbt keeps a trailing stop's anchor at the best price since entry, so the stop
# is returned as the fraction of that anchor that puts it on the ratcheted level.
#
# ⚠️ vectorbt refuses a negative stop fraction, so a level above a long's best price (or
# below a short's) cannot be stated. That happens only when the level has passed the
# market: the stop is then at the anchor, which the open is already at or through, so it
# fills at the open — the same price a stop above the market fills at. The one difference
# is a bar that gaps past the best price: there the open step leaves it, and the rest of
# the bar fills it at the same open.


@njit(cache=True)
def trail_level_nb(c, level, steps_per_bar, ratchet, owner):
    """vectorbt `adjust_sl_func_nb`: move a stop to each completed bar's level, never back.

    Args:
        c: vectorbt's `AdjustSLContext`.
        level: `(n_bars, n_cols)` level per fill bar (row t is bar t - 1's level), NaN for
            none.
        steps_per_bar: 2 on split bars (the context's indices are steps), else 1.
        ratchet: `(n_cols,)` float scratch array, NaN: each position's stop so far.
        owner: `(n_cols,)` int64 scratch array, -1: the entry each ratchet belongs to.

    Returns:
        `(stop fraction, trailing)` for vectorbt.
    """
    if c.position_now == 0 or c.init_i < 0 or not c.curr_trail or np.isnan(c.curr_stop):
        return c.curr_stop, c.curr_trail
    col = c.col
    if owner[col] != c.init_i:
        # A new position: its stop starts at the signal bar's level.
        owner[col] = c.init_i
        ratchet[col] = level[c.init_i // steps_per_bar, col]
    candidate = level[c.i // steps_per_bar, col]
    if not np.isnan(candidate):
        if np.isnan(ratchet[col]):
            ratchet[col] = candidate
        elif c.position_now > 0:
            ratchet[col] = max(ratchet[col], candidate)
        else:
            ratchet[col] = min(ratchet[col], candidate)
    stop = ratchet[col]
    anchor = c.curr_price
    if np.isnan(stop) or not (anchor > 0):
        return c.curr_stop, c.curr_trail
    if c.position_now > 0:
        fraction = (anchor - stop) / anchor
    else:
        fraction = (stop - anchor) / anchor
    if fraction < 0.0:
        fraction = 0.0
    return fraction, True
