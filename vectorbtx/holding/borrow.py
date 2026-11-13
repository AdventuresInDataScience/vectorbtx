"""Section G — the borrow fee on a short in something traded directly (G6).

To short a security you trade directly (`mode="equity"`: a share, ETF, fund or commodity
product), you borrow it, and the lender charges for that.

* **It follows how the instrument is traded, not what it is.** A CFD or spread bet
  borrows nothing: its short side already pays (or earns) funding at
  `benchmark - markup`, which is the whole cost of holding it, so a borrow fee there would
  charge twice. An index can only be traded that way, so it never pays one.
  `validate_spec` rejects a `borrow_rate` outside `equity`.
* **Always a charge.** Only the short pays it, and nobody is ever credited it.
* **On the short's market value**, accrued like financing: daily, with the weekend on the
  instrument's triple day, over the market's day count.
* **No default rate.** The rate differs by security (small for a liquid large cap or a
  big ETF, far larger for a hard-to-borrow name) and can change at short notice, and no
  free source publishes it. A default would be an invented number that looks
  authoritative. So the rate is the user's `Instrument.borrow_rate`, and a short held
  with none set warns rather than borrowing for free in silence. `borrow_rate=0.0`
  states that the borrow is free.

Pure arithmetic, plus the one warning — no simulation needed to test any of it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..rates.table import DAY_COUNT, annual_to_period


def borrow_charge(short_value, annual_rate, day_multiplier=1.0, day_count: float = DAY_COUNT):
    """Cash adjustment for one accrual period. Never positive.

    Args:
        short_value: market value of the short position. Its sign is ignored.
        annual_rate: the borrow fee, annualised (`0.01` is 1% a year).
        day_multiplier: 1 normally, 3 on the triple day, 0 on non-trading days.
        day_count: ACT/360 by default; pass the market's own (`day_count_for`).

    Returns:
        Amount to **add** to cash: `-|short_value| x rate x days / day_count`.

    Examples:
        >>> borrow_charge(36_000, 0.01, day_multiplier=1, day_count=360)
        -1.0
    """
    rate = np.asarray(annual_rate, dtype=float)
    amount = np.abs(np.asarray(short_value, dtype=float)) * annual_to_period(
        rate, day_multiplier, day_count
    )
    return -amount


def borrow_series(short_value, annual_rate, multipliers, day_count: float = DAY_COUNT):
    """Vectorised `borrow_charge` across a whole backtest index.

    `short_value` is the short's market value per bar and zero wherever the position is
    flat or long, so the fee accrues only while a short is held.
    """
    return borrow_charge(short_value, annual_rate, np.asarray(multipliers, dtype=float), day_count)


def warn_unpriced_shorts(universe, position, columns, stacklevel: int = 3) -> list:
    """Warn about every column held short directly (`equity`) with no `borrow_rate`.

    Args:
        universe: the `Universe`.
        position: signed position per bar and column (`pf.assets()`).
        columns: the columns to check.

    Returns:
        The columns warned about, so callers and tests can see them.
    """
    if position is None:
        return []
    frame = pd.DataFrame(position)
    unpriced = []
    for col in columns:
        if col not in universe:
            continue
        spec = universe[col].filled()
        if "borrow" not in spec.active_costs or spec.borrow_rate is not None:
            continue
        if col in frame.columns:
            held = frame[col].to_numpy(dtype=float)
        elif frame.shape[1] == 1 and len(columns) == 1:
            held = frame.iloc[:, 0].to_numpy(dtype=float)       # an unnamed single series
        else:
            continue
        if np.any(held < 0):
            unpriced.append(col)
    if unpriced:
        warnings.warn(
            f"Short positions in {unpriced} borrowed the securities for free: no "
            f"borrow_rate is set. Shorting something you trade directly means borrowing it, "
            f"and the lender charges a fee, so the short side here looks cheaper than it "
            f"is. Set borrow_rate on each instrument as an annual fraction (0.005 for 0.5% a "
            f"year), or borrow_rate=0.0 to state that the borrow is free.",
            stacklevel=stacklevel,
        )
    return unpriced
