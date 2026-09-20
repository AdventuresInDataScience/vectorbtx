"""Section G — FX carry.

The interest differential between the two currencies in a pair, using the **tom-next**
rate rather than an interpolated differential.

Carry applies to **any** FX position regardless of leverage, because you are long one
currency and short another either way. That is what makes it conceptually distinct from
financing, and why the two must not be merged: separate modules, separate rate tables,
separate multiplier tables.
"""

from __future__ import annotations

import numpy as np

from ..rates.table import DAY_COUNT, annual_to_period


def tom_next_differential(base_rate, quote_rate, markup: float = 0.0, is_long: bool = True):
    """The tom-next differential actually applied to a position.

    Long the pair means long the base currency and short the quote, so you earn the base
    rate and pay the quote rate. The markup is adverse in both directions.

    Args:
        base_rate: annualised rate for the base currency.
        quote_rate: annualised rate for the quote currency.
        markup: broker markup, always adverse.
        is_long: direction of the position.

    Examples:
        >>> tom_next_differential(0.05, 0.02, markup=0.005, is_long=True)
        0.025
    """
    diff = np.asarray(base_rate, dtype=float) - np.asarray(quote_rate, dtype=float)
    if not is_long:
        diff = -diff
    return diff - markup


def carry_charge(notional, tom_next_diff, day_multiplier=1.0, day_count: float = DAY_COUNT):
    """Cash adjustment from carry for one accrual period, **signed**.

    Args:
        notional: full position notional. Not the borrowed portion — carry does not work
            that way.
        tom_next_diff: the differential from `tom_next_differential`.
        day_multiplier: 1 normally, 3 on the triple day, 0 on non-trading days.
        day_count: ACT/360 by default.

    Returns:
        Amount to add to cash — positive for positive carry, negative for negative carry.

    Notes:
        ACT/360 slightly **overstates** a received credit, so a positive-carry position
        looks marginally better than reality. The usual "overstates = conservative"
        reasoning inverts here.

    Examples:
        >>> carry_charge(100_000, 0.025, day_multiplier=1)
        6.944444444444445
    """
    return np.abs(np.asarray(notional, dtype=float)) * annual_to_period(
        tom_next_diff, day_multiplier, day_count
    )


def carry_series(notional, base_rates, quote_rates, multipliers, markup: float = 0.0,
                 is_long: bool = True, day_count: float = DAY_COUNT):
    """Vectorised `carry_charge` across a whole backtest index."""
    diff = tom_next_differential(base_rates, quote_rates, markup, is_long)
    return carry_charge(notional, diff, multipliers, day_count)
