"""Section G — leverage financing.

Interest on borrowed money, applied **only to the borrowed portion**. A £10,000 position
fully covered by £10,000 cash finances nothing; £50,000 against £10,000 finances £40,000.

The rate is a benchmark plus a broker markup, and **the markup is always adverse**: pay
`benchmark + markup` financing a long, receive `benchmark - markup` when short. A simple
addition credits shorts too generously and flatters short-heavy backtests.

Pure arithmetic — no simulation needed to test any of it.
"""

from __future__ import annotations

import numpy as np

from ..rates.table import DAY_COUNT, annual_to_period


def financing_rate(benchmark_rate, markup: float, is_long: bool):
    """The adverse rate actually applied.

    `benchmark + markup` when long, `benchmark - markup` when short. Never a simple
    addition.

    Examples:
        >>> financing_rate(0.05, 0.03, is_long=True)
        0.08
        >>> financing_rate(0.05, 0.03, is_long=False)
        0.02
    """
    rate = np.asarray(benchmark_rate, dtype=float)
    return rate + markup if is_long else rate - markup


def financing_charge(borrowed, benchmark_rate, markup: float, is_long: bool,
                     day_multiplier=1.0, day_count: float = DAY_COUNT):
    """Cash adjustment for one accrual period, **signed**.

    Args:
        borrowed: the borrowed notional. Zero for a cash-covered position.
        benchmark_rate: annualised benchmark (Section M).
        markup: broker markup, applied adversely.
        is_long: direction of the position.
        day_multiplier: 1 normally, 3 on the triple day, 0 on non-trading days.
        day_count: ACT/360 by default.

    Returns:
        Amount to **add** to cash. Negative when financing a long; positive when a short
        earns, and negative when `benchmark - markup` goes below zero, which is correct.

    Examples:
        >>> financing_charge(40_000, 0.05, 0.03, is_long=True, day_multiplier=1)
        -8.888888888888889
        >>> financing_charge(40_000, 0.05, 0.03, is_long=False, day_multiplier=1)
        2.2222222222222223
    """
    rate = financing_rate(benchmark_rate, markup, is_long)
    amount = np.abs(np.asarray(borrowed, dtype=float)) * annual_to_period(
        rate, day_multiplier, day_count
    )
    return -amount if is_long else amount


def financing_series(borrowed, benchmark_rates, markup, is_long, multipliers,
                     day_count: float = DAY_COUNT):
    """Vectorised `financing_charge` across a whole backtest index.

    The D4 interaction, stated once: apply the position's **native-currency rate** to the
    **base-currency-equivalent notional**. Convert the amount, not the rate.
    """
    rate = financing_rate(np.asarray(benchmark_rates, dtype=float), markup, is_long)
    amount = np.abs(np.asarray(borrowed, dtype=float)) * annual_to_period(
        rate, np.asarray(multipliers, dtype=float), day_count
    )
    return -amount if is_long else amount
