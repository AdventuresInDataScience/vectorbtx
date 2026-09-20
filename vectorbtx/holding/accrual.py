"""Section G — accrual timing.

Both holding costs charge at a broker-defined end-of-day, on **trading days only**, with
a **3x multiplier on Wednesday** covering the weekend (FX spot settles T+2, so rolling a
Wednesday position settles the following Monday). Weekends and holidays carry no separate
charge.

Do not additionally charge calendar days. Combining the two conventions bills nine days a
week.

End-of-day is **inert on daily bars** — one bar per day means the cutoff cannot change
which bar a charge lands on. It matters only on intraday data. Documented here so nobody
debugs a setting that cannot have an effect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WEEKDAYS = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}


def day_multipliers(index, triple_day: str = "WED", end_of_day: str = "22:00", holidays=None,
                    triple_multiplier: float = 3.0) -> np.ndarray:
    """Accrual multiplier per bar: 1 on trading days, 3 on the triple day, 0 otherwise.

    Args:
        index: the backtest index.
        triple_day: weekday carrying the weekend roll. Configurable, not hardcoded — UK
            brokers use Wednesday but this is a broker property, not a law.
        end_of_day: the broker's cutoff. **Inert on daily bars.**
        holidays: optional iterable of dates carrying no charge.
        triple_multiplier: the multiplier on the triple day, default 3.

    Returns:
        `np.ndarray` of multipliers, one per bar.

    Examples:
        >>> day_multipliers(pd.date_range("2024-01-01", periods=7))
        array([1., 1., 3., 1., 1., 0., 0.])     # Mon..Sun, triple on Wed
    """
    idx = pd.DatetimeIndex(index)
    target = WEEKDAYS.get(str(triple_day).upper())
    if target is None:
        raise ValueError(f"Unknown triple_day {triple_day!r}. Expected one of {sorted(WEEKDAYS)}.")

    weekday = idx.weekday.to_numpy() if hasattr(idx.weekday, "to_numpy") else np.asarray(idx.weekday)
    mult = np.ones(len(idx), dtype=float)
    mult[weekday >= 5] = 0.0
    mult[weekday == target] = triple_multiplier

    if holidays is not None:
        # ⚠️ Off by default, and deliberately so. A market holiday does not stop interest
        # accruing on the money: brokers fund any position open at the cut-off. The plan
        # read "weekends and holidays carry no *separate* charge" — true, because the
        # triple day already covers them — as "holidays carry no charge at all", which
        # leaves roughly eight days a year unfunded per instrument. Pass `holidays`
        # explicitly if a broker genuinely suspends funding.
        holiday_index = pd.DatetimeIndex(holidays).normalize()
        mult[np.isin(idx.normalize(), holiday_index)] = 0.0

    if _is_intraday(idx):
        mult = _collapse_intraday(idx, mult, end_of_day)
    return mult


def _is_intraday(idx: pd.DatetimeIndex) -> bool:
    if len(idx) < 2:
        return False
    return bool((idx[1] - idx[0]) < pd.Timedelta("1D"))


def _collapse_intraday(idx: pd.DatetimeIndex, mult: np.ndarray, end_of_day: str) -> np.ndarray:
    """On intraday bars, charge only the bar at or after the broker's cutoff."""
    cutoff = pd.Timestamp(end_of_day).time()
    out = np.zeros_like(mult)
    frame = pd.DataFrame({"m": mult, "t": idx.time}, index=idx)
    for _, day in frame.groupby(idx.normalize()):
        eligible = day.index[[t >= cutoff for t in day["t"]]]
        pick = eligible[0] if len(eligible) else day.index[-1]
        out[idx.get_loc(pick)] = frame.loc[pick, "m"]
    return out


def trading_day_count(index, **kwargs) -> float:
    """Total accrual days over the index, for reconciling a hand-computed total."""
    return float(day_multipliers(index, **kwargs).sum())
