"""Section G — holding-cost arithmetic, testable without any simulation."""

import numpy as np
import pandas as pd
import pytest

from vectorbtx.holding.accrual import day_multipliers, trading_day_count
from vectorbtx.holding.carry import carry_charge, tom_next_differential
from vectorbtx.holding.financing import financing_charge, financing_rate, financing_series
from vectorbtx.rates.table import annual_to_period

WEEK = pd.date_range("2024-01-01", periods=7, freq="D")  # Mon .. Sun


def test_day_count_conversion_is_explicit():
    """ACT/360: one day at 5.25% annualised."""
    assert annual_to_period(0.0525, 1) == pytest.approx(0.0525 / 360)
    assert annual_to_period(0.0525, 3) == pytest.approx(0.0525 * 3 / 360)


def test_multipliers_are_one_three_zero():
    assert list(day_multipliers(WEEK)) == [1, 1, 3, 1, 1, 0, 0]


def test_triple_day_is_configurable_not_hardcoded():
    assert list(day_multipliers(WEEK, triple_day="FRI")) == [1, 1, 1, 1, 3, 0, 0]


def test_no_charge_on_weekends_or_holidays():
    mult = day_multipliers(WEEK, holidays=["2024-01-02"])
    assert mult[1] == 0.0        # holiday
    assert mult[5] == mult[6] == 0.0   # weekend


def test_week_bills_seven_days_not_nine():
    """1+1+3+1+1 = 7. Charging calendar days on top would bill nine days a week."""
    assert trading_day_count(WEEK) == 7.0


def test_long_pays_benchmark_plus_markup():
    assert financing_rate(0.05, 0.03, is_long=True) == pytest.approx(0.08)
    assert financing_charge(40_000, 0.05, 0.03, True, 1) == pytest.approx(-40_000 * 0.08 / 360)


def test_short_receives_benchmark_minus_markup():
    """A simple addition here would credit shorts far too generously."""
    assert financing_rate(0.05, 0.03, is_long=False) == pytest.approx(0.02)
    assert financing_charge(40_000, 0.05, 0.03, False, 1) == pytest.approx(40_000 * 0.02 / 360)


def test_short_is_charged_when_markup_exceeds_benchmark():
    assert financing_charge(40_000, 0.01, 0.03, False, 1) < 0


def test_cash_covered_position_finances_nothing():
    assert financing_charge(0.0, 0.05, 0.03, True, 1) == pytest.approx(0.0)


def test_triple_day_multiplier_applies():
    one = financing_charge(40_000, 0.05, 0.03, True, 1)
    three = financing_charge(40_000, 0.05, 0.03, True, 3)
    assert three == pytest.approx(3 * one)


def test_total_over_a_week_matches_hand_computation():
    """One long £40,000 borrowed, held Mon-Sun at 5% + 3%."""
    mult = day_multipliers(WEEK)
    total = financing_series(40_000, np.full(7, 0.05), 0.03, True, mult).sum()
    expected = -40_000 * 0.08 * 7 / 360
    assert total == pytest.approx(expected)


def test_carry_applies_to_an_unleveraged_position():
    """Carry does not depend on borrowing — you are long one currency and short another."""
    diff = tom_next_differential(0.05, 0.02, markup=0.005, is_long=True)
    assert diff == pytest.approx(0.025)
    assert carry_charge(100_000, diff, 1) == pytest.approx(100_000 * 0.025 / 360)


def test_carry_markup_is_adverse_both_directions():
    long_diff = tom_next_differential(0.05, 0.02, markup=0.005, is_long=True)
    short_diff = tom_next_differential(0.05, 0.02, markup=0.005, is_long=False)
    assert long_diff == pytest.approx(0.03 - 0.005)
    assert short_diff == pytest.approx(-0.03 - 0.005)


def test_carry_and_financing_stay_separate():
    """Different inputs: carry takes full notional, financing takes the borrowed portion."""
    fin = financing_charge(0.0, 0.05, 0.03, True, 1)
    car = carry_charge(100_000, tom_next_differential(0.05, 0.02), 1)
    assert fin == 0.0 and car != 0.0


# ############# Rate/price index alignment ############# #


def test_a_rate_table_that_does_not_overlap_the_index_is_rejected():
    """The library's worst failure mode, caught on the way in rather than in the output.

    Yahoo hands back a **tz-aware** index; FRED hands back a **tz-naive** one. Their
    intersection is empty, so every rate reindexes to NaN and `ffill`/`bfill` have nothing
    to carry. Nothing downstream notices: NaN rates give NaN carry, a NaN equity curve,
    and `pf.stats()` that still prints a believable total return with the entire cost
    model silently contributing zero. Found by running strategy 01 against live data.
    """
    from vectorbtx.holding.callback import _rate_for

    tz_aware = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    tz_naive = pd.date_range("2024-01-01", periods=10, freq="D")
    rates = pd.DataFrame({"USD": np.full(10, 0.05)}, index=tz_naive)

    with pytest.raises(ValueError, match="does not overlap"):
        _rate_for(rates, "USD", tz_aware)
    # The message must name the actual cause, not just the symptom.
    with pytest.raises(ValueError, match="tz-aware.*but the rate index is tz-naive"):
        _rate_for(rates, "USD", tz_aware)


def test_a_matching_index_still_resolves_normally():
    """The guard must not fire on the ordinary case."""
    from vectorbtx.holding.callback import _rate_for

    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    rates = pd.DataFrame({"USD": np.full(10, 0.05)}, index=idx)
    assert _rate_for(rates, "USD", idx).tolist() == pytest.approx([0.05] * 10)


def test_a_partially_covering_rate_table_is_filled_not_rejected():
    """A short rate series is normal — holidays and publication lag. Only *no* overlap
    is an error."""
    from vectorbtx.holding.callback import _rate_for

    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    rates = pd.DataFrame({"USD": [0.05, 0.06]}, index=idx[3:5])
    out = _rate_for(rates, "USD", idx)
    assert not np.isnan(out).any()
    assert out[0] == pytest.approx(0.05)   # bfilled from the first known value
    assert out[-1] == pytest.approx(0.06)  # ffilled from the last
