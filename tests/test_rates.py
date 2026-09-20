"""Section M — day count, alignment and the no-API-key path."""

import pandas as pd
import pytest

from vectorbtx.rates.table import (
    DEFAULT_SERIES,
    align_rates,
    annual_to_period,
    rate_table,
    static_rate_table,
)

IDX = pd.date_range("2024-01-01", periods=7, freq="D")


def test_default_series_per_currency():
    assert DEFAULT_SERIES == {
        "GBP": "SONIA", "USD": "SOFR", "EUR": "ESTR", "JPY": "TONA", "CHF": "SARON",
    }


def test_act_360_day_count_worked_example():
    """One daily accrual from an annualised rate, hand-computed."""
    assert annual_to_period(0.0525, 1) == pytest.approx(0.00014583333, rel=1e-6)
    assert annual_to_period(0.0525, 1, day_count=365) != annual_to_period(0.0525, 1)


def test_library_works_with_a_static_rate_and_no_api_key():
    table = static_rate_table(["GBP", "USD"], IDX, rate=0.05)
    assert table.shape == (7, 2)
    assert (table["GBP"] == 0.05).all()


def test_static_source_accepts_per_currency_rates():
    table = static_rate_table({"GBP": 0.0525, "USD": 0.0533}, IDX)
    assert table["USD"].iloc[0] == 0.0533


def test_rate_table_static_source():
    table = rate_table(["GBP"], IDX, source="static", rate=0.04)
    assert (table["GBP"] == 0.04).all()


def test_business_day_rates_align_to_calendar_days():
    business = pd.DataFrame({"GBP": [0.05, 0.05]},
                            index=pd.to_datetime(["2024-01-01", "2024-01-02"]))
    aligned = align_rates(business, IDX, fill_policy="ffill")
    assert len(aligned) == 7
    assert aligned["GBP"].notna().all()


def test_strict_alignment_raises_on_gaps():
    business = pd.DataFrame({"GBP": [0.05]}, index=pd.to_datetime(["2024-01-03"]))
    with pytest.raises(ValueError, match="business days"):
        align_rates(business, IDX, fill_policy="strict")
