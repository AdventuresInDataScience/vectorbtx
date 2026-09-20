"""Rates: FRED benchmark data and the currency-keyed rate table (Section M)."""

from .fred import FREDData, cache_dir
from .table import (
    DAY_COUNT,
    DEFAULT_SERIES,
    FRED_SERIES,
    align_rates,
    annual_to_period,
    rate_table,
    static_rate_table,
)

__all__ = [
    "FREDData",
    "cache_dir",
    "rate_table",
    "static_rate_table",
    "align_rates",
    "annual_to_period",
    "DEFAULT_SERIES",
    "FRED_SERIES",
    "DAY_COUNT",
]
