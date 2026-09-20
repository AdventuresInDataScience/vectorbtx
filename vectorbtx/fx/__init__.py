"""FX: base-currency pre-conversion and the shared rate table (Section I)."""

from .convert import (
    ConvertedFrame,
    check_ohlc_ordering,
    convert_amount,
    convert_fixed_costs,
    to_base_currency,
)
from .table import FXError, FXTable

__all__ = [
    "FXTable",
    "FXError",
    "to_base_currency",
    "convert_amount",
    "convert_fixed_costs",
    "check_ohlc_ordering",
    "ConvertedFrame",
]
