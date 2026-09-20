"""Modes: user units <-> engine size (Section H)."""

from .base import ModeAdapter, adapters, get_adapter, register
from .cfd import CFDAdapter
from .equity import EquityAdapter
from .fx_spot import FXSpotAdapter
from .helpers import (
    pnl,
    points_to_price_fraction,
    price_fraction_to_points,
    size_for_notional,
    size_for_risk,
    value_per_point,
)
from .spreadbet import SpreadBetAdapter

__all__ = [
    "ModeAdapter",
    "get_adapter",
    "adapters",
    "register",
    "SpreadBetAdapter",
    "CFDAdapter",
    "FXSpotAdapter",
    "EquityAdapter",
    "points_to_price_fraction",
    "price_fraction_to_points",
    "size_for_risk",
    "size_for_notional",
    "value_per_point",
    "pnl",
]
