"""Costs: per-trade cost modules composed into Order fields (Section E)."""

from .base import CostError, CostModule
from .commission import COMMISSION, Commission
from .compose import MODULES, resolve_costs
from .helpers import equivalent_flat, equivalent_pct
from .slippage import SLIPPAGE, Slippage, auto_slippage, realised_volatility
from .spread import SPREAD, Spread

__all__ = [
    "CostModule",
    "CostError",
    "Commission",
    "COMMISSION",
    "Spread",
    "SPREAD",
    "Slippage",
    "SLIPPAGE",
    "auto_slippage",
    "realised_volatility",
    "resolve_costs",
    "MODULES",
    "equivalent_pct",
    "equivalent_flat",
]
