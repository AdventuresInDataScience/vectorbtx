"""The `vbt.x` namespace — vectorbtX constructors, loaders and free functions.

Everything a user needs that is not bound to a `Portfolio` or a pandas object lives here,
so `vbt.x.<tab>` is a complete inventory of the extras surface.
"""

# --- instruments (C) ---
from .instruments.spec import Instrument
from .instruments.defaults import (
    MODES,
    mode_defaults,
    max_leverage,
    default_cost_stack,
    financing_basis,
    triple_day,
    fx_pip,
)

# --- universe / resolution (C2) ---
from .resolve.load import Universe, load_universe
from .resolve.export import export_universe

# --- quantization (D) ---
from .quantize.quantizer import quantize, to_order_params, drift_report

# --- modes (H) ---
from .modes.base import get_adapter, adapters
from .modes.helpers import (
    points_to_price_fraction,
    price_fraction_to_points,
    size_for_risk,
    size_for_notional,
    value_per_point,
    pnl,
)

# --- fx (I) ---
from .fx.table import FXTable
from .fx.convert import to_base_currency, convert_amount, convert_fixed_costs, as_fx_table

# --- costs (E) ---
from .costs.compose import resolve_costs
from .costs.helpers import equivalent_pct, equivalent_flat

# --- margin (F) ---
from .margin.cap import leverage_cap, max_size_for
from .margin.buying_power import buying_power, leverage_of
from .margin.requirement import margin_required, notional
from .margin.exposure import borrowed_portion, margin_level, min_margin_level

# --- construction (J) ---
from .construct.weights import equal_weight, price_weight, cap_weight
from .construct.schedule import (
    calendar_schedule,
    threshold_schedule,
    always_schedule,
    drifted_weights,
)
from .construct.rebalance import apply_leverage_budget, to_targets, validate_weights

# --- rates (M) ---
from .rates.fred import FREDData
from .rates.table import rate_table, static_rate_table, annual_to_period, day_count_for

# --- holding costs (G) ---
from .holding.financing import financing_charge, financing_rate, financing_series
from .holding.carry import carry_charge, carry_series, tom_next_differential
from .holding.accrual import day_multipliers

# --- aggregation ---
from .metrics.aggregate import aggregate

# --- front door (N) ---
from .frontend.backtest import PortfolioBacktest
from .frontend.single import SingleBacktest
from .frontend.prepare import BacktestError

__all__ = [
    "Instrument",
    "MODES",
    "mode_defaults",
    "max_leverage",
    "default_cost_stack",
    "financing_basis",
    "triple_day",
    "fx_pip",
    "Universe",
    "load_universe",
    "export_universe",
    "quantize",
    "to_order_params",
    "drift_report",
    "get_adapter",
    "adapters",
    "points_to_price_fraction",
    "price_fraction_to_points",
    "size_for_risk",
    "size_for_notional",
    "value_per_point",
    "pnl",
    "FXTable",
    "to_base_currency",
    "as_fx_table",
    "convert_amount",
    "convert_fixed_costs",
    "resolve_costs",
    "equivalent_pct",
    "equivalent_flat",
    "leverage_cap",
    "max_size_for",
    "buying_power",
    "leverage_of",
    "margin_required",
    "notional",
    "borrowed_portion",
    "margin_level",
    "min_margin_level",
    "equal_weight",
    "price_weight",
    "cap_weight",
    "calendar_schedule",
    "threshold_schedule",
    "always_schedule",
    "drifted_weights",
    "apply_leverage_budget",
    "to_targets",
    "validate_weights",
    "FREDData",
    "rate_table",
    "static_rate_table",
    "annual_to_period",
    "day_count_for",
    "financing_charge",
    "financing_series",
    "financing_rate",
    "carry_charge",
    "carry_series",
    "tom_next_differential",
    "day_multipliers",
    "aggregate",
    "PortfolioBacktest",
    "SingleBacktest",
    "BacktestError",
]
