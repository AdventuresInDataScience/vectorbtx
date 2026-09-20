"""Section C — the instrument specification.

One frozen dataclass describing a tradable instrument. Every later section reads from
this and nothing else, so it is the vocabulary the whole library speaks.

A frozen dataclass rather than vectorbt's `Config`: fewer moving parts, better IDE
support, no config-copying semantics, and it is ours to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Literal, Optional

MODE = Literal["spreadbet", "cfd", "fx_spot", "equity"]
ASSET_CLASS = Literal["stocks", "commodities", "fx", "index"]
ROUNDING = Literal["floor", "ceil", "nearest"]
PRICE_FIELD = Literal["close", "adj_close"]
LEVERAGE_TYPE = Literal["instrument_level", "portfolio_level"]


@dataclass(frozen=True)
class Instrument:
    """A single tradable instrument.

    Attributes:
        mode: how user units convert to engine size. The one required field.
        asset_class: sub-type within a mode. Drives the default maximum leverage and the
            default cost stack (an FX CFD carries; an index CFD does not).
        point_size: **the single unit primitive** — the price increment worth one point.
            Pip position derives from it; there are deliberately no separate `tick_size`
            or `pip_decimal_place` fields. For a 5-decimal FX quote `point_size = 0.0001`
            (the pip) while price precision stays at 5 decimals: the fractional 5th digit
            must not be truncated, because a meaningful part of the spread lives there.
        contract_size: units of underlying per lot. CFD and FX spot only — **spread bets
            have none**, the stake is the size.
        increment: quantization step in user units (0.01 lots, £0.01/pt, 1 share).
        min_size: smallest order a broker will fill, in user units.
        allow_fractional: equity only; fractional shares.
        rounding: per-instrument override of the global rounding direction.
        quote_currency: what the instrument is priced in.
        base_currency: FX spot only — the other side of the pair.
        cost_currency: what *fixed* costs are quoted in. Separate from `quote_currency`
            on purpose: D4 converts a US equity's prices to GBP, but a $1 minimum
            commission is still quoted in dollars. Without this field Section E cannot
            know which fixed costs need converting.
        price_field: D10. Equity defaults to `adj_close`; nominal-point modes to `close`.
        commission_pct, commission_flat, commission_per_unit, commission_min: Section E.
        spread_points: the **full** bid/ask gap in points. Half is charged per side.
        slippage: a fraction of price, or the keyword `"auto"`.
        slippage_max, slippage_coef: parameters for `slippage="auto"`.
        financing: a fraction (annualised), or the keyword `"auto"` to pull from Section M.
        active_costs: which cost modules apply. Empty → the mode's default stack.
        leverage: maximum leverage. None → the per-mode default.
        leverage_type: per-ticker cap, or one portfolio-wide budget.
        triple_day, end_of_day: accrual timing. Broker-and-instrument properties, so they
            live here rather than in global settings.
        adv: average daily volume, for `slippage="auto"` when no volume column exists.
        meta: pass-through metadata preserved on round-trip export.
    """

    mode: MODE
    asset_class: Optional[ASSET_CLASS] = None

    # unit definition
    point_size: Optional[float] = None
    contract_size: Optional[float] = None

    # quantization
    increment: Optional[float] = None
    min_size: Optional[float] = None
    allow_fractional: bool = False
    rounding: Optional[ROUNDING] = None

    # currency
    quote_currency: Optional[str] = None
    base_currency: Optional[str] = None
    cost_currency: Optional[str] = None

    # data
    price_field: Optional[PRICE_FIELD] = None

    # costs
    commission_pct: Optional[float] = None
    commission_flat: Optional[float] = None
    commission_per_unit: Optional[float] = None
    commission_min: Optional[float] = None
    spread_points: Optional[float] = None
    slippage: Optional[object] = None
    slippage_max: Optional[float] = None
    slippage_coef: Optional[float] = None
    financing: Optional[object] = None
    adv: Optional[float] = None

    active_costs: frozenset = frozenset()

    # leverage
    leverage: Optional[float] = None
    leverage_type: LEVERAGE_TYPE = "instrument_level"

    # accrual
    triple_day: Optional[str] = None
    end_of_day: str = "22:00"

    meta: tuple = ()

    def __post_init__(self):
        from .validate import validate_spec

        validate_spec(self)

    # ############# Helpers ############# #

    def replace(self, **kwargs) -> "Instrument":
        """Return a copy with fields replaced. Re-validates."""
        return replace(self, **kwargs)

    def filled(self) -> "Instrument":
        """Return a copy with every unspecified field resolved to its mode default."""
        from .defaults import fill_defaults

        return fill_defaults(self)

    def to_dict(self) -> dict:
        """Plain dict of every field, for export and round-tripping."""
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, frozenset):
                v = "|".join(sorted(v))
            if f.name == "meta":
                v = dict(v)
            out[f.name] = v
        return out

    @property
    def is_nominal_points(self) -> bool:
        """True for modes whose P&L is priced in nominal points (spread bet, CFD).

        These cannot use an adjusted price series (D10): adjusting rescales the whole
        history on every dividend, so the number of points between two dates depends on
        when the data was pulled.
        """
        return self.mode in ("spreadbet", "cfd")

    @property
    def is_derivative(self) -> bool:
        """True where there is no ownership of the underlying."""
        return self.mode in ("spreadbet", "cfd")
