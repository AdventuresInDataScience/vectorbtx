"""Section C — validation.

Everything here rejects at construction. The rule of thumb: any condition that would
otherwise produce a plausible-looking wrong number rather than an exception belongs in
this file.
"""

from __future__ import annotations

import warnings

from .defaults import ASSET_CLASSES, MODES, max_leverage


class SpecError(ValueError):
    """Raised when an `Instrument` describes something no broker would trade."""


def _positive(value, name: str, allow_none: bool = True):
    if value is None:
        if allow_none:
            return
        raise SpecError(f"{name} is required")
    if not isinstance(value, (int, float)) or value != value:
        raise SpecError(f"{name} must be a finite number, got {value!r}")
    if value <= 0:
        raise SpecError(f"{name} must be > 0, got {value!r} (mirrors vectorbt nb.py:414)")


def validate_spec(spec) -> None:
    """Validate one `Instrument`. Called from `Instrument.__post_init__`."""
    if spec.mode not in MODES:
        raise SpecError(f"Unknown mode {spec.mode!r}. Expected one of {MODES}.")
    if spec.asset_class is not None and spec.asset_class not in ASSET_CLASSES:
        raise SpecError(f"Unknown asset_class {spec.asset_class!r}. Expected one of {ASSET_CLASSES}.")

    _positive(spec.point_size, "point_size")
    _positive(spec.increment, "increment")
    _positive(spec.min_size, "min_size")

    # --- contract size ---
    if spec.mode == "spreadbet":
        if spec.contract_size is not None:
            raise SpecError(
                "Spread bets have no contract_size — the stake IS the size "
                "(engine_size = stake_per_point / point_size). Remove contract_size."
            )
    else:
        _positive(spec.contract_size, "contract_size")
        if spec.mode in ("cfd", "fx_spot") and spec.contract_size is None:
            # Only reachable when a caller bypasses defaults; kept explicit anyway.
            raise SpecError(f"{spec.mode} requires a contract_size")

    # --- quantization ---
    if spec.min_size is not None and spec.increment is not None:
        ratio = spec.min_size / spec.increment
        if abs(ratio - round(ratio)) > 1e-9:
            raise SpecError(
                f"min_size ({spec.min_size}) is not a whole multiple of increment "
                f"({spec.increment}); no order could ever satisfy both."
            )
    if spec.allow_fractional and spec.mode != "equity":
        # Non-equity modes are already fractional by construction; the flag would be inert.
        pass

    if spec.rounding is not None and spec.rounding not in ("floor", "ceil", "nearest"):
        raise SpecError(f"Unknown rounding {spec.rounding!r}")

    # --- currency ---
    if spec.mode == "fx_spot":
        if spec.quote_currency is not None and spec.base_currency is not None:
            if spec.quote_currency == spec.base_currency:
                raise SpecError("fx_spot base_currency and quote_currency must differ")
    elif spec.base_currency is not None and spec.asset_class != "fx":
        raise SpecError(
            f"base_currency is meaningful only for an FX underlying, not "
            f"{spec.mode!r}/{spec.asset_class!r}. Use quote_currency for what the "
            f"instrument is priced in."
        )
    elif spec.base_currency is not None and spec.base_currency == spec.quote_currency:
        raise SpecError("base_currency and quote_currency must differ")

    # --- data ---
    if spec.price_field is not None and spec.price_field not in ("close", "adj_close"):
        raise SpecError(f"Unknown price_field {spec.price_field!r}")
    if spec.is_nominal_points and spec.price_field == "adj_close":
        raise SpecError(
            f"{spec.mode!r} prices P&L in nominal points, so an adjusted close is wrong "
            f"(D10): adjustment rescales the whole history on every dividend, so the "
            f"number of points between two dates changes with when the data was pulled. "
            f"Use price_field='close'."
        )

    # --- leverage ---
    if spec.leverage_type not in ("instrument_level", "portfolio_level"):
        raise SpecError(f"Unknown leverage_type {spec.leverage_type!r}")
    if spec.leverage is not None:
        _positive(spec.leverage, "leverage")
        cap = max_leverage(spec.mode, spec.asset_class)
        if spec.leverage > cap:
            warnings.warn(
                f"leverage={spec.leverage} exceeds the {spec.mode}/{spec.asset_class} "
                f"default maximum of {cap}. Permitted, but check your broker allows it.",
                stacklevel=3,
            )

    # --- costs ---
    for name in ("commission_pct", "commission_flat", "commission_per_unit", "commission_min"):
        v = getattr(spec, name)
        if v is not None and v < 0:
            raise SpecError(f"{name} must be >= 0, got {v!r}")
    if spec.spread_points is not None and spec.spread_points < 0:
        raise SpecError(f"spread_points must be >= 0, got {spec.spread_points!r}")

    # ⚠️ An FX-underlying position has ONE overnight charge and it is the tom-next swap,
    # which already embeds both currencies' interest rates. A leverage-driven financing
    # charge on top bills the same money twice — and it is invisible, because both are
    # plausible-looking numbers in the same direction.
    # Carry needs BOTH sides of the pair. An FX spread bet or FX CFD is denominated in the
    # account currency, but the carry it earns or pays is still the differential between
    # the two currencies of the underlying pair — so it must be able to name them. Before
    # this, base_currency was rejected outside fx_spot and every FX derivative silently
    # accrued zero carry.
    if "carry" in spec.active_costs and spec.asset_class == "fx":
        if spec.base_currency is None or spec.quote_currency is None:
            warnings.warn(
                f"{spec.mode}/fx has carry active but no currency pair "
                f"(base_currency={spec.base_currency!r}, "
                f"quote_currency={spec.quote_currency!r}). Carry will resolve to zero. "
                f"Set both so the tom-next differential can be computed.",
                stacklevel=3,
            )

    # A point_size that contradicts the pip convention is almost always a copy-paste from
    # another pair, and it silently rescales every stop, target and risk-sized position.
    if spec.asset_class == "fx" and spec.point_size is not None and spec.quote_currency:
        from .defaults import fx_pip

        expected = fx_pip(spec.quote_currency)
        if abs(float(spec.point_size) - expected) > expected * 1e-9:
            warnings.warn(
                f"point_size={spec.point_size} for a {spec.quote_currency}-quoted pair, but "
                f"the pip convention for {spec.quote_currency} is {expected}. Every stop, "
                f"target and risk-sized position will be scaled by "
                f"{float(spec.point_size) / expected:g}x. Omit point_size to have it derived "
                f"from the quote currency, or keep it if you really do mean points rather "
                f"than pips.",
                stacklevel=3,
            )

    if {"carry", "financing"} <= set(spec.active_costs):
        raise SpecError(
            "active_costs contains both 'carry' and 'financing'. These are alternatives, "
            "not additions: carry (tom-next) already prices both currencies' interest, so "
            "adding financing charges the same cost twice. Use 'carry' for an FX "
            "underlying and 'financing' for everything else."
        )

    from ..resolve.keywords import check_keyword

    check_keyword("slippage", spec.slippage)
    check_keyword("financing", spec.financing)


def validate_universe(specs: dict) -> None:
    """Validate rules that only make sense across a whole universe.

    Currently one rule, and an important one: `portfolio_level` leverage is a *joint*
    constraint, so per-ticker leverage values combined with it must be rejected rather
    than silently resolved by picking one.
    """
    portfolio_level = {
        col: s for col, s in specs.items() if s.leverage_type == "portfolio_level"
    }
    if not portfolio_level:
        return
    values = {s.leverage for s in portfolio_level.values() if s.leverage is not None}
    if len(values) > 1:
        detail = ", ".join(f"{c}={s.leverage}" for c, s in portfolio_level.items())
        raise SpecError(
            f"leverage_type='portfolio_level' is a single portfolio-wide budget, but "
            f"per-ticker leverage values differ ({detail}). Set one shared value, or "
            f"switch to leverage_type='instrument_level'."
        )
