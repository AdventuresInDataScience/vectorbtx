"""Section C2 — type coercion for the DataFrame front door.

The dict path arrives correctly typed; CSV and Excel do not. Expect `increment` as
`"0.01"`, `allow_fractional` as `"FALSE"`/`"0"`/`"no"`, `mode` with trailing whitespace,
Excel reformatting numbers or eating leading zeros, and a numeric-looking instrument name
coming back as a float.

Every failure here produces a plausible-looking wrong number rather than an exception, so
coercion is strict and loud: anything ambiguous raises and names the cell.
"""

from __future__ import annotations

import math

import numpy as np

TRUE_STRINGS = {"true", "t", "yes", "y", "1"}
FALSE_STRINGS = {"false", "f", "no", "n", "0"}

BOOL_FIELDS = ("allow_fractional",)
STR_FIELDS = (
    "mode",
    "asset_class",
    "rounding",
    "quote_currency",
    "base_currency",
    "cost_currency",
    "price_field",
    "leverage_type",
    "triple_day",
    "end_of_day",
)
KEYWORD_FIELDS = ("slippage", "financing")
SET_FIELDS = ("active_costs",)
COST_MODULES = frozenset({"commission", "spread", "slippage", "financing", "carry"})


class CoercionError(ValueError):
    """Raised when a spreadsheet cell cannot be coerced unambiguously."""


def is_blank(value) -> bool:
    """True for an unspecified cell.

    A blank cell is **not** NaN. vectorbt uses NaN and inf meaningfully in `Order` fields,
    so a blank becoming NaN can accidentally *mean* something. Blanks become `None`
    internally and are resolved against mode defaults; NaN appears only at emit time.
    """
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    try:
        return bool(value is np.nan)
    except Exception:  # pragma: no cover
        return False


def coerce_str(value, field: str, where: str = ""):
    """Coerce to a stripped lowercase string, or None if blank."""
    if is_blank(value):
        return None
    s = str(value).strip()
    if field in ("quote_currency", "base_currency", "cost_currency", "triple_day"):
        return s.upper()
    if field == "end_of_day":
        return s
    return s.lower()


def coerce_bool(value, field: str, where: str = ""):
    """Coerce to a bool, or None if blank. Raises on anything ambiguous."""
    if is_blank(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if value in (0, 1):
            return bool(value)
        raise CoercionError(f"{field}={value!r} {where}: expected a boolean, got a number")
    s = str(value).strip().lower()
    if s in TRUE_STRINGS:
        return True
    if s in FALSE_STRINGS:
        return False
    raise CoercionError(
        f"{field}={value!r} {where}: cannot read as a boolean. "
        f"Use one of {sorted(TRUE_STRINGS | FALSE_STRINGS)}."
    )


def coerce_float(value, field: str, where: str = ""):
    """Coerce to a float, or None if blank. Raises on anything unparseable."""
    if is_blank(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        raise CoercionError(f"{field}={value!r} {where}: expected a number, got a boolean")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    s = str(value).strip().replace(",", "").replace("%", "")
    try:
        return float(s)
    except ValueError as exc:
        raise CoercionError(f"{field}={value!r} {where}: cannot read as a number") from exc


def coerce_keyword(value, field: str, where: str = ""):
    """Coerce a keyword-capable cell: numeric first, then the parameter's keyword set.

    Order matters. Coercing numerics first means `'0.01'` is a number, not a failed
    keyword lookup; checking the keyword set second means a typo raises instead of
    silently reaching a default.
    """
    from .keywords import check_keyword

    if is_blank(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    ):
        return float(value)
    s = str(value).strip()
    try:
        return float(s.replace(",", "").replace("%", ""))
    except ValueError:
        pass
    check_keyword(field, s.lower(), where)
    return s.lower()


def coerce_name(value, where: str = "") -> str:
    """Coerce an instrument-name cell to a string.

    The name column goes through the same reader as everything else, so a numeric-looking
    ticker arrives as a float. `1234` must not become `"1234.0"` and fail to join.
    """
    if is_blank(value):
        raise CoercionError(f"blank instrument name {where}")
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value).strip()


def coerce_set(value, field: str = "active_costs", where: str = ""):
    """Coerce a `"spread|slippage"` cell back to a frozenset of cost-module names.

    This is the round-trip partner of `Instrument.to_dict`, which joins the set with
    `"|"`. An unrecognised module name raises rather than being dropped — a silently
    ignored `"slipage"` would disable a cost and flatter the backtest.
    """
    if is_blank(value):
        return frozenset()
    if isinstance(value, (set, frozenset, list, tuple)):
        parts = [str(p).strip().lower() for p in value]
    else:
        parts = [p.strip().lower() for p in str(value).split("|") if p.strip()]
    unknown = set(parts) - COST_MODULES
    if unknown:
        raise CoercionError(
            f"{field}={value!r} {where}: unknown cost module(s) {sorted(unknown)}. "
            f"Expected any of {sorted(COST_MODULES)}."
        )
    return frozenset(parts)


def coerce_field(field: str, value, where: str = ""):
    """Coerce one cell according to its field's declared type."""
    if field in SET_FIELDS:
        return coerce_set(value, field, where)
    if field in BOOL_FIELDS:
        return coerce_bool(value, field, where)
    if field in STR_FIELDS:
        return coerce_str(value, field, where)
    if field in KEYWORD_FIELDS:
        return coerce_keyword(value, field, where)
    return coerce_float(value, field, where)
