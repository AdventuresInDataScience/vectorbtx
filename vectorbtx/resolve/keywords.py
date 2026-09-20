"""Section C2 — keyword cells (D8).

A cell holds either a literal or a keyword. Keywords are **per-parameter, not global**:
`'auto'` for slippage means volume-and-volatility scaled, for financing it means pull
from Section M. The vocabulary starts deliberately small — two entries — because a large
keyword set invented up front is a liability.

Unrecognised strings are fatal. A typo'd `'atuo'` falling through to a default is exactly
the silent-wrong-number failure this design exists to prevent.
"""

from __future__ import annotations

#: parameter name -> the set of keywords it accepts.
KEYWORDS = {
    "slippage": frozenset({"auto"}),
    "financing": frozenset({"auto"}),
}


class KeywordError(ValueError):
    """Raised for an unrecognised keyword, or a valid keyword on the wrong parameter."""


def is_keyword(param: str, value) -> bool:
    """True if `value` is a recognised keyword for `param`."""
    return isinstance(value, str) and value in KEYWORDS.get(param, frozenset())


def check_keyword(param: str, value, where: str = "") -> None:
    """Raise unless `value` is numeric, None, or a keyword valid for `param`."""
    if value is None or not isinstance(value, str):
        return
    allowed = KEYWORDS.get(param, frozenset())
    if value in allowed:
        return
    loc = f" ({where})" if where else ""
    for other, kws in KEYWORDS.items():
        if value in kws:
            raise KeywordError(
                f"{param}={value!r}{loc} — {value!r} is a valid keyword for "
                f"{other!r}, not for {param!r}."
            )
    if allowed:
        raise KeywordError(
            f"{param}={value!r}{loc} is not a recognised keyword. "
            f"Expected a number or one of {sorted(allowed)}."
        )
    raise KeywordError(f"{param}={value!r}{loc} must be a number; {param!r} takes no keywords.")
