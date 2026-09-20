"""Section E — flat/percentage convenience.

Retail equity commission is often a flat fee (£10 per trade regardless of size), which
maps cleanly onto `fixed_fees`. But users frequently want a single percentage instead,
and £10 on £10,000 is 0.1%. These two functions do that translation and nothing else.
"""

from __future__ import annotations


def equivalent_pct(flat_fee: float, notional: float) -> float:
    """The percentage equivalent of a flat fee at a given notional.

    Args:
        flat_fee: the flat charge, e.g. 10.0.
        notional: the trade's notional value, e.g. 10000.0.

    Returns:
        Fraction of notional, e.g. 0.001.

    Examples:
        >>> equivalent_pct(10, 10_000)
        0.001
    """
    if notional == 0:
        raise ValueError("notional must be non-zero")
    return flat_fee / notional


def equivalent_flat(pct: float, notional: float) -> float:
    """The flat fee equivalent of a percentage at a given notional.

    Examples:
        >>> equivalent_flat(0.001, 10_000)
        10.0
    """
    return pct * notional
