"""Section H — FX spot.

Currency pairs sized in lots: standard 100,000 units of base, mini 10,000, micro 1,000.
Price moves are measured in pips.

Modern quoting is **5 decimal places, not 4** (3 for JPY, not 2). The pip remains the 4th
decimal (2nd for JPY) and the 5th gives fractional-pip precision. Arithmetically this
changes nothing, but the extra digit must not be truncated — a meaningful part of the
spread lives there, so no rounding happens anywhere in this adapter.

Overnight positions earn or pay carry, which applies regardless of leverage because you
are long one currency and short another either way.
"""

from .base import _Base, register


class FXSpotAdapter(_Base):
    """Lots <-> engine size (units of base currency)."""

    name = "fx_spot"
    unit = "lots"

    def to_engine_units(self, user_size, spec, price=None):
        """`lots * contract_size`, contract_size defaulting to a 100,000-unit standard lot."""
        return user_size * spec.contract_size

    def from_engine_units(self, engine_size, spec, price=None):
        """`engine_size / contract_size`."""
        return engine_size / spec.contract_size


register(FXSpotAdapter())
