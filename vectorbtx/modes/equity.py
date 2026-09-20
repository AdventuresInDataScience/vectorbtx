"""Section H — equity.

Buy N shares at price P; you own the asset. This is the model vectorbt natively assumes,
so the adapter is the identity. Retail commission is often a flat fee rather than a
percentage — see Section E, which maps that onto `fixed_fees`.

Equity is the mode where D10's `adj_close` default earns its keep: shares x price change
on an adjusted series gives true total return.
"""

from .base import _Base, register


class EquityAdapter(_Base):
    """Shares <-> engine size. The identity."""

    name = "equity"
    unit = "shares"

    def to_engine_units(self, user_size, spec, price=None):
        """Identity."""
        return user_size

    def from_engine_units(self, engine_size, spec, price=None):
        """Identity."""
        return engine_size


register(EquityAdapter())
