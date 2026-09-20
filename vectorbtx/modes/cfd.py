"""Section H — CFD.

A derivative tracking an underlying, sized in **lots**, where one lot represents a fixed
contract size of the underlying:

    engine_size = lots * contract_size

`contract_size` alone is sufficient to describe any CFD — they all trade the same way.
Margin required is `notional / leverage`, and overnight positions incur financing.
"""

from .base import _Base, register


class CFDAdapter(_Base):
    """Lots <-> engine size."""

    name = "cfd"
    unit = "lots"

    def to_engine_units(self, user_size, spec, price=None):
        """`lots * contract_size`."""
        return user_size * spec.contract_size

    def from_engine_units(self, engine_size, spec, price=None):
        """`engine_size / contract_size`."""
        return engine_size / spec.contract_size


register(CFDAdapter())
