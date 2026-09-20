"""Section H — spread bet.

Sized in **£ per point**. P&L is `price_change_in_points x stake_per_point`, there is no
ownership, and — critically — **there is no contract size**. The stake *is* the size:

    engine_size = stake_per_point / point_size

Verified against three independent cases:

| Instrument            | point_size | £1/point ->            | expected |
|-----------------------|------------|------------------------|----------|
| UK stock (GBP series) | 0.01       | 100 shares             | 100 ✓    |
| EURUSD                | 0.0001     | 10,000 units = 0.1 lot | ~0.1 ✓   |
| Micro FX @ £0.10      | 0.0001     | 1,000 units = 0.01 lot | micro ✓  |

All three fall out of that one line — the "100 contract size" for UK stocks is *derived*,
not configured. Commodities take the same £1/point treatment as equities, so there is no
special case.

Spread bets are account-currency-denominated by construction, so D4's pre-conversion does
not apply to them.
"""

from .base import _Base, register


class SpreadBetAdapter(_Base):
    """£/point <-> engine size."""

    name = "spreadbet"
    unit = "£/point"

    def to_engine_units(self, user_size, spec, price=None):
        """`stake_per_point / point_size`."""
        return user_size / spec.point_size

    def from_engine_units(self, engine_size, spec, price=None):
        """`engine_size * point_size`."""
        return engine_size * spec.point_size


register(SpreadBetAdapter())
