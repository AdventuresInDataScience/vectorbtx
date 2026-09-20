"""Section H — the mode adapter protocol.

Each adapter is essentially two functions. **Resist making them anything more**: the
moment an adapter makes a decision rather than converting a unit, the abstraction has
failed.

Every mode collapses to the same engine identity:

    P&L = engine_size x price_change

so an economically identical position expressed as a spread bet, a CFD, an FX spot
position or shares must produce identical P&L. That is the cross-mode equivalence test,
and it is the section's real deliverable.
"""

from __future__ import annotations

from typing import Protocol


class ModeAdapter(Protocol):
    """Converts between the units a user trades in and the size the engine simulates."""

    name: str

    def to_engine_units(self, user_size: float, spec, price=None) -> float:
        """User units -> engine size."""

    def from_engine_units(self, engine_size: float, spec, price=None) -> float:
        """Engine size -> user units. Must round-trip with `to_engine_units`."""

    def unit_name(self, spec) -> str:
        """Human-readable name for the user unit, for reporting."""


class _Base:
    """Shared plumbing. Adapters are stateless, so one instance each is enough."""

    name = "base"
    unit = "units"

    def unit_name(self, spec) -> str:
        return self.unit

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"


_REGISTRY: dict = {}


def register(adapter) -> None:
    """Register an adapter instance under its `name`."""
    _REGISTRY[adapter.name] = adapter


def get_adapter(mode: str) -> ModeAdapter:
    """Look up the adapter for a mode.

    Args:
        mode: one of `"spreadbet"`, `"cfd"`, `"fx_spot"`, `"equity"`, or an `Instrument`
            (whose `.mode` is used) — callers nearly always have the spec to hand.

    Returns:
        The adapter instance.

    Examples:
        >>> get_adapter("spreadbet").to_engine_units(1.0, uk_stock_spec)
        100.0
        >>> get_adapter(uk_stock_spec).to_engine_units(1.0, uk_stock_spec)
        100.0
    """
    if not _REGISTRY:
        from . import cfd, equity, fx_spot, spreadbet  # noqa: F401  (self-registering)
    mode = getattr(mode, "mode", mode)
    if mode not in _REGISTRY:
        raise ValueError(f"No adapter for mode {mode!r}. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[mode]


def adapters() -> dict:
    """Every registered adapter, keyed by mode."""
    if not _REGISTRY:
        get_adapter("equity")
    return dict(_REGISTRY)
