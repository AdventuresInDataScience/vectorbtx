"""Section E — the cost-module interface.

Each module emits a **partial dict** targeting one of `fees`, `fixed_fees` or `slippage`.
A cost engine returning a single scalar total would be the wrong shape: vectorbt applies
those three fields differently, and collapsing them loses that.

Mapping:

* percentage commission     -> `fees`
* flat or per-unit commission -> `fixed_fees`
* spread AND slippage       -> `slippage`  (only one field exists, so they are **summed**)

That last line is the least obvious constraint in the section.
"""

from __future__ import annotations

from typing import Protocol


class CostModule(Protocol):
    """A composable transaction-cost model."""

    name: str

    def applies(self, spec) -> bool:
        """True if this module is in the instrument's active stack."""

    def contribute(self, spec, price, ctx) -> dict:
        """Partial `Order` kwargs, e.g. `{"fees": 0.001}` or `{"slippage": series}`."""


class _Base:
    """Shared plumbing for cost modules."""

    name = "base"

    def applies(self, spec) -> bool:
        return self.name in spec.filled().active_costs

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"


class CostError(ValueError):
    """Raised when a cost cannot be resolved — always naming the instrument."""
