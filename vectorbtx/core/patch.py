"""Section B — import-time patching (D5).

`ReturnsAccessor` and `Trades` are patched **in place**. The accessor decorator sits on
`ReturnsSRAccessor` / `ReturnsDFAccessor`, not on the base `ReturnsAccessor`, so patching
the base propagates to both without re-registration and therefore without a UserWarning.

Every attachment goes through `attach`, which **raises rather than overwrites**. That is
the only defence against silently shadowing a future upstream method of the same name.
"""

from __future__ import annotations

import typing as _t


class ShadowError(RuntimeError):
    """Raised when vectorbtX would overwrite an existing upstream attribute."""


def attach(cls: type, name: str, obj: _t.Any, overwrite: bool = False) -> None:
    """Set `cls.name = obj`, raising if the name already exists upstream."""
    if not overwrite and hasattr(cls, name):
        raise ShadowError(
            f"{cls.__name__}.{name} already exists upstream — vectorbtX would shadow it. "
            f"Rename the vectorbtX version or remove it."
        )
    setattr(cls, name, obj)


def merge_metrics(cls: type, entries: dict) -> None:
    """Merge `_metrics` entries into a `StatsBuilderMixin` subclass, in place.

    `_metrics` is a plain mutable `Config` deep-copied per instance; upstream explicitly
    sanctions in-place mutation (stats_builder.py:85).
    """
    for key, cfg in entries.items():
        if key in cls._metrics:
            raise ShadowError(
                f"{cls.__name__}._metrics['{key}'] already exists upstream — "
                f"vectorbtX would shadow it."
            )
        cls._metrics[key] = cfg


_APPLIED = False


def apply_patches() -> None:
    """Apply every vectorbtX patch. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return

    from ..metrics.returns.register import register_returns_metrics
    from ..metrics.trades.register import register_trade_metrics
    from .accessors import register_accessors

    register_returns_metrics()
    register_trade_metrics()
    register_accessors()

    _APPLIED = True
