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


#: Stamped onto everything `attach` sets, so a later collision can tell *whose* attribute
#: it hit. The patched classes (`ReturnsAccessor`, `Trades`) are upstream's and therefore
#: shared, so this tag survives even when `vectorbtx` itself has been imported twice —
#: which is precisely the case the plain `hasattr` check cannot distinguish.
_ATTACHED_BY_VECTORBTX = "__vectorbtx_attached__"


def attach(cls: type, name: str, obj: _t.Any, overwrite: bool = False) -> None:
    """Set `cls.name = obj`, raising if the name is already taken.

    The two ways that happens need different fixes, so they get different messages:
    upstream grew a method of the same name (rename ours), or vectorbtX attached it
    already (something re-ran the patching).
    """
    if not overwrite and hasattr(cls, name):
        if getattr(getattr(cls, name), _ATTACHED_BY_VECTORBTX, False):
            raise ShadowError(
                f"{cls.__name__}.{name} was attached by vectorbtX, not by upstream, so "
                f"apply_patches() has run twice against the same class. That means the "
                f"`vectorbtx` package exists as two distinct module objects, each with "
                f"its own `_APPLIED` flag — check `sys.modules` for more than one "
                f"spelling. The import name is lowercase `vectorbtx`."
            )
        raise ShadowError(
            f"{cls.__name__}.{name} already exists upstream — vectorbtX would shadow it. "
            f"Rename the vectorbtX version or remove it."
        )
    try:
        setattr(obj, _ATTACHED_BY_VECTORBTX, True)
    except (AttributeError, TypeError):  # pragma: no cover - slots/builtins
        pass  # best-effort: without the tag we simply fall back to the upstream message
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
