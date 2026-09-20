"""Core: import hook, Portfolio subclass, accessor patching, settings."""

from .patch import apply_patches, attach, merge_metrics, ShadowError
from .settings import register_defaults, DEFAULTS, get

__all__ = [
    "apply_patches",
    "attach",
    "merge_metrics",
    "ShadowError",
    "register_defaults",
    "DEFAULTS",
    "get",
]
