"""K1 — returns metrics: Ulcer, UPI, M2, Treynor, geometric Sharpe, SNR, K-ratio."""

from . import nb
from .register import register_returns_metrics, METRICS, METHODS

__all__ = ["nb", "register_returns_metrics", "METRICS", "METHODS"]
