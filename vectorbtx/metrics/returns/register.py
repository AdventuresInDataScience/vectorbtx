"""Section K1 — attach the seven returns metrics to `ReturnsAccessor` (D5).

Methods are `setattr`-ed onto the *base* `ReturnsAccessor`, which is undecorated; the
accessor decorator sits on `ReturnsSRAccessor` / `ReturnsDFAccessor`, so patching the base
reaches both without re-registration and without a UserWarning.

Every attachment goes through `core.patch.attach`, which raises rather than overwrites.
"""

from __future__ import annotations

import numpy as np

from ...core.patch import attach, merge_metrics
from . import nb


def _bm(self, benchmark_rets):
    """Resolve and broadcast benchmark returns to the accessor's shape."""
    from vectorbt.base.reshape_fns import broadcast_to, to_2d_array

    if benchmark_rets is None:
        benchmark_rets = self.benchmark_rets
    if benchmark_rets is None:
        raise ValueError("benchmark_rets is required for this metric")
    return broadcast_to(to_2d_array(benchmark_rets), to_2d_array(self.obj))


def _wrap(self, result, name, wrap_kwargs):
    from vectorbt.utils.config import merge_dicts

    wrap_kwargs = merge_dicts(dict(name_or_index=name), wrap_kwargs)
    return self.wrapper.wrap_reduced(result, group_by=False, **wrap_kwargs)


# ############# Accessor methods ############# #


def ulcer_index(self, wrap_kwargs=None):
    """Ulcer Index. See `vectorbtx.metrics.returns.nb.ulcer_index_1d_nb`."""
    return _wrap(self, nb.ulcer_index_nb(self.to_2d_array()), "ulcer_index", wrap_kwargs)


def rolling_ulcer_index(self, window=None, minp=None, wrap_kwargs=None):
    """Rolling Ulcer Index."""
    window = self.defaults["window"] if window is None else window
    minp = self.defaults["minp"] if minp is None else minp
    out = nb.rolling_ulcer_index_nb(self.to_2d_array(), window, minp)
    return self.wrapper.wrap(out, group_by=False, **(wrap_kwargs or {}))


def ulcer_performance_index(self, risk_free=None, wrap_kwargs=None):
    """Ulcer Performance Index (Martin ratio)."""
    risk_free = self.defaults["risk_free"] if risk_free is None else risk_free
    out = nb.ulcer_performance_index_nb(self.to_2d_array(), self.ann_factor, risk_free)
    return _wrap(self, out, "ulcer_performance_index", wrap_kwargs)


def m2(self, benchmark_rets=None, risk_free=None, wrap_kwargs=None):
    """M² (Modigliani–Modigliani). Requires a benchmark."""
    risk_free = self.defaults["risk_free"] if risk_free is None else risk_free
    out = nb.m2_nb(self.to_2d_array(), _bm(self, benchmark_rets), self.ann_factor, risk_free)
    return _wrap(self, out, "m2", wrap_kwargs)


def treynor_ratio(self, benchmark_rets=None, risk_free=None, wrap_kwargs=None):
    """Treynor ratio. Requires a benchmark."""
    risk_free = self.defaults["risk_free"] if risk_free is None else risk_free
    out = nb.treynor_ratio_nb(self.to_2d_array(), _bm(self, benchmark_rets), self.ann_factor, risk_free)
    return _wrap(self, out, "treynor_ratio", wrap_kwargs)


def geometric_sharpe_ratio(self, risk_free=None, ddof=None, wrap_kwargs=None):
    """Sharpe ratio on log returns (upstream's is arithmetic)."""
    risk_free = self.defaults["risk_free"] if risk_free is None else risk_free
    ddof = self.defaults["ddof"] if ddof is None else ddof
    out = nb.geometric_sharpe_ratio_nb(self.to_2d_array(), self.ann_factor, risk_free, ddof)
    return _wrap(self, out, "geometric_sharpe_ratio", wrap_kwargs)


def snr(self, wrap_kwargs=None):
    """Efficiency ratio — net movement over path length, bounded 0–1."""
    return _wrap(self, nb.snr_nb(self.to_2d_array()), "snr", wrap_kwargs)


def k_ratio(self, version=3, wrap_kwargs=None):
    """K-ratio. `version` selects the 1996 / 2003 / 2013 normalisation. Default 3."""
    out = nb.k_ratio_nb(self.to_2d_array(), self.ann_factor, version)
    return _wrap(self, out, f"k_ratio_v{version}", wrap_kwargs)


def k_ratio_v1(self, wrap_kwargs=None):
    """K-ratio, Kestner (1996) normalisation."""
    return k_ratio(self, version=1, wrap_kwargs=wrap_kwargs)


def k_ratio_v2(self, wrap_kwargs=None):
    """K-ratio, Kestner (2003) normalisation."""
    return k_ratio(self, version=2, wrap_kwargs=wrap_kwargs)


def k_ratio_v3(self, wrap_kwargs=None):
    """K-ratio, Kestner (2013) normalisation."""
    return k_ratio(self, version=3, wrap_kwargs=wrap_kwargs)


def rolling_k_ratio(self, window=None, minp=None, version=3, wrap_kwargs=None):
    """Rolling K-ratio."""
    window = self.defaults["window"] if window is None else window
    minp = self.defaults["minp"] if minp is None else minp
    out = nb.rolling_k_ratio_nb(self.to_2d_array(), window, minp, self.ann_factor, version)
    return self.wrapper.wrap(out, group_by=False, **(wrap_kwargs or {}))


def rolling_snr(self, window=None, minp=None, wrap_kwargs=None):
    """Rolling efficiency ratio."""
    window = self.defaults["window"] if window is None else window
    minp = self.defaults["minp"] if minp is None else minp
    out = nb.rolling_snr_nb(self.to_2d_array(), window, minp)
    return self.wrapper.wrap(out, group_by=False, **(wrap_kwargs or {}))


def rolling_geometric_sharpe_ratio(self, window=None, minp=None, risk_free=None, wrap_kwargs=None):
    """Rolling geometric Sharpe ratio."""
    window = self.defaults["window"] if window is None else window
    minp = self.defaults["minp"] if minp is None else minp
    risk_free = self.defaults["risk_free"] if risk_free is None else risk_free
    out = nb.rolling_geometric_sharpe_ratio_nb(
        self.to_2d_array(), window, minp, self.ann_factor, risk_free
    )
    return self.wrapper.wrap(out, group_by=False, **(wrap_kwargs or {}))


METHODS = dict(
    ulcer_index=ulcer_index,
    rolling_ulcer_index=rolling_ulcer_index,
    ulcer_performance_index=ulcer_performance_index,
    m2=m2,
    treynor_ratio=treynor_ratio,
    geometric_sharpe_ratio=geometric_sharpe_ratio,
    rolling_geometric_sharpe_ratio=rolling_geometric_sharpe_ratio,
    snr=snr,
    rolling_snr=rolling_snr,
    k_ratio=k_ratio,
    k_ratio_v1=k_ratio_v1,
    k_ratio_v2=k_ratio_v2,
    k_ratio_v3=k_ratio_v3,
    rolling_k_ratio=rolling_k_ratio,
)

METRICS = dict(
    ulcer_index=dict(
        title="Ulcer Index [%]",
        calc_func="ulcer_index",
        post_calc_func=lambda self, out, settings: out * 100,
        tags=["vectorbtx", "returns", "risk"],
    ),
    ulcer_performance_index=dict(
        title="Ulcer Performance Index",
        calc_func="ulcer_performance_index",
        check_has_freq=True,
        check_has_year_freq=True,
        tags=["vectorbtx", "returns", "risk"],
    ),
    m2=dict(
        title="M2 [%]",
        calc_func="m2",
        post_calc_func=lambda self, out, settings: out * 100,
        check_has_freq=True,
        check_has_year_freq=True,
        check_has_benchmark_rets=True,
        tags=["vectorbtx", "returns", "benchmark"],
    ),
    treynor_ratio=dict(
        title="Treynor Ratio",
        calc_func="treynor_ratio",
        check_has_freq=True,
        check_has_year_freq=True,
        check_has_benchmark_rets=True,
        tags=["vectorbtx", "returns", "benchmark"],
    ),
    geometric_sharpe_ratio=dict(
        title="Geometric Sharpe Ratio",
        calc_func="geometric_sharpe_ratio",
        check_has_freq=True,
        check_has_year_freq=True,
        tags=["vectorbtx", "returns"],
    ),
    snr=dict(
        title="Efficiency Ratio (SNR)",
        calc_func="snr",
        tags=["vectorbtx", "returns"],
    ),
    k_ratio=dict(
        title="K-Ratio (v3)",
        calc_func="k_ratio_v3",
        check_has_freq=True,
        check_has_year_freq=True,
        tags=["vectorbtx", "returns"],
    ),
)

_DONE = False


def register_returns_metrics() -> None:
    """Attach K1's methods and `_metrics` entries to `ReturnsAccessor`. Idempotent."""
    global _DONE
    if _DONE:
        return
    from vectorbt.returns.accessors import ReturnsAccessor

    for name, fn in METHODS.items():
        fn.__qualname__ = f"ReturnsAccessor.{name}"
        attach(ReturnsAccessor, name, fn)
    merge_metrics(ReturnsAccessor, METRICS)
    _DONE = True
