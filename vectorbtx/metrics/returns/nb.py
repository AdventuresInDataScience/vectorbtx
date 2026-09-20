"""Section K1 — returns-metric kernels.

Seven genuine gaps in vectorbt 1.1.0: Ulcer Index, Ulcer Performance Index, M²,
Treynor ratio, geometric Sharpe, SNR (efficiency ratio) and the K-ratio family.

Conventions follow `vectorbt/returns/nb.py` exactly — a `_1d_nb` core, an `_nb` column
loop, a `rolling_*_nb` variant — so the additions read as native. All ratios are
computed in *fractions*; the `_metrics` entries do any ×100 display scaling.

Nothing here imports vectorbt, so every kernel is unit-testable on its own.
"""

from __future__ import annotations

import numpy as np

from ...core._numba import njit

# ############# Primitives ############# #


@njit(cache=True)
def cum_curve_1d_nb(returns: np.ndarray, start_value: float = 1.0) -> np.ndarray:
    """Equity curve from simple returns, starting at `start_value`."""
    out = np.empty(returns.shape[0] + 1, dtype=np.float64)
    out[0] = start_value
    v = start_value
    for i in range(returns.shape[0]):
        r = returns[i]
        if not np.isnan(r):
            v = v * (1.0 + r)
        out[i + 1] = v
    return out


@njit(cache=True)
def annualized_return_1d_nb(returns: np.ndarray, ann_factor: float) -> float:
    """Geometric annualized return. Mirrors upstream `annualized_return_1d_nb`."""
    curve = cum_curve_1d_nb(returns, 1.0)
    n = returns.shape[0]
    if n == 0 or curve[-1] <= 0.0:
        return np.nan
    return curve[-1] ** (ann_factor / n) - 1.0


@njit(cache=True)
def annualized_volatility_1d_nb(returns: np.ndarray, ann_factor: float, ddof: int = 1) -> float:
    """Annualized standard deviation of simple returns."""
    r = returns[~np.isnan(returns)]
    if r.shape[0] <= ddof:
        return np.nan
    return np.std(r) * np.sqrt(r.shape[0] / (r.shape[0] - ddof)) * np.sqrt(ann_factor)


@njit(cache=True)
def linreg_1d_nb(y: np.ndarray):
    """Least-squares regression of `y` on `t = 0..n-1`.

    Returns `(slope, stderr_of_slope, correlation)`. The shared primitive behind the
    K-ratio family (K1) and LR Correlation / LR Standard Error (K2) — three metrics for
    the price of one.
    """
    n = y.shape[0]
    if n < 3:
        return np.nan, np.nan, np.nan
    x = np.arange(n).astype(np.float64)
    xm = x.mean()
    ym = y.mean()
    sxx = np.sum((x - xm) ** 2)
    syy = np.sum((y - ym) ** 2)
    sxy = np.sum((x - xm) * (y - ym))
    if sxx == 0.0:
        return np.nan, np.nan, np.nan
    slope = sxy / sxx
    intercept = ym - slope * xm
    resid = y - (intercept + slope * x)
    sse = np.sum(resid**2)
    stderr = np.sqrt(sse / (n - 2)) / np.sqrt(sxx)
    corr = np.nan if syy == 0.0 else sxy / np.sqrt(sxx * syy)
    return slope, stderr, corr


@njit(cache=True)
def beta_1d_nb(returns: np.ndarray, benchmark_rets: np.ndarray) -> float:
    """Covariance beta. Duplicated from upstream so K1 stays import-free."""
    mask = (~np.isnan(returns)) & (~np.isnan(benchmark_rets))
    r = returns[mask]
    b = benchmark_rets[mask]
    if r.shape[0] < 2:
        return np.nan
    bv = np.var(b)
    if bv == 0.0:
        return np.nan
    return np.mean((r - r.mean()) * (b - b.mean())) / bv


# ############# Ulcer Index ############# #


@njit(cache=True)
def ulcer_index_1d_nb(returns: np.ndarray) -> float:
    """Ulcer Index — Martin (1987).

    Equity curve → running maximum → **percentage** drawdown → RMS.
    `UI = sqrt(mean(D_i^2))`, with `D_i` a fraction (0.05 = 5% below peak).

    Percentage, not absolute: squaring absolute drawdowns would make the metric scale
    with account size and stop being comparable between portfolios.
    """
    curve = cum_curve_1d_nb(returns, 1.0)
    n = curve.shape[0]
    if n < 2:
        return np.nan
    peak = curve[0]
    acc = 0.0
    for i in range(1, n):
        if curve[i] > peak:
            peak = curve[i]
        if peak > 0.0:
            dd = curve[i] / peak - 1.0
        else:
            dd = 0.0
        acc += dd * dd
    return np.sqrt(acc / (n - 1))


@njit(cache=True)
def ulcer_performance_index_1d_nb(
    returns: np.ndarray, ann_factor: float, risk_free: float = 0.0
) -> float:
    """Ulcer Performance Index (Martin ratio) — `(annualized return - rf) / UI`."""
    ui = ulcer_index_1d_nb(returns)
    if ui == 0.0 or np.isnan(ui):
        return np.nan
    return (annualized_return_1d_nb(returns, ann_factor) - risk_free) / ui


# ############# M-squared ############# #


@njit(cache=True)
def m2_1d_nb(
    returns: np.ndarray, benchmark_rets: np.ndarray, ann_factor: float, risk_free: float = 0.0
) -> float:
    """M² (Modigliani–Modigliani) — `rf + Sharpe_p * sigma_benchmark`.

    The portfolio's return rescaled to the benchmark's volatility, so it reads as a
    return rather than a ratio.
    """
    r = returns[~np.isnan(returns)]
    if r.shape[0] < 2:
        return np.nan
    sd = np.std(r) * np.sqrt(r.shape[0] / (r.shape[0] - 1))
    if sd == 0.0:
        return np.nan
    sharpe = (np.mean(r) - risk_free / ann_factor) / sd * np.sqrt(ann_factor)
    return risk_free + sharpe * annualized_volatility_1d_nb(benchmark_rets, ann_factor)


# ############# Treynor ############# #


@njit(cache=True)
def treynor_ratio_1d_nb(
    returns: np.ndarray, benchmark_rets: np.ndarray, ann_factor: float, risk_free: float = 0.0
) -> float:
    """Treynor ratio — excess annualized return per unit of systematic risk (beta)."""
    beta = beta_1d_nb(returns, benchmark_rets)
    if beta == 0.0 or np.isnan(beta):
        return np.nan
    return (annualized_return_1d_nb(returns, ann_factor) - risk_free) / beta


# ############# Geometric Sharpe ############# #


@njit(cache=True)
def geometric_sharpe_ratio_1d_nb(
    returns: np.ndarray, ann_factor: float, risk_free: float = 0.0, ddof: int = 1
) -> float:
    """Sharpe ratio computed on **log** returns.

    Upstream's `sharpe_ratio` is arithmetic (`mean/std * sqrt(ann_factor)`), so this is a
    genuine gap rather than a restatement.
    """
    r = returns[~np.isnan(returns)]
    r = r[r > -1.0]
    if r.shape[0] <= ddof:
        return np.nan
    lr = np.log(1.0 + r)
    sd = np.std(lr) * np.sqrt(r.shape[0] / (r.shape[0] - ddof))
    if sd == 0.0:
        return np.nan
    return (np.mean(lr) - np.log(1.0 + risk_free) / ann_factor) / sd * np.sqrt(ann_factor)


# ############# SNR / efficiency ratio ############# #


@njit(cache=True)
def snr_1d_nb(returns: np.ndarray) -> float:
    """Signal-to-noise as the **efficiency ratio**.

    `|P_end - P_start| / sum(|P_t - P_(t-1)|)` on the equity curve: net directional
    movement over total path length, bounded 0–1.

    Chosen over the mean/sigma reading (which duplicates Sharpe) and the decibel form
    (unconventional in finance). Complements the K-ratio: the K-ratio measures trend
    consistency, this measures path directness.
    """
    curve = cum_curve_1d_nb(returns, 1.0)
    n = curve.shape[0]
    if n < 2:
        return np.nan
    path = 0.0
    for i in range(1, n):
        path += abs(curve[i] - curve[i - 1])
    if path == 0.0:
        return np.nan
    return abs(curve[-1] - curve[0]) / path


# ############# K-ratio ############# #


@njit(cache=True)
def _log_equity_1d_nb(returns: np.ndarray) -> np.ndarray:
    """Cumulative log-return curve, the K-ratio regression's dependent variable."""
    n = returns.shape[0]
    out = np.empty(n, dtype=np.float64)
    acc = 0.0
    for i in range(n):
        r = returns[i]
        if not np.isnan(r) and r > -1.0:
            acc += np.log(1.0 + r)
        out[i] = acc
    return out


@njit(cache=True)
def k_ratio_1d_nb(returns: np.ndarray, ann_factor: float = 252.0, version: int = 3) -> float:
    """K-ratio — regression of cumulative log returns on time, `slope / stderr(slope)`.

    The three published variants differ **only in normalisation**:

    * `version=1` — Kestner (1996): `t / n`. The raw t-statistic grows roughly as
      `n ** 1.5` (the slope is stable while its standard error shrinks as `n ** -1.5`),
      so dividing by `n` leaves v1 still growing with `sqrt(n)`: **not comparable across
      backtests of different lengths.**
    * `version=2` — Kestner (2003): v1 adjusted for return frequency, `* sqrt(ann_factor)`.
      Comparable across data frequencies, still length-dependent.
    * `version=3` — Kestner (2013): v2 normalised by a further `sqrt(n)`, cancelling the
      residual growth. Independent of both length and frequency. **The default.**
    """
    y = _log_equity_1d_nb(returns)
    slope, stderr, _ = linreg_1d_nb(y)
    if np.isnan(slope) or np.isnan(stderr) or stderr == 0.0:
        return np.nan
    n = float(y.shape[0])
    t = slope / stderr
    if version == 1:
        return t / n
    if version == 2:
        return t / n * np.sqrt(ann_factor)
    return t / n * np.sqrt(ann_factor) / np.sqrt(n)


# ############# Column loops ############# #


def _colwise(func_1d, arr_2d, *args):
    """Apply a `_1d_nb` kernel down every column."""
    arr_2d = np.asarray(arr_2d, dtype=np.float64)
    if arr_2d.ndim == 1:
        arr_2d = arr_2d[:, None]
    out = np.empty(arr_2d.shape[1], dtype=np.float64)
    for c in range(arr_2d.shape[1]):
        out[c] = func_1d(arr_2d[:, c], *args)
    return out


def _colwise_bm(func_1d, arr_2d, bm_2d, *args):
    """Apply a benchmark-taking `_1d_nb` kernel down every column."""
    arr_2d = np.asarray(arr_2d, dtype=np.float64)
    bm_2d = np.asarray(bm_2d, dtype=np.float64)
    if arr_2d.ndim == 1:
        arr_2d = arr_2d[:, None]
    if bm_2d.ndim == 1:
        bm_2d = bm_2d[:, None]
    if bm_2d.shape[1] == 1 and arr_2d.shape[1] > 1:
        bm_2d = np.repeat(bm_2d, arr_2d.shape[1], axis=1)
    out = np.empty(arr_2d.shape[1], dtype=np.float64)
    for c in range(arr_2d.shape[1]):
        out[c] = func_1d(arr_2d[:, c], bm_2d[:, c], *args)
    return out


def ulcer_index_nb(returns_2d):
    """Column loop over `ulcer_index_1d_nb`."""
    return _colwise(ulcer_index_1d_nb, returns_2d)


def ulcer_performance_index_nb(returns_2d, ann_factor, risk_free=0.0):
    """Column loop over `ulcer_performance_index_1d_nb`."""
    return _colwise(ulcer_performance_index_1d_nb, returns_2d, ann_factor, risk_free)


def m2_nb(returns_2d, benchmark_2d, ann_factor, risk_free=0.0):
    """Column loop over `m2_1d_nb`."""
    return _colwise_bm(m2_1d_nb, returns_2d, benchmark_2d, ann_factor, risk_free)


def treynor_ratio_nb(returns_2d, benchmark_2d, ann_factor, risk_free=0.0):
    """Column loop over `treynor_ratio_1d_nb`."""
    return _colwise_bm(treynor_ratio_1d_nb, returns_2d, benchmark_2d, ann_factor, risk_free)


def geometric_sharpe_ratio_nb(returns_2d, ann_factor, risk_free=0.0, ddof=1):
    """Column loop over `geometric_sharpe_ratio_1d_nb`."""
    return _colwise(geometric_sharpe_ratio_1d_nb, returns_2d, ann_factor, risk_free, ddof)


def snr_nb(returns_2d):
    """Column loop over `snr_1d_nb`."""
    return _colwise(snr_1d_nb, returns_2d)


def k_ratio_nb(returns_2d, ann_factor=252.0, version=3):
    """Column loop over `k_ratio_1d_nb`."""
    return _colwise(k_ratio_1d_nb, returns_2d, ann_factor, version)


# ############# Rolling variants ############# #


def _rolling(func_1d, arr_2d, window, minp, *args):
    """Rolling application of a `_1d_nb` kernel, matching upstream's minp semantics."""
    arr_2d = np.asarray(arr_2d, dtype=np.float64)
    if arr_2d.ndim == 1:
        arr_2d = arr_2d[:, None]
    n, m = arr_2d.shape
    minp = window if minp is None else minp
    out = np.full((n, m), np.nan, dtype=np.float64)
    for c in range(m):
        for i in range(n):
            lo = max(0, i - window + 1)
            if i - lo + 1 < minp:
                continue
            out[i, c] = func_1d(arr_2d[lo : i + 1, c], *args)
    return out


def rolling_ulcer_index_nb(returns_2d, window, minp=None):
    """Rolling Ulcer Index."""
    return _rolling(ulcer_index_1d_nb, returns_2d, window, minp)




def rolling_geometric_sharpe_ratio_nb(returns_2d, window, minp=None, ann_factor=252.0, risk_free=0.0):
    """Rolling geometric Sharpe ratio."""
    return _rolling(geometric_sharpe_ratio_1d_nb, returns_2d, window, minp, ann_factor, risk_free)


def rolling_snr_nb(returns_2d, window, minp=None):
    """Rolling efficiency ratio."""
    return _rolling(snr_1d_nb, returns_2d, window, minp)


def rolling_k_ratio_nb(returns_2d, window, minp=None, ann_factor=252.0, version=3):
    """Rolling K-ratio."""
    return _rolling(k_ratio_1d_nb, returns_2d, window, minp, ann_factor, version)
