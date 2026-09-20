"""Multi-asset / multi-run aggregate stats — `vbt.x.aggregate`.

vectorbt has no wrapper for combining stats across independently-simulated columns or
runs. This is that wrapper, and nothing more.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate(results, how: str = "geometric", weights=None) -> pd.Series:
    """Aggregate several stats Series (or a stats DataFrame) into one.

    Args:
        results: list of `pd.Series` (e.g. `pf.stats()` per run), or a DataFrame whose
            columns are runs.
        how: `"geometric"` (compound return-like rows, mean the rest), `"mean"`,
            `"median"`, or `"sum"`.
        weights: optional per-run weights, used by `"mean"` and `"geometric"`.

    Returns:
        `pd.Series` indexed like the inputs.

    Notes:
        `"geometric"` compounds any row whose label contains "Return" and averages
        everything else — the only sensible default when the rows are a mix of returns,
        ratios and durations.
    """
    if isinstance(results, pd.DataFrame):
        frame = results
    else:
        frame = pd.concat(list(results), axis=1)
    numeric = frame.apply(pd.to_numeric, errors="coerce")

    if weights is None:
        w = np.ones(frame.shape[1]) / frame.shape[1]
    else:
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()

    if how == "sum":
        return numeric.sum(axis=1)
    if how == "median":
        return numeric.median(axis=1)
    if how == "mean":
        return numeric.mul(w, axis=1).sum(axis=1)
    if how != "geometric":
        raise ValueError(f"Unknown aggregation {how!r}")

    out = numeric.mul(w, axis=1).sum(axis=1)
    for label in numeric.index:
        if "Return" not in str(label):
            continue
        row = numeric.loc[label].to_numpy(dtype=float)
        pct = "[%]" in str(label)
        vals = row / 100.0 if pct else row
        finite = np.isfinite(vals)
        if not finite.any():
            continue
        g = np.exp(np.sum(w[finite] * np.log1p(vals[finite])) / w[finite].sum()) - 1.0
        out.loc[label] = g * 100.0 if pct else g
    return out
