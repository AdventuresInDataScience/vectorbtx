"""Section J — rebalance schedules.

Boolean masks over the backtest index, composable with `&` and `|`. "Monthly, but only if
drift exceeds 5%" is the common real-world rule and is just `calendar & threshold`, so it
is supported explicitly rather than left for users to reinvent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def always_schedule(index) -> pd.Series:
    """Rebalance every bar. Rarely what you want; useful as a baseline."""
    return pd.Series(True, index=pd.Index(index))


def calendar_schedule(index, freq: str = "M", on: str = "first") -> pd.Series:
    """Calendar rebalancing — a `DatetimeIndex` resample turned into a boolean mask.

    Args:
        index: the backtest index.
        freq: `"W"`, `"M"`, `"Q"`, `"A"` or any pandas period alias.
        on: `"first"` or `"last"` bar of each period.

    Returns:
        Boolean `pd.Series`, True on rebalance bars.

    Examples:
        >>> calendar_schedule(index, freq="M").sum()      # one per month
        12
    """
    idx = pd.DatetimeIndex(index)
    step, base = _split_multiple(freq)
    periods = idx.to_period(_alias(base))
    mask = pd.Series(False, index=idx)
    grouped = pd.Series(np.arange(len(idx)), index=idx).groupby(periods)
    picks = grouped.first() if on == "first" else grouped.last()

    # ⚠️ `to_period` DISCARDS a leading multiplier: "6M" periodises identically to "M".
    # Left alone, `calendar_schedule(index, "6M")` returns a monthly mask — a six-fold
    # over-rebalancing that still runs, still looks plausible, and quietly pays six times
    # the transaction costs. Take every `step`-th period instead.
    if step > 1:
        picks = picks.iloc[::step]

    mask.iloc[picks.to_numpy()] = True
    return mask


def _split_multiple(freq):
    """Split `"6M"` into `(6, "M")`. A bare alias gives a step of 1."""
    text = str(freq).strip()
    digits = ""
    while text and text[0].isdigit():
        digits += text[0]
        text = text[1:]
    return (int(digits) if digits else 1), (text or "M")


def _alias(freq: str) -> str:
    """Normalise a frequency alias to a Period alias.

    pandas has renamed several offset aliases over time (`M` -> `ME`, `Q` -> `QE`,
    `A` -> `YE`) while Period still expects the old spellings. Accept either, so a
    schedule written against one pandas version keeps working on the next.
    """
    return {
        "M": "M", "ME": "M", "MS": "M",
        "Q": "Q", "QE": "Q", "QS": "Q",
        "A": "Y", "AS": "Y", "Y": "Y", "YE": "Y", "YS": "Y",
        "W": "W", "D": "D",
    }.get(str(freq).upper(), freq)


def threshold_schedule(current, target, tolerance: float = 0.05) -> pd.Series:
    """Threshold rebalancing on **relative** drift, per asset.

    Args:
        current: DataFrame of actual weights.
        target: DataFrame of target weights.
        tolerance: relative drift that triggers, default 5%.

    Returns:
        Boolean `pd.Series`, True on any bar where at least one asset has drifted.

    Notes:
        Relative, not absolute. A 1% position drifting to 1.5% is the 50% move it is;
        absolute drift under-triggers badly on small positions.
    """
    cur = pd.DataFrame(current).astype(float)
    tgt = pd.DataFrame(target).astype(float).reindex_like(cur)
    with np.errstate(divide="ignore", invalid="ignore"):
        drift = np.abs(cur - tgt) / tgt.replace(0.0, np.nan)
    return (drift > tolerance).any(axis=1).fillna(False)


def drifted_weights(weights, prices, schedule=None) -> pd.DataFrame:
    """Weights a buy-and-hold portfolio drifts to between rebalances.

    Args:
        weights: target weights.
        prices: price frame.
        schedule: boolean mask of rebalance bars. Weights reset to target on each of
            them and drift from there.

    Returns:
        DataFrame of drifted weights, normalised to sum to 1.

    Notes:
        **Pass the schedule.** Without it, drift accumulates from the very first bar,
        which is only meaningful across a single holding period. Over a multi-year
        backtest it grows without bound, so any threshold test on it degenerates to
        "always true" — and a composed `calendar & threshold` schedule silently collapses
        back to plain calendar rebalancing.
    """
    w = pd.DataFrame(weights).astype(float)
    px = pd.DataFrame(prices).astype(float).reindex_like(w)

    if schedule is None:
        anchors = np.zeros(len(w), dtype=int)
    else:
        mask = pd.Series(schedule).reindex(w.index).fillna(False).to_numpy(dtype=bool)
        positions = np.where(mask, np.arange(len(w)), -1)
        # Anchor to the last rebalance STRICTLY BEFORE this bar. Including the current
        # bar would make drift exactly zero on every rebalance bar — so a composed
        # `calendar & threshold` schedule could never fire at all.
        previous = np.full(len(w), -1, dtype=int)
        previous[1:] = positions[:-1]
        anchors = np.maximum.accumulate(previous)
        anchors[anchors < 0] = 0

    px_anchor = px.to_numpy()[anchors]
    growth = px.to_numpy() / px_anchor
    drifted = w.to_numpy()[anchors] * growth
    out = pd.DataFrame(drifted, index=w.index, columns=w.columns)
    return out.div(out.sum(axis=1), axis=0)
