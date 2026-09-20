"""Section J — target emission, the leverage budget, and drift.

You never compute the buy/sell orders. `SizeType.TargetPercent` derives them from target
weights; combined with `cash_sharing=True` and `group_by`, the messy part is already
solved upstream. This module is a weight-matrix builder, not a new simulation path.

Four hazards live here, in descending order of how likely they are to be written
backwards:

1. **NaN off-schedule, never forward-fill.** Supply a target only on rebalance bars; NaN
   elsewhere means "no order" and the position drifts as intended. Forward-filling
   rebalances *every bar* — which still runs, still looks plausible, and quietly accrues
   transaction costs while producing a suspiciously smooth equity curve.
2. **`call_seq="auto"` is load-bearing.** It orders sells before buys within a segment so
   the sells fund the buys. Without it a rebalance can fail for lack of cash despite the
   weights being perfectly fundable.
3. **Weights must sum sanely.** Under 1 leaves the remainder in cash, which is fine and
   often intended. Over 1 is implicit leverage that must route through the `leverage_type`
   budget rather than entering through the back door.
4. **`SizeType.Percent` is not a substitute.** Upstream explicitly warns it misbehaves
   with `call_seq="auto"` and does not support position reversal.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def apply_leverage_budget(weights, universe, equity=None) -> pd.DataFrame:
    """Scale a weight matrix to fit the portfolio-level notional budget (Section F).

    Args:
        weights: DataFrame of target weights.
        universe: the `Universe`. Only specs with `leverage_type="portfolio_level"`
            contribute a budget.
        equity: unused; accepted so callers can pass it uniformly.

    Returns:
        DataFrame of weights, rows scaled down where gross exposure exceeds the budget.

    Notes:
        This is where `leverage_type="portfolio_level"` is actually enforced. Weights are
        known ahead of time, so scaling them to fit is exact and cheap — unlike
        `max_size`, which is per-column and has no knowledge of its siblings.

        With `group_by` and cash sharing, "portfolio" means **the group**: the
        cash-sharing unit is the only coherent budget boundary.

    Examples:
        >>> apply_leverage_budget(weights, universe)   # 2x budget, weights summing to 3
        # every row scaled by 2/3
    """
    w = pd.DataFrame(weights).astype(float)
    budgets = [
        spec.filled().leverage
        for col, spec in universe.specs.items()
        if col in w.columns and spec.leverage_type == "portfolio_level"
    ]
    gross = w.abs().sum(axis=1)

    if not budgets:
        if (gross > 1.0 + 1e-9).any():
            warnings.warn(
                f"Weights sum above 1 on {int((gross > 1 + 1e-9).sum())} bars (max "
                f"{gross.max():.3f}), which is implicit leverage. Set "
                f"leverage_type='portfolio_level' on the universe so it routes through "
                f"the budget rather than entering through the back door.",
                stacklevel=2,
            )
        return w

    budget = min(budgets)
    scale = np.where(gross > budget, budget / gross.replace(0.0, np.nan), 1.0)
    scale = np.nan_to_num(scale, nan=1.0)
    return w.mul(scale, axis=0)


def to_targets(weights, schedule=None, universe=None, price=None, equity=None, min_trade=None):
    """Turn weights and a schedule into the target matrix `from_orders` consumes.

    Args:
        weights: DataFrame of target weights.
        schedule: boolean Series, True on rebalance bars. None → every bar.
        universe: the `Universe`, for quantization.
        price: price frame, needed to quantize weights into fillable sizes.
        equity: account equity, needed for the same reason.
        min_trade: suppress rebalance trades whose weight change is below this.

    Returns:
        `(targets, drift)` — targets with **NaN off-schedule**, and a drift frame
        combining quantization and suppression.

    Notes:
        Quantize **before** emitting targets. Quantizing after the engine has computed
        target percentages loses the drift information entirely.

        The minimum-trade filter is available but off by default. Suppressed trades
        accumulate as tracking error, so their drift is routed through Section D's
        reporting rather than being allowed to vanish.
    """
    w = pd.DataFrame(weights).astype(float)

    if schedule is None:
        mask = pd.Series(True, index=w.index)
    else:
        mask = pd.Series(schedule).reindex(w.index).fillna(False).astype(bool)

    _warn_if_forward_filled(w, mask)

    drift = pd.DataFrame(0.0, index=w.index, columns=w.columns)
    targets = w.copy()

    if universe is not None and price is not None and equity is not None:
        targets, drift = _quantize_weights(targets, universe, price, equity)

    if min_trade:
        targets, suppressed = _apply_min_trade(targets, mask, min_trade)
        drift = drift.add(suppressed, fill_value=0.0)

    # NaN off-schedule. These two lines are the whole section.
    mask2d = pd.DataFrame(
        np.repeat(mask.to_numpy()[:, None], targets.shape[1], axis=1),
        index=targets.index,
        columns=targets.columns,
    )
    targets = targets.where(mask2d, np.nan)
    drift = drift.where(mask2d, 0.0)
    return targets, drift


def _warn_if_forward_filled(weights, mask) -> None:
    """Catch the classic input error: weights already forward-filled to every bar.

    The signature of a forward-filled input is that values **change only on rebalance
    bars** and are held flat in between. Weights that never change at all — equal
    weighting, most obviously — are a static allocation, not an error, and must not warn
    or the check becomes noise people learn to ignore.
    """
    if mask.all():
        return
    frame = pd.DataFrame(weights)
    off = ~mask.to_numpy()
    if len(frame) <= 2 or not frame[off].notna().all().all():
        return

    # Values must be unchanged from the previous bar at every off-schedule bar...
    step = frame.diff().abs().sum(axis=1).to_numpy()
    off_after_first = off.copy()
    off_after_first[0] = False
    if off_after_first.sum() == 0 or (step[off_after_first] > 1e-12).any():
        return
    # ...and must actually change somewhere, or this is a static allocation, not a ffill.
    if (step[1:] > 1e-12).sum() == 0:
        return

    warnings.warn(
        "Weights appear to be forward-filled across off-schedule bars: they change only "
        "on rebalance bars and are held flat in between. Targets are NaN off-schedule by "
        "design; a constant non-NaN value there would rebalance every bar. The input is "
        "being masked, so this is a warning rather than an error — but check it is what "
        "you meant.",
        stacklevel=3,
    )


def _quantize_weights(weights, universe, price, equity):
    """Round target weights to the nearest achievable fill, recording the drift."""
    from ..modes.base import get_adapter
    from ..quantize.quantizer import quantize

    px = pd.DataFrame(price).reindex_like(weights).astype(float)
    out = weights.copy()
    drift = pd.DataFrame(0.0, index=weights.index, columns=weights.columns)

    for col in weights.columns:
        if col not in universe:
            continue
        spec = universe[col].filled()
        adapter = get_adapter(spec.mode)
        notional = weights[col] * float(equity)
        with np.errstate(divide="ignore", invalid="ignore"):
            engine_size = notional / px[col]
        user_size = adapter.from_engine_units(engine_size, spec, px[col])
        q_user, _ = quantize(user_size, spec)
        q_engine = adapter.to_engine_units(np.asarray(q_user, dtype=float), spec, px[col])
        achieved = pd.Series(q_engine, index=weights.index) * px[col] / float(equity)
        out[col] = achieved
        drift[col] = weights[col] - achieved
    return out, drift


def _apply_min_trade(targets, mask, min_trade):
    """Suppress rebalance trades below `min_trade`, recording what was skipped."""
    out = targets.copy()
    suppressed = pd.DataFrame(0.0, index=targets.index, columns=targets.columns)
    last = None
    for i, (ts, row) in enumerate(targets.iterrows()):
        if not mask.iloc[i]:
            continue
        if last is None:
            last = row
            continue
        change = (row - last).abs()
        small = change < min_trade
        if small.any():
            out.loc[ts, small] = last[small]
            suppressed.loc[ts, small] = (row - last)[small]
        last = out.loc[ts]
    return out, suppressed


def validate_weights(weights, tolerance: float = 1e-6) -> None:
    """Check that weights sum sanely, warning rather than failing on the benign case.

    Under 1 leaves the remainder in cash — fine, and often intended. Over 1 is **implicit
    leverage**, which must route through the `leverage_type` budget rather than entering
    through the back door, so it warns more pointedly.
    """
    gross = pd.DataFrame(weights).abs().sum(axis=1)
    if (gross < 1.0 - tolerance).any():
        warnings.warn(
            "Weights sum below 1 on some bars; the remainder is held in cash. "
            "Often intended — noting it so it is not a surprise.",
            stacklevel=2,
        )
    if (gross > 1.0 + tolerance).any():
        warnings.warn(
            f"Weights sum above 1 on some bars (max gross {float(gross.max()):.2f}x). "
            f"That is implicit leverage. Route it through leverage_type='portfolio_level' "
            f"and apply_leverage_budget so the budget is explicit, rather than levering "
            f"the portfolio by accident.",
            stacklevel=2,
        )
