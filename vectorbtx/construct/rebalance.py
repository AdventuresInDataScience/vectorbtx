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
        price: price frame, needed to quantize weights into fillable sizes, and by
            `min_trade` to tell how far each holding has drifted.
        equity: account equity, needed for quantization.
        min_trade: skip a column's rebalance trade when its target is within this weight
            of the holding just before the rebalance (see Notes).

    Returns:
        `(targets, drift)` — targets with **NaN off-schedule** (and on a skipped trade),
        and a drift frame combining quantization and suppression.

    Notes:
        Quantize **before** emitting targets. Quantizing after the engine has computed
        target percentages loses the drift information entirely.

        **`min_trade` is a weight gap.** On each rebalance bar, a column whose target is
        within `min_trade` of the weight it has drifted to is given no order (NaN), so its
        position keeps drifting; the gap left is recorded as suppressed drift. The first
        rebalance always trades in full: it builds the book. A trade to a target of zero
        is a trade like any other, so a holding within `min_trade` of nothing is left.

        Here the holding is an **estimate**: the book drifted with the prices since each
        trade, ignoring costs and rounding. `Portfolio.from_weights` (and so the front
        door) makes it exact by measuring the simulated holding at each rebalance.

        ⚠️ This compared each target with the previous TARGET until OpenStrategy Studio
        issue 21, and wrote the old target back, so the engine still traded from the
        drifted holding back to it. Fixed weights never change, so `min_trade` did nothing
        at all; moving weights had only their small changes frozen.
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
        if price is None:
            raise ValueError(
                "`min_trade` compares each target with the weight its holding has drifted "
                "to, which needs `price=`."
            )
        skip, gap = estimated_min_trade_skips(targets, mask, price, min_trade)
        targets = targets.where(~skip)
        drift = drift.add(gap.where(skip, 0.0), fill_value=0.0)

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


def _schedule_mask(schedule, index) -> pd.Series:
    """The rebalance bars as a boolean Series on `index`; None is every bar."""
    if schedule is None:
        return pd.Series(True, index=index)
    return pd.Series(schedule).reindex(index).fillna(False).astype(bool)


def estimated_min_trade_skips(targets, mask, price, min_trade):
    """Which rebalance trades `min_trade` skips, with the holding estimated from drift.

    The book is followed bar by bar as a fraction of one unit of equity: each column's
    holding moves with its price, cash stays put, and on each rebalance bar a column within
    `min_trade` of its target is left while the others trade to theirs. Costs, rounding and
    partial fills are ignored, so this can disagree with the simulated holding by a little;
    `measured_min_trade_skips` measures the real one.

    Args:
        targets: target weights, NaN where no order is wanted.
        mask: boolean Series of rebalance bars.
        price: the price trades fill and the book is valued at, on each bar.
        min_trade: the weight gap below which a trade is skipped.

    Returns:
        `(skip, gap)`: boolean frame of skipped trades, and each rebalance cell's target
        minus the holding just before it (0 elsewhere).
    """
    t = pd.DataFrame(targets).astype(float)
    px = pd.DataFrame(price).reindex(index=t.index, columns=t.columns).to_numpy(dtype=float)
    on = pd.Series(mask).reindex(t.index).fillna(False).to_numpy(dtype=bool)
    wanted = t.to_numpy(dtype=float)

    skip = np.zeros(wanted.shape, dtype=bool)
    gap = np.zeros(wanted.shape)
    units = np.zeros(wanted.shape[1])
    cash = 1.0
    first = True
    for i in range(len(t)):
        if not on[i]:
            continue
        p = px[i]
        held = np.where(units != 0, units * p, 0.0)
        value = cash + np.nansum(held)
        current = held / value if value > 0 else np.zeros_like(held)
        target = wanted[i]
        live = np.isfinite(target) & np.isfinite(p) & (p > 0)
        gap[i] = np.where(live, target - current, 0.0)
        if not first:
            skip[i] = live & (np.abs(target - current) < min_trade)
        trade = live & ~skip[i]
        new_units = np.where(trade, target * value / np.where(trade, p, 1.0), units)
        cash -= np.nansum(np.where(trade, (new_units - units) * p, 0.0))
        units = new_units
        first = False
    return (pd.DataFrame(skip, index=t.index, columns=t.columns),
            pd.DataFrame(gap, index=t.index, columns=t.columns))


def measured_min_trade_skips(pf, targets, mask, price, min_trade):
    """Which rebalance trades `min_trade` skips, measured on a simulated portfolio.

    On each rebalance bar, each column's holding is read as the engine sees it before
    trading: the position after the previous bar, valued at this bar's price, as a share
    of its group's value at that price (what `TargetPercent` is a fraction of).

    Args:
        pf: the portfolio simulated with the previous estimate's skips.
        targets, mask, price, min_trade: as `estimated_min_trade_skips`.

    Returns:
        `(skip, gap)`, as `estimated_min_trade_skips`.
    """
    t = pd.DataFrame(targets).astype(float)
    index, columns = t.index, list(t.columns)
    px = pd.DataFrame(price).reindex(index=index, columns=columns).to_numpy(dtype=float)
    on = pd.Series(mask).reindex(index).fillna(False).to_numpy(dtype=bool)
    wanted = t.to_numpy(dtype=float)

    assets = pd.DataFrame(pf.assets(), index=index, columns=columns).to_numpy(dtype=float)
    groups = np.asarray(pf.wrapper.grouper.get_groups())
    cash = np.asarray(pd.DataFrame(pf.cash()).to_numpy(dtype=float))
    start = np.atleast_1d(np.asarray(pf.get_init_cash(), dtype=float))
    if not pf.cash_sharing:
        groups = np.arange(len(columns))
        cash = pd.DataFrame(pf.cash(group_by=False)).to_numpy(dtype=float)
        start = np.broadcast_to(np.asarray(pf.get_init_cash(group_by=False), dtype=float),
                                (len(columns),))

    skip = np.zeros(wanted.shape, dtype=bool)
    gap = np.zeros(wanted.shape)
    first = True
    for i in range(len(index)):
        if not on[i]:
            continue
        before = assets[i - 1] if i > 0 else np.zeros(len(columns))
        held = np.where(before != 0, before * px[i], 0.0)
        cash_before = cash[i - 1] if i > 0 else start
        value = np.array([cash_before[groups[k]] + np.nansum(held[groups == groups[k]])
                          for k in range(len(columns))])
        with np.errstate(divide="ignore", invalid="ignore"):
            current = np.where(value > 0, held / value, 0.0)
        live = np.isfinite(wanted[i]) & np.isfinite(px[i])
        gap[i] = np.where(live, wanted[i] - current, 0.0)
        if not first:
            skip[i] = live & (np.abs(wanted[i] - current) < min_trade)
        first = False
    return (pd.DataFrame(skip, index=index, columns=columns),
            pd.DataFrame(gap, index=index, columns=columns))


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
