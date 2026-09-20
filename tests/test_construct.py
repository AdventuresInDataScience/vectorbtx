"""Section J — weights, schedules, the leverage budget and target emission."""

import warnings
import numpy as np
import pandas as pd
import pytest

from vectorbtx.construct.rebalance import apply_leverage_budget, to_targets
from vectorbtx.construct.schedule import (
    calendar_schedule,
    drifted_weights,
    threshold_schedule,
)
from vectorbtx.construct.weights import cap_weight, equal_weight, price_weight
from vectorbtx.instruments.spec import Instrument
from vectorbtx.resolve.load import load_universe

IDX = pd.date_range("2024-01-01", periods=90, freq="D")
COLS = ["A", "B"]


def test_equal_weight():
    weights = equal_weight(COLS, IDX)
    assert weights.iloc[0].tolist() == [0.5, 0.5]


def test_price_weight():
    prices = pd.DataFrame({"A": [30.0], "B": [70.0]})
    assert price_weight(prices).iloc[0].tolist() == [0.3, 0.7]


def test_cap_weight_recomputed_each_bar():
    caps = pd.DataFrame({"A": [100.0, 300.0], "B": [100.0, 100.0]})
    weights = cap_weight(caps)
    assert weights.iloc[0].tolist() == [0.5, 0.5]
    assert weights.iloc[1].tolist() == [0.75, 0.25]


def test_calendar_schedule_triggers_once_per_month():
    schedule = calendar_schedule(IDX, freq="M")
    assert schedule.sum() == 3  # Jan, Feb, Mar


def test_threshold_schedule_uses_relative_drift():
    """A 1% position drifting to 1.5% is the 50% move it is."""
    target = pd.DataFrame({"A": [0.01], "B": [0.99]})
    current = pd.DataFrame({"A": [0.015], "B": [0.985]})
    assert threshold_schedule(current, target, tolerance=0.05).iloc[0]


def test_schedules_compose_with_and():
    calendar = calendar_schedule(IDX, freq="M")
    always_drifted = pd.Series(True, index=IDX)
    combined = calendar & always_drifted
    assert combined.sum() == calendar.sum()
    never = pd.Series(False, index=IDX)
    assert (calendar & never).sum() == 0


def test_off_schedule_bars_are_nan():
    """The single most likely thing to get written backwards in this section."""
    weights = equal_weight(COLS, IDX)
    schedule = calendar_schedule(IDX, freq="M")
    targets, _ = to_targets(weights, schedule)
    assert targets.notna().all(axis=1).sum() == schedule.sum()
    assert targets[~schedule].isna().all().all()


def test_forward_filled_input_warns():
    """A forward-filled input is flat between rebalances but changes overall."""
    schedule = calendar_schedule(IDX, freq="M")
    sparse = pd.DataFrame(np.nan, index=IDX, columns=COLS)
    rebal_rows = np.flatnonzero(schedule.to_numpy())
    for n, row in enumerate(rebal_rows):
        # A different allocation at each rebalance, so the frame is not constant.
        sparse.iloc[row] = [0.5 + 0.05 * n, 0.5 - 0.05 * n]
    forward_filled = sparse.ffill().bfill()

    with pytest.warns(UserWarning, match="forward-filled"):
        to_targets(forward_filled, schedule)


def test_constant_weights_do_not_warn():
    """Equal weighting is constant by design, not a forward-fill mistake.

    Warning on it would make the check noise that people learn to ignore, which is worse
    than not having it.
    """
    weights = equal_weight(COLS, IDX)
    schedule = calendar_schedule(IDX, freq="M")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        to_targets(weights, schedule)
    assert not [w for w in caught if "forward-filled" in str(w.message)]


def test_portfolio_level_budget_scales_weights():
    universe = load_universe({
        "A": Instrument(mode="cfd", contract_size=1.0, leverage=2.0,
                        leverage_type="portfolio_level"),
        "B": Instrument(mode="cfd", contract_size=1.0, leverage=2.0,
                        leverage_type="portfolio_level"),
    })
    weights = pd.DataFrame({"A": [1.5], "B": [1.5]})   # gross 3.0
    scaled = apply_leverage_budget(weights, universe)
    assert scaled.abs().sum(axis=1).iloc[0] == pytest.approx(2.0)
    assert scaled["A"].iloc[0] == pytest.approx(1.0)


def test_weights_within_budget_are_untouched():
    universe = load_universe({
        "A": Instrument(mode="cfd", contract_size=1.0, leverage=2.0,
                        leverage_type="portfolio_level"),
    })
    weights = pd.DataFrame({"A": [0.8]})
    assert apply_leverage_budget(weights, universe)["A"].iloc[0] == pytest.approx(0.8)


def test_weights_over_one_without_a_budget_warn():
    universe = load_universe({"A": Instrument(mode="equity")})
    weights = pd.DataFrame({"A": [1.5]})
    with pytest.warns(UserWarning, match="implicit leverage"):
        apply_leverage_budget(weights, universe)


def test_quantization_happens_before_target_emission():
    """Quantizing after the engine computes percentages would lose the drift."""
    universe = load_universe({"A": Instrument(mode="equity", point_size=0.01)})
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    weights = pd.DataFrame({"A": [0.5, 0.5]}, index=idx)
    price = pd.DataFrame({"A": [333.0, 333.0]}, index=idx)
    targets, drift = to_targets(weights, None, universe=universe, price=price, equity=10_000)
    # 10,000 x 0.5 / 333 = 15.015 shares -> floors to 15
    assert targets["A"].iloc[0] == pytest.approx(15 * 333 / 10_000)
    assert drift["A"].iloc[0] > 0


def test_min_trade_suppression_reports_its_drift():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    weights = pd.DataFrame({"A": [0.50, 0.505, 0.70]}, index=idx)
    targets, drift = to_targets(weights, None, min_trade=0.05)
    assert targets["A"].iloc[1] == pytest.approx(0.50)   # suppressed
    assert drift["A"].iloc[1] != 0.0                     # but recorded
    assert targets["A"].iloc[2] == pytest.approx(0.70)   # large trade goes through


def test_drifted_weights_normalise():
    prices = pd.DataFrame({"A": [100.0, 120.0], "B": [100.0, 100.0]})
    weights = pd.DataFrame({"A": [0.5, 0.5], "B": [0.5, 0.5]})
    drifted = drifted_weights(weights, prices)
    assert drifted.sum(axis=1).iloc[1] == pytest.approx(1.0)
    assert drifted["A"].iloc[1] > 0.5


def test_drift_resets_at_each_rebalance():
    """Without the schedule, drift accumulates from bar zero and any threshold test
    degenerates to 'always true' — silently collapsing calendar & threshold back to
    plain calendar rebalancing."""
    idx = pd.date_range("2020-01-01", periods=750, freq="B")
    trend = pd.DataFrame({
        "A": np.exp(np.linspace(0, 1.2, 750)),      # steadily outgrows B
        "B": np.ones(750),
    }, index=idx)
    weights = pd.DataFrame(0.5, index=idx, columns=["A", "B"])
    monthly = calendar_schedule(idx, freq="M")

    anchored = drifted_weights(weights, trend, schedule=monthly)
    unanchored = drifted_weights(weights, trend)

    # Anchored drift stays small because it resets every month.
    assert anchored["A"].max() < 0.56
    # Unanchored drift runs away over the whole history.
    assert unanchored["A"].max() > 0.70

    tight = threshold_schedule(anchored, weights, tolerance=0.05)
    loose = threshold_schedule(unanchored, weights, tolerance=0.05)
    assert tight.sum() < loose.sum()


def test_drift_is_nonzero_on_the_rebalance_bar_itself():
    """If the anchor included the current bar, drift there would be exactly zero and a
    composed `calendar & threshold` schedule could never fire."""
    idx = pd.date_range("2020-01-01", periods=400, freq="B")
    trend = pd.DataFrame({"A": np.exp(np.linspace(0, 0.8, 400)),
                          "B": np.ones(400)}, index=idx)
    weights = pd.DataFrame(0.5, index=idx, columns=["A", "B"])
    monthly = calendar_schedule(idx, freq="M")

    drifted = drifted_weights(weights, trend, schedule=monthly)
    on_rebalance = drifted[monthly.to_numpy()]
    # Skip the first bar, which has no prior rebalance to drift from.
    assert (on_rebalance["A"].iloc[1:] - 0.5).abs().min() > 1e-6

    composed = monthly & threshold_schedule(drifted, weights, tolerance=0.01)
    assert composed.sum() > 0
