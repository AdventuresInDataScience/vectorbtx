"""Section K2 — MT4/MT5 trade metrics against hand-computed values."""

import numpy as np
import pytest

from vectorbtx.metrics.trades import calc

PNL = np.array([100.0, -50.0, 30.0, 40.0, -20.0, -10.0, 60.0])
INIT = 1000.0


def test_one_line_derivations():
    assert calc.total_net_profit(PNL) == 150.0
    assert calc.gross_profit(PNL) == 230.0
    assert calc.gross_loss(PNL) == -80.0
    assert calc.win_rate(PNL) == pytest.approx(4 / 7)
    assert calc.loss_rate(PNL) == pytest.approx(3 / 7)


def test_balance_curve_is_closed_trades_only():
    curve = calc.balance_curve(PNL, INIT)
    assert curve[0] == INIT
    assert curve[-1] == INIT + PNL.sum()
    assert len(curve) == len(PNL) + 1


def test_absolute_drawdown_is_currency_from_initial_deposit():
    losing = np.array([-100.0, -50.0, 200.0])
    assert calc.absolute_drawdown(losing, 1000.0) == 150.0


def test_balance_drawdown_is_a_fraction():
    losing = np.array([-100.0, -50.0, 200.0])
    assert calc.balance_drawdown(losing, 1000.0) == pytest.approx(0.15)
    assert calc.balance_drawdown_abs(losing, 1000.0) == pytest.approx(150.0)


def test_recovery_factor_is_profit_over_drawdown():
    dd = calc.balance_drawdown_abs(PNL, INIT)
    assert calc.recovery_factor(PNL, INIT) == pytest.approx(150.0 / dd)


def test_runs_decomposition():
    assert calc.runs(PNL) == [
        (True, 1, 100.0),
        (False, 1, -50.0),
        (True, 2, 70.0),
        (False, 2, -30.0),
        (True, 1, 60.0),
    ]


def test_longest_streak_is_not_the_most_profitable():
    """The distinction the plan calls out: these are different metrics."""
    assert calc.max_consecutive_money(PNL, winning=True) == 70.0
    assert calc.maximal_consecutive(PNL, winning=True) == 100.0
    # Streak COUNTS are native in vectorbt (winning_streak.max()); the plan says do
    # not rebuild them. Only the monetary total below is a genuine gap.


def test_average_consecutive():
    assert calc.avg_consecutive(PNL, winning=True) == pytest.approx((1 + 2 + 1) / 3)
    assert calc.avg_consecutive(PNL, winning=False) == pytest.approx((1 + 2) / 2)


def test_z_score_formula():
    a = PNL
    n, wins = len(a), int((a > 0).sum())
    losses = n - wins
    r = 1 + int(np.sum(np.sign(a[1:]) != np.sign(a[:-1])))
    p = 2.0 * wins * losses
    expected = (n * (r - 0.5) - p) / np.sqrt(p * (p - n) / (n - 1))
    assert calc.z_score(a) == pytest.approx(expected)


def test_z_score_sign_reads_as_clustering():
    """Negative Z means outcomes cluster into streaks; positive means they alternate."""
    alternating = np.array([10.0, -10.0] * 12)
    clustered = np.array([10.0] * 12 + [-10.0] * 12)
    assert calc.z_score(alternating) > calc.z_score(clustered)


def test_hpr_conventions_match_mt():
    rets = np.array([0.02, -0.01, 0.03])
    assert calc.ahpr(rets) == pytest.approx(np.mean(1 + rets))
    assert calc.ghpr(rets) == pytest.approx(np.prod(1 + rets) ** (1 / 3))


def test_lr_metrics_come_from_the_shared_regression():
    assert -1.0 <= calc.lr_correlation(PNL, INIT) <= 1.0
    assert calc.lr_standard_error(PNL, INIT) >= 0.0


def test_empty_input_returns_nan_not_zero():
    assert np.isnan(calc.total_net_profit(np.array([])))
    assert np.isnan(calc.z_score(np.array([1.0])))
