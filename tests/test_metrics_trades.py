"""Section K2 — MT4/MT5 trade metrics against hand-computed values."""

import numpy as np
import pandas as pd
import pytest

from conftest import needs_vectorbt
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


# ############# The open-position criterion ############# #


@needs_vectorbt
def test_balance_and_equity_drawdown_differ_on_a_portfolio_with_open_positions():
    """Section K2's last "Done when", never implemented as a check until now.

    The two drawdowns answer different questions and MT4/MT5 users expect both:

    * **Balance** drawdown runs on `init_cash + cumsum(closed-trade pnl)`. An open
      position has contributed no closed pnl, so it cannot move the balance curve.
    * **Equity** drawdown is mark-to-market and follows the position down.

    One asset, bought on the first bar and never sold, priced 100 → 60 → 90:

    * Closed trades: **none**. The balance curve is flat at £10,000, so balance
      drawdown is **0**.
    * Equity peaks at £10,000 and troughs at £6,000, so equity drawdown is
      **0.4** — a 40% fall the balance curve cannot see.

    If these two ever come out equal, one of them has been wired to the wrong series,
    which is the failure this criterion exists to catch.
    """
    import vectorbtx as vbt

    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    close = pd.DataFrame({"A": [100.0, 60.0, 90.0]}, index=idx)
    size = pd.DataFrame({"A": [100.0, np.nan, np.nan]}, index=idx)  # buy once, hold

    pf = vbt.Portfolio.from_orders(close, size=size, init_cash=10_000, freq="D")

    assert len(pf.trades.records) == 1, "expected exactly one position"
    assert pf.trades.records["status"][0] == 0, "the position must still be open"

    equity_dd = abs(float(np.ravel(np.asarray(pf.max_drawdown()))[0]))
    balance_dd = float(
        np.ravel(np.asarray(pf.trades.balance_drawdown(init_cash=10_000.0)))[0]
    )

    assert equity_dd == pytest.approx(0.4)
    assert balance_dd == pytest.approx(0.0)
    assert equity_dd > balance_dd


@needs_vectorbt
def test_the_hardcoded_closed_status_matches_the_upstream_enum():
    """`register._STATUS_CLOSED` is a literal so the module imports without vectorbt.

    That is only safe while it still agrees with upstream, so pin it here rather than
    discovering the drift through wrong numbers.
    """
    from vectorbt.portfolio.enums import TradeStatus

    from vectorbtx.metrics.trades.register import _STATUS_CLOSED

    assert _STATUS_CLOSED == TradeStatus.Closed
