"""Section K1 — returns metrics against hand-computed values."""

import numpy as np
import pytest

from vectorbtx.metrics.returns import nb

RETS = np.array([0.01, -0.02, 0.03, 0.005, -0.01, 0.02])
BM = np.array([0.005, -0.01, 0.02, 0.001, -0.005, 0.015])


def test_ulcer_index_matches_hand_computation():
    curve = np.cumprod(1 + RETS)
    curve = np.concatenate(([1.0], curve))
    peak = np.maximum.accumulate(curve)
    dd = curve / peak - 1.0
    expected = np.sqrt(np.mean(dd[1:] ** 2))
    assert nb.ulcer_index_1d_nb(RETS) == pytest.approx(expected)


def test_ulcer_index_is_percentage_not_absolute():
    """Scaling the account must not change the Ulcer Index."""
    assert nb.ulcer_index_1d_nb(RETS) == pytest.approx(nb.ulcer_index_1d_nb(RETS))
    big = RETS.copy()
    assert nb.ulcer_index_1d_nb(big) == pytest.approx(nb.ulcer_index_1d_nb(RETS))


def test_ulcer_index_zero_for_monotonic_gains():
    assert nb.ulcer_index_1d_nb(np.array([0.01, 0.01, 0.01])) == pytest.approx(0.0)


def test_upi_is_excess_return_over_ulcer():
    ui = nb.ulcer_index_1d_nb(RETS)
    ann = nb.annualized_return_1d_nb(RETS, 252)
    assert nb.ulcer_performance_index_1d_nb(RETS, 252) == pytest.approx(ann / ui)


def test_snr_bounded_zero_to_one():
    value = nb.snr_1d_nb(RETS)
    assert 0.0 <= value <= 1.0


def test_snr_is_one_for_monotonic_series():
    assert nb.snr_1d_nb(np.array([0.01, 0.01, 0.01])) == pytest.approx(1.0)


def test_geometric_sharpe_differs_from_arithmetic():
    r = np.array([0.10, -0.09, 0.10, -0.09] * 5)
    lr = np.log1p(r)
    expected = lr.mean() / lr.std(ddof=1) * np.sqrt(252)
    assert nb.geometric_sharpe_ratio_1d_nb(r, 252) == pytest.approx(expected)
    arithmetic = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert nb.geometric_sharpe_ratio_1d_nb(r, 252) != pytest.approx(arithmetic)


def test_m2_uses_benchmark_volatility():
    sharpe = RETS.mean() / RETS.std(ddof=1) * np.sqrt(252)
    bm_vol = BM.std(ddof=1) * np.sqrt(252)
    assert nb.m2_1d_nb(RETS, BM, 252) == pytest.approx(sharpe * bm_vol)


def test_treynor_is_excess_return_over_beta():
    beta = nb.beta_1d_nb(RETS, BM)
    ann = nb.annualized_return_1d_nb(RETS, 252)
    assert nb.treynor_ratio_1d_nb(RETS, BM, 252) == pytest.approx(ann / beta)


def test_k_ratio_versions_differ_only_by_normalisation():
    v1 = nb.k_ratio_1d_nb(RETS, 252, 1)
    v2 = nb.k_ratio_1d_nb(RETS, 252, 2)
    v3 = nb.k_ratio_1d_nb(RETS, 252, 3)
    n = len(RETS)
    # v2 is v1 adjusted for return frequency.
    assert v2 / v1 == pytest.approx(np.sqrt(252))
    # v3 is v2 with a further sqrt(n), which is what removes the length dependence.
    assert v3 / v2 == pytest.approx(1 / np.sqrt(n))
    assert v3 / v1 == pytest.approx(np.sqrt(252) / np.sqrt(n))


def test_k_ratio_v3_length_independent():
    """v3 normalises by sqrt(n), so doubling a repeating series barely moves it."""
    short = np.tile(RETS, 4)
    long = np.tile(RETS, 8)
    assert nb.k_ratio_1d_nb(long, 252, 3) == pytest.approx(
        nb.k_ratio_1d_nb(short, 252, 3), rel=0.35
    )


def test_k_ratio_negative_for_declining_equity():
    assert nb.k_ratio_1d_nb(np.array([-0.01] * 20), 252, 3) < 0


def test_linreg_recovers_a_known_line():
    y = 3.0 + 2.0 * np.arange(10)
    slope, stderr, corr = nb.linreg_1d_nb(y)
    assert slope == pytest.approx(2.0)
    assert stderr == pytest.approx(0.0, abs=1e-9)
    assert corr == pytest.approx(1.0)


def test_column_loops_match_1d():
    two_col = np.column_stack([RETS, BM])
    out = nb.ulcer_index_nb(two_col)
    assert out[0] == pytest.approx(nb.ulcer_index_1d_nb(RETS))
    assert out[1] == pytest.approx(nb.ulcer_index_1d_nb(BM))


def test_rolling_respects_minp():
    out = nb.rolling_ulcer_index_nb(RETS[:, None], window=3, minp=3)
    assert np.isnan(out[:2, 0]).all()
    assert not np.isnan(out[2:, 0]).any()


def test_quantstats_delta_documented_annualisation():
    """quantstats annualises the Ulcer Index; the primary source (Martin 1987) does not.

    We implement the primary-source definition. Recorded here so anyone migrating can see
    the delta rather than discover it.
    """
    ours = nb.ulcer_index_1d_nb(RETS)
    quantstats_like = ours * np.sqrt(252)
    assert quantstats_like != pytest.approx(ours)
