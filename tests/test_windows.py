"""Section N — several windows of one history in one backtest (OpenStrategy Studio issue 42).

The gate: every window's portfolio from one `windows=` call equals the same call made on that
window alone, with every per-bar input sliced `.loc[start:end]`: the same order records (ids
included), close, value, returns, trades, holding costs and stats. Each configuration below is a
route through the front door the windows must not change.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("vectorbt")

import vectorbtx as vbt  # noqa: E402
from vectorbtx.frontend.prepare import BacktestError  # noqa: E402

N = 360
INDEX = pd.bdate_range("2020-01-01", periods=N)
#: Overlapping windows, one starting on the first bar and one ending on the last.
WINDOWS = [(INDEX[0], INDEX[119]), (INDEX[100], INDEX[239]), (INDEX[230], INDEX[-1])]
SPREADBET = dict(mode="spreadbet", point_size=0.01, spread_points=2.0, leverage=5.0)


def _prices(columns=("A", "B"), seed=7, drift=0.0):
    """Open, high, low and close frames of a random walk, high and low bracketing both."""
    rng = np.random.default_rng(seed)
    close, open_ = {}, {}
    for k, column in enumerate(columns):
        steps = rng.normal(drift, 0.012, N)
        c = 100.0 * (1 + 0.3 * k) * np.exp(np.cumsum(steps))
        o = np.concatenate([[c[0]], c[:-1]]) * np.exp(rng.normal(0.0, 0.003, N))
        close[column], open_[column] = c, o
    close = pd.DataFrame(close, index=INDEX)
    open_ = pd.DataFrame(open_, index=INDEX)
    wiggle = pd.DataFrame(np.abs(rng.normal(0.0, 0.004, close.shape)), index=INDEX, columns=close.columns)
    high = np.maximum(open_, close) * (1 + wiggle)
    low = np.minimum(open_, close) * (1 - wiggle)
    return dict(close=close, open=open_, high=high, low=low)


def _signals(columns=("A", "B"), seed=11, rate=0.06, short=False):
    rng = np.random.default_rng(seed)
    shape = (N, len(columns))

    def mask():
        return pd.DataFrame(rng.random(shape) < rate, index=INDEX, columns=list(columns))

    out = dict(entries=mask(), exits=mask())
    if short:
        out.update(short_entries=mask(), short_exits=mask())
    return out


def _single(kwargs):
    """The same inputs as Series, for `SingleBacktest`."""
    out = {}
    for key, value in kwargs.items():
        out[key] = value.iloc[:, 0] if isinstance(value, pd.DataFrame) and value.shape[1] == 1 else value
    return out


def _sliced(value, start, end):
    """One input as the call on a window alone gets it: everything per bar sliced `.loc[start:end]`."""
    if isinstance(value, (pd.DataFrame, pd.Series)) and isinstance(value.index, pd.DatetimeIndex):
        return value.loc[start:end]
    if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[0] == N:
        a, b = INDEX.slice_locs(start, end)
        return np.ascontiguousarray(value[a:b])
    if isinstance(value, tuple):
        return tuple(_sliced(item, start, end) for item in value)
    return value


def _assert_same(pf, alone):
    """Two portfolios that must be the same backtest."""
    assert list(pf.wrapper.columns) == list(alone.wrapper.columns)
    assert pf.wrapper.index.equals(alone.wrapper.index)
    a, b = pf.order_records, alone.order_records
    assert a.dtype == b.dtype and len(a) == len(b), (len(a), len(b))
    for name in a.dtype.names:
        np.testing.assert_array_equal(a[name], b[name], err_msg=name)
    pd.testing.assert_frame_equal(pd.DataFrame(pf.close), pd.DataFrame(alone.close))
    np.testing.assert_array_equal(np.asarray(pf.value()), np.asarray(alone.value()))
    np.testing.assert_array_equal(np.asarray(pf.returns()), np.asarray(alone.returns()))
    np.testing.assert_array_equal(np.asarray(pf.get_init_cash()), np.asarray(alone.get_init_cash()))
    assert int(np.asarray(pf.trades.count()).sum()) == int(np.asarray(alone.trades.count()).sum())
    if alone.holding_costs is None:
        assert pf.holding_costs is None
    else:
        pd.testing.assert_frame_equal(pd.DataFrame(pf.holding_costs), pd.DataFrame(alone.holding_costs))
    columns = list(alone.wrapper.columns)
    if len(columns) > 1 and not alone.wrapper.grouper.is_grouped():
        for column in columns:
            pd.testing.assert_series_equal(pf.stats(column=column), alone.stats(column=column))
    else:
        pd.testing.assert_series_equal(pf.stats(), alone.stats())


def _gate(kwargs, front=vbt.x.PortfolioBacktest, windows=WINDOWS):
    """Every window of one call against the call on that window alone; returns the windows' portfolios."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pfs = front(**kwargs, windows=windows)
        assert isinstance(pfs, list) and len(pfs) == len(windows)
        for (start, end), pf in zip(windows, pfs):
            alone = front(**{key: _sliced(value, start, end) for key, value in kwargs.items()})
            _assert_same(pf, alone)
    return pfs


# ############# The gate, route by route ############# #


def test_a_fixed_stake_single_instrument():
    kwargs = _single(dict(**_prices(("A",)), **_signals(("A",)), **SPREADBET, stake=1.0, init_cash=10_000,
                          freq="D"))
    pfs = _gate(kwargs, front=vbt.x.SingleBacktest)
    assert all(int(np.asarray(pf.trades.count()).sum()) > 0 for pf in pfs)


@pytest.mark.parametrize("risk_of", ["balance", "equity"])
def test_risk_sizing_from_the_account_with_a_stop_and_target_in_points(risk_of):
    # The re-sizing loop runs: every entry is a share of the account the run has made so far.
    kwargs = _single(dict(**_prices(("A",)), **_signals(("A",)), **SPREADBET, risk_pct=0.02, risk_of=risk_of,
                          stop_points=150.0, target_points=300.0, init_cash=10_000, freq="D"))
    _gate(kwargs, front=vbt.x.SingleBacktest)


def test_vectorbts_own_stops_in_percent():
    kwargs = _single(dict(**_prices(("A",)), **_signals(("A",)), **SPREADBET, risk_pct=0.05, risk_of="equity",
                          sl_stop=0.02, tp_stop=0.04, init_cash=10_000, freq="D"))
    _gate(kwargs, front=vbt.x.SingleBacktest)


def test_a_trailing_stop_in_points():
    kwargs = _single(dict(**_prices(("A",)), **_signals(("A",)), **SPREADBET, risk_pct=0.02, risk_of="balance",
                          trail_points=200.0, init_cash=10_000, freq="D"))
    _gate(kwargs, front=vbt.x.SingleBacktest)


def test_a_stop_at_a_level_that_is_missing_while_it_warms_up():
    prices = _prices(("A",))
    level = prices["low"].rolling(15).min()          # NaN for the first 14 bars: those entries are skipped
    kwargs = _single(dict(**prices, **_signals(("A",)), **SPREADBET, risk_amount=100.0, stop_price=level,
                          init_cash=10_000, freq="D"))
    _gate(kwargs, front=vbt.x.SingleBacktest)


def test_long_and_short_in_separate_accounts():
    kwargs = dict(**_prices(), **_signals(short=True), **SPREADBET, risk_pct=0.02, risk_of="equity",
                  stop_points=150.0, cash="per_instrument", init_cash=10_000, freq="D")
    _gate(kwargs)


@pytest.mark.parametrize("priority", [["B", "A"], "random"])
def test_one_shared_account_per_window(priority):
    kwargs = dict(**_prices(), **_signals(rate=0.1), **SPREADBET, risk_pct=0.03, risk_of="balance",
                  stop_points=150.0, cash="shared", entry_priority=priority, seed=3, init_cash=10_000,
                  freq="D")
    pfs = _gate(kwargs)
    assert all(pf.cash_sharing and pf.wrapper.grouper.is_grouped() for pf in pfs)


def test_holding_costs_with_a_static_rate():
    rates = vbt.x.static_rate_table(["GBP"], INDEX, rate=0.05)
    kwargs = _single(dict(**_prices(("A",)), **_signals(("A",)), **SPREADBET, quote_currency="GBP",
                          account_currency="GBP", rates=rates, risk_pct=0.05, risk_of="balance",
                          stop_points=200.0, init_cash=10_000, freq="D"))
    pfs = _gate(kwargs, front=vbt.x.SingleBacktest)
    assert all(float(pf.holding_costs.to_numpy().sum()) < 0 for pf in pfs)


def test_a_levered_account_that_outgrows_its_starting_loan():
    # Issue 41's grown margin loan: a trending market, 20% risked over a 3% stop at up to 10x.
    prices = _prices(("A",), seed=5, drift=0.004)
    kwargs = _single(dict(**prices, **_signals(("A",), rate=0.15), mode="spreadbet", point_size=0.01,
                          spread_points=1.0, leverage=10.0, risk_pct=0.2, risk_of="balance", sl_stop=0.03,
                          init_cash=10_000, freq="D"))
    pfs = _gate(kwargs, front=vbt.x.SingleBacktest)
    starting = 10_000 * (10.0 - 1.0)
    assert any(float(np.max(pf.margin_loan)) > starting * (1 + 1e-9) for pf in pfs)


def test_automatic_slippage_measures_each_windows_own_history():
    prices = _prices(("A",))
    volume = pd.DataFrame({"A": np.random.default_rng(2).uniform(5e4, 2e5, N)}, index=INDEX)
    kwargs = _single(dict(**prices, **_signals(("A",)), mode="equity", point_size=0.01, slippage="auto",
                          commission_pct=0.001, volume=volume, notional=5_000.0, init_cash=10_000, freq="D"))
    _gate(kwargs, front=vbt.x.SingleBacktest)


def test_prices_converted_to_the_account_currency():
    from vectorbtx.fx.convert import to_base_currency

    prices = _prices(("A",))
    gbpusd = pd.Series(1.25 + 0.05 * np.sin(np.arange(N) / 20.0), index=INDEX, name="GBPUSD")
    converted = {name: to_base_currency(frame, "GBP", fx=gbpusd, quote_currency="USD", validate=False)
                 for name, frame in prices.items()}
    kwargs = dict(**converted, **_signals(("A",)), mode="spreadbet", point_size=0.01, spread_points=2.0,
                  leverage=5.0, quote_currency="USD", account_currency="GBP", risk_pct=0.02, risk_of="equity",
                  stop_points=150.0, init_cash=10_000, freq="D")
    _gate(kwargs)


def test_a_stop_callback_reading_its_own_per_bar_array():
    from numba import njit

    @njit
    def tighten_nb(c, factor):
        # Reads its own array by bar and column, as the Studio's extra exits do.
        return c.curr_stop * factor[c.i, c.col], c.curr_trail

    prices = _prices()
    factor = np.ascontiguousarray(np.where(np.arange(N)[:, None] % 7 == 0, 0.5, 1.0) * np.ones((N, 2)))
    kwargs = dict(**prices, **_signals(), **SPREADBET, stake=1.0, sl_stop=0.03, adjust_sl_func_nb=tighten_nb,
                  adjust_sl_args=(factor,), cash="per_instrument", init_cash=10_000, freq="D")
    _gate(kwargs)


def test_windows_that_do_not_overlap_cover_the_history_as_folds():
    cuts = np.linspace(0, N, 7).astype(int)
    folds = [(INDEX[a], INDEX[b - 1]) for a, b in zip(cuts[:-1], cuts[1:])]
    kwargs = _single(dict(**_prices(("A",)), **_signals(("A",)), **SPREADBET, risk_pct=0.02, risk_of="balance",
                          stop_points=150.0, init_cash=10_000, freq="D"))
    _gate(kwargs, front=vbt.x.SingleBacktest, windows=folds)


# ############# What windows say and refuse ############# #


def test_a_warning_names_the_window_it_is_about():
    # A stake under the minimum in the second window only: its entries are not placed.
    kwargs = _single(dict(**_prices(("A",)), **_signals(("A",)), **SPREADBET, risk_amount=0.5, stop_points=150.0,
                          init_cash=10_000, freq="D"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        vbt.x.SingleBacktest(**kwargs, windows=WINDOWS)
    said = " ".join(str(item.message) for item in caught)
    assert "A (window 2)" in said


@pytest.mark.parametrize("windows, words", [
    ([(INDEX[10], INDEX[10])], "at least 2"),
    ([(INDEX[20], INDEX[10])], "at least 2"),
    ([(INDEX[0] - pd.Timedelta(days=5), INDEX[10])], "outside"),
    ([INDEX[3]], "pair"),
    ([], "empty"),
    ("2020-01-01", "list"),
])
def test_a_bad_window_is_refused(windows, words):
    kwargs = dict(**_prices(("A",)), **_signals(("A",)), **SPREADBET, stake=1.0, init_cash=10_000, freq="D")
    with pytest.raises(BacktestError, match=words):
        vbt.x.PortfolioBacktest(**kwargs, windows=windows)


def test_a_buy_and_hold_and_a_rebalanced_portfolio_are_refused():
    prices = _prices()
    with pytest.raises(BacktestError, match="buy-and-hold"):
        vbt.x.PortfolioBacktest(**prices, **SPREADBET, stake=1.0, init_cash=10_000, freq="D", windows=WINDOWS)
    weights = pd.DataFrame(0.5, index=INDEX, columns=["A", "B"])
    with pytest.raises(BacktestError, match="rebalanced"):
        vbt.x.PortfolioBacktest(**prices, mode="equity", point_size=0.01, weights=weights, init_cash=10_000,
                                freq="D", windows=WINDOWS)


def test_a_raw_grouping_is_refused():
    kwargs = dict(**_prices(), **_signals(), **SPREADBET, stake=1.0, init_cash=10_000, freq="D")
    with pytest.raises(BacktestError, match="cash='shared'"):
        vbt.x.PortfolioBacktest(**kwargs, cash_sharing=True, group_by=True, windows=WINDOWS)
