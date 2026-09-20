"""Section N — the front-door API.

Covers the acceptance criteria in PLAN.md §N "Done when", plus the failure modes the
layer exists to make loud.
"""

import numpy as np
import pandas as pd
import pytest

try:
    import vectorbt  # noqa: F401

    HAS_VECTORBT = True
except ImportError:  # pragma: no cover
    HAS_VECTORBT = False

needs_vectorbt = pytest.mark.skipif(not HAS_VECTORBT, reason="requires vectorbt")

from vectorbtx.frontend.prepare import (
    BacktestError,
    resolve_size,
    strategy_of,
    to_universe,
)
from vectorbtx.instruments.spec import Instrument

IDX = pd.date_range("2024-01-01", periods=20, freq="B")
FLAT = pd.DataFrame({"A": np.full(20, 100.0)}, index=IDX)
#: Orders fill at the next bar's open, so every front-door call needs one.
FLAT_OPEN = FLAT.copy()
#: high/low are mandatory too — stops are checked against them.
FLAT_HIGH = FLAT * 1.01
FLAT_LOW = FLAT * 0.99
COLS = ["VOD.L", "MSFT"]
SHARED = dict(mode="equity", point_size=0.01, commission_flat=10.0)


# ############# Four supply forms ############# #


def test_all_four_instrument_forms_agree(tmp_path):
    """PLAN §N: 'All four instrument supply forms produce identical Universe objects'."""
    flat = to_universe(None, columns=COLS, **SHARED)
    overrides = to_universe({c: {} for c in COLS}, columns=COLS, **SHARED)
    objects = to_universe({c: Instrument(**SHARED) for c in COLS})

    path = tmp_path / "u.csv"
    pd.DataFrame(
        [{"name": c, "mode": "equity", "point_size": 0.01, "commission_flat": 10.0}
         for c in COLS]
    ).to_csv(path, index=False)
    from_csv = to_universe(path)

    assert flat == overrides == objects == from_csv


def test_per_symbol_mode_overrides_the_shared_one():
    """How one CFD sits in a book of equities."""
    universe = to_universe(
        {"VOD.L": {}, "NVDA": dict(mode="cfd", contract_size=1.0)},
        columns=["VOD.L", "NVDA"], **SHARED,
    )
    assert universe["VOD.L"].mode == "equity"
    assert universe["NVDA"].mode == "cfd"


# ############# Mode legality ############# #


@pytest.mark.parametrize(
    "mode,arg,value",
    [
        ("spreadbet", "contract_size", 1.0),
        ("spreadbet", "allow_fractional", True),
        ("equity", "contract_size", 1.0),
        ("equity", "base_currency", "USD"),
        ("cfd", "allow_fractional", True),
    ],
)
def test_rejected_arguments_raise_naming_mode_and_argument(mode, arg, value):
    """PLAN §N: 'Every mode's rejected arguments raise, naming mode and argument'."""
    kwargs = dict(mode=mode, point_size=0.01, **{arg: value})
    if mode == "cfd":
        kwargs["contract_size"] = 1.0
    with pytest.raises(BacktestError) as exc:
        to_universe(None, columns=["A"], **kwargs)
    assert arg in str(exc.value) and mode in str(exc.value)


def test_missing_mode_raises():
    with pytest.raises(BacktestError, match="mode.*required"):
        to_universe(None, columns=["A"], point_size=0.01)


def test_spreadbet_commission_is_forced_to_zero():
    """The fee is baked into a wider spread; a spread bet with commission is a misdescription."""
    universe = to_universe(None, columns=["A"], mode="spreadbet", point_size=0.01,
                           commission_pct=0.01)
    assert universe["A"].commission_pct == 0.0


# ############# Strategy dispatch ############# #


def test_strategy_is_implied_by_arguments():
    assert strategy_of(entries=1) == "signals"
    assert strategy_of(weights=1) == "weights"
    assert strategy_of() == "hold"


def test_signals_and_weights_together_raise():
    with pytest.raises(BacktestError, match="Both signal"):
        strategy_of(entries=1, weights=1)


# ############# Sizing ############# #


def test_two_sizing_arguments_raise_naming_both():
    """PLAN §N: 'Two sizing arguments raise, naming both'."""
    universe = to_universe(None, columns=["A"], mode="equity", point_size=0.01)
    with pytest.raises(BacktestError) as exc:
        resolve_size(universe, FLAT, ["A"], stake=1.0, notional=100.0)
    assert "stake" in str(exc.value) and "notional" in str(exc.value)


def test_risk_sizing_without_a_stop_raises():
    universe = to_universe(None, columns=["A"], mode="equity", point_size=0.01)
    with pytest.raises(BacktestError, match="stop_points"):
        resolve_size(universe, FLAT, ["A"], risk_pct=0.01, init_cash=10_000)


def test_risk_sizing_gives_identical_monetary_risk_across_modes():
    """The cross-mode identity, at the front door: same risk, same money."""
    price = pd.DataFrame({"A": np.full(20, 1.10)}, index=IDX)
    sizes = {}
    for mode, extra in [
        ("spreadbet", {}),
        ("cfd", dict(contract_size=1.0)),
        ("fx_spot", dict(contract_size=1.0, quote_currency="USD", base_currency="EUR")),
        ("equity", {}),
    ]:
        universe = to_universe(None, columns=["A"], mode=mode, point_size=0.0001, **extra)
        engine = resolve_size(universe, price, ["A"], risk_amount=100.0, stop_points=60)
        sizes[mode] = float(engine.iloc[0, 0])
    # engine size is the common currency: 100 / (60 * 0.0001)
    assert set(np.round(list(sizes.values()), 6)) == {round(100 / (60 * 0.0001), 6)}


# ############# Collisions ############# #


@needs_vectorbt
def test_colliding_passthrough_raises_naming_the_field():
    """PLAN §N: 'A colliding passthrough raises, naming the field'."""
    import vectorbtx as vbt

    with pytest.raises(BacktestError, match="fees"):
        vbt.x.PortfolioBacktest(close=FLAT, open=FLAT_OPEN, high=FLAT_HIGH, low=FLAT_LOW, mode="equity", point_size=0.01,
                                commission_pct=0.001, fees=0.0,
                                stake=1.0, init_cash=10_000, freq="D")


# ############# SingleBacktest ############# #


@needs_vectorbt
def test_single_backtest_rejects_multi_column_price():
    import vectorbtx as vbt

    two = pd.DataFrame({"A": np.full(20, 1.0), "B": np.full(20, 1.0)}, index=IDX)
    with pytest.raises(BacktestError, match="PortfolioBacktest"):
        vbt.x.SingleBacktest(close=two, open=two, high=two * 1.01, low=two * 0.99, mode="equity", point_size=0.01)


@needs_vectorbt
def test_single_backtest_rejects_portfolio_only_arguments():
    import vectorbtx as vbt

    with pytest.raises(BacktestError, match="PortfolioBacktest"):
        vbt.x.SingleBacktest(close=FLAT["A"], open=FLAT["A"], high=FLAT["A"] * 1.01, low=FLAT["A"] * 0.99, mode="equity", point_size=0.01, weights=FLAT)


# ############# End to end ############# #


@needs_vectorbt
def test_spreadbet_runs_with_no_universe_no_engine_units_no_splat():
    """PLAN §N: the flagship criterion — a spread bet in a handful of lines."""
    import vectorbtx as vbt

    n = 120
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    rng = np.random.default_rng(7)
    px = pd.Series(1.10 * np.exp(np.cumsum(rng.normal(0, 0.004, n))), index=idx, name="EURUSD")
    fast, slow = px.rolling(5).mean(), px.rolling(20).mean()

    pf = vbt.x.SingleBacktest(
        close=px, open=px, high=px * 1.01, low=px * 0.99,
        entries=(fast > slow) & (fast.shift(1) <= slow.shift(1)),
        exits=(fast < slow) & (fast.shift(1) >= slow.shift(1)),
        mode="spreadbet", asset_class="fx", point_size=0.0001,
        spread_points=0.9, stake=1.0, init_cash=10_000, freq="D",
    )
    assert pf.x.unit_names() == {"EURUSD": "£/point"}
    assert not pf.x.user_size().empty


@needs_vectorbt
def test_rebalanced_investing_reports_drift():
    import vectorbtx as vbt

    n = 200
    idx = pd.date_range("2021-01-04", periods=n, freq="B")
    rng = np.random.default_rng(3)
    px = pd.DataFrame(
        {c: 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.008, n))) for c in ("X", "Y")},
        index=idx,
    )
    weights = px.copy()
    weights["X"], weights["Y"] = 0.6, 0.4

    pf = vbt.x.PortfolioBacktest(
        close=px, open=px, high=px * 1.01, low=px * 0.99, weights=weights, schedule=vbt.x.calendar_schedule(px.index, freq="M"),
        mode="equity", point_size=0.01, commission_flat=5.0,
        init_cash=250_000, freq="D",
    )
    assert len(pf.orders.records) > 0
    assert pf.drift is not None


@needs_vectorbt
def test_leverage_is_funded_not_merely_capped():
    """The headline fix. `buy_nb` bounds every long by cash (`nb.py:113`), so capping with
    `max_size` alone left a 30x spread bet filled at about 1x. The engine must be given
    buying power, with the loan subtracted back out of reported equity."""
    import vectorbtx as vbt

    idx = pd.date_range("2024-01-03", periods=10, freq="B")
    px = pd.DataFrame({"EURUSD": np.full(10, 1.10)}, index=idx)
    entries = pd.DataFrame(False, index=idx, columns=["EURUSD"])
    entries.iloc[1] = True

    pf = vbt.x.SingleBacktest(
        # A narrow intrabar range on purpose: a 60-pip stop on 1.10 is ~55bp, so a +/-1%
        # bar would stop the position out, and this test is about funding, not stops.
        close=px, open=px, high=px * 1.0005, low=px * 0.9995,
        entries=entries, exits=pd.DataFrame(False, index=idx, columns=["EURUSD"]),
        mode="spreadbet", asset_class="fx", point_size=0.0001,
        min_size=1.0, increment=0.01,
        risk_pct=0.01, stop_points=60, leverage=30,
        init_cash=10_000, freq="D",
    )
    # £100 risked over 60 pips -> £1.667/pt, floored to the £0.01 increment.
    assert float(pf.x.user_size().abs().max().iloc[0]) == pytest.approx(1.66)
    # Equity is reported, not buying power.
    assert float(np.ravel(np.asarray(pf.init_cash))[0]) == pytest.approx(10_000.0)
    assert float(np.ravel(np.asarray(pf.value()))[-1]) == pytest.approx(10_000.0, rel=1e-3)


@needs_vectorbt
def test_margin_accounting_closes_exactly():
    """cash + asset_value == equity, and the drawn loan is exactly the negative cash."""
    import vectorbtx as vbt

    idx = pd.date_range("2024-01-03", periods=10, freq="B")
    px = pd.DataFrame({"A": np.full(10, 100.0)}, index=idx)
    size = pd.DataFrame(np.nan, index=idx, columns=["A"])
    size.iloc[0] = [400.0]                       # £40,000 notional on £10,000 equity

    pf = vbt.x.PortfolioBacktest(
        close=px, open=px, high=px * 1.01, low=px * 0.99, mode="equity", point_size=0.01, leverage=5.0,
        size=size, init_cash=10_000, freq="D",
    )
    cash = float(np.ravel(np.asarray(pf.cash()))[-1])
    borrowed = float(np.ravel(np.asarray(pf.x.borrowed_portion()))[-1])
    value = float(np.ravel(np.asarray(pf.value()))[-1])

    assert cash < 0                              # the loan is drawn
    assert borrowed == pytest.approx(-cash)      # borrowed IS the negative cash
    assert value == pytest.approx(cash + 40_000.0)
    assert value == pytest.approx(10_000.0, rel=1e-3)


@needs_vectorbt
def test_borrowed_portion_excludes_the_cash_covered_part():
    """£40,000 against £10,000 borrows £30,000, not £40,000. Charging financing on the
    asset value would bill the cash-covered part too."""
    import vectorbtx as vbt

    idx = pd.date_range("2024-01-03", periods=10, freq="B")
    px = pd.DataFrame({"A": np.full(10, 100.0)}, index=idx)
    size = pd.DataFrame(np.nan, index=idx, columns=["A"])
    size.iloc[0] = [400.0]

    pf = vbt.x.PortfolioBacktest(
        close=px, open=px, high=px * 1.01, low=px * 0.99, mode="equity", point_size=0.01, leverage=5.0,
        size=size, init_cash=10_000, freq="D",
    )
    assert float(np.ravel(np.asarray(pf.x.borrowed_portion()))[-1]) == pytest.approx(
        30_000.0, rel=0.02
    )


@needs_vectorbt
def test_financing_now_fires_on_a_levered_long():
    """Before the fix a long could never exceed cash, so `borrowed_portion` was ~zero and
    leverage financing silently never applied."""
    import vectorbtx as vbt

    idx = pd.date_range("2024-01-01", periods=25, freq="B")
    px = pd.DataFrame({"A": np.full(25, 100.0)}, index=idx)
    universe = vbt.load_universe(
        {"A": vbt.Instrument(mode="equity", quote_currency="GBP", leverage=5.0)}
    )
    size = pd.DataFrame(np.nan, index=idx, columns=["A"])
    size.iloc[0] = [400.0]

    pf = vbt.x.PortfolioBacktest(
        close=px, open=px, high=px * 1.01, low=px * 0.99, instruments=universe, size=size,
        rates=vbt.x.static_rate_table(["GBP"], idx, rate=0.05),
        init_cash=10_000, freq="D",
    )
    assert pf.holding_costs.to_numpy().sum() < -50      # a real, material charge


@needs_vectorbt
def test_exceeding_funded_buying_power_still_warns():
    """Leverage is funded, but asking for more than the leverage allows is a partial fill
    rather than a rejection, so it must still be said out loud."""
    import vectorbtx as vbt

    idx = pd.date_range("2024-01-03", periods=10, freq="B")
    px = pd.DataFrame({"A": np.full(10, 100.0)}, index=idx)
    with pytest.warns(UserWarning, match="funded buying power"):
        vbt.x.PortfolioBacktest(
            close=px, open=px, high=px * 1.01, low=px * 0.99, mode="equity", point_size=0.01, leverage=2.0,
            notional=50_000, init_cash=10_000, freq="D",
        )


@needs_vectorbt
def test_unlevered_portfolio_carries_no_loan():
    """The whole mechanism must be inert at 1x — §1.3 drop-in guarantee."""
    import vectorbtx as vbt

    idx = pd.date_range("2024-01-03", periods=10, freq="B")
    px = pd.DataFrame({"A": np.full(10, 100.0)}, index=idx)
    pf = vbt.x.PortfolioBacktest(
        close=px, open=px, high=px * 1.01, low=px * 0.99, mode="equity", point_size=0.01, leverage=1.0,
        stake=10.0, init_cash=10_000, freq="D",
    )
    assert float(np.ravel(np.asarray(pf.margin_loan))[0]) == pytest.approx(0.0)
    assert float(np.ravel(np.asarray(pf.init_cash))[0]) == pytest.approx(10_000.0)


# ############# Next-open execution ############# #


@needs_vectorbt
def test_orders_fill_at_the_next_bar_open_never_the_signal_close():
    """A signal from bar `t`'s close cannot be acted on until bar `t+1` opens.

    The fixture makes the difference impossible to miss: close is flat at 100 throughout,
    but the open gaps to 110 on the bar after the signal. Filling at the signal bar's close
    would buy at 100 and pocket the whole gap — a profit no one could have made.
    """
    import vectorbtx as vbt

    idx = pd.date_range("2024-01-01", periods=8, freq="B")
    close = pd.DataFrame({"A": np.full(8, 100.0)}, index=idx)
    op = pd.DataFrame({"A": [100.0, 100.0] + [110.0] * 6}, index=idx)
    entries = pd.DataFrame(False, index=idx, columns=["A"])
    entries.iloc[1] = True

    pf = vbt.x.PortfolioBacktest(
        close=close, open=op, high=close * 1.01, low=op * 0.99,
        entries=entries, exits=pd.DataFrame(False, index=idx, columns=["A"]),
        mode="equity", point_size=0.01, stake=10.0, init_cash=10_000, freq="D",
    )
    records = pf.orders.records
    assert list(records["idx"]) == [2]                  # shifted one bar
    assert float(records["price"].iloc[0]) > 109.0      # filled at the gapped open


@needs_vectorbt
def test_open_is_required_with_no_way_to_opt_out():
    """There is no `execution=` argument. Filling at the signal close is lookahead, so the
    API does not offer it — the omission has to be an error, not a default."""
    import vectorbtx as vbt

    entries = pd.DataFrame(False, index=IDX, columns=["A"])
    with pytest.raises(BacktestError, match="next bar's open"):
        vbt.x.PortfolioBacktest(
            close=FLAT, entries=entries, mode="equity", point_size=0.01,
            stake=1.0, init_cash=10_000, freq="D",
        )


@needs_vectorbt
def test_rebalance_schedule_is_shifted_too():
    """A target computed from today's close is executed tomorrow morning."""
    import vectorbtx as vbt

    idx = pd.date_range("2024-01-01", periods=40, freq="B")
    px = pd.DataFrame({"X": np.full(40, 100.0), "Y": np.full(40, 50.0)}, index=idx)
    weights = px.copy()
    weights["X"], weights["Y"] = 0.6, 0.4
    schedule = pd.Series(False, index=idx)
    schedule.iloc[10] = True

    pf = vbt.x.PortfolioBacktest(
        close=px, open=px, high=px * 1.01, low=px * 0.99, weights=weights, schedule=schedule,
        mode="equity", point_size=0.01, init_cash=100_000, freq="D",
    )
    assert set(pf.orders.records["idx"]) == {11}


@needs_vectorbt
@pytest.mark.parametrize("leverage", [1.0, 2.0, 5.0])
def test_weights_mean_fraction_of_equity_not_buying_power(leverage):
    """A 50/50 book is fully invested, never levered, whatever `leverage` says.

    `TargetPercent` is a fraction of the account value the ENGINE sees. Funding the engine
    at `equity x leverage` turned a 60/40 book into 120/80 — fully invested by the user's
    reckoning and 2x levered in fact, on equity mode's *default* 2x. Rescaling the weights
    does not fix it either: the loan is constant while equity moves, so the ratio drifts.
    """
    import vectorbtx as vbt

    idx = pd.date_range("2021-01-04", periods=200, freq="B")
    px = pd.DataFrame({"X": np.full(200, 100.0), "Y": np.full(200, 50.0)}, index=idx)
    weights = px.copy()
    weights["X"], weights["Y"] = 0.5, 0.5

    pf = vbt.x.PortfolioBacktest(
        close=px, open=px, high=px * 1.01, low=px * 0.99, weights=weights,
        schedule=vbt.x.calendar_schedule(idx, freq="M"),
        mode="equity", point_size=0.01, leverage=leverage,
        init_cash=250_000, freq="D",
    )
    exposure = float(pf.asset_value(group_by=False).iloc[-1].sum())
    equity = float(pf.value(group_by=True).iloc[-1])
    assert exposure / equity == pytest.approx(1.0, abs=0.05)


@needs_vectorbt
def test_weights_above_one_warn_rather_than_silently_underfilling():
    """Rebalancing is funded at equity, so excess exposure simply does not fill."""
    import vectorbtx as vbt

    idx = pd.date_range("2021-01-04", periods=60, freq="B")
    px = pd.DataFrame({"X": np.full(60, 100.0)}, index=idx)
    weights = px.copy()
    weights["X"] = 1.5

    with pytest.warns(UserWarning, match="more exposure than the account holds"):
        vbt.x.PortfolioBacktest(
            close=px, open=px, high=px * 1.01, low=px * 0.99, weights=weights, schedule=vbt.x.calendar_schedule(idx, freq="M"),
            mode="equity", point_size=0.01, init_cash=100_000, freq="D",
        )


# ############# Mandatory intrabar extremes ############# #


@needs_vectorbt
def test_high_and_low_are_required():
    """Without them a stop can only be compared against the close, so it fires late or not
    at all, and drawdown misses every intrabar excursion. Both flatter the result."""
    import vectorbtx as vbt

    entries = pd.DataFrame(False, index=IDX, columns=["A"])
    with pytest.raises(BacktestError, match="high.*low.*required"):
        vbt.x.PortfolioBacktest(
            close=FLAT, open=FLAT_OPEN, entries=entries, mode="equity", point_size=0.01,
            stake=1.0, init_cash=10_000, freq="D",
        )


@needs_vectorbt
def test_intrabar_drawdown_is_never_smaller_than_close_to_close():
    """The peak comes from the highs and the trough from the lows. Taking both from the
    lows would depress the running peak too and report a *smaller* drawdown."""
    import vectorbtx as vbt

    idx = pd.date_range("2024-01-01", periods=60, freq="B")
    rng = np.random.default_rng(2)
    close = pd.DataFrame({"A": 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 60)))}, index=idx)
    open_ = close.shift(1).bfill()
    high = pd.concat([close, open_], axis=1).max(axis=1).to_frame("A") * 1.01
    low = pd.concat([close, open_], axis=1).min(axis=1).to_frame("A") * 0.97
    entries = pd.DataFrame(False, index=idx, columns=["A"])
    entries.iloc[1] = True

    pf = vbt.x.PortfolioBacktest(
        close=close, open=open_, high=high, low=low,
        entries=entries, exits=pd.DataFrame(False, index=idx, columns=["A"]),
        mode="equity", point_size=0.01, stake=50.0, init_cash=20_000, freq="D",
    )
    close_to_close = abs(float(np.ravel(np.asarray(pf.max_drawdown()))[0]))
    assert pf.x.intrabar_drawdown() >= close_to_close


@needs_vectorbt
def test_high_low_not_bracketing_close_warns():
    """Usually means the series were adjusted or FX-converted inconsistently, which makes
    stops fire at prices that never traded."""
    import vectorbtx as vbt

    entries = pd.DataFrame(False, index=IDX, columns=["A"])
    with pytest.warns(UserWarning, match="bracket close"):
        vbt.x.PortfolioBacktest(
            close=FLAT, open=FLAT_OPEN, high=FLAT * 0.9, low=FLAT * 0.8,   # high below close
            entries=entries, mode="equity", point_size=0.01,
            stake=1.0, init_cash=10_000, freq="D",
        )


@needs_vectorbt
def test_price_series_must_be_passed_by_keyword():
    """Four same-shaped frames in a row is exactly the argument list where a silent
    transposition — high and low swapped, open where close belongs — produces a plausible
    wrong answer rather than an error. Keyword-only makes the mistake unspellable."""
    import vectorbtx as vbt

    entries = pd.DataFrame(False, index=IDX, columns=["A"])
    with pytest.raises(TypeError, match="positional"):
        vbt.x.PortfolioBacktest(
            FLAT, FLAT_OPEN, FLAT_HIGH, FLAT_LOW,
            entries=entries, mode="equity", point_size=0.01,
            stake=1.0, init_cash=10_000, freq="D",
        )
    with pytest.raises(TypeError, match="positional"):
        vbt.x.SingleBacktest(FLAT["A"], FLAT_OPEN["A"], mode="equity", point_size=0.01)
