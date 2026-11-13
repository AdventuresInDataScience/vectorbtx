"""The borrow fee on a short in something traded directly (Section G, G6).

It follows how the instrument is traded: shorting a security you trade directly
(`mode="equity"`: a share, ETF, fund or commodity product) borrows it, and the lender
charges for that. A CFD or spread bet borrows nothing; its short already pays funding at
`benchmark - markup`. The rate is the user's (`Instrument.borrow_rate`), because none is
published; always a charge, only while short; a short held with no rate warns.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

try:
    import vectorbt  # noqa: F401

    HAS_VECTORBT = True
except ImportError:  # pragma: no cover
    HAS_VECTORBT = False

needs_vectorbt = pytest.mark.skipif(not HAS_VECTORBT, reason="requires vectorbt")

INDEX = pd.bdate_range("2024-01-01", periods=12)          # bar 3 is a Thursday: one day
THURSDAY = 3
DIRECT = {"mode": "equity", "quote_currency": "USD", "spread_points": 0.0, "slippage": 0.0,
          "leverage": 2.0}
SHARE_CFD = {"mode": "cfd", "contract_size": 1.0, "quote_currency": "USD", "spread_points": 0.0}


def _flat(columns=("X",)):
    return pd.DataFrame({c: np.full(len(INDEX), 100.0) for c in columns}, index=INDEX)


def _run(side, instrument, rates=None, **kwargs):
    """100 units opened at bar 1's open and held to the end, at a flat 100: 10,000."""
    import vectorbtx as vbt

    c = _flat()
    none = pd.DataFrame(False, index=INDEX, columns=["X"])
    signal = none.copy()
    signal.iloc[0] = True
    signals = (dict(entries=signal, exits=none) if side == "long"
               else dict(entries=none, exits=none, short_entries=signal, short_exits=none))
    return vbt.x.PortfolioBacktest(close=c, open=c, high=c + 0.5, low=c - 0.5,
                                   instruments={"X": instrument}, size=100, init_cash=10_000,
                                   freq="D", rates=rates, **signals, **kwargs)


def _borrow_warnings(caught):
    return [w for w in caught if "borrow_rate" in str(w.message)]


def _costs(pf):
    return 0.0 if pf.holding_costs is None else float(pf.holding_costs.to_numpy().sum())


# ############# The arithmetic ############# #


def test_the_charge_is_rate_times_value_over_the_day_count_and_never_positive():
    from vectorbtx.holding.borrow import borrow_charge

    assert float(borrow_charge(36_000, 0.01, day_multiplier=1, day_count=360)) == pytest.approx(-1.0)
    assert float(borrow_charge(-36_000, 0.01, day_multiplier=1, day_count=360)) == pytest.approx(-1.0)
    assert float(borrow_charge(36_000, 0.01, day_multiplier=3, day_count=360)) == pytest.approx(-3.0)
    assert float(borrow_charge(36_500, 0.01, day_multiplier=1, day_count=365)) == pytest.approx(-1.0)
    assert float(borrow_charge(36_000, 0.0)) == 0.0


def test_the_series_accrues_only_where_a_short_is_held():
    from vectorbtx.holding.borrow import borrow_series

    got = borrow_series([0.0, 36_000, 36_000, 0.0], 0.01, [1, 1, 3, 1], day_count=360)
    assert got.tolist() == pytest.approx([0.0, -1.0, -3.0, 0.0])


# ############# Where it applies: how the instrument is traded ############# #


@pytest.mark.parametrize("asset_class", [None, "stocks", "index", "commodities"])
def test_direct_trading_carries_the_fee_whatever_is_traded(asset_class):
    """A share, an index fund or ETF, a commodity product: all borrowed to be shorted."""
    import vectorbtx as vbt

    assert "borrow" in vbt.x.default_cost_stack("equity", asset_class)
    vbt.Instrument(mode="equity", asset_class=asset_class, borrow_rate=0.005)   # accepted


@pytest.mark.parametrize("mode, asset_class", [
    ("cfd", "stocks"), ("cfd", "index"), ("cfd", "fx"),
    ("spreadbet", "stocks"), ("spreadbet", "index"), ("spreadbet", "fx"), ("fx_spot", None),
])
def test_cfds_spread_bets_and_fx_carry_no_fee(mode, asset_class):
    """Their short side already pays funding at benchmark - markup; a fee would be twice."""
    import vectorbtx as vbt

    assert "borrow" not in vbt.x.default_cost_stack(mode, asset_class)


@pytest.mark.parametrize("fields, match", [
    (dict(mode="equity", borrow_rate=-0.01), "never received"),
    (dict(mode="equity", borrow_rate=float("nan")), "finite"),
    (dict(mode="equity", borrow_rate="1%"), "finite"),
    (dict(mode="cfd", contract_size=1.0, borrow_rate=0.01), "CFD or spread bet borrows nothing"),
    (dict(mode="spreadbet", borrow_rate=0.01), "CFD or spread bet borrows nothing"),
    (dict(mode="fx_spot", contract_size=100_000.0, borrow_rate=0.01), "trade directly"),
    (dict(mode="cfd", contract_size=1.0, active_costs=frozenset({"spread", "borrow"})),
     "includes 'borrow'"),
    (dict(mode="equity", borrow_rate=0.01, active_costs=frozenset({"spread", "financing"})),
     "silently ignored"),
])
def test_a_borrow_rate_that_cannot_mean_anything_raises(fields, match):
    import vectorbtx as vbt
    from vectorbtx.instruments.validate import SpecError

    with pytest.raises(SpecError, match=match):
        vbt.Instrument(**fields)


def test_a_spreadsheet_can_set_the_rate_and_the_module():
    import vectorbtx as vbt

    table = pd.DataFrame({"name": ["VOD"], "mode": ["equity"], "borrow_rate": ["0.0125"],
                          "active_costs": ["commission|spread|financing|borrow"]})
    spec = vbt.load_universe(table)["VOD"]
    assert spec.borrow_rate == pytest.approx(0.0125)
    assert "borrow" in spec.active_costs


# ############# Through the front door ############# #


@needs_vectorbt
def test_a_direct_short_pays_the_fee_on_its_market_value_every_day_it_is_held():
    pf = _run("short", {**DIRECT, "borrow_rate": 0.036})
    # 100 units at 100 is 10,000; 3.6% a year over 360 days is 1.00 a day.
    assert float(pf.holding_costs["X"].iloc[THURSDAY]) == pytest.approx(-1.0)
    assert float(pf.holding_costs["X"].iloc[0]) == 0.0          # not yet short
    # Held from bar 1 to bar 11: nine weekdays plus two Fridays at three days each.
    assert float(pf.holding_costs["X"].sum()) == pytest.approx(-15.0)
    # It reaches the totals (issue 15), and trade P&L stays before holding costs.
    assert float(np.ravel(pf.total_profit())[0]) == pytest.approx(-15.0)
    assert float(np.ravel(pf.trades.pnl.sum())[0]) == pytest.approx(0.0)


@needs_vectorbt
def test_a_direct_long_pays_no_borrow_fee():
    pf = _run("long", {**DIRECT, "borrow_rate": 0.036})
    assert _costs(pf) == 0.0


@needs_vectorbt
def test_a_cfd_short_pays_its_funding_and_nothing_else():
    """The short side of a CFD earns benchmark - markup; there is no borrow fee on top."""
    import vectorbtx as vbt

    rates = vbt.x.static_rate_table(["USD"], INDEX, {"USD": 0.05})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pf = _run("short", dict(SHARE_CFD), rates=rates, financing_markup=0.03)
    assert float(pf.holding_costs["X"].iloc[THURSDAY]) == pytest.approx(10_000 * 0.02 / 360)
    assert not _borrow_warnings(caught)


@needs_vectorbt
def test_the_fee_needs_no_rate_table_and_is_not_financing():
    """A direct short owes no overdraft, so it pays no financing even with rates; the fee
    is its only holding cost, with or without a rate table."""
    import vectorbtx as vbt

    rates = vbt.x.static_rate_table(["USD"], INDEX, {"USD": 0.05})
    without = _run("short", {**DIRECT, "borrow_rate": 0.036})
    with_rates = _run("short", {**DIRECT, "borrow_rate": 0.036}, rates=rates)
    assert _costs(without) == pytest.approx(-15.0)
    assert _costs(with_rates) == pytest.approx(-15.0)


@needs_vectorbt
@pytest.mark.parametrize("rates", [False, True])
def test_a_direct_short_with_no_rate_warns_and_is_not_charged(rates):
    """Whether or not other holding costs are computed, a free borrow is never silent."""
    import vectorbtx as vbt

    table = vbt.x.static_rate_table(["USD"], INDEX, {"USD": 0.05}) if rates else None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pf = _run("short", dict(DIRECT), rates=table)
    assert len(_borrow_warnings(caught)) == 1
    assert "['X']" in str(_borrow_warnings(caught)[0].message)
    assert _costs(pf) == 0.0


@needs_vectorbt
@pytest.mark.parametrize("side, instrument", [
    ("short", {**DIRECT, "borrow_rate": 0.0}),                        # stated free
    ("long", dict(DIRECT)),                                           # nothing borrowed
    ("short", dict(SHARE_CFD)),                                       # a CFD borrows nothing
    ("short", {"mode": "spreadbet", "quote_currency": "USD", "spread_points": 0.0}),
])
def test_no_warning_where_there_is_no_fee_to_miss(side, instrument):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _run(side, instrument)
    assert not _borrow_warnings(caught)


# ############# The replay path ############# #


def _replay(units, **instrument):
    import vectorbtx as vbt

    c = _flat()
    universe = vbt.load_universe({"X": vbt.Instrument(mode="equity", quote_currency="GBP",
                                                      spread_points=0.0, **instrument)})
    size = pd.DataFrame(np.nan, index=INDEX, columns=["X"])
    size.iloc[0] = float(units)
    return vbt.Portfolio.from_holding_costs(c, size, universe, init_cash=10_000, freq="D")


@needs_vectorbt
def test_the_replay_path_charges_the_fee_without_a_rate_table():
    pf = _replay(-150, leverage=2.0, borrow_rate=0.0365)
    short_value = abs(float(pd.DataFrame(pf.asset_value(group_by=False))["X"].iloc[THURSDAY]))
    assert float(pf.holding_costs["X"].iloc[THURSDAY]) == pytest.approx(-short_value * 0.0365 / 365)


@needs_vectorbt
def test_the_replay_path_charges_no_financing_without_a_rate_table():
    """With no benchmark every rate read as 0%, so financing came to the markup alone."""
    pf = _replay(150, leverage=2.0)                            # 15,000 on 10,000: 5,000 owed
    assert float(pf.x.borrowed_portion().to_numpy().max()) > 4_000
    assert float(pf.holding_costs.to_numpy().sum()) == 0.0


@needs_vectorbt
def test_the_replay_path_warns_about_a_short_with_no_rate():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _replay(-150, leverage=2.0)
    assert len(_borrow_warnings(caught)) == 1


# ############# `financing` is reserved ############# #


def test_financing_resolves_to_auto_and_a_number_warns_that_it_is_unused():
    import vectorbtx as vbt

    assert vbt.Instrument(mode="cfd", contract_size=1.0).filled().financing == "auto"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        vbt.Instrument(mode="cfd", contract_size=1.0, financing="auto")
        vbt.Instrument(mode="cfd", contract_size=1.0)
    assert not [w for w in caught if "financing=" in str(w.message)]
    with pytest.warns(UserWarning, match="reserved for future use"):
        vbt.Instrument(mode="cfd", contract_size=1.0, financing=0.10)
