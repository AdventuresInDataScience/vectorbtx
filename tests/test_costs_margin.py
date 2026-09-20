"""Sections E and F — cost composition and leverage caps."""

import numpy as np
import pandas as pd
import pytest

from vectorbtx.costs.commission import COMMISSION
from vectorbtx.costs.compose import resolve_costs
from vectorbtx.costs.helpers import equivalent_flat, equivalent_pct
from vectorbtx.costs.slippage import auto_slippage
from vectorbtx.costs.spread import SPREAD
from vectorbtx.fx.table import FXTable
from vectorbtx.instruments.spec import Instrument
from vectorbtx.margin.cap import cap_user_size, leverage_cap
from vectorbtx.margin.requirement import margin_required, notional
from vectorbtx.resolve.load import load_universe

IDX = pd.date_range("2024-01-01", periods=4, freq="D")
PRICES = pd.DataFrame({"UKX": [7500.0, 7520, 7480, 7550], "MSFT": [300.0, 305, 310, 315]}, index=IDX)


def test_flat_to_percentage_helper():
    assert equivalent_pct(10, 10_000) == pytest.approx(0.001)
    assert equivalent_flat(0.001, 10_000) == pytest.approx(10.0)


def test_spread_is_half_the_gap_per_side():
    spec = Instrument(mode="spreadbet", point_size=1.0, spread_points=4.0).filled()
    out = SPREAD.contribute(spec, PRICES["UKX"], {})
    assert out["slippage"][0] == pytest.approx(0.5 * 4 * 1.0 / 7500)


def test_spread_bet_has_no_commission():
    spec = Instrument(mode="spreadbet", point_size=1.0).filled()
    out = COMMISSION.contribute(spec, PRICES["UKX"], {})
    assert out["fees"] == 0.0 and out["fixed_fees"] == 0.0


def test_flat_equity_commission_maps_to_fixed_fees():
    spec = Instrument(mode="equity", commission_flat=10.0, quote_currency="GBP").filled()
    out = COMMISSION.contribute(spec, PRICES["MSFT"], {})
    assert out["fixed_fees"] == pytest.approx(10.0)


def test_minimum_commission_is_precomputed_into_fixed_fees():
    """max(pct x notional, min) is a conditional; fees + fixed_fees are additive."""
    spec = Instrument(mode="equity", commission_pct=0.001, commission_min=5.0,
                      quote_currency="GBP").filled()
    out = COMMISSION.contribute(spec, PRICES["MSFT"], {"notional": np.array([1000.0, 20_000.0])})
    assert out["fees"] == 0.0
    assert out["fixed_fees"][0] == pytest.approx(5.0)     # minimum binds
    assert out["fixed_fees"][1] == pytest.approx(20.0)    # percentage binds


def test_minimum_commission_without_notional_fails_loudly():
    from vectorbtx.costs.base import CostError

    spec = Instrument(mode="equity", commission_pct=0.001, commission_min=5.0).filled()
    with pytest.raises(CostError, match="needs the order notional"):
        COMMISSION.contribute(spec, PRICES["MSFT"], {"column": "MSFT"})


def test_foreign_fixed_fee_is_converted():
    table = FXTable.constant({"USD": 1.0, "GBP": 1.25}, IDX)
    spec = Instrument(mode="equity", commission_flat=10.0, quote_currency="USD",
                      cost_currency="USD").filled()
    out = COMMISSION.contribute(
        spec, PRICES["MSFT"], {"fx_table": table, "base_currency": "GBP", "column": "MSFT"}
    )
    assert float(np.ravel(out["fixed_fees"])[0]) == pytest.approx(10.0 * (1 / 1.25))


def test_spread_and_slippage_are_summed_not_overwritten():
    universe = load_universe({
        "UKX": Instrument(mode="spreadbet", point_size=1.0, spread_points=4.0, slippage=0.001),
    })
    out = resolve_costs(universe, PRICES[["UKX"]], base_currency="GBP")
    expected = 0.5 * 4 * 1.0 / 7500 + 0.001
    assert float(np.ravel(out["slippage"])[0]) == pytest.approx(expected)


def test_auto_slippage_is_square_root_in_size():
    price = pd.Series(np.linspace(100, 110, 40), index=pd.date_range("2024-01-01", periods=40))
    volume = pd.Series(1e6, index=price.index)
    small = auto_slippage(price, volume, 1_000).iloc[-1]
    big = auto_slippage(price, volume, 4_000).iloc[-1]
    assert big / small == pytest.approx(2.0, rel=1e-6)


def test_auto_slippage_is_clipped():
    price = pd.Series(np.linspace(100, 200, 40), index=pd.date_range("2024-01-01", periods=40))
    volume = pd.Series(10.0, index=price.index)
    assert auto_slippage(price, volume, 1e9, slippage_max=0.005).max() <= 0.005


def test_auto_slippage_fails_cleanly_without_volume():
    from vectorbtx.costs.base import CostError
    from vectorbtx.costs.slippage import SLIPPAGE

    spec = Instrument(mode="equity", slippage="auto").filled()
    with pytest.raises(CostError, match="needs a volume series"):
        SLIPPAGE.contribute(spec, PRICES["MSFT"], {"column": "MSFT", "size": 100})


def test_fx_markup_reaches_slippage():
    universe = load_universe({"MSFT": Instrument(mode="equity", spread_points=0.0)})
    priced = PRICES[["MSFT"]].copy()
    from vectorbtx.fx.convert import ConvertedFrame

    frame = ConvertedFrame(priced)
    frame.fx_markup = 0.005
    out = resolve_costs(universe, frame, base_currency="GBP")
    assert float(np.ravel(out["slippage"])[0]) == pytest.approx(0.005)


# --- Section F ---

CFD_5X = Instrument(mode="cfd", point_size=0.01, contract_size=1.0, leverage=5.0).filled()


def test_worked_leverage_example():
    """5x CFD: 100 units at 500 -> £50,000 notional, £10,000 margin."""
    assert notional(100.0, 500.0, CFD_5X) == pytest.approx(50_000.0)
    assert margin_required(100.0, 500.0, CFD_5X) == pytest.approx(10_000.0)
    assert leverage_cap(CFD_5X, 10_000, 500.0) == pytest.approx(100.0)


def test_max_size_clips_an_over_leveraged_order():
    capped, was_capped = cap_user_size(500.0, CFD_5X, 10_000, 500.0)
    assert capped == pytest.approx(100.0)
    assert was_capped


def test_within_cap_is_untouched():
    capped, was_capped = cap_user_size(50.0, CFD_5X, 10_000, 500.0)
    assert capped == pytest.approx(50.0)
    assert not was_capped


def test_margin_required_per_mode():
    spreadbet = Instrument(mode="spreadbet", point_size=0.01, leverage=5.0).filled()
    assert margin_required(1.0, 500.0, spreadbet) == pytest.approx(10_000.0)
    equity = Instrument(mode="equity", point_size=0.01, leverage=2.0).filled()
    assert margin_required(100.0, 500.0, equity) == pytest.approx(25_000.0)


# --- Section F: borrowed portion, the three cases the plan requires ----------------
# Split out as pure arithmetic so they need no simulation.

from vectorbtx.margin.exposure import borrowed_amount


def test_cash_covered_position_borrows_nothing():
    """The premise of Section G: a fully covered position finances nothing."""
    assert borrowed_amount(10_000, 10_000) == 0.0
    assert borrowed_amount(10_000, 25_000) == 0.0


def test_partially_financed_position():
    """£50,000 against £10,000 cash borrows £40,000 — the worked example."""
    assert borrowed_amount(50_000, 10_000) == 40_000.0


def test_fully_leveraged_position_borrows_everything():
    assert borrowed_amount(100_000, 0) == 100_000.0


def test_shorts_borrow_on_absolute_exposure():
    assert borrowed_amount(-50_000, 10_000) == 40_000.0


def test_negative_cash_does_not_reduce_borrowing():
    """Clipped at zero: owing money cannot cover a position."""
    assert borrowed_amount(50_000, -5_000) == 50_000.0


def test_borrowed_amount_is_elementwise():
    got = borrowed_amount([10_000, 50_000, 100_000], [10_000, 10_000, 0])
    assert list(got) == [0.0, 40_000.0, 100_000.0]


# --- Section F: raise_reject is genuine vectorbt passthrough -----------------------


def test_raise_reject_is_a_real_vectorbt_parameter():
    """Our docs promise reject-instead-of-clip semantics; confirm vectorbt supplies it,
    so the promise is passthrough rather than an invention."""
    try:
        from vectorbt.portfolio.enums import Order
    except ImportError:
        import pytest

        pytest.skip("requires vectorbt")
    assert "raise_reject" in Order._fields
