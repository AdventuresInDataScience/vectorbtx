"""Section H — the cross-mode equivalence test, this section's real deliverable.

An economically identical position — same underlying, same exposure, same price move —
expressed as a spread bet, a CFD, an FX spot position and shares. All four must produce
identical P&L to floating-point tolerance.
"""

import numpy as np
import pandas as pd
import pytest

from vectorbtx.instruments.spec import Instrument
from vectorbtx.modes.base import get_adapter
from vectorbtx.modes.helpers import (
    pnl,
    points_to_price_fraction,
    price_fraction_to_points,
    size_for_notional,
    size_for_risk,
    value_per_point,
)
from vectorbtx.quantize.quantizer import quantize

ENTRY, EXIT = 500.0, 510.0
NOTIONAL = 50_000.0

SPREADBET = Instrument(mode="spreadbet", point_size=0.01, quote_currency="GBP").filled()
CFD = Instrument(mode="cfd", point_size=0.01, contract_size=1.0, quote_currency="GBP").filled()
EQUITY = Instrument(mode="equity", point_size=0.01, quote_currency="GBP").filled()
FX = Instrument(
    mode="fx_spot", point_size=0.0001, contract_size=100_000.0,
    quote_currency="USD", base_currency="EUR",
).filled()

ALL = [SPREADBET, CFD, EQUITY]


def _user_size_for(spec, engine_size):
    return get_adapter(spec.mode).from_engine_units(engine_size, spec)


def test_cross_mode_equivalence():
    """THE test. Same exposure, same move, four expressions, one P&L."""
    engine_size = NOTIONAL / ENTRY
    results = {
        spec.mode: pnl(_user_size_for(spec, engine_size), ENTRY, EXIT, spec) for spec in ALL
    }
    values = list(results.values())
    assert values[0] == pytest.approx(1000.0)
    for value in values[1:]:
        assert value == pytest.approx(values[0], rel=1e-12)


def test_spread_bet_conversion_uk_stock():
    """£1/point on a GBP series with point_size 0.01 is 100 shares."""
    assert get_adapter("spreadbet").to_engine_units(1.0, SPREADBET) == pytest.approx(100.0)


def test_spread_bet_conversion_eurusd():
    """£1/pip on EURUSD is 10,000 units, i.e. 0.1 standard lots."""
    spec = Instrument(mode="spreadbet", point_size=0.0001).filled()
    units = get_adapter("spreadbet").to_engine_units(1.0, spec)
    assert units == pytest.approx(10_000.0)
    assert units / 100_000.0 == pytest.approx(0.1)


def test_spread_bet_conversion_micro_fx():
    """£0.10/pip is 1,000 units, i.e. a micro lot."""
    spec = Instrument(mode="spreadbet", point_size=0.0001, min_size=0.1).filled()
    units = get_adapter("spreadbet").to_engine_units(0.10, spec)
    assert units == pytest.approx(1_000.0)
    assert units / 100_000.0 == pytest.approx(0.01)


@pytest.mark.parametrize("spec", [SPREADBET, CFD, EQUITY, FX])
def test_adapters_round_trip(spec):
    adapter = get_adapter(spec.mode)
    for value in (0.01, 1.0, 137.4829):
        back = adapter.from_engine_units(adapter.to_engine_units(value, spec), spec)
        assert back == pytest.approx(value)


def test_point_stops_convert_to_price_fractions():
    frac = points_to_price_fraction(50, 1.23456, FX)
    assert frac == pytest.approx(50 * 0.0001 / 1.23456)
    assert price_fraction_to_points(frac, 1.23456, FX) == pytest.approx(50)


def test_five_decimal_quote_keeps_its_fractional_pip():
    """The 5th decimal must survive conversion — a real part of the spread lives there."""
    price = 1.23456
    frac = points_to_price_fraction(1, price, FX)
    recovered = price * (1 + frac)
    assert recovered == pytest.approx(price + 0.0001, abs=1e-12)
    assert price_fraction_to_points(frac, price, FX) == pytest.approx(1.0)
    truncated = round(price, 4)
    assert points_to_price_fraction(1, price, FX) != points_to_price_fraction(1, truncated, FX)


@pytest.mark.parametrize("spec", ALL)
def test_risk_sizing_gives_identical_monetary_risk(spec):
    size = size_for_risk(100.0, 50, spec)
    stop_price = ENTRY - 50 * spec.point_size
    assert abs(pnl(size, ENTRY, stop_price, spec)) == pytest.approx(100.0)


def test_value_per_point_is_the_stake_for_a_spread_bet():
    assert value_per_point(2.5, SPREADBET) == pytest.approx(2.5)


def test_fractional_stops_are_fx_invariant():
    """D4 scales the whole series uniformly, so a fractional stop is unchanged."""
    # A uniform FX rescaling multiplies both the price and the point size. The *number*
    # of points to the stop is unchanged, so the resulting fraction must be too.
    scaled = points_to_price_fraction(50, 1.23456 * 1.27,
                                      Instrument(mode="fx_spot", point_size=0.0001 * 1.27,
                                                 contract_size=100_000.0,
                                                 quote_currency="USD",
                                                 base_currency="EUR").filled())
    assert scaled == pytest.approx(points_to_price_fraction(50, 1.23456, FX))


# --- index preservation -----------------------------------------------------------
# A helper that takes a price series must return something aligned to it. A bare ndarray
# looks correct in isolation, then reindexes against a dated frame downstream and yields
# a union-shaped result instead of raising — the exact silent-wrong-shape failure this
# library exists to avoid.

_IDX = pd.date_range("2024-01-01", periods=50, freq="B")
_PRICE = pd.Series(np.linspace(100, 120, 50), index=_IDX)


def test_helpers_preserve_the_price_index():
    for spec in (SPREADBET, CFD, EQUITY, FX):
        for out in (
            size_for_risk(1000.0, 50, spec, _PRICE),
            size_for_notional(10_000.0, _PRICE, spec),
            points_to_price_fraction(50, _PRICE, spec),
            price_fraction_to_points(0.01, _PRICE, spec),
        ):
            assert isinstance(out, pd.Series)
            assert out.index.equals(_IDX)


def test_sizes_multiply_without_reindexing():
    """The failure this guards: close * size silently becoming a 2n-row union."""
    size = size_for_notional(10_000.0, _PRICE, EQUITY)
    quantized, _ = quantize(size, EQUITY)
    engine = get_adapter(EQUITY).to_engine_units(quantized, EQUITY, _PRICE)
    notional = _PRICE * engine
    assert len(notional) == len(_IDX)
