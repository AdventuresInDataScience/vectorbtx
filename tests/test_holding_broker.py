"""Section G — fixtures worked from **broker documentation**, not from the plan.

Every other test in this suite encodes the plan's assumptions, which is exactly why none
of them caught the defects in `docs/dev/AUDIT-G.md`. These are derived from IG's published
costs schedules and reproduce their worked examples to the penny, so they fail if the
model drifts away from what a broker actually charges rather than away from what the plan
said a broker charges.

Sources are named per test. Where a number is IG's own, it is reproduced exactly.
"""

import numpy as np
import pandas as pd
import pytest

from vectorbtx.holding.accrual import day_multipliers, trading_day_count
from vectorbtx.holding.carry import carry_charge, tom_next_differential
from vectorbtx.holding.financing import financing_charge
from vectorbtx.instruments.defaults import (
    default_cost_stack,
    financing_basis,
    triple_day,
)
from vectorbtx.instruments.spec import Instrument
from vectorbtx.instruments.validate import SpecError
from vectorbtx.rates.table import day_count_for

WEEK = pd.date_range("2024-01-01", periods=7, freq="D")  # Mon .. Sun


# ############# G1 — financing basis (fixed) ############# #


def test_ig_share_cfd_worked_example_to_the_penny():
    """IG's own worked example, reproduced exactly.

    Long 1500 contracts of an Australian share CFD, contract value AUD 1, closing price
    83.90, 1-month benchmark 1.89%, admin fee 3.0%, 360-day divisor:

        1500 x 1 x 83.90 x (3.0% + 1.89%) / 360 = 17.09 AUD

    Note what the formula contains: contracts x value x **price**. Position value
    throughout, with no reference to account cash. This is the fixture that would have
    caught the borrowed-portion error.
    """
    notional = 1500 * 1 * 83.90            # 125,850 AUD
    charge = financing_charge(
        notional, benchmark_rate=0.0189, markup=0.030, is_long=True,
        day_multiplier=1, day_count=360.0,
    )
    assert float(-charge) == pytest.approx(17.09, abs=0.01)


def test_financing_basis_is_notional_for_derivatives_and_debit_for_equity():
    """The distinction the plan collapsed."""
    assert financing_basis("equity") == "borrowed"
    assert financing_basis("cfd") == "notional"
    assert financing_basis("spreadbet") == "notional"


def test_unlevered_cfd_still_finances_in_full():
    """The sharpest consequence of G1: a broker funds the whole position regardless of
    how much margin sits behind it, so an unlevered CFD is *not* free to hold. Under the
    plan's borrowed-portion rule this charge was exactly zero."""
    charge = financing_charge(
        20_000, benchmark_rate=0.0375, markup=0.025, is_long=True,
        day_multiplier=1, day_count=360.0,
    )
    assert float(-charge) == pytest.approx(20_000 * 0.0625 / 360, rel=1e-9)
    assert float(charge) < 0


# ############# G2 — carry and financing are alternatives ############# #


@pytest.mark.parametrize("mode", ["cfd", "spreadbet"])
def test_fx_underlying_gets_carry_only_never_both(mode):
    """An FX position has ONE overnight charge and it is the tom-next swap, which already
    prices both currencies' interest. Financing on top bills the same money twice."""
    stack = default_cost_stack(mode, "fx")
    assert "carry" in stack
    assert "financing" not in stack


@pytest.mark.parametrize("mode,asset_class", [("cfd", "stocks"), ("spreadbet", "index")])
def test_non_fx_underlying_gets_financing_only(mode, asset_class):
    stack = default_cost_stack(mode, asset_class)
    assert "financing" in stack
    assert "carry" not in stack


def test_combining_carry_and_financing_is_rejected():
    """Guard so the double-charge cannot be reintroduced by hand."""
    with pytest.raises(SpecError, match="both 'carry' and 'financing'"):
        Instrument(
            mode="spreadbet", point_size=0.0001, asset_class="fx",
            active_costs=frozenset({"carry", "financing"}),
        )


# ############# G3 — the triple day follows the asset class ############# #


def test_fx_rolls_on_wednesday_everything_else_on_friday():
    """FX is settlement-driven (spot settles T+2, so Wednesday's roll reaches Monday).
    Shares and indices settle nothing — the triple charge just covers the calendar
    weekend, so it lands on the last trading day before it."""
    assert triple_day("fx") == "WED"
    assert triple_day("stocks") == "FRI"
    assert triple_day("index") == "FRI"
    assert triple_day("commodities") == "FRI"


def test_an_fx_spread_bet_does_not_inherit_the_equity_convention():
    """Resolving the triple day from the mode alone would give every FX spread bet
    Friday, because the spread-bet mode defaults to stocks."""
    assert Instrument(mode="spreadbet", point_size=0.0001, asset_class="fx").filled().triple_day == "WED"
    assert Instrument(mode="spreadbet", point_size=0.01, asset_class="index").filled().triple_day == "FRI"


def test_thursday_to_friday_share_position_is_charged_three_days():
    """The per-trade error the Wednesday default caused. Weekly totals are identical
    either way, so only a short holding period exposes it."""
    fri = day_multipliers(WEEK, triple_day="FRI")
    wed = day_multipliers(WEEK, triple_day="WED")
    assert fri[4] == 3.0 and wed[4] == 1.0        # Friday
    assert trading_day_count(WEEK, triple_day="FRI") == trading_day_count(WEEK, triple_day="WED")


# ############# G4 — the divisor follows the market's denomination ############# #


def test_gbp_markets_use_a_365_day_divisor():
    """Not a house convention: IG uses 365 for the FTSE 100 and other GBP, SGD and ZAR
    denominated markets, and 360 for everything else."""
    assert day_count_for("GBP") == 365.0
    assert day_count_for("SGD") == 365.0
    assert day_count_for("ZAR") == 365.0
    assert day_count_for("USD") == 360.0
    assert day_count_for("EUR") == 360.0
    assert day_count_for(None) == 360.0


def test_uniform_360_overstates_a_gbp_position_by_about_1_4_percent():
    """The size of the bias the plan's uniform ACT/360 introduced."""
    args = dict(benchmark_rate=0.05, markup=0.03, is_long=True, day_multiplier=1)
    correct = float(financing_charge(100_000, day_count=365.0, **args))
    plan = float(financing_charge(100_000, day_count=360.0, **args))
    assert abs(plan) > abs(correct)
    assert abs(plan) / abs(correct) == pytest.approx(365 / 360, rel=1e-9)


# ############# G5 — the admin fee on tom-next ############# #


def test_carry_markup_is_adverse_on_both_sides():
    """Brokers add an admin fee to the tom-next rate. Left at zero, every FX carry number
    is missing the broker's entire margin on it."""
    long_diff = tom_next_differential(0.05, 0.02, markup=0.01, is_long=True)
    short_diff = tom_next_differential(0.05, 0.02, markup=0.01, is_long=False)
    assert float(long_diff) == pytest.approx(0.03 - 0.01)
    assert float(short_diff) == pytest.approx(-0.03 - 0.01)   # adverse, not symmetric


def test_carry_markup_defaults_to_one_percent():
    from vectorbtx.core.settings import get

    assert get("carry_markup") == pytest.approx(0.01)


def test_positive_carry_can_be_flipped_negative_by_the_admin_fee():
    """A thin positive differential does not survive a 1% fee — the reason the default
    mattering is not a rounding argument."""
    diff = tom_next_differential(0.030, 0.025, markup=0.01, is_long=True)
    assert float(carry_charge(100_000, diff, day_multiplier=1)) < 0


# ############# G9/G10 — confirmed, recorded so they are not re-litigated ############# #


def test_short_pays_when_the_admin_fee_exceeds_the_benchmark():
    """IG expresses the short side as admin fee minus benchmark, which is algebraically
    the plan's `benchmark - markup` received. In a low-rate period both sides pay."""
    assert float(financing_charge(100_000, 0.01, 0.03, is_long=False, day_multiplier=1)) < 0
    assert float(financing_charge(100_000, 0.06, 0.03, is_long=False, day_multiplier=1)) > 0


def test_a_week_funds_seven_days_under_either_convention():
    """1+1+3+1+1 for FX, 1+1+1+1+3 for shares. Charging calendar days on top would bill
    nine days a week."""
    assert trading_day_count(WEEK, triple_day="WED") == 7.0
    assert trading_day_count(WEEK, triple_day="FRI") == 7.0


# ############# G8 — holidays ############# #


def test_holidays_are_funded_unless_explicitly_excluded():
    """A market holiday does not stop interest accruing on the money."""
    assert day_multipliers(WEEK)[1] == 1.0                              # ordinary Tuesday
    assert day_multipliers(WEEK, holidays=["2024-01-02"])[1] == 0.0     # opt in to skip


# ############# FX pip convention ############# #


@pytest.mark.parametrize("quote,expected", [
    ("USD", 0.0001), ("EUR", 0.0001), ("GBP", 0.0001), ("CHF", 0.0001),
    ("JPY", 0.01), ("KRW", 0.01),
])
def test_pip_follows_the_quote_currency(quote, expected):
    """The pip is the fourth decimal of the quote for almost every pair, and the second for
    JPY-quoted ones. It is a market convention, so it is derived rather than configured."""
    from vectorbtx.instruments.defaults import fx_pip

    assert fx_pip(quote) == expected


def test_a_mixed_book_needs_no_per_pair_point_size():
    """The reason this matters: one shared block of terms covers EURUSD and USDJPY together."""
    spec_usd = Instrument(mode="spreadbet", asset_class="fx",
                          base_currency="EUR", quote_currency="USD").filled()
    spec_jpy = Instrument(mode="spreadbet", asset_class="fx",
                          base_currency="USD", quote_currency="JPY").filled()
    assert spec_usd.point_size == 0.0001
    assert spec_jpy.point_size == 0.01


def test_contradicting_the_pip_convention_warns():
    """Using 0.0001 on a JPY pair makes every stop 100x too tight — silently, and in a
    direction that produces more trades rather than fewer."""
    with pytest.warns(UserWarning, match="pip convention"):
        Instrument(mode="spreadbet", asset_class="fx", point_size=0.0001,
                   base_currency="USD", quote_currency="JPY")


def test_a_jpy_stop_is_scaled_correctly():
    """30 pips on USDJPY at 150.00 is 0.30 of price, not 0.003."""
    import numpy as np
    import pandas as pd

    from vectorbtx.modes.helpers import points_to_price_fraction

    spec = Instrument(mode="spreadbet", asset_class="fx",
                      base_currency="USD", quote_currency="JPY").filled()
    price = pd.Series(np.full(3, 150.0))
    fraction = float(np.ravel(points_to_price_fraction(30, price, spec))[0])
    assert fraction == pytest.approx(0.30 / 150.0)
