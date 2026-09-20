"""Section C — one valid spec per mode, and a test for every validation rule."""

import pytest

from vectorbtx.instruments.defaults import default_cost_stack, max_leverage, mode_defaults
from vectorbtx.instruments.spec import Instrument
from vectorbtx.instruments.validate import SpecError, validate_universe

# --- D7 fixtures: one fully-populated spec per mode, with Part 8 defaults ---

UK_SPREADBET = Instrument(mode="spreadbet", point_size=0.01, quote_currency="GBP")
INDEX_SPREADBET = Instrument(mode="spreadbet", point_size=1.0, asset_class="index")
EURUSD_CFD = Instrument(
    mode="cfd", asset_class="fx", point_size=0.0001, contract_size=100_000.0,
    quote_currency="USD",
)
EURUSD_SPOT = Instrument(
    mode="fx_spot", point_size=0.0001, contract_size=100_000.0,
    quote_currency="USD", base_currency="EUR",
)
MSFT = Instrument(mode="equity", point_size=0.01, quote_currency="USD", cost_currency="USD")

ALL_MODES = [UK_SPREADBET, EURUSD_CFD, EURUSD_SPOT, MSFT]


def test_every_mode_has_a_valid_default_spec():
    for spec in ALL_MODES:
        filled = spec.filled()
        assert filled.increment > 0
        assert filled.min_size > 0
        assert filled.leverage > 0
        assert filled.active_costs


def test_part8_sizing_defaults():
    assert mode_defaults("spreadbet")["min_size"] == 1.0
    assert mode_defaults("spreadbet")["increment"] == 0.01
    assert mode_defaults("cfd")["min_size"] == 0.01
    assert mode_defaults("fx_spot")["contract_size"] == 100_000.0
    assert mode_defaults("equity")["allow_fractional"] is False


def test_part8_leverage_defaults():
    assert max_leverage("equity") == 2.0
    assert max_leverage("fx_spot") == 2.0
    assert max_leverage("spreadbet", "stocks") == 5.0
    assert max_leverage("spreadbet", "commodities") == 10.0
    assert max_leverage("cfd", "fx") == 30.0


def test_commodities_behave_exactly_as_equities_for_sizing():
    stock = Instrument(mode="spreadbet", point_size=0.01, asset_class="stocks").filled()
    commodity = Instrument(mode="spreadbet", point_size=0.01, asset_class="commodities").filled()
    assert stock.min_size == commodity.min_size == 1.0
    assert stock.increment == commodity.increment == 0.01


def test_spread_bet_rejects_contract_size():
    with pytest.raises(SpecError, match="no contract_size"):
        Instrument(mode="spreadbet", point_size=0.01, contract_size=100.0)


def test_rejects_non_positive_increment():
    with pytest.raises(SpecError, match="must be > 0"):
        Instrument(mode="equity", increment=0.0)
    with pytest.raises(SpecError, match="must be > 0"):
        Instrument(mode="equity", increment=-1.0)


def test_rejects_min_size_not_a_multiple_of_increment():
    with pytest.raises(SpecError, match="whole multiple"):
        Instrument(mode="equity", increment=0.5, min_size=0.7)


def test_rejects_unresolvable_currency_combination():
    with pytest.raises(SpecError, match="base_currency is meaningful only"):
        Instrument(mode="equity", quote_currency="USD", base_currency="EUR")
    with pytest.raises(SpecError, match="must differ"):
        Instrument(mode="fx_spot", contract_size=100_000.0,
                   quote_currency="USD", base_currency="USD")


def test_rejects_adjusted_close_on_nominal_point_modes():
    with pytest.raises(SpecError, match="nominal points"):
        Instrument(mode="spreadbet", point_size=0.01, price_field="adj_close")
    with pytest.raises(SpecError, match="nominal points"):
        Instrument(mode="cfd", contract_size=1.0, price_field="adj_close")


def test_equity_defaults_to_adjusted_close():
    assert MSFT.filled().price_field == "adj_close"
    assert UK_SPREADBET.filled().price_field == "close"


def test_rejects_unknown_mode_and_asset_class():
    with pytest.raises(SpecError, match="Unknown mode"):
        Instrument(mode="futures")
    with pytest.raises(SpecError, match="Unknown asset_class"):
        Instrument(mode="equity", asset_class="crypto")


def test_rejects_bad_keyword_in_a_spec_field():
    from vectorbtx.resolve.keywords import KeywordError

    with pytest.raises(KeywordError, match="not a recognised keyword"):
        Instrument(mode="equity", slippage="atuo")


def test_portfolio_level_rejects_per_ticker_leverage():
    specs = {
        "A": Instrument(mode="cfd", contract_size=1.0, leverage=5.0,
                        leverage_type="portfolio_level"),
        "B": Instrument(mode="cfd", contract_size=1.0, leverage=3.0,
                        leverage_type="portfolio_level"),
    }
    with pytest.raises(SpecError, match="portfolio-wide budget"):
        validate_universe({k: v.filled() for k, v in specs.items()})


def test_default_cost_stacks_match_the_plan():
    assert "commission" not in default_cost_stack("spreadbet", "stocks")
    assert "carry" in default_cost_stack("spreadbet", "fx")
    assert "carry" in default_cost_stack("cfd", "fx")
    assert "carry" not in default_cost_stack("cfd", "stocks")
    assert "carry" in default_cost_stack("fx_spot")
    assert "financing" in default_cost_stack("equity")


def test_cost_currency_defaults_to_quote_currency():
    spec = Instrument(mode="equity", quote_currency="USD").filled()
    assert spec.cost_currency == "USD"


def test_allow_fractional_changes_the_fill_increment():
    """Otherwise the flag is decorative: the spec says fractional, the fill says 1 share."""
    whole = Instrument(mode="equity", point_size=0.01).filled()
    fractional = Instrument(mode="equity", point_size=0.01, allow_fractional=True).filled()

    assert whole.increment == 1.0
    assert whole.min_size == 1.0
    assert fractional.increment < 1.0
    assert fractional.min_size == fractional.increment

    from vectorbtx.quantize.quantizer import quantize

    assert quantize(10.4, whole)[0] == 10.0
    assert quantize(10.4, fractional)[0] == pytest.approx(10.4)


def test_explicit_increment_still_wins_over_allow_fractional():
    spec = Instrument(mode="equity", point_size=0.01,
                      allow_fractional=True, increment=0.01, min_size=0.01).filled()
    assert spec.increment == 0.01
