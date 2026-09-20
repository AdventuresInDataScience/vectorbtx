"""Section I — the D4 identity, OHLC ordering, adverse markup and table validation."""

import numpy as np
import pandas as pd
import pytest

from vectorbtx.fx.convert import check_ohlc_ordering, convert_amount, to_base_currency
from vectorbtx.fx.table import FXTable

IDX = pd.date_range("2024-01-01", periods=3, freq="D")
CANONICAL = pd.DataFrame(
    {"GBP": [1.27, 1.25, 1.24], "EUR": [1.09, 1.10, 1.08], "USD": [1.0, 1.0, 1.0]}, index=IDX
)
TABLE = FXTable.from_canonical(CANONICAL, hub="USD")


def test_reciprocals_are_exact_by_construction():
    product = TABLE.get("USD", "GBP") * TABLE.get("GBP", "USD")
    assert np.allclose(product.to_numpy(), 1.0)


def test_triangulation_holds():
    direct = TABLE.get("EUR", "GBP")
    via = TABLE.get("EUR", "USD") * TABLE.get("USD", "GBP")
    assert np.allclose(direct.to_numpy(), via.to_numpy())


def test_conversion_identity_with_opposite_moves():
    """D7: price up in USD while GBP strengthens — opposite directions hide sign errors."""
    price = pd.Series([300.0, 330.0, 360.0], index=IDX, name="MSFT")
    converted = to_base_currency(price, "GBP", TABLE, quote_currency="USD", validate=False)
    rate = TABLE.get("USD", "GBP")
    shares = 10
    manual = shares * (price.iloc[2] * rate.iloc[2] - price.iloc[0] * rate.iloc[0])
    engine = shares * (converted.iloc[2] - converted.iloc[0])
    assert engine == pytest.approx(manual, rel=1e-12)


def test_fx_markup_is_adverse_in_both_directions():
    out = convert_amount(100.0, "USD", "GBP", TABLE, markup=0.005)
    back = convert_amount(out, "GBP", "USD", TABLE, markup=0.005)
    assert back < 100.0
    assert 100.0 - back == pytest.approx(100.0 * (1 - 0.995**2), rel=1e-9)


def test_markup_is_carried_for_slippage_not_folded_into_the_series():
    """A uniform scaling would cancel out of every return, so it must be a cost."""
    price = pd.DataFrame({"MSFT": [300.0, 330.0]}, index=IDX[:2])
    converted = to_base_currency(price, "GBP", TABLE, quote_currency="USD", fx_markup=0.005,
                                 validate=False)
    assert converted.fx_markup == 0.005


def test_ohlc_ordering_survives_conversion():
    frame = pd.DataFrame(
        {"open": [100.0, 101], "high": [102.0, 103], "low": [99.0, 100], "close": [101.0, 102]},
        index=IDX[:2],
    )
    converted = to_base_currency(frame, "GBP", TABLE, quote_currency="USD")
    check_ohlc_ordering(converted)


def test_mixed_rates_across_ohlc_are_caught():
    """One rate per bar. Different rates per field break H >= C >= L."""
    frame = pd.DataFrame(
        {"open": [100.0], "high": [102.0], "low": [99.0], "close": [130.0]}, index=IDX[:1]
    )
    with pytest.raises(ValueError, match="OHLC ordering broken"):
        check_ohlc_ordering(frame)


def test_corrupted_table_is_caught_by_validation():
    corrupt = TABLE.rates.copy()
    corrupt[("USD", "GBP")] = corrupt[("USD", "GBP")] * 1.05
    with pytest.warns(UserWarning, match="Reciprocal check failed"):
        FXTable(corrupt, hub="USD")


def test_calendar_mismatch_warns_rather_than_silently_filling():
    target = pd.date_range("2024-01-01", periods=5, freq="D")
    with pytest.warns(UserWarning, match="forward-filled"):
        TABLE.align(TABLE.get("USD", "GBP"), target)


def test_strict_fill_policy_raises():
    target = pd.date_range("2024-01-01", periods=5, freq="D")
    with pytest.raises(Exception, match="no FX rate"):
        TABLE.align(TABLE.get("USD", "GBP"), target, fill_policy="strict")


def test_original_series_is_retrievable_for_decomposition():
    price = pd.DataFrame({"MSFT": [300.0, 330.0, 360.0]}, index=IDX)
    converted = to_base_currency(price, "GBP", TABLE, quote_currency="USD", validate=False)
    assert converted.native["MSFT"].iloc[0] == 300.0
    parts = converted.decompose()
    assert set(parts) == {"total", "asset", "fx"}


def test_explicit_conversion_is_required():
    price = pd.Series([1.0], index=IDX[:1])
    with pytest.raises(ValueError, match="quote_currency \\(or instruments\\) is required"):
        to_base_currency(price, "GBP", TABLE)


def test_cross_pair_is_just_another_lookup():
    assert TABLE.get("EUR", "GBP").iloc[0] == pytest.approx(1.09 / 1.27)
