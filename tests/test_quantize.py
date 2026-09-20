"""Section D — quantization per mode, drift, and sub-minimum rejection."""

import numpy as np
import pytest

from vectorbtx.instruments.spec import Instrument
from vectorbtx.quantize.quantizer import drift_report, quantize, to_order_params

SPREADBET = Instrument(mode="spreadbet", point_size=0.01).filled()
CFD = Instrument(mode="cfd", point_size=0.01, contract_size=1.0).filled()
EQUITY = Instrument(mode="equity", point_size=0.01).filled()


def test_spread_bet_increment_is_a_penny_per_point():
    size, drift = quantize(2.3456, SPREADBET)
    assert size == pytest.approx(2.34)
    assert drift == pytest.approx(0.0056)


def test_cfd_increment_is_a_hundredth_of_a_lot():
    size, _ = quantize(0.12345, CFD)
    assert size == pytest.approx(0.12)


def test_equity_increment_is_a_whole_share():
    size, drift = quantize(137.4829, EQUITY)
    assert size == 137.0
    assert drift == pytest.approx(0.4829)


def test_floor_ceil_and_nearest():
    assert quantize(2.345, SPREADBET, "floor")[0] == pytest.approx(2.34)
    assert quantize(2.341, SPREADBET, "ceil")[0] == pytest.approx(2.35)
    assert quantize(2.346, SPREADBET, "nearest")[0] == pytest.approx(2.35)


def test_sub_minimum_spread_bet_rejects_rather_than_rounding_to_zero():
    """A £0.50/point stake is below the £1 minimum: rejected, not enlarged."""
    size, drift = quantize(0.5, SPREADBET)
    assert size == 0.0
    assert drift == pytest.approx(0.5)


def test_sub_minimum_cfd_rejects():
    size, _ = quantize(0.005, CFD)
    assert size == 0.0


def test_sign_is_preserved_and_floor_shrinks_magnitude():
    """Floor always means a smaller position, in both directions."""
    assert quantize(-2.3456, SPREADBET)[0] == pytest.approx(-2.34)
    assert abs(quantize(-2.3456, SPREADBET)[0]) < 2.3456


def test_order_params_are_in_engine_units():
    params = to_order_params(SPREADBET)
    assert params["size_granularity"] == pytest.approx(0.01 / 0.01)
    assert params["min_size"] == pytest.approx(1.0 / 0.01)


def test_non_floor_rounding_disables_engine_quantization():
    params = to_order_params(SPREADBET, direction="nearest")
    assert np.isnan(params["size_granularity"])


def test_rounding_drift_and_max_size_drift_are_distinguishable():
    """max_size partially fills, so its drift has a different cause and a separate record."""
    intended = np.array([[10.7]])
    quantized, _ = quantize(intended, EQUITY)
    capped = np.minimum(quantized, 5.0)
    rounding = drift_report(intended, quantized)
    capping = drift_report(quantized, capped)
    assert rounding.iloc[0, 0] == pytest.approx(0.7)
    assert capping.iloc[0, 0] == pytest.approx(5.0)


def test_quantize_preserves_pandas_types():
    import pandas as pd

    series = pd.Series([2.3456, 5.678], index=["a", "b"])
    out, drift = quantize(series, SPREADBET)
    assert isinstance(out, pd.Series)
    assert list(out.index) == ["a", "b"]
