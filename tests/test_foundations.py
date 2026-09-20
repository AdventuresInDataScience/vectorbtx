"""Section B — the collision guard and the drop-in guarantee.

The plan requires both to be proven by tests that deliberately exercise them, not
asserted. The collision guard is testable without vectorbt; drop-in equivalence is not.
"""

import sys
import types

import pytest

from vectorbtx.core.patch import ShadowError, attach, merge_metrics

try:
    import vectorbt  # noqa: F401

    HAS_VECTORBT = True
except ImportError:
    HAS_VECTORBT = False

needs_vectorbt = pytest.mark.skipif(not HAS_VECTORBT, reason="requires vectorbt")


# --- the collision guard (plan §B: "proven by a test that deliberately triggers it") ---


class _Host:
    existing = "upstream value"
    _metrics = {"already_here": dict(title="Upstream")}


def test_attach_raises_rather_than_shadowing_an_upstream_attribute():
    """This is the whole defence against silently overwriting a future upstream method."""
    with pytest.raises(ShadowError, match="already exists upstream"):
        attach(_Host, "existing", lambda self: None)
    assert _Host.existing == "upstream value"


def test_attach_succeeds_on_a_free_name():
    attach(_Host, "brand_new", lambda self: 42)
    assert _Host.brand_new(None) == 42
    del _Host.brand_new


def test_attach_overwrite_is_opt_in_and_explicit():
    attach(_Host, "existing", lambda self: "ours", overwrite=True)
    assert _Host.existing(None) == "ours"
    _Host.existing = "upstream value"


def test_merge_metrics_raises_on_a_key_collision():
    with pytest.raises(ShadowError, match="already exists upstream"):
        merge_metrics(_Host, {"already_here": dict(title="Ours")})
    assert _Host._metrics["already_here"]["title"] == "Upstream"


def test_merge_metrics_adds_new_keys():
    merge_metrics(_Host, {"fresh": dict(title="Fresh")})
    assert _Host._metrics["fresh"]["title"] == "Fresh"
    del _Host._metrics["fresh"]


@needs_vectorbt
def test_patching_is_idempotent():
    """apply_patches runs on every import; a second call must not raise."""
    from vectorbtx.metrics.returns.register import register_returns_metrics
    from vectorbtx.metrics.trades.register import register_trade_metrics

    register_returns_metrics()
    register_returns_metrics()
    register_trade_metrics()
    register_trade_metrics()


# --- drop-in equivalence (plan §B / §1.3, and §L "drop-in equivalence test") ---


@needs_vectorbt
def test_unmodified_vectorbt_script_is_bit_identical_under_vectorbtx():
    """The contract: `import vectorbtX as vbt` changes nothing for plain vectorbt code."""
    import numpy as np
    import pandas as pd
    import vectorbt as plain
    import vectorbtX as x

    index = pd.date_range("2020-01-01", periods=200, freq="D")
    rng = np.random.default_rng(0)
    price = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 200))), index=index)

    fast = price.rolling(10).mean()
    slow = price.rolling(50).mean()
    entries = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    exits = (fast < slow) & (fast.shift(1) >= slow.shift(1))

    kwargs = dict(init_cash=100, fees=0.001, freq="D")
    a = plain.Portfolio.from_signals(price, entries, exits, **kwargs)
    b = x.Portfolio.from_signals(price, entries, exits, **kwargs)

    np.testing.assert_array_equal(a.order_records, b.order_records)
    assert a.total_profit() == b.total_profit()
    np.testing.assert_array_equal(a.value().to_numpy(), b.value().to_numpy())


@needs_vectorbt
def test_native_stats_still_contain_every_upstream_metric():
    """Additive means additive: nothing upstream may disappear from .stats()."""
    import pandas as pd
    import vectorbt as plain
    import vectorbtX as x

    index = pd.date_range("2020-01-01", periods=100, freq="D")
    price = pd.Series(range(100, 200), index=index, dtype=float)

    a = plain.Portfolio.from_holding(price, init_cash=100, freq="D").stats()
    b = x.Portfolio.from_holding(price, init_cash=100, freq="D").stats()

    missing = [k for k in a.index if k not in b.index]
    assert not missing, f"vectorbtX dropped upstream metrics: {missing}"
