"""Section G — regression tests for the four simulation-path defects.

These are the tests that would have caught the bugs the v0.1 build shipped with. Every
one of them needs vectorbt importable; the pure-arithmetic tests live in
`test_holding.py` and deliberately do not.

Set `NUMBA_DISABLE_JIT=1` to run these without a numba toolchain — vectorbt's own
`is_numba_func` check honours it for any function whose name ends `_nb`, which every
kernel here does.
"""

import numpy as np
import pandas as pd
import pytest

try:  # the runner executes conftest under its own name; fall back to a local check
    import vectorbt  # noqa: F401

    HAS_VECTORBT = True
except ImportError:  # pragma: no cover
    HAS_VECTORBT = False

needs_vectorbt = pytest.mark.skipif(not HAS_VECTORBT, reason="requires vectorbt")

IDX = pd.date_range("2024-01-01", periods=12, freq="D")
FLAT = pd.DataFrame({"A": np.full(12, 100.0)}, index=IDX)


def _universe():
    import vectorbtx as vbt

    return vbt.load_universe(
        {"A": vbt.Instrument(mode="equity", quote_currency="GBP", leverage=2.0)}
    )


def _size(amount=150.0, columns=("A",)):
    size = pd.DataFrame(np.nan, index=IDX, columns=list(columns))
    size.iloc[0] = [amount] * len(columns)
    return size


def _final_value(pf):
    return float(np.ravel(np.asarray(pf.value()))[-1])


# ############# Bug 1 — Config type destroyed by merge_with ############# #


@needs_vectorbt
def test_replace_survives_the_carried_state():
    """`Config.merge_with` returns a plain dict; assigning it to `_config` breaks
    `.replace()`, which every constructor in this module ends with."""
    import vectorbtx as vbt
    from vectorbt.utils.config import Config

    pf = vbt.Portfolio.from_orders(FLAT, size=_size(1.0), init_cash=10_000, freq="D")
    assert isinstance(pf.config, Config)
    assert pf.replace(drift="sentinel").drift == "sentinel"


@needs_vectorbt
def test_indexing_preserves_config_type():
    """`indexing_func` routes through `.replace()`, so it fails the same way."""
    import vectorbtx as vbt
    from vectorbt.utils.config import Config

    close = pd.DataFrame({"A": np.full(12, 100.0), "B": np.full(12, 50.0)}, index=IDX)
    pf = vbt.Portfolio.from_orders(close, size=_size(1.0, ("A", "B")),
                                   init_cash=10_000, freq="D")
    assert isinstance(pf["A"].config, Config)


# ############# Bug 3 — last_cash is group-indexed under cash sharing ############# #


def test_post_segment_writes_to_the_group_under_cash_sharing():
    """`last_cash = init_cash.copy()` has one entry per GROUP when cash_sharing=True.

    Plan §5.3 sketches `c.last_cash[col] -= ...`, which runs off the end of that array.
    Pure enough to test without vectorbt.
    """
    from collections import namedtuple

    from vectorbtx.holding.callback import post_segment_func_nb

    Ctx = namedtuple("Ctx", "cash_sharing last_cash group from_col to_col i")
    charges = np.array([[-1.0, -2.0]])

    shared = np.array([1000.0])
    post_segment_func_nb(Ctx(True, shared, 0, 0, 2, 0), charges)
    assert shared[0] == pytest.approx(997.0)

    per_col = np.array([1000.0, 1000.0])
    post_segment_func_nb(Ctx(False, per_col, 0, 0, 2, 0), charges)
    assert list(per_col) == pytest.approx([999.0, 998.0])


# ############# Bug 2 — the replay path ############# #


@needs_vectorbt
@pytest.mark.parametrize("cash_sharing", [False, True])
def test_holding_cost_path_runs_and_places_the_same_orders(cash_sharing):
    """There is no `nb.order_func_nb` upstream, and pass 2 previously dropped `size`
    and every resolved cost, so it replayed an empty portfolio."""
    import vectorbtx as vbt

    close = pd.DataFrame({"A": np.full(12, 100.0), "B": np.full(12, 50.0)}, index=IDX)
    universe = vbt.load_universe({
        "A": vbt.Instrument(mode="equity", quote_currency="GBP", leverage=2.0),
        "B": vbt.Instrument(mode="equity", quote_currency="GBP", leverage=2.0),
    })
    rates = vbt.x.static_rate_table(["GBP"], IDX, rate=0.05)

    pf = vbt.Portfolio.from_holding_costs(
        close, _size(120.0, ("A", "B")), universe, rates=rates,
        init_cash=10_000, cash_sharing=cash_sharing,
        group_by=True if cash_sharing else None, freq="D",
    )
    assert len(pf.orders.records) == 2          # the replay really placed the orders
    assert pf.holding_costs.to_numpy().sum() < 0


@needs_vectorbt
def test_unreplayable_order_field_raises():
    """A field pass 2 cannot materialise must fail loudly, not be dropped."""
    from vectorbtx.holding.callback import _order_arrays

    with pytest.raises(ValueError, match="Cannot replay"):
        _order_arrays({"reject_prob": 0.5}, (4, 1), ["A"])


# ############# Bug 4 — charges must reach the reported equity curve ############# #


@needs_vectorbt
def test_charges_appear_in_reported_equity():
    """`cash_flow` is rebuilt from ORDER RECORDS, so a `last_cash` mutation is invisible
    to `value()`, `returns()` and every metric unless it is folded back in."""
    import vectorbtx as vbt

    universe = _universe()

    def run(rate):
        rates = vbt.x.static_rate_table(["GBP"], IDX, rate=rate)
        pf = vbt.Portfolio.from_holding_costs(
            FLAT, _size(), universe, rates=rates, init_cash=10_000, freq="D"
        )
        return _final_value(pf), float(pf.holding_costs.to_numpy().sum())

    v0, c0 = run(0.0)
    v5, c5 = run(0.05)
    assert (v5 - v0) == pytest.approx(c5 - c0, abs=1e-9)
    assert v5 < v0                                   # a real cost, not a wash


@needs_vectorbt
def test_charges_reach_the_totals_too():
    """`total_profit` is also rebuilt from order records, so `final_value`, `total_return`
    and stats' End Value need the charges added, as `cash_flow` does (Studio issue 15)."""
    import vectorbtx as vbt

    rates = vbt.x.static_rate_table(["GBP"], IDX, rate=0.05)
    pf = vbt.Portfolio.from_holding_costs(
        FLAT, _size(), _universe(), rates=rates, init_cash=10_000, freq="D"
    )
    last = _final_value(pf)
    assert float(pf.holding_costs.to_numpy().sum()) < 0
    assert float(np.ravel(pf.final_value())[0]) == pytest.approx(last)
    assert float(np.ravel(pf.total_return())[0]) == pytest.approx(last / 10_000 - 1)
    assert float(pf.stats()["End Value"]) == pytest.approx(last)


@needs_vectorbt
def test_no_holding_costs_leaves_cash_flow_untouched():
    """The override must be inert for every ordinary portfolio — §1.3 drop-in guarantee."""
    import vectorbtx as vbt

    pf = vbt.Portfolio.from_orders(FLAT, size=_size(1.0), init_cash=10_000, freq="D")
    assert pf.holding_costs is None
    plain = np.asarray(pf.cash_flow(), dtype=float)
    assert np.isfinite(plain).all()


@needs_vectorbt
def test_short_is_credited_not_charged():
    """The adverse markup must survive the round trip through the simulation, per bar.

    On a CFD, which is financed on its full notional. Margin share dealing (`equity`)
    finances only the account's overdraft, and a short sale adds cash rather than
    borrowing it, so a share short is financed on nothing; see the test below. (This test
    used `equity` once, and passed only because a short's value was counted as borrowed.)
    """
    import vectorbtx as vbt

    universe = vbt.load_universe(
        {"A": vbt.Instrument(mode="cfd", quote_currency="GBP", contract_size=1.0, leverage=5.0)}
    )
    rates = vbt.x.static_rate_table(["GBP"], IDX, rate=0.05)

    def charges(amount):
        pf = vbt.Portfolio.from_holding_costs(
            FLAT, _size(amount), universe, rates=rates, init_cash=10_000, freq="D"
        )
        assert len(pf.orders.records) == 1
        return float(pf.holding_costs.to_numpy().sum())

    assert charges(150.0) < 0        # long pays benchmark + markup
    assert charges(-150.0) > 0       # short receives benchmark - markup


@needs_vectorbt
def test_a_share_short_borrows_no_cash_so_pays_no_financing():
    import vectorbtx as vbt

    rates = vbt.x.static_rate_table(["GBP"], IDX, rate=0.05)
    pf = vbt.Portfolio.from_holding_costs(
        FLAT, _size(-150.0), _universe(), rates=rates, init_cash=10_000, freq="D"
    )
    assert len(pf.orders.records) == 1
    assert float(pf.x.borrowed_portion().to_numpy().max()) == 0.0
    assert float(pf.holding_costs.to_numpy().sum()) == 0.0
