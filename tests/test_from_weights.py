"""Section J — `Portfolio.from_weights`, against hand-computed numbers.

These cover the two Section J "Done when" criteria that were never implemented as checks:

* "`from_weights` reproduces a hand-computed two-asset monthly rebalance"
* "A cross-currency portfolio rebalances to correct base-currency weights"

Both were stubbed out as `CANNOT VERIFY` while vectorbt was unavailable and stayed that
way. Until this file existed, `from_weights` had **no direct test at all** — Section J was
exercised only indirectly, through `vbt.x.PortfolioBacktest` in `test_frontend.py`.

Every expected number below is derived by hand in the docstring that precedes it. None was
read off the implementation, which is the only way a test like this is worth having.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import needs_vectorbt

IDX = pd.date_range("2024-01-01", periods=60, freq="D")
FEB = pd.Timestamp("2024-02-01")


def _bar(pf, frame, ts):
    """One bar of a per-column frame, as a plain dict."""
    return pd.DataFrame(frame).loc[ts].to_dict()


# ############# Criterion: hand-computed two-asset monthly rebalance ############# #


@needs_vectorbt
def test_from_weights_reproduces_a_hand_computed_two_asset_monthly_rebalance():
    """50/50 across A (flat) and B (doubles), rebalanced on the first of each month.

    Hand-computation, with £10,000 and no costs:

    * **1 Jan, rebalance 1.** Value £10,000, target 50% each → £5,000 each.
      A at £100 → **50 shares**. B at £100 → **50 shares**.
    * **B doubles to £200 on 1 Feb.** Pre-trade: A 50 x £100 = £5,000;
      B 50 x £200 = £10,000. Value **£15,000**, weights 1/3 and 2/3 — drifted.
    * **1 Feb, rebalance 2.** Target 50% of £15,000 = £7,500 each.
      A: £7,500 / £100 = **75 shares** (buy 25). B: £7,500 / £200 = **37.5 shares**
      (sell 12.5). Value unchanged at £15,000, weights back to 50/50.

    The sell leg is deliberately fractional: it catches a rebalance that silently rounds.
    """
    import vectorbtx as vbt
    from vectorbtx.construct import calendar_schedule

    close = pd.DataFrame({
        "A": pd.Series(100.0, index=IDX),
        "B": pd.Series(np.where(IDX >= FEB, 200.0, 100.0), index=IDX),
    })
    weights = pd.DataFrame(0.5, index=IDX, columns=["A", "B"])
    schedule = calendar_schedule(IDX, "M", "first")

    # Exactly two rebalances in a 60-day window starting 1 Jan — otherwise the arithmetic
    # below is not the whole story.
    assert list(IDX[schedule.to_numpy()]) == [IDX[0], FEB]

    pf = vbt.Portfolio.from_weights(
        close, weights, schedule=schedule, init_cash=10_000, freq="D"
    )
    assets = pd.DataFrame(pf.assets())

    assert _bar(pf, assets, IDX[0]) == pytest.approx({"A": 50.0, "B": 50.0})
    assert _bar(pf, assets, FEB) == pytest.approx({"A": 75.0, "B": 37.5})

    value = float(np.ravel(np.asarray(pf.value()))[list(IDX).index(FEB)])
    assert value == pytest.approx(15_000.0)

    # ...and the point of rebalancing: both legs are worth the same again.
    held = _bar(pf, assets, FEB)
    assert held["A"] * 100.0 == pytest.approx(held["B"] * 200.0) == pytest.approx(7_500.0)


@needs_vectorbt
def test_an_unrebalanced_portfolio_is_left_to_drift():
    """The control. Without the 1 Feb rebalance the share counts must not move.

    Same data, but the schedule fires only on 1 Jan. A stays at 50 shares and B at 50, so
    the value is still £15,000 — identical total, different composition. If this test and
    the one above ever agree on `assets`, the schedule is being ignored.
    """
    import vectorbtx as vbt

    close = pd.DataFrame({
        "A": pd.Series(100.0, index=IDX),
        "B": pd.Series(np.where(IDX >= FEB, 200.0, 100.0), index=IDX),
    })
    weights = pd.DataFrame(0.5, index=IDX, columns=["A", "B"])
    schedule = pd.Series(IDX == IDX[0], index=IDX)

    pf = vbt.Portfolio.from_weights(
        close, weights, schedule=schedule, init_cash=10_000, freq="D"
    )
    assets = pd.DataFrame(pf.assets())

    assert _bar(pf, assets, FEB) == pytest.approx({"A": 50.0, "B": 50.0})
    value = float(np.ravel(np.asarray(pf.value()))[list(IDX).index(FEB)])
    assert value == pytest.approx(15_000.0)


# ############# Criterion: cross-currency rebalance in base currency ############# #


@needs_vectorbt
def test_a_cross_currency_portfolio_rebalances_to_correct_base_currency_weights():
    """A USD asset and a GBP asset, rebalanced 50/50 in GBP, with only the FX rate moving.

    This is the D4 design under test: prices are converted to base currency *before*
    simulation, so an FX move must reach the target weights exactly as a price move would.

    Hand-computation, base GBP, £16,000, no costs and no FX markup:

    * Rate is quoted USD-per-GBP. In January GBP = 1.25, so **$100 → £80**.
      In February GBP = 1.00, so the same **$100 → £100** — GBP has weakened and the
      US holding is worth more in base terms.
    * **1 Jan.** Both assets are £80. 50% of £16,000 = £8,000 each →
      **100 shares of each**.
    * **1 Feb, pre-trade.** US: 100 x £100 = £10,000. UK: 100 x £80 = £8,000.
      Value **£18,000**, weights 5/9 and 4/9 — drifted purely on FX.
    * **1 Feb, rebalance.** £9,000 each. US: £9,000 / £100 = **90 shares** (sell 10).
      UK: £9,000 / £80 = **112.5 shares** (buy 12.5).

    The UK asset's own price never moves, so every trade in it is caused by the currency.
    """
    import vectorbtx as vbt
    from vectorbtx.construct import calendar_schedule
    from vectorbtx.fx.convert import to_base_currency
    from vectorbtx.fx.table import FXTable

    canonical = pd.DataFrame(
        {"GBP": np.where(IDX >= FEB, 1.00, 1.25), "USD": 1.0}, index=IDX
    )
    table = FXTable.from_canonical(canonical, hub="USD")

    us_in_usd = pd.Series(100.0, index=IDX, name="US")
    us_in_gbp = to_base_currency(
        us_in_usd, "GBP", table, quote_currency="USD", fx_markup=0.0, validate=False
    )
    # Cross-check the conversion against the hand-computation before relying on it.
    assert float(us_in_gbp.iloc[0]) == pytest.approx(80.0)
    assert float(us_in_gbp.loc[FEB]) == pytest.approx(100.0)

    close = pd.DataFrame({"US": us_in_gbp, "UK": pd.Series(80.0, index=IDX)})
    weights = pd.DataFrame(0.5, index=IDX, columns=["US", "UK"])
    schedule = calendar_schedule(IDX, "M", "first")

    pf = vbt.Portfolio.from_weights(
        close, weights, schedule=schedule, init_cash=16_000, freq="D"
    )
    assets = pd.DataFrame(pf.assets())

    assert _bar(pf, assets, IDX[0]) == pytest.approx({"US": 100.0, "UK": 100.0})
    assert _bar(pf, assets, FEB) == pytest.approx({"US": 90.0, "UK": 112.5})

    value = float(np.ravel(np.asarray(pf.value()))[list(IDX).index(FEB)])
    assert value == pytest.approx(18_000.0)

    # The weights are correct *in base currency* — the whole point of the criterion.
    held = _bar(pf, assets, FEB)
    assert held["US"] * 100.0 == pytest.approx(9_000.0)
    assert held["UK"] * 80.0 == pytest.approx(9_000.0)
