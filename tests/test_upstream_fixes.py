"""Regression tests for the issues OpenStrategy Studio's phase 0 found in the front door.

One section per issue, numbered as in the Studio's `docs/UPSTREAM.md`. Most of them were
**wrong numbers with no error**, so each test pins the number, not just the absence of a
crash.
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

pytestmark = pytest.mark.skipif(not HAS_VECTORBT, reason="requires vectorbt")

EQUITY = {"mode": "equity", "point_size": 1.0, "quote_currency": "USD", "spread_points": 0.0,
          "leverage": 1.0}
FX_SPREADBET = {"mode": "spreadbet", "asset_class": "fx", "point_size": 0.0001,
                "base_currency": "EUR", "quote_currency": "USD", "spread_points": 0.0,
                "leverage": 30}


def frames(opens, highs=None, lows=None, closes=None, columns=("X",)):
    """OHLC frames with the same values in every column."""
    index = pd.bdate_range("2024-01-01", periods=len(opens))
    highs = highs if highs is not None else [v + 0.5 for v in opens]
    lows = lows if lows is not None else [v - 0.5 for v in opens]
    closes = closes if closes is not None else opens

    def make(values):
        return pd.DataFrame({c: np.asarray(values, dtype=float) for c in columns}, index=index)

    return make(opens), make(highs), make(lows), make(closes)


def mask(index, columns, at=()):
    """A boolean frame, True at each (column, bar) pair."""
    out = pd.DataFrame(False, index=index, columns=list(columns))
    for column, bar in at:
        out.iloc[bar, out.columns.get_loc(column)] = True
    return out


def run(o, h, l, c, **kwargs):
    import vectorbtx as vbt

    return vbt.x.PortfolioBacktest(close=c, open=o, high=h, low=l, freq="D", **kwargs)


# ############# 1. Short signals shift to the next bar ############# #


def test_short_signals_fill_at_the_next_open_like_longs():
    """A short signal at bar 1's close used to fill at bar 1's open: lookahead."""
    o, h, l, c = frames([100, 101, 102, 103, 104, 105])
    signal, none = mask(c.index, ["X"], [("X", 1)]), mask(c.index, ["X"])
    fills = {}
    for side, kw in (("long", dict(entries=signal, exits=none)),
                     ("short", dict(short_entries=signal, short_exits=none))):
        pf = run(o, h, l, c, instruments={"X": EQUITY}, size=1, init_cash=10_000, **kw)
        order = pf.orders.records_readable.iloc[0]
        fills[side] = (c.index.get_loc(order["Timestamp"]), float(order["Price"]))
    assert fills["long"] == fills["short"] == (2, 102.0)


def test_short_signals_alone_are_a_signal_strategy():
    """Passing only the short pair must not fall through to buy-and-hold."""
    o, h, l, c = frames([100] * 6)
    pf = run(o, h, l, c, short_entries=mask(c.index, ["X"], [("X", 0)]),
             short_exits=mask(c.index, ["X"], [("X", 3)]),
             instruments={"X": EQUITY}, size=1, init_cash=10_000)
    assert pf.orders.records_readable["Side"].tolist() == ["Sell", "Buy"]
    assert pf.orders.records["idx"].tolist() == [1, 4]


# ############# 2-4. FX carry ############# #


def _implied_carry(single, instrument, **extra):
    """Annual carry rate charged on a long EURUSD at EUR 3% / USD 4.5%."""
    import vectorbtx as vbt

    o, h, l, c = frames([1.10] * 12, [1.1005] * 12, [1.0995] * 12, columns=("EURUSD",))
    rates = vbt.x.static_rate_table(["EUR", "USD"], c.index, {"EUR": 0.03, "USD": 0.045})
    entries, exits = mask(c.index, ["EURUSD"], [("EURUSD", 0)]), mask(c.index, ["EURUSD"])
    common = dict(size=10_000, rates=rates, init_cash=10_000, freq="D", **extra)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if single:
            pf = vbt.x.SingleBacktest(close=c["EURUSD"], open=o["EURUSD"], high=h["EURUSD"],
                                      low=l["EURUSD"], entries=entries["EURUSD"],
                                      exits=exits["EURUSD"], **instrument, **common)
        else:
            pf = vbt.x.PortfolioBacktest(close=c, open=o, high=h, low=l, entries=entries,
                                         exits=exits, instruments={"EURUSD": instrument}, **common)
    nights = pd.DataFrame(pf.holding_costs).iloc[:, 0]
    nights = nights[nights != 0].abs()
    one_night = float(nights.min()) if len(nights) else 0.0     # a weekday: multiplier 1
    return one_night * 360 / (10_000 * 1.10), [str(w.message) for w in caught]


def test_flat_base_currency_on_an_fx_instrument_names_the_pair():
    """`SingleBacktest(base_currency="EUR")` set the ACCOUNT to EUR and charged the whole
    USD rate. It is the pair's base, so carry is the differential plus the markup."""
    single, messages = _implied_carry(True, FX_SPREADBET)
    portfolio, _ = _implied_carry(False, FX_SPREADBET)
    assert single == pytest.approx(portfolio) == pytest.approx(0.015 + 0.01)
    assert not any("no currency pair" in m for m in messages)


def test_account_currency_and_base_currency_cannot_disagree_off_fx():
    import vectorbtx as vbt
    from vectorbtx.frontend.prepare import BacktestError

    o, h, l, c = frames([100] * 4)
    with pytest.raises(BacktestError, match="account_currency"):
        vbt.x.PortfolioBacktest(close=c, open=o, high=h, low=l, instruments={"X": EQUITY},
                                account_currency="GBP", base_currency="USD", init_cash=1_000)


def test_missing_currency_leg_skips_carry_and_says_so():
    """A missing leg used to be priced at 0%, so a long paid the whole quote rate."""
    instrument = {k: v for k, v in FX_SPREADBET.items() if k != "base_currency"}
    implied, messages = _implied_carry(False, instrument)
    assert implied == 0.0
    assert any("Carry skipped" in m for m in messages)


def test_carry_markup_defaults_to_the_setting():
    import vectorbtx as vbt

    setting = vbt.settings["vectorbtx"]["carry_markup"]
    implied, _ = _implied_carry(False, FX_SPREADBET)
    assert setting > 0
    assert implied == pytest.approx(0.015 + setting)
    explicit, _ = _implied_carry(False, FX_SPREADBET, carry_markup=0.0)
    assert explicit == pytest.approx(0.015)


# ############# 5-6. Stops ############# #


def test_point_stop_is_anchored_at_the_fill_open_not_the_entry_close():
    """Filled at 100 on a bar closing at 103, a 2-point stop sat at 100.94 and fired."""
    o, h, l, c = frames([100, 100, 103, 103, 103, 103], [100.5, 103.5, 103.5, 103.5, 103.5, 103.5],
                        [99.5, 99.5, 100.5, 102.5, 102.5, 102.5], [100, 103, 103, 103, 103, 103])
    pf = run(o, h, l, c, entries=mask(c.index, ["X"], [("X", 0)]), exits=mask(c.index, ["X"]),
             instruments={"X": EQUITY}, size=1, stop_points=2, init_cash=10_000)
    assert pf.orders.records_readable["Side"].tolist() == ["Buy"]


@pytest.mark.parametrize("which", ["stop", "target"])
def test_stop_and_target_exits_pay_the_spread(which):
    """vectorbt's "stoplimit" exit applies no slippage, and the spread lives there."""
    instrument = dict(EQUITY, spread_points=1.0)          # half a point per side
    after = 95 if which == "stop" else 103                 # gaps through the level
    o, h, l, c = frames([100, 100, 100, after, after, after])
    stops = dict(stop_points=2) if which == "stop" else dict(target_points=2)
    pf = run(o, h, l, c, entries=mask(c.index, ["X"], [("X", 0)]), exits=mask(c.index, ["X"]),
             instruments={"X": instrument}, size=1, init_cash=10_000, **stops)
    assert float(pf.orders.records_readable.iloc[1]["Price"]) == pytest.approx(after - 0.5)


# ############# 7. Buy and hold ############# #


def test_unsized_hold_is_funded_at_equity_not_buying_power():
    """A 30x spread bet's hold carried 30x the account's equity."""
    o, h, l, c = frames([1.10] * 10, [1.1005] * 10, [1.0995] * 10, columns=("EURUSD",))
    pf = run(o, h, l, c, instruments={"EURUSD": FX_SPREADBET}, init_cash=1_000_000)
    exposure = float((pd.DataFrame(pf.assets()) * c).sum(axis=1).iloc[2])
    equity = float(pd.DataFrame(pf.value()).sum(axis=1).iloc[2])
    assert exposure / equity == pytest.approx(1.0, abs=1e-3)


def test_hold_that_cannot_reach_the_minimum_size_warns():
    """£10,000 at 1x cannot buy £1/pt of EURUSD; a silent flat line reads as no market."""
    o, h, l, c = frames([1.10] * 10, [1.1005] * 10, [1.0995] * 10, columns=("EURUSD",))
    with pytest.warns(UserWarning, match="placed no order"):
        run(o, h, l, c, instruments={"EURUSD": FX_SPREADBET}, init_cash=10_000)


# ############# 8. Weights above the cash ############# #


def test_weights_above_the_cash_scale_pro_rata():
    """0.75 / 0.75 used to fill 0.75 / 0.25: the first column in full, the last starved."""
    o, h, l, c = frames([100] * 8, columns=("A", "B"))
    schedule = pd.Series(False, index=c.index)
    schedule.iloc[0] = True
    with pytest.warns(UserWarning, match="scaled down pro rata"):
        pf = run(o, h, l, c, instruments={"A": EQUITY, "B": EQUITY},
                 weights=pd.DataFrame(0.75, index=c.index, columns=["A", "B"]),
                 schedule=schedule, init_cash=100_000)
    held = (pd.DataFrame(pf.assets()) * c).iloc[2] / float(pd.DataFrame(pf.value()).sum(axis=1).iloc[2])
    assert held.tolist() == pytest.approx([0.5, 0.5])


def test_fundable_long_short_book_is_left_alone():
    """Without `lock_cash` a short's proceeds fund the longs, so 140/40 fits at 1x."""
    o, h, l, c = frames([100] * 8, columns=("A", "B", "C"))
    schedule = pd.Series(False, index=c.index)
    schedule.iloc[0] = True
    weights = pd.DataFrame([[0.7, 0.7, -0.4]] * len(c), index=c.index, columns=["A", "B", "C"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pf = run(o, h, l, c, instruments={k: EQUITY for k in "ABC"}, weights=weights,
                 schedule=schedule, init_cash=100_000)
    assert not any("pro rata" in str(w.message) for w in caught)
    held = (pd.DataFrame(pf.assets()) * c).iloc[2] / float(pd.DataFrame(pf.value()).sum(axis=1).iloc[2])
    assert held.tolist() == pytest.approx([0.7, 0.7, -0.4])


# ############# 9. FX conversion markup ############# #


def test_fx_markup_is_charged_on_converted_columns_only():
    """The markup vanished in `pd.DataFrame(open)`. Now it is charged — but only where a
    conversion happened: a GBP holding in a GBP account crossed no exchange."""
    import vectorbtx as vbt

    o, h, l, c = frames([100] * 6, [101] * 6, [99] * 6, columns=("VOD.L", "MSFT"))
    fx = pd.Series(1.25, index=c.index, name="GBPUSD")
    book = {"VOD.L": {"mode": "equity", "point_size": 0.01, "quote_currency": "GBP", "spread_points": 0.0},
            "MSFT": {"mode": "equity", "point_size": 0.01, "quote_currency": "USD", "spread_points": 0.0}}
    universe = vbt.x.load_universe({k: vbt.x.Instrument(**v) for k, v in book.items()})
    conv = [vbt.x.to_base_currency(f, base="GBP", fx=fx, instruments=universe, fx_markup=0.05)
            for f in (o, h, l, c)]
    pf = vbt.x.PortfolioBacktest(close=conv[3], open=conv[0], high=conv[1], low=conv[2],
                                 entries=mask(c.index, book, [("VOD.L", 0), ("MSFT", 0)]),
                                 exits=mask(c.index, book), instruments=book, size=1,
                                 init_cash=10_000, freq="D", account_currency="GBP", fx=fx)
    prices = dict(pf.orders.records_readable[["Column", "Price"]].values.tolist())
    assert prices == pytest.approx({"VOD.L": 100.0, "MSFT": 80.0 * 1.05})


def test_fx_markup_is_a_real_argument():
    o, h, l, c = frames([100] * 6)
    pf = run(o, h, l, c, entries=mask(c.index, ["X"], [("X", 0)]), exits=mask(c.index, ["X"]),
             instruments={"X": EQUITY}, size=1, init_cash=10_000, fx_markup=0.05)
    assert float(pf.orders.records_readable.iloc[0]["Price"]) == pytest.approx(105.0)


def test_converted_series_keeps_its_markup_through_single_backtest():
    import vectorbtx as vbt

    o, h, l, c = frames([100] * 6)
    fx = pd.Series(1.25, index=c.index, name="GBPUSD")
    conv = vbt.x.to_base_currency(c["X"], base="GBP", fx=fx, quote_currency="USD", fx_markup=0.02)
    pf = vbt.x.SingleBacktest(close=conv, open=conv, high=conv * 1.01, low=conv * 0.99,
                              entries=mask(c.index, ["X"], [("X", 0)])["X"],
                              exits=mask(c.index, ["X"])["X"], mode="equity", point_size=0.01,
                              spread_points=0.0, size=1, init_cash=10_000, freq="D")
    assert float(pf.orders.records_readable.iloc[0]["Price"]) == pytest.approx(80.0 * 1.02)


def test_to_base_currency_markup_defaults_to_the_setting():
    import vectorbtx as vbt

    o, _, _, _ = frames([100] * 4)
    fx = pd.Series(1.25, index=o.index, name="GBPUSD")
    before = vbt.settings["vectorbtx"]["fx_markup"]
    vbt.settings["vectorbtx"]["fx_markup"] = 0.0123
    try:
        out = vbt.x.to_base_currency(o, base="GBP", fx=fx, quote_currency="USD")
    finally:
        vbt.settings["vectorbtx"]["fx_markup"] = before
    assert out.fx_markup == pytest.approx(0.0123)


# ############# 10. Scalar column selection on a levered portfolio ############# #


def test_scalar_column_selection_on_a_levered_portfolio():
    o, h, l, c = frames([1.10, 1.11, 1.12, 1.11, 1.13, 1.14], columns=("A", "B"))
    pf = run(o, h, l, c, entries=mask(c.index, ["A", "B"], [("A", 0), ("B", 0)]),
             exits=mask(c.index, ["A", "B"]), instruments={"A": FX_SPREADBET, "B": FX_SPREADBET},
             size=10_000, init_cash=10_000)
    assert float(pf["A"].total_return()) == pytest.approx(float(np.ravel(pf[["A"]].total_return())[0]))


# ############# 11. Intrabar drawdown ############# #


def _reference_intrabar(pf):
    assets = pd.DataFrame(pf.assets())
    cash = pd.DataFrame(pf.cash()).sum(axis=1)
    highs = pd.DataFrame(pf.high).reindex(columns=assets.columns)
    lows = pd.DataFrame(pf.low).reindex(columns=assets.columns)
    longs, shorts = assets.clip(lower=0), assets.clip(upper=0)
    best = (longs * highs + shorts * lows).sum(axis=1) + cash
    worst = (longs * lows + shorts * highs).sum(axis=1) + cash
    return float((1 - worst / best.cummax()).max())


def test_intrabar_drawdown_counts_shared_cash_once():
    """Shared cash was counted once per column: 0.88% intrabar against 2.71% close."""
    rng = np.random.default_rng(3)
    index = pd.bdate_range("2024-01-01", periods=120)
    columns = ["A", "B", "C", "D"]
    c = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (120, 4)), axis=0)),
                     index=index, columns=columns)
    o, h, l = c.shift(1).fillna(c), c * 1.004, c * 0.996
    schedule = pd.Series(False, index=index)
    schedule.iloc[0] = True
    pf = run(o, h, l, c, instruments={k: EQUITY for k in columns},
             weights=pd.DataFrame(0.25, index=index, columns=columns), schedule=schedule,
             init_cash=100_000)
    value = pd.DataFrame(pf.value()).sum(axis=1)
    close_to_close = float((1 - value / value.cummax()).max())
    assert pf.x.intrabar_drawdown() >= close_to_close
    assert pf.x.intrabar_drawdown() == pytest.approx(_reference_intrabar(pf))


def test_intrabar_drawdown_brackets_a_short_by_the_high():
    """A short of 10 at 100 with a spike to 104 reported a GAIN (-0.1%)."""
    o, h, l, c = frames([100] * 8, [100.5, 100.5, 100.5, 104, 100.5, 100.5, 100.5, 100.5],
                        [99.5] * 8, [100] * 8)
    none = mask(c.index, ["X"])
    pf = run(o, h, l, c, entries=none, exits=none, short_entries=mask(c.index, ["X"], [("X", 0)]),
             short_exits=none, instruments={"X": EQUITY}, size=10, init_cash=10_000)
    # Cash 11,000 once short. Peak: covered at the low, 11,000 - 995. Trough: at the
    # spike, 11,000 - 1,040.
    assert pf.x.intrabar_drawdown() == pytest.approx(1 - 9_960 / 10_005)
    series = pf.x.intrabar_drawdown_series()
    assert len(series) == len(c)
    assert float(series.max()) == pf.x.intrabar_drawdown()


# ############# 12. pandas 4 ############# #


def test_accrual_does_not_use_the_removed_weekday_alias():
    import inspect

    from vectorbtx.holding import accrual

    assert ".weekday" not in inspect.getsource(accrual)


# ############# 13. Shared cash with an entry priority ############# #

COLUMNS = ["A", "B", "C", "D"]


def _bar_three(**kwargs):
    """4 instruments at a flat 100, 10,000 cash, 40 shares (40%) per entry.

    Bar 0 signals A and B (filled at bar 1, leaving 2,000). Bar 2 signals A to exit and C
    and D to enter (filled at bar 3). A's exit frees 4,000: exactly one of C and D fits.
    """
    o, h, l, c = frames([100] * 8, [100.5] * 8, [99.5] * 8, columns=COLUMNS)
    pf = run(o, h, l, c, entries=mask(c.index, COLUMNS, [("A", 0), ("B", 0), ("C", 2), ("D", 2)]),
             exits=mask(c.index, COLUMNS, [("A", 2)]), instruments={k: EQUITY for k in COLUMNS},
             size=40, init_cash=10_000, **kwargs)
    orders = pf.orders.records_readable
    filled = orders[orders["Timestamp"] == c.index[3]]
    return {str(r.Column): (r.Side, float(r.Size)) for r in filled.itertuples()}, pf


def test_shared_cash_exits_first_then_priority_list():
    got, _ = _bar_three(cash="shared", entry_priority=["D", "C", "A", "B"], when_cash_short="skip")
    assert got == {"A": ("Sell", 40.0), "D": ("Buy", 40.0)}


def test_shared_cash_skip_is_the_default():
    got, _ = _bar_three(cash="shared", entry_priority=["D", "C"])
    assert got == {"A": ("Sell", 40.0), "D": ("Buy", 40.0)}


def test_shared_cash_priority_by_score_shifts_with_the_signals():
    index = pd.bdate_range("2024-01-01", periods=8)
    scores = pd.DataFrame(0.0, index=index, columns=COLUMNS)
    scores.loc[index[2], ["C", "D"]] = [2.0, 1.0]        # scored on the signal bar
    got, _ = _bar_three(cash="shared", entry_priority=scores)
    assert got == {"A": ("Sell", 40.0), "C": ("Buy", 40.0)}


def test_shared_cash_partial_fills_the_remainder():
    got, _ = _bar_three(cash="shared", entry_priority=["D", "C", "A", "B"], when_cash_short="partial")
    assert got["D"] == ("Buy", 40.0) and got["C"] == ("Buy", 20.0)


def test_shared_cash_random_priority_is_reproducible_per_seed():
    first, _ = _bar_three(cash="shared", entry_priority="random", seed=7)
    second, _ = _bar_three(cash="shared", entry_priority="random", seed=7)
    assert first == second
    assert [side for side, _ in first.values()].count("Buy") == 1


def test_shared_cash_is_one_pool_and_per_instrument_is_n():
    _, shared = _bar_three(cash="shared")
    _, separate = _bar_three(cash="per_instrument")
    assert float(np.sum(np.asarray(shared.get_init_cash(), dtype=float))) == pytest.approx(10_000)
    assert float(np.sum(np.asarray(separate.get_init_cash(), dtype=float))) == pytest.approx(40_000)


def test_short_exits_free_cash_before_short_entries():
    o, h, l, c = frames([100] * 8, [100.5] * 8, [99.5] * 8, columns=COLUMNS)
    none = mask(c.index, COLUMNS)
    pf = run(o, h, l, c, entries=none, exits=none,
             short_entries=mask(c.index, COLUMNS, [("A", 0), ("B", 0), ("C", 2), ("D", 2)]),
             short_exits=mask(c.index, COLUMNS, [("A", 2)]), instruments={k: EQUITY for k in COLUMNS},
             size=40, init_cash=10_000, cash="shared", entry_priority=["D", "C"], lock_cash=True)
    orders = pf.orders.records_readable
    filled = orders[orders["Timestamp"] == c.index[3]]
    assert {str(r.Column): r.Side for r in filled.itertuples()} == {"A": "Buy", "D": "Sell"}


def test_call_seq_puts_closing_orders_first_and_reads_direction():
    from vectorbtx.frontend.cash import shared_call_seq

    index = pd.bdate_range("2024-01-01", periods=2)
    entries = mask(index, COLUMNS, [("B", 1)])
    exits = mask(index, COLUMNS, [("C", 1)])
    # longonly: C's exit closes, so it goes first; B's entry next; then the rest.
    assert shared_call_seq(index, COLUMNS, entries, exits, direction="longonly")[1].tolist() == [2, 1, 0, 3]
    # both: C's "exit" is a short entry, so B and C compete, and the priority decides.
    seq = shared_call_seq(index, COLUMNS, entries, exits, entry_priority=["C", "B"], direction="both")
    assert seq[1].tolist() == [2, 1, 0, 3]
    seq = shared_call_seq(index, COLUMNS, entries, exits, entry_priority=["B", "C"], direction="both")
    assert seq[1].tolist() == [1, 2, 0, 3]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(cash="per_instrument", entry_priority=["A"]), "needs `cash='shared'`"),
        (dict(entry_priority=["A"]), "needs `cash='shared'`"),
        (dict(cash="shared", entry_priority=["Z"]), "not in the backtest"),
        (dict(cash="shared", entry_priority=["A"], call_seq="auto"), "not both"),
        (dict(cash="shared", group_by=[0, 0, 1, 1]), "ONE account"),
        (dict(cash="shared", cash_sharing=False), "contradicts"),
        (dict(when_cash_short="skip", allow_partial=True), "not both"),
        (dict(cash="pooled"), "must be one of"),
        (dict(when_cash_short="shrink"), "must be one of"),
    ],
)
def test_cash_arguments_that_cannot_mean_anything_raise(kwargs, match):
    from vectorbtx.frontend.prepare import BacktestError

    with pytest.raises(BacktestError, match=match):
        _bar_three(**kwargs)


def test_shared_cash_gap_stop_funds_an_entry_at_that_open():
    """A stop that gaps through at the open frees its cash AT the open, so an entry at that
    same open can use it. A stop hit intrabar comes after the open, so it must not."""
    cols = ["A", "B"]
    flat_b = dict(o=100.0, h=100.5, l=99.5, c=100.0)

    def account(a_open, a_low):
        o, h, l, c = frames([100] * 6, [100.5] * 6, [99.5] * 6, columns=cols)
        o.loc[o.index[3], "A"], l.loc[l.index[3], "A"] = a_open, a_low
        h.loc[h.index[3], "A"] = max(a_open, 100) + 0.5
        c.loc[c.index[3], "A"] = a_open
        for frame, key in ((o, "o"), (h, "h"), (l, "l"), (c, "c")):
            frame["B"] = flat_b[key]
        pf = run(o, h, l, c, entries=mask(c.index, cols, [("A", 0), ("B", 2)]),
                 exits=mask(c.index, cols), instruments={k: EQUITY for k in cols}, size=60,
                 stop_points=5, init_cash=10_000, cash="shared")
        orders = pf.orders.records_readable
        return {(str(r.Column), r.Side) for r in orders[orders["Timestamp"] == c.index[3]].itertuples()}

    assert account(a_open=90, a_low=89.5) == {("A", "Sell"), ("B", "Buy")}      # gap at the open
    assert account(a_open=100, a_low=94) == {("A", "Sell")}                       # intrabar


def test_single_backtest_rejects_the_portfolio_cash_arguments():
    import vectorbtx as vbt
    from vectorbtx.frontend.prepare import BacktestError

    o, h, l, c = frames([100] * 4)
    with pytest.raises(BacktestError, match="cash"):
        vbt.x.SingleBacktest(close=c["X"], open=o["X"], high=h["X"], low=l["X"],
                             entries=mask(c.index, ["X"])["X"], mode="equity", point_size=1.0,
                             size=1, init_cash=1_000, cash="shared")


# ############# Stops on the entry bar (split bars) ############# #


def _orders(pf):
    """(bar, side, price) for every order, in record order."""
    r = pf.orders.records_readable
    return [(pf.wrapper.index.get_loc(t), s, round(float(p), 6))
            for t, s, p in zip(r["Timestamp"], r["Side"], r["Price"])]


@pytest.mark.parametrize("side", ["long", "short"])
def test_stop_is_checked_on_the_entry_bar(side):
    """Whole bars check a stop from the bar AFTER entry, so a stop the entry bar traded
    through was missed. Bar 1 opens at 100 and reaches 97 / 103 later in the bar."""
    o, h, l, c = frames([100] * 6, [101, 103 if side == "short" else 101] + [101] * 4,
                        [99, 97 if side == "long" else 99] + [99] * 4)
    none = mask(c.index, ["X"])
    signal = mask(c.index, ["X"], [("X", 0)])
    kw = dict(entries=signal, exits=none) if side == "long" else dict(
        entries=none, exits=none, short_entries=signal, short_exits=none)
    pf = run(o, h, l, c, instruments={"X": EQUITY}, size=1, stop_points=2, init_cash=10_000, **kw)
    stop = 98.0 if side == "long" else 102.0
    assert [(bar, price) for bar, _, price in _orders(pf)] == [(1, 100.0), (1, stop)]
    assert pd.DataFrame(pf.assets())["X"].abs().sum() == 0        # flat by the close of bar 1


def test_target_is_checked_on_the_entry_bar_and_a_stop_beats_it():
    """With both in range on one bar the order of the high and the low is unknown, so the
    stop wins: the pessimistic reading."""
    o, h, l, c = frames([100] * 6, [101, 103, 101, 101, 101, 101], [99] * 6)
    signal, none = mask(c.index, ["X"], [("X", 0)]), mask(c.index, ["X"])
    pf = run(o, h, l, c, entries=signal, exits=none, instruments={"X": EQUITY}, size=1,
             target_points=2, init_cash=10_000)
    assert _orders(pf) == [(1, "Buy", 100.0), (1, "Sell", 102.0)]
    l.iloc[1] = 97.0
    pf = run(o, h, l, c, entries=signal, exits=none, instruments={"X": EQUITY}, size=1,
             stop_points=2, target_points=2, init_cash=10_000)
    assert _orders(pf) == [(1, "Buy", 100.0), (1, "Sell", 98.0)]


def test_an_exit_signal_at_the_open_beats_a_later_intrabar_stop():
    """The signal fills at the open; the stop could only be hit after it. Whole bars let
    the stop win, at the stop price."""
    o, h, l, c = frames([100] * 6, [101] * 6, [99, 99, 99, 95, 99, 99])
    pf = run(o, h, l, c, entries=mask(c.index, ["X"], [("X", 0)]),
             exits=mask(c.index, ["X"], [("X", 2)]), instruments={"X": EQUITY}, size=1,
             stop_points=2, init_cash=10_000)
    assert _orders(pf) == [(1, "Buy", 100.0), (3, "Sell", 100.0)]


def test_a_same_bar_round_trip_carries_nothing_overnight():
    import vectorbtx as vbt

    o, h, l, c = frames([1.10] * 8, [1.1010] * 8, [1.0990, 1.0950] + [1.0990] * 6,
                        columns=("EURUSD",))
    rates = vbt.x.static_rate_table(["EUR", "USD"], c.index, {"EUR": 0.03, "USD": 0.045})
    pf = run(o, h, l, c, entries=mask(c.index, ["EURUSD"], [("EURUSD", 0)]),
             exits=mask(c.index, ["EURUSD"]), instruments={"EURUSD": FX_SPREADBET},
             size=10_000, stop_points=20, rates=rates, init_cash=10_000)
    assert [bar for bar, _, _ in _orders(pf)] == [1, 1]
    assert float(pd.DataFrame(pf.holding_costs).abs().sum().sum()) == 0.0


def test_split_bars_change_nothing_when_no_entry_bar_reaches_its_stop():
    """The split must only ADD the entry-bar check. With bar ranges too narrow for any
    entry bar to reach a 3% stop, it reproduces a whole-bar simulation exactly."""
    from vectorbtx.core.portfolio import Portfolio
    from vectorbtx.frontend.backtest import _next_bar

    rng = np.random.default_rng(11)
    index = pd.bdate_range("2023-01-02", periods=300)
    columns = ["A", "B", "C"]
    c = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (300, 3)), axis=0)),
                     index=index, columns=columns)
    o = c.shift(1).fillna(c)
    h, l = np.maximum(o, c) * 1.002, np.minimum(o, c) * 0.998
    entries = pd.DataFrame(rng.random((300, 3)) < 0.05, index=index, columns=columns)
    none = mask(index, columns)

    split = run(o, h, l, c, entries=entries, exits=none, instruments={k: EQUITY for k in columns},
                size=10, stop_points=3.0, init_cash=10_000)
    whole = Portfolio.from_signals(c, entries=_next_bar(entries, c), exits=_next_bar(none, c),
                                   size=10, open=o, high=h, low=l, price=o, sl_stop=3.0 / o,
                                   stop_entry_price="price", stop_exit_price="stopmarket",
                                   init_cash=10_000, freq="D")
    assert len(_orders(split)) > 10                  # enough stop exits to mean something
    assert sorted(_orders(split)) == sorted(_orders(whole))
    assert np.allclose(pd.DataFrame(split.value()).to_numpy(), pd.DataFrame(whole.value()).to_numpy())


def test_split_bar_portfolio_supports_every_report():
    o, h, l, c = frames([100] * 6, [101] * 6, [99, 97, 99, 99, 99, 99])
    pf = run(o, h, l, c, entries=mask(c.index, ["X"], [("X", 0)]), exits=mask(c.index, ["X"]),
             instruments={"X": EQUITY}, size=1, stop_points=2, init_cash=10_000)
    assert pf.wrapper.index.equals(c.index)
    assert float(pf.stats()["Total Trades"]) == 1
    assert len(pf.trades.stats()) and len(pf.returns_stats())
    assert pf.x.intrabar_drawdown() >= 0


# ############# Columns inside a shared account ############# #


def _shared_book():
    rng = np.random.default_rng(5)
    index = pd.bdate_range("2023-01-02", periods=200)
    columns = ["A", "B", "C"]
    c = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (200, 3)), axis=0)),
                     index=index, columns=columns)
    o = c.shift(1).fillna(c)
    h, l = np.maximum(o, c) * 1.01, np.minimum(o, c) * 0.99
    entries = pd.DataFrame(rng.random((200, 3)) < 0.05, index=index, columns=columns)
    exits = pd.DataFrame(rng.random((200, 3)) < 0.03, index=index, columns=columns)
    return run(o, h, l, c, entries=entries, exits=exits, instruments={k: EQUITY for k in columns},
               size=10, stop_points=3.0, init_cash=10_000, cash="shared")


def test_a_column_of_a_shared_account_is_its_contribution():
    """`pf["A"]` raised on a shared account. It is now A's contribution: the account's
    equity and A's trades only, so the columns add up to the account exactly."""
    pf = _shared_book()
    contributions = {k: float(pf[k].total_return()) for k in ["A", "B", "C"]}
    assert sum(contributions.values()) == pytest.approx(float(pf.total_return()))
    assert sum(float(pf[k].total_profit()) for k in "ABC") == pytest.approx(float(pf.total_profit()))
    assert list(contributions.values()) == pytest.approx(pf.total_return(group_by=False).tolist())
    assert float(np.ravel(pf["A"].get_init_cash())[0]) == pytest.approx(10_000)
    assert float(pf["A"].stats()["Total Trades"]) > 0


def test_several_columns_of_a_shared_account_share_its_equity():
    pf = _shared_book()
    pair = pf[["A", "B"]]
    assert float(np.ravel(pair.get_init_cash())[0]) == pytest.approx(10_000)
    assert float(np.ravel(pair.total_return())[0]) == pytest.approx(
        float(pf["A"].total_return()) + float(pf["B"].total_return()))


def test_a_levered_shared_account_adds_up_too():
    o, h, l, c = frames([1.10, 1.11, 1.12, 1.11, 1.13, 1.14], [1.15] * 6, [1.05] * 6,
                        columns=("A", "B"))
    pf = run(o, h, l, c, entries=mask(c.index, ["A", "B"], [("A", 0), ("B", 0)]),
             exits=mask(c.index, ["A", "B"]), instruments={"A": FX_SPREADBET, "B": FX_SPREADBET},
             size=10_000, init_cash=10_000, cash="shared")
    assert float(pf["A"].total_return() + pf["B"].total_return()) == pytest.approx(
        float(pf.total_return()))


def test_selection_across_accounts_raises_and_groups_still_select():
    from vectorbtx.core.portfolio import AccountSelectionError

    o, h, l, c = frames([100] * 6, columns=("A", "B", "C", "D"))
    pf = run(o, h, l, c, entries=mask(c.index, "ABCD", [("A", 0), ("C", 0)]),
             exits=mask(c.index, "ABCD"), instruments={k: EQUITY for k in "ABCD"}, size=1,
             init_cash=10_000, cash_sharing=True, group_by=[0, 0, 1, 1])
    assert float(np.ravel(pf["C"].get_init_cash())[0]) == pytest.approx(10_000)
    with pytest.raises(AccountSelectionError, match="more than one"):
        pf[["A", "C"]]
    assert list(pf[0].wrapper.columns) == ["A", "B"]           # a group label, as upstream
    with pytest.raises(KeyError):
        pf["Z"]


# ############# 15. Holding costs reach the totals ############# #


def _one(value):
    """The single number in a one-column result (a Series, under a one-column DataFrame)."""
    return float(np.ravel(np.asarray(value, dtype=float))[0])


def _carry_only():
    """A long EURUSD spread bet in a flat market with no spread: carry is all that moves."""
    import vectorbtx as vbt

    o, h, l, c = frames([1.10] * 12, [1.1005] * 12, [1.0995] * 12, columns=("EURUSD",))
    rates = vbt.x.static_rate_table(["EUR", "USD"], c.index, {"EUR": 0.03, "USD": 0.045})
    return run(o, h, l, c, entries=mask(c.index, ["EURUSD"], [("EURUSD", 0)]),
               exits=mask(c.index, ["EURUSD"]), instruments={"EURUSD": FX_SPREADBET},
               size=10_000, rates=rates, init_cash=10_000)


def _carry_book(cash):
    """Two FX spread bets on random walks, with stops (so split bars) and carry."""
    import vectorbtx as vbt

    rng = np.random.default_rng(3)
    index = pd.bdate_range("2024-01-01", periods=120)
    columns = ["A", "B"]
    c = pd.DataFrame(1.10 * np.exp(np.cumsum(rng.normal(0, 0.003, (120, 2)), axis=0)),
                     index=index, columns=columns)
    o = c.shift(1).fillna(c)
    h, l = np.maximum(o, c) * 1.002, np.minimum(o, c) * 0.998
    entries = pd.DataFrame(rng.random((120, 2)) < 0.08, index=index, columns=columns)
    exits = pd.DataFrame(rng.random((120, 2)) < 0.05, index=index, columns=columns)
    rates = vbt.x.static_rate_table(["EUR", "USD"], index, {"EUR": 0.03, "USD": 0.045})
    return run(o, h, l, c, entries=entries, exits=exits,
               instruments={k: FX_SPREADBET for k in columns}, size=10_000, stop_points=30,
               rates=rates, init_cash=10_000, cash=cash)


def test_holding_costs_reach_total_profit_final_value_total_return_and_stats():
    """Studio issue 15: carry reached `value()` but not `total_profit`, so `final_value`,
    `total_return` and stats' End Value and Total Return reported the result before carry
    while the equity curve, Sharpe and drawdown in the same tear sheet included it."""
    pf = _carry_only()
    charged = float(pf.holding_costs.to_numpy().sum())
    last = float(pd.DataFrame(pf.value()).iloc[-1, 0])
    assert charged < -1                                  # a real charge, so the test can fail
    assert _one(pf.total_profit()) == pytest.approx(charged)
    assert _one(pf.final_value()) == pytest.approx(last)
    assert _one(pf.total_return()) == pytest.approx(last / 10_000 - 1)
    stats = pf.stats()
    assert float(stats["End Value"]) == pytest.approx(last)
    assert float(stats["Total Return [%]"]) == pytest.approx((last / 10_000 - 1) * 100)


def test_trade_pnl_stays_before_holding_costs():
    """By design: a charge accrues per bar on whatever is held, so it has no trade to
    belong to. Trade P&L and the MT5 trade metrics leave it out; the totals add it."""
    pf = _carry_only()
    trade_pnl = _one(pf.trades.pnl.sum())
    assert trade_pnl == pytest.approx(0.0, abs=1e-9)    # a flat market with no spread
    assert _one(pf.total_profit()) == pytest.approx(
        trade_pnl + float(pf.holding_costs.to_numpy().sum()))


@pytest.mark.parametrize("cash", ["shared", "per_instrument"])
def test_totals_match_the_equity_curve_with_holding_costs(cash):
    """Per account and per column, on split bars: the totals are the curve's last value."""
    pf = _carry_book(cash)
    assert pf.holding_costs.to_numpy().sum() < 0
    for group_by in (None, False):
        last = pd.DataFrame(pf.value(group_by=group_by)).iloc[-1].to_numpy(dtype=float)
        init = np.ravel(np.asarray(pf.get_init_cash(group_by=group_by), dtype=float))
        assert np.ravel(pf.final_value(group_by=group_by)) == pytest.approx(last)
        assert np.ravel(pf.total_return(group_by=group_by)) == pytest.approx(last / init - 1)


def test_contributions_to_a_shared_account_carry_their_own_holding_costs():
    pf = _carry_book("shared")
    assert sum(float(pf[k].total_profit()) for k in "AB") == pytest.approx(_one(pf.total_profit()))
    assert [float(pf[k].total_profit()) for k in "AB"] == pytest.approx(
        pf.total_profit(group_by=False).tolist())
    a = pf["A"]
    assert _one(a.final_value()) == pytest.approx(float(pd.DataFrame(a.value()).iloc[-1, 0]))


# ############# 16. Holding costs per instrument in a shared account ############# #


def _by_column(pf):
    """Every order as (column, bar, side, price), sorted: record order differs by cash model."""
    r = pf.orders.records_readable
    return sorted((str(k), pf.wrapper.index.get_loc(t), s, round(float(p), 9))
                  for k, t, s, p in zip(r["Column"], r["Timestamp"], r["Side"], r["Price"]))


def test_an_instrument_in_a_shared_account_pays_carry_on_its_own_position_only():
    """Carry was read off `asset_value()`, which a shared account reports per group, so
    every instrument paid carry on the whole account's position: B, holding nothing, paid
    exactly what A paid. Separate accounts charged B nothing, as they should."""
    import vectorbtx as vbt

    o, h, l, c = frames([1.10] * 12, [1.1005] * 12, [1.0995] * 12, columns=("A", "B"))
    rates = vbt.x.static_rate_table(["EUR", "USD"], c.index, {"EUR": 0.03, "USD": 0.045})
    charged = {}
    for cash in ("per_instrument", "shared"):
        pf = run(o, h, l, c, entries=mask(c.index, ["A", "B"], [("A", 0)]),
                 exits=mask(c.index, ["A", "B"]), instruments={"A": FX_SPREADBET, "B": FX_SPREADBET},
                 size=10_000, rates=rates, init_cash=10_000, cash=cash)
        charged[cash] = pf.holding_costs.sum()
    assert charged["shared"]["A"] < -1                   # a real charge, so the test can fail
    assert charged["shared"]["B"] == 0.0
    assert charged["shared"].tolist() == pytest.approx(charged["per_instrument"].tolist())


def test_the_same_orders_pay_the_same_carry_whichever_cash_model():
    """Carry depends on each position, not on whose cash funded it. With cash to spare
    both models place the same orders, so they must be charged the same, bar by bar."""
    shared, separate = _carry_book("shared"), _carry_book("per_instrument")
    assert _by_column(shared) == _by_column(separate)
    assert shared.holding_costs.to_numpy() == pytest.approx(separate.holding_costs.to_numpy())


def test_the_replay_path_charges_each_instrument_its_own_carry_in_a_shared_account():
    import vectorbtx as vbt

    o, h, l, c = frames([1.10] * 12, [1.1005] * 12, [1.0995] * 12, columns=("A", "B"))
    universe = vbt.load_universe({k: vbt.Instrument(**FX_SPREADBET) for k in "AB"})
    rates = vbt.x.static_rate_table(["EUR", "USD"], c.index, {"EUR": 0.03, "USD": 0.045})
    # The minimum stake is 10,000 units, and this level-3 path takes `init_cash` literally
    # (no buying power), so the account must cover the full 11,000 notional.
    size = pd.DataFrame(np.nan, index=c.index, columns=["A", "B"])
    size.iloc[0] = [10_000.0, np.nan]
    charged = {}
    for shared in (False, True):
        pf = vbt.Portfolio.from_holding_costs(c, size, universe, rates=rates, init_cash=50_000,
                                              cash_sharing=shared, group_by=shared or None, freq="D")
        assert len(pf.orders.records) == 1
        charged[shared] = pf.holding_costs.sum()
    assert charged[True]["A"] < 0
    assert charged[True]["B"] == 0.0
    assert charged[True].tolist() == pytest.approx(charged[False].tolist())


def test_charges_refuse_a_per_account_frame_for_several_instruments():
    """The old fallback handed a one-column frame to every column; it now says what to pass."""
    import vectorbtx as vbt
    from vectorbtx.holding.callback import build_charges

    o, h, l, c = frames([1.10] * 4, columns=("A", "B"))
    universe = vbt.load_universe({k: vbt.Instrument(**FX_SPREADBET) for k in "AB"})
    rates = vbt.x.static_rate_table(["EUR", "USD"], c.index, {"EUR": 0.03, "USD": 0.045})
    account = pd.DataFrame({"group": [11_000.0] * 4}, index=c.index)
    with pytest.raises(ValueError, match="per column"):
        build_charges(universe, c.index, notional=account, rates=rates, columns=["A", "B"])


EQUITY_2X = {"mode": "equity", "point_size": 1.0, "quote_currency": "GBP", "spread_points": 0.0,
             "leverage": 2.0}


def _margin_account(units, cash="shared"):
    """10,000 of equity at 2x, flat prices of 100, buying `units` of A and B at bar 1's open.
    Financing is 5% + 3% = 8% a year on the debit, /365 for GBP."""
    import vectorbtx as vbt

    o, h, l, c = frames([100.0] * 12, columns=("A", "B"))
    entries = mask(c.index, ["A", "B"], [(k, 0) for k in units])
    size = pd.DataFrame({k: float(units.get(k, 0)) for k in "AB"}, index=c.index)
    rates = vbt.x.static_rate_table(["GBP"], c.index, {"GBP": 0.05})
    return run(o, h, l, c, entries=entries, exits=mask(c.index, ["A", "B"]),
               instruments={k: EQUITY_2X for k in "AB"}, size=size, rates=rates,
               init_cash=10_000, cash=cash, financing_markup=0.03)


@pytest.mark.parametrize("units, split", [
    ({"A": 80, "B": 80}, [3_000, 3_000]),     # neither alone exceeds the equity: was 0, 0
    ({"A": 120, "B": 40}, [4_500, 1_500]),    # by long market value
    ({"A": 150}, [5_000, 0]),                 # unchanged: one instrument borrows it all
])
def test_a_shared_margin_account_is_financed_on_its_debit_split_by_long_value(units, split):
    """Each instrument's borrowing was read as if it had the account's whole equity to
    itself, so two that together borrowed 6,000 against 10,000 showed none and paid none.
    The account's debit is now split between them in proportion to their long value."""
    pf = _margin_account(units)
    before_charges = pf.replace(holding_costs=None)     # charges are computed from this
    thursday = 3                                         # one day's accrual, no weekend
    assert before_charges.x.borrowed_portion().iloc[thursday].tolist() == pytest.approx(split)
    assert float(before_charges.x.borrowed_portion(group_by=None).iloc[thursday, 0]) == \
        pytest.approx(sum(split))
    assert pf.holding_costs.iloc[thursday].tolist() == pytest.approx(
        [-amount * 0.08 / 365 for amount in split])


def test_the_split_adds_up_and_falls_back_to_equal_shares():
    """Two accounts, (A, B) and (C, D). Shorts take no share; an account that owes money
    but holds no longs (only after losses beyond the account) splits it equally."""
    from vectorbtx.margin.exposure import split_by_long_value

    debit = [[6_000.0, 0.0],
             [6_000.0, 1_000.0],
             [0.0, 500.0]]
    long_value = [[12_000.0, 4_000.0, -3_000.0, np.nan],
                  [10_000.0, -2_000.0, 0.0, np.nan],     # D not yet priced
                  [5_000.0, 5_000.0, 5_000.0, 0.0]]
    got = split_by_long_value(debit, long_value, [0, 0, 1, 1])
    np.testing.assert_allclose(got, [[4_500, 1_500, 0, 0],
                                     [6_000, 0, 500, 500],
                                     [0, 0, 500, 0]])
    assert got[:, :2].sum(axis=1).tolist() == pytest.approx([6_000, 6_000, 0])
    assert got[:, 2:].sum(axis=1).tolist() == pytest.approx([0, 1_000, 500])


def test_separate_accounts_grouped_add_their_borrowing_up():
    """Summing the group's cash let B's spare 10,000 pay off A's 5,000 debt."""
    pf = _margin_account({"A": 150}, cash="per_instrument").replace(holding_costs=None)
    assert pf.x.borrowed_portion().iloc[3].tolist() == pytest.approx([5_000, 0])
    assert float(pf.x.borrowed_portion(group_by=True).iloc[3, 0]) == pytest.approx(5_000)


# ############# 17. Borrowing without a margin loan; leverage on the replay path ############# #


def test_a_fully_paid_share_portfolio_borrows_nothing_and_pays_no_financing():
    """The rebalancing path runs without a margin loan, and borrowing then fell back to
    `|asset value| - free cash` measured after the purchase: a fully invested 60/40 book
    counted every pound as borrowed and paid margin financing on all of it."""
    import vectorbtx as vbt

    o, h, l, c = frames([100.0] * 12, columns=("A", "B"))
    weights = pd.DataFrame({"A": 0.6, "B": 0.4}, index=c.index)
    rates = vbt.x.static_rate_table(["GBP"], c.index, {"GBP": 0.05})
    pf = run(o, h, l, c, instruments={k: EQUITY_2X for k in "AB"}, weights=weights,
             rates=rates, init_cash=10_000, financing_markup=0.03)
    invested = float(pd.DataFrame(pf.asset_value(group_by=False)).iloc[-1].sum())
    assert invested == pytest.approx(10_000, rel=1e-2)          # fully invested, all paid for
    assert float(pf.x.borrowed_portion().to_numpy().max()) == 0.0
    assert float(pf.holding_costs.to_numpy().sum()) == 0.0


def test_a_cash_paid_position_built_outside_the_front_door_borrows_nothing():
    """8,000 bought from 10,000 leaves 2,000 of cash; it counted as 6,000 borrowed."""
    import vectorbtx as vbt

    c = frames([100.0] * 6)[3]
    size = pd.DataFrame(np.nan, index=c.index, columns=["X"])
    size.iloc[0] = 80.0
    pf = vbt.Portfolio.from_orders(c, size=size, init_cash=10_000, freq="D")
    assert float(pd.DataFrame(pf.cash()).iloc[-1, 0]) == pytest.approx(2_000)
    assert float(pf.x.borrowed_portion().to_numpy().max()) == 0.0


def _replay(units, cash_sharing=False, leverage=2.0):
    """The level-3 replay path: buy `units` at 100 on bar 0 on 10,000 of equity; GBP at
    5% plus a 3% markup is 8% a year on any overdraft, /365."""
    import vectorbtx as vbt

    c = frames([100.0] * 12, columns=("A", "B"))[3]
    universe = vbt.load_universe({
        k: vbt.Instrument(mode="equity", quote_currency="GBP", leverage=leverage) for k in "AB"
    })
    size = pd.DataFrame(np.nan, index=c.index, columns=["A", "B"])
    size.iloc[0] = [float(units.get(k, np.nan)) for k in "AB"]
    rates = vbt.x.static_rate_table(["GBP"], c.index, {"GBP": 0.05})
    return vbt.Portfolio.from_holding_costs(
        c, size, universe, rates=rates, markup=0.03, init_cash=10_000,
        cash_sharing=cash_sharing, group_by=True if cash_sharing else None, freq="D",
    )


def _spent(pf):
    """What every order cost, fill price (with slippage) times size plus fees."""
    r = pf.orders.records
    return float((r["size"] * r["price"] + r["fees"]).sum())


def test_the_replay_path_funds_leverage_like_the_front_door():
    """It handed `init_cash` to the engine as its cash, so a margin account could never
    borrow: 150 shares at 100 on 10,000 at 2x part-filled at 100 shares, and margin
    financing never fired. Now the engine gets the buying power and equity is reported."""
    pf = _replay({"A": 150})
    thursday = 3
    owed = _spent(pf) - 10_000       # 15,000 plus the default slippage, less the equity
    assert float(pd.DataFrame(pf.assets())["A"].iloc[-1]) == pytest.approx(150)
    assert np.ravel(pf.get_init_cash()).tolist() == pytest.approx([10_000, 10_000])
    assert pf.replace(holding_costs=None).x.borrowed_portion().iloc[thursday].tolist() == \
        pytest.approx([owed, 0])
    assert pf.holding_costs.iloc[thursday].tolist() == pytest.approx([-owed * 0.08 / 365, 0])
    slippage = _spent(pf) - 150 * 100
    assert float(pd.DataFrame(pf.value())["A"].iloc[-1]) == pytest.approx(
        10_000 - slippage + float(pf.holding_costs["A"].sum()))     # equity, not buying power


def test_the_replay_path_shares_one_funded_account_and_splits_its_debit():
    pf = _replay({"A": 80, "B": 80}, cash_sharing=True)
    thursday = 3
    each = (_spent(pf) - 10_000) / 2     # equal holdings take equal shares of the overdraft
    assert pf.replace(holding_costs=None).x.borrowed_portion().iloc[thursday].tolist() == \
        pytest.approx([each, each])
    assert pf.holding_costs.iloc[thursday].tolist() == pytest.approx([-each * 0.08 / 365] * 2)


def test_an_unlevered_replay_borrows_nothing():
    """At 1x, 8,000 of 10,000 is paid for; it was financed as 6,000 borrowed."""
    pf = _replay({"A": 80}, leverage=1.0)
    assert float(pf.x.borrowed_portion().to_numpy().max()) == 0.0
    assert float(pf.holding_costs.to_numpy().sum()) == 0.0


# ############# Stop-adjustment callbacks on split bars (Studio issue 14) ############# #

if HAS_VECTORBT:
    from numba import njit

    @njit
    def _count_calls_sl_nb(c, calls):
        """Count calls per bar while in a position. The last slot counts any `c.i` that is
        not a bar index — guarded, because numba does not bounds-check."""
        if c.position_now != 0:
            if 0 <= c.i < calls.shape[0] - 1:
                calls[c.i] += 1.0
            else:
                calls[calls.shape[0] - 1] += 1.0
        return c.curr_stop, c.curr_trail

    @njit
    def _time_stop_sl_nb(c, bars):
        """After `bars` bars in a position, move the stop to the entry price; an open below
        it exits there, at the open."""
        if c.position_now != 0 and c.i - c.init_i >= bars:
            return 0.0, False
        return c.curr_stop, c.curr_trail

    @njit
    def _reads_an_array_sl_nb(c, values):
        return c.curr_stop, c.curr_trail


def _narrow_random_bars(n=300, columns=("A", "B")):
    """Bars too narrow for any entry bar to reach a 3% stop."""
    rng = np.random.default_rng(21)
    index = pd.bdate_range("2023-01-02", periods=n)
    c = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, len(columns))), axis=0)),
                     index=index, columns=list(columns))
    o = c.shift(1).fillna(c)
    h, l = np.maximum(o, c) * 1.002, np.minimum(o, c) * 0.998
    entries = pd.DataFrame(rng.random(c.shape) < 0.05, index=index, columns=list(columns))
    return o, h, l, c, entries


def _whole_bars(o, h, l, c, entries, stop_fraction, **callback):
    """The same strategy on whole bars, straight through vectorbt."""
    from vectorbtx.core.portfolio import Portfolio
    from vectorbtx.frontend.backtest import _next_bar

    return Portfolio.from_signals(c, entries=_next_bar(entries, c), exits=_next_bar(mask(c.index, c.columns), c),
                                  size=10, open=o, high=h, low=l, price=o, sl_stop=stop_fraction,
                                  stop_entry_price="price", stop_exit_price="stopmarket",
                                  init_cash=10_000, freq="D", **callback)


def test_a_stop_adjusting_callback_still_checks_the_entry_bar():
    """Studio issue 14: a callback used to rule out split bars, so the entry bar's range
    went unchecked. Bar 1 opens at 100 and trades to 97 against a 2-point stop."""
    o, h, l, c = frames([100] * 6, [101] * 6, [99, 97, 99, 99, 99, 99])
    calls = np.zeros(len(c) + 1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pf = run(o, h, l, c, entries=mask(c.index, ["X"], [("X", 0)]), exits=mask(c.index, ["X"]),
                 instruments={"X": EQUITY}, size=1, stop_points=2, init_cash=10_000,
                 adjust_sl_func_nb=_count_calls_sl_nb, adjust_sl_args=(calls,))
    assert _orders(pf) == [(1, "Buy", 100.0), (1, "Sell", 98.0)]
    assert not any("NOT checked" in str(w.message) for w in caught)
    assert calls[-1] == 0                      # never handed a step index


def test_callbacks_are_called_exactly_as_on_whole_bars():
    """Once per bar, at the open, with bar indices: the same calls on the same bars as a
    whole-bar run, and the same orders when no entry bar reaches its stop."""
    o, h, l, c, entries = _narrow_random_bars()
    split_calls = np.zeros((len(c) + 1))
    whole_calls = np.zeros((len(c) + 1))
    split = run(o, h, l, c, entries=entries, exits=mask(c.index, c.columns),
                instruments={k: EQUITY for k in c.columns}, size=10, stop_points=3.0,
                init_cash=10_000, adjust_sl_func_nb=_count_calls_sl_nb, adjust_sl_args=(split_calls,))
    whole = _whole_bars(o, h, l, c, entries, 3.0 / o,
                        adjust_sl_func_nb=_count_calls_sl_nb, adjust_sl_args=(whole_calls,))
    assert split_calls[-1] == 0
    assert split_calls.sum() > 50
    assert split_calls.tolist() == whole_calls.tolist()
    assert sorted(_orders(split)) == sorted(_orders(whole))


def test_a_time_stop_callback_exits_at_the_open_as_on_whole_bars():
    """Bar counting (`c.i - c.init_i`) and the exit at the open behave as on whole bars."""
    o, h, l, c = frames([100, 100, 99.5, 99, 98.5, 98, 97.5, 97, 96.5, 96],
                        [100.2, 100.2, 99.7, 99.2, 98.7, 98.2, 97.7, 97.2, 96.7, 96.2],
                        [99.8, 99.8, 99.3, 98.8, 98.3, 97.8, 97.3, 96.8, 96.3, 95.8])
    entries = mask(c.index, ["X"], [("X", 0)])
    split = run(o, h, l, c, entries=entries, exits=mask(c.index, ["X"]), instruments={"X": EQUITY},
                size=10, stop_points=20, init_cash=10_000,
                adjust_sl_func_nb=_time_stop_sl_nb, adjust_sl_args=(3,))
    whole = _whole_bars(o, h, l, c, entries, 20.0 / o, adjust_sl_func_nb=_time_stop_sl_nb, adjust_sl_args=(3,))
    # Entered at bar 1; three bars later the stop moves to the entry, and bar 4 opens below it.
    assert _orders(split) == _orders(whole) == [(1, "Buy", 100.0), (4, "Sell", 98.5)]


def test_templated_callback_arguments_fall_back_with_a_warning():
    """Templates are filled from the two-step arrays, which a bar view cannot index."""
    import vectorbtx as vbt

    o, h, l, c = frames([100] * 6, [101] * 6, [99] * 6)
    with pytest.warns(UserWarning, match="NOT checked on the bar a position is entered"):
        run(o, h, l, c, entries=mask(c.index, ["X"], [("X", 0)]), exits=mask(c.index, ["X"]),
            instruments={"X": EQUITY}, size=1, stop_points=2, init_cash=10_000,
            adjust_sl_func_nb=_reads_an_array_sl_nb, adjust_sl_args=(vbt.Rep("high"),))
