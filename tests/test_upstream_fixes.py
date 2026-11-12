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
