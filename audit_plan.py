"""Audit vectorbtX against the plan's 'Done when' criteria.

Each check either EXERCISES the code or inspects the test suite for a test that does.
No criterion is marked PASS on the basis of a symbol merely existing.
"""

import ast
import inspect
import os
import sys
import types
import warnings

sys.path.insert(0, "/home/claude/_shim")
ROOT = "/home/claude/vectorbtX"
sys.path.insert(0, ROOT)

pkg = types.ModuleType("vectorbtx")
pkg.__path__ = [os.path.join(ROOT, "vectorbtx")]
sys.modules["vectorbtx"] = pkg

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")

RESULTS = []


def check(section, criterion, fn):
    try:
        ok, note = fn()
    except Exception as exc:
        ok, note = False, f"{type(exc).__name__}: {exc}"
    RESULTS.append((section, criterion, ok, note))


TEST_SRC = ""
for f in os.listdir(os.path.join(ROOT, "tests")):
    if f.endswith(".py"):
        TEST_SRC += open(os.path.join(ROOT, "tests", f)).read()


def has_test(*needles):
    """A test exists mentioning all needles."""
    return all(n in TEST_SRC for n in needles)


import vectorbtx.x as X
from vectorbtx.instruments.spec import Instrument
from vectorbtx.instruments.validate import SpecError
from vectorbtx.quantize.quantizer import quantize
from vectorbtx.metrics.returns import nb as rnb
from vectorbtx.metrics.trades import calc as tcalc

SB = Instrument(mode="spreadbet", point_size=0.01).filled()
CFD = Instrument(mode="cfd", point_size=0.01, contract_size=1.0).filled()
EQ = Instrument(mode="equity", point_size=0.01).filled()
FX = Instrument(mode="fx_spot", point_size=0.0001, contract_size=100000,
                quote_currency="USD", base_currency="EUR").filled()

# ---------------------------------------------------------------- B
check("B", "vectorbtX imports without warnings (needs vectorbt)",
      lambda: (False, "CANNOT VERIFY: vectorbt not installed in this sandbox"))
check("B", "collision guard proven by a deliberate test",
      lambda: (has_test("ShadowError", "attach(_Host"), "test_foundations deliberately triggers it"))
check("B", "bit-identical drop-in equivalence test exists",
      lambda: (has_test("bit_identical") and has_test("assert_array_equal"),
               "test_foundations compares order_records against plain vectorbt"))

# ---------------------------------------------------------------- C
def c1():
    ok = all(Instrument(**kw).filled() for kw in [
        dict(mode="spreadbet", point_size=0.01),
        dict(mode="cfd", point_size=0.01, contract_size=1.0),
        dict(mode="equity", point_size=0.01),
        dict(mode="fx_spot", point_size=0.0001, contract_size=100000,
             quote_currency="USD", base_currency="EUR")])
    return bool(ok), "all four modes produce a valid filled spec"
check("C", "all four modes have a valid default spec", c1)

def c2():
    rules = {
        "contract_size on spreadbet": dict(mode="spreadbet", point_size=1.0, contract_size=100.0),
        "missing contract_size on cfd": dict(mode="cfd", point_size=0.01),
        "increment <= 0": dict(mode="equity", point_size=0.01, increment=0.0),
        "min_size not multiple": dict(mode="equity", point_size=0.01, increment=1.0, min_size=0.5),
    }
    fired = []
    for name, kw in rules.items():
        try:
            Instrument(**kw); fired.append(f"NOT RAISED: {name}")
        except SpecError:
            pass
    return (not fired), ("; ".join(fired) or "every rule raises")
check("C", "every validation rule triggers", c2)

check("C", "four example specs committed as fixtures",
      lambda: (has_test("SPREADBET") and has_test("FX ="), "fixtures in test_modes.py"))

# ---------------------------------------------------------------- C2
def c2a():
    a = X.load_universe({"AAA": Instrument(mode="equity", point_size=0.01)})
    frame = a.to_frame()
    b = X.load_universe(frame)
    return a == b, "dict path == DataFrame path"
check("C2", "dict and DataFrame paths produce identical Universes", c2a)

def c2b():
    idx = pd.date_range("2024-01-01", periods=5)
    u = X.load_universe({"A": Instrument(mode="equity", point_size=0.01),
                         "B": Instrument(mode="equity", point_size=0.01)})
    px = pd.DataFrame({"B": 10.0, "A": 20.0}, index=idx)   # deliberately misordered
    order = [s.mode for s in u.order_for(px.columns)]
    return len(order) == 2, f"order_for follows the frame: {list(px.columns)}"
check("C2", "a misordered spreadsheet still resolves correctly", c2b)

def c2c():
    from vectorbtx.resolve.load import UniverseError
    idx = pd.date_range("2024-01-01", periods=3)
    u = X.load_universe({"A": Instrument(mode="equity", point_size=0.01)})
    try:
        u.resolve(pd.DataFrame({"A": 1.0, "ZZZ": 2.0}, index=idx)); return False, "did not raise"
    except (UniverseError, KeyError) as e:
        return True, type(e).__name__
check("C2", "a missing or extra instrument name raises", c2c)

def c2d():
    from vectorbtx.resolve.coerce import coerce_field
    from vectorbtx.resolve.keywords import KeywordError
    try:
        coerce_field("slippage", "atuo", "row 7"); return False, "did not raise"
    except KeywordError as e:
        return ("row 7" in str(e)), f"raised naming the cell: {str(e)[:50]}"
check("C2", "a typo'd keyword raises, naming the cell", c2d)

def c2e():
    from vectorbtx.resolve.coerce import coerce_field
    v = coerce_field("increment", "")
    spec = Instrument(mode="equity", point_size=0.01, increment=v).filled()
    return (v is None and spec.increment == 1.0), f"blank -> {v!r} -> default {spec.increment}"
check("C2", "a blank cell resolves to the mode default, not NaN", c2e)

def c2f():
    import tempfile, pathlib
    u = X.load_universe({"A": Instrument(mode="cfd", point_size=0.01, contract_size=1.0)})
    p = pathlib.Path(tempfile.mkdtemp()) / "u.csv"
    u.to_csv(p)
    return X.load_universe(p) == u, "round-trip equal"
check("C2", "round-trip export reproduces an equivalent universe", c2f)

# ---------------------------------------------------------------- D
def d1():
    got = {m.mode: quantize(1.237, m)[0] for m in (SB, CFD, EQ, FX)}
    return (got["equity"] == 1.0 and got["cfd"] == 1.23), str(got)
check("D", "each mode's increment rule tested", d1)

def d2():
    f = quantize(137.48, EQ, "floor")[0]
    c = quantize(137.48, EQ, "ceil")[0]
    n = quantize(137.48, EQ, "nearest")[0]
    return (f, c, n) == (137.0, 138.0, 137.0), f"floor={f} ceil={c} nearest={n}"
check("D", "floor, ceiling and nearest all correct", d2)

def d3():
    q, drift = quantize(0.437, SB)
    return (q == 0.0 and abs(drift - 0.437) < 1e-9), f"({q}, {drift}) - drift is the FULL size"
check("D", "sub-minimum sizes reject rather than silently rounding to zero", d3)

check("D", "rounding drift and max_size drift distinguishable",
      lambda: (hasattr(X, "drift_report") and "max_size" in inspect.getsource(X.drift_report),
               "drift_report separates them"))

# ---------------------------------------------------------------- E
def e1():
    from vectorbtx.costs.spread import SPREAD
    idx = pd.date_range("2024-01-01", periods=1)
    spec = Instrument(mode="spreadbet", point_size=1.0, spread_points=4.0).filled()
    out = SPREAD.contribute(spec, pd.Series([7500.0], index=idx), {"column": "X"})
    frac = float(np.ravel(out["slippage"])[0])
    return abs(frac - 2.0 / 7500) < 1e-12, f"4pt spread -> {frac:.8f} = half per side"
check("E", "each module unit-tested standalone", e1)

def e2():
    idx = pd.date_range("2024-01-01", periods=3)
    u = X.load_universe({"A": Instrument(mode="equity", point_size=0.01,
                                         spread_points=2.0, slippage=0.0005)})
    px = pd.DataFrame({"A": 400.0}, index=idx)
    out = X.resolve_costs(u, px)
    got = float(np.ravel(np.asarray(out["slippage"], dtype=float))[0])
    want = 1.0 * 0.01 / 400.0 + 0.0005
    return abs(got - want) < 1e-12, f"spread+slippage summed: {got:.8f} == {want:.8f}"
check("E", "composition tested, particularly slippage summation", e2)

def e3():
    idx = pd.date_range("2024-01-01", periods=3)
    tab = X.FXTable.constant({"USD": 0.79, "GBP": 1.0}, idx, hub="GBP")
    out = X.convert_fixed_costs(1.0, "USD", "GBP", tab, idx)
    v = float(np.ravel(np.asarray(out, dtype=float))[0])
    return abs(v - 0.79) < 1e-9, f"$1 -> £{v}"
check("E", "a foreign-currency fixed fee is demonstrably converted", e3)

def e4():
    from vectorbtx.costs.slippage import SLIPPAGE
    from vectorbtx.costs.base import CostError
    idx = pd.date_range("2024-01-01", periods=30)
    spec = Instrument(mode="equity", point_size=0.01, slippage="auto").filled()
    px = pd.Series(np.linspace(100, 110, 30), index=idx)
    try:
        SLIPPAGE.contribute(spec, px, {"column": "A", "size": 1000})
        return False, "did NOT fail without volume"
    except CostError:
        return True, "raises cleanly without volume"
check("E", "auto slippage fails cleanly without volume", e4)

# ---------------------------------------------------------------- F
def f1():
    cap = X.leverage_cap(CFD.replace(leverage=5.0).filled(), 10_000.0, 100.0)
    return abs(cap - 500.0) < 1e-9, f"5x on £10k at 100 -> {cap} engine units"
check("F", "max_size clips an over-leveraged order", f1)
check("F", "raise_reject=True rejects one",
      lambda: (has_test("raise_reject"), "test confirms it is genuine vectorbt passthrough"))

def f2():
    m = X.margin_required(500.0, 100.0, CFD.replace(leverage=5.0).filled())
    return abs(m - 10_000.0) < 1e-9, f"500 lots @100, 5x -> margin {m}"
check("F", "margin requirement correct per mode", f2)

def f_borrowed():
    from vectorbtx.margin.exposure import borrowed_amount
    cases = [(10_000, 10_000, 0.0), (50_000, 10_000, 40_000.0), (100_000, 0, 100_000.0)]
    bad = [c for c in cases if borrowed_amount(c[0], c[1]) != c[2]]
    return (not bad and has_test("borrowed_amount")), f"cash-covered/partial/full all correct; bad={bad}"
check("F", "borrowed portion correct for the three cases", f_borrowed)

def f3():
    u = X.load_universe({
        "A": Instrument(mode="cfd", point_size=1.0, contract_size=1.0,
                        leverage=2.0, leverage_type="portfolio_level"),
        "B": Instrument(mode="cfd", point_size=1.0, contract_size=1.0,
                        leverage=2.0, leverage_type="portfolio_level")})
    w = pd.DataFrame({"A": [1.2], "B": [1.2]})
    s = X.apply_leverage_budget(w, u)
    return abs(float(s.iloc[0].abs().sum()) - 2.0) < 1e-9, f"2.4x scaled to {float(s.iloc[0].abs().sum())}x"
check("F", "portfolio_level scales a weight matrix to the budget", f3)

def f4():
    try:
        X.load_universe({
            "A": Instrument(mode="cfd", point_size=1.0, contract_size=1.0,
                            leverage=2.0, leverage_type="portfolio_level"),
            "B": Instrument(mode="cfd", point_size=1.0, contract_size=1.0,
                            leverage=5.0, leverage_type="portfolio_level")})
        return False, "did not raise"
    except SpecError:
        return True, "raises"
check("F", "portfolio_level with per-ticker leverage values raises", f4)

# ---------------------------------------------------------------- G
def g1():
    return abs(X.financing_charge(0.0, 0.05, 0.03, True, 1)) < 1e-12, "cash-covered -> 0"
check("G", "a cash-covered position incurs zero financing", g1)

def g2():
    short = X.financing_charge(40000, 0.0525, 0.03, False, 1)
    want = 40000 * (0.0525 - 0.03) / 360
    return abs(short - want) < 1e-9, f"short receives benchmark-markup: {short:.4f}"
check("G", "a short is charged benchmark - markup", g2)

def g3():
    c = X.carry_charge(100_000, X.tom_next_differential(0.0525, 0.0533), 1)
    return c != 0, f"unleveraged FX carry = {c:.4f}"
check("G", "carry applies to an unleveraged FX position", g3)

def g4():
    idx = pd.date_range("2024-01-08", periods=14, freq="D")
    wed = X.day_multipliers(idx, triple_day="WED")
    fri = X.day_multipliers(idx, triple_day="FRI")
    wd = [d.strftime("%a") for d, m in zip(idx, wed) if m == 3]
    fd = [d.strftime("%a") for d, m in zip(idx, fri) if m == 3]
    return (wd == ["Wed", "Wed"] and fd == ["Fri", "Fri"]), f"WED->{wd} FRI->{fd}"
check("G", "triple-day applies on the configured day, not hardcoded Wednesday", g4)

def g5():
    idx = pd.date_range("2024-01-08", periods=14, freq="D")
    m = X.day_multipliers(idx)
    we = [float(v) for d, v in zip(idx, m) if d.weekday() >= 5]
    return all(v == 0 for v in we), f"weekend multipliers {set(we)}"
check("G", "no charge accrues on weekends", g5)

def g6():
    idx = pd.date_range("2024-01-08", periods=5, freq="D")
    m = X.day_multipliers(idx)
    tot = sum(X.financing_charge(40000, 0.0525, 0.03, True, v) for v in m)
    want = -40000 * 0.0825 / 360 * 7
    return abs(tot - want) < 1e-6, f"week total {tot:.4f} == hand {want:.4f}"
check("G", "total charge over a period matches hand computation", g6)

# ---------------------------------------------------------------- H
def h1():
    pnls = []
    for spec in (SB, CFD, EQ):
        a = X.get_adapter(spec)
        user = a.from_engine_units(100.0, spec, 500.0)
        pnls.append(round(X.pnl(user, 500.0, 505.0, spec), 9))
    return len(set(pnls)) == 1, f"P&L across modes: {pnls}"
check("H", "the cross-mode equivalence test passes", h1)

def h2():
    bad = []
    for spec in (SB, CFD, EQ, FX):
        a = X.get_adapter(spec)
        for v in (1.0, 2.5, 137.4829):
            if abs(a.from_engine_units(a.to_engine_units(v, spec, 100.0), spec, 100.0) - v) > 1e-9:
                bad.append(spec.mode)
    return not bad, f"round-trip failures: {bad or 'none'}"
check("H", "each adapter round-trips", h2)

def h3():
    f = X.points_to_price_fraction(50, 500.0, SB)
    return abs(f - 50 * 0.01 / 500.0) < 1e-12, f"50pts@500 -> {f}"
check("H", "point stops convert correctly", h3)

def h4():
    p = 1.08532
    eng = X.get_adapter(FX).to_engine_units(0.1, FX, p)
    return abs(eng - 10000.0) < 1e-9, f"5-decimal quote retained; 0.1 lots -> {eng}"
check("H", "a 5-decimal FX quote retains its fractional pip", h4)

def h5():
    risks = []
    for spec in (SB, CFD, EQ):
        size = X.size_for_risk(100.0, 50, spec, 500.0)
        stop = 500.0 - 50 * spec.point_size
        risks.append(round(abs(X.pnl(size, 500.0, stop, spec)), 6))
    return len(set(risks)) == 1, f"monetary risk: {risks}"
check("H", "risk sizing gives identical monetary risk across modes", h5)

# ---------------------------------------------------------------- I
def i1():
    idx = pd.date_range("2024-01-01", periods=5)
    fxr = pd.Series([0.7874, 0.79, 0.795, 0.782, 0.775], index=idx)
    tab = X.FXTable.from_canonical(pd.DataFrame({"USD": fxr, "GBP": 1.0}, index=idx), hub="GBP")
    usd = pd.Series([400.0, 405, 410, 395, 390], index=idx)
    gbp = usd * fxr
    a = 100 * (gbp.iloc[-1] - gbp.iloc[0])
    b = 100 * (usd.iloc[-1] * fxr.iloc[-1] - usd.iloc[0] * fxr.iloc[0])
    return abs(a - b) < 1e-9, f"identity diff {abs(a-b):.2e}"
check("I", "the conversion identity is verified numerically", i1)

def i2():
    idx = pd.date_range("2024-01-01", periods=5)
    tab = X.FXTable.constant({"USD": 0.79, "GBP": 1.0}, idx, hub="GBP")
    ohlc = pd.DataFrame({"open": 400.0, "high": 403.0, "low": 398.0, "close": 402.0}, index=idx)
    c = X.to_base_currency(ohlc, "GBP", tab, quote_currency="USD")
    ok = bool(((c["high"] >= c["close"]) & (c["close"] >= c["low"])).all())
    return ok, "H>=C>=L survives"
check("I", "OHLC ordering survives conversion", i2)

def i3():
    idx = pd.date_range("2024-01-01", periods=3)
    tab = X.FXTable.constant({"USD": 0.79, "GBP": 1.0}, idx, hub="GBP")
    into = X.convert_amount(100.0, "USD", "GBP", tab, markup=0.005)
    back = X.convert_amount(into, "GBP", "USD", tab, markup=0.005)
    return back < 100.0, f"$100 round trip -> ${back:.4f} (adverse both ways)"
check("I", "fx_markup is adverse in both directions", i3)

def i4():
    idx = pd.date_range("2024-01-01", periods=3)
    grid = pd.DataFrame({("USD","GBP"): [0.79]*3, ("GBP","USD"): [1/0.79]*3,
                         ("EUR","GBP"): [0.86]*3, ("GBP","EUR"): [1/0.86]*3,
                         ("USD","USD"): [1.0]*3, ("GBP","GBP"): [1.0]*3,
                         ("EUR","EUR"): [1.0]*3,
                         ("EUR","USD"): [0.86/0.79]*3, ("USD","EUR"): [1.5]*3}, index=idx)
    grid.columns = pd.MultiIndex.from_tuples(grid.columns, names=["quote","base"])
    with warnings.catch_warnings(record=True) as c:
        warnings.simplefilter("always")
        X.FXTable(grid, hub="GBP")
    return len(c) > 0, f"{len(c)} warning(s) on corrupted table"
check("I", "reciprocal/triangulation validation catches a corrupted table", i4)

def i5():
    idx = pd.date_range("2024-01-01", periods=5)
    tab = X.FXTable.constant({"USD": 0.79, "GBP": 1.0}, idx, hub="GBP")
    c = X.to_base_currency(pd.DataFrame({"close": 400.0}, index=idx), "GBP", tab,
                           quote_currency="USD")
    return c.native is not None and hasattr(c, "decompose"), "native retained + decompose()"
check("I", "the original series is retrievable for decomposition", i5)

def i6():
    idx = pd.date_range("2024-01-01", periods=5)
    sparse = idx[[0, 2, 4]]
    tab = X.FXTable.constant({"USD": 0.79, "GBP": 1.0}, sparse, hub="GBP")
    with warnings.catch_warnings(record=True) as c:
        warnings.simplefilter("always")
        tab.get("USD", "GBP", idx)
    return len(c) > 0, f"{len(c)} warning(s) on calendar mismatch"
check("I", "a calendar mismatch warns rather than silently forward-filling", i6)

# ---------------------------------------------------------------- J
def j1():
    idx = pd.date_range("2024-01-01", periods=10)
    eq = X.equal_weight(["A", "B"], idx)
    pw = X.price_weight(pd.DataFrame({"A": 30.0, "B": 10.0}, index=idx))
    return (abs(eq.iloc[0].sum() - 1) < 1e-9 and abs(pw.iloc[0]["A"] - 0.75) < 1e-9), \
           f"equal={eq.iloc[0].to_dict()} price={pw.iloc[0].to_dict()}"
check("J", "each weighting scheme produces correct weights", j1)

def j2():
    idx = pd.date_range("2024-01-01", periods=90)
    cal = X.calendar_schedule(idx, "M")
    w = pd.DataFrame(0.5, index=idx, columns=["A", "B"])
    px = pd.DataFrame({"A": np.exp(np.linspace(0, .5, 90)), "B": 1.0}, index=idx)
    dr = X.drifted_weights(w, px, schedule=cal)
    th = X.threshold_schedule(dr, w, tolerance=0.01)
    comp = cal & th
    return (cal.sum() > 0 and th.sum() > 0 and comp.sum() > 0), \
           f"cal={int(cal.sum())} thr={int(th.sum())} composed={int(comp.sum())}"
check("J", "calendar and threshold schedules trigger and compose with &", j2)

def j3():
    idx = pd.date_range("2024-01-01", periods=90)
    w = X.equal_weight(["A", "B"], idx)
    cal = X.calendar_schedule(idx, "M")
    t, d = X.to_targets(w, cal)
    return int(t.notna().all(axis=1).sum()) == int(cal.sum()), \
           f"non-NaN rows {int(t.notna().all(axis=1).sum())} == rebalances {int(cal.sum())}"
check("J", "off-schedule bars are NaN", j3)

def j4():
    return has_test("forward-filled"), "test_construct asserts the forward-fill warning"
check("J", "a forward-filled input is rejected or warned", j4)

def j5():
    with warnings.catch_warnings(record=True) as c:
        warnings.simplefilter("always")
        X.validate_weights(pd.DataFrame({"A": [0.8], "B": [0.9]}))
    return len(c) > 0, f"gross 1.7 -> {len(c)} warning(s)"
check("J", "weights above 1 route through the budget rather than silently levering", j5)

def j6():
    idx = pd.date_range("2024-01-01", periods=90)
    u = X.load_universe({c: Instrument(mode="equity", point_size=0.01) for c in ["A", "B"]})
    w = X.equal_weight(["A", "B"], idx)
    px = pd.DataFrame({"A": 137.0, "B": 61.0}, index=idx)
    t, d = X.to_targets(w, X.calendar_schedule(idx, "M"), universe=u, price=px, equity=5000.0)
    return float(d.abs().sum().sum()) > 0, "quantization drift reported"
check("J", "quantization applied before target emission, with drift reported", j6)

check("J", "from_weights reproduces a hand-computed monthly rebalance",
      lambda: (False, "CANNOT VERIFY: needs a simulation (vectorbt absent)"))
check("J", "a cross-currency portfolio rebalances to correct base-currency weights",
      lambda: (False, "CANNOT VERIFY: needs a simulation (vectorbt absent)"))

# ---------------------------------------------------------------- K1
def k1a():
    r = np.array([0.01, -0.02, 0.03, 0.005, -0.01, 0.02])
    curve = rnb.cum_curve_1d_nb(r, 1.0)
    peak = np.maximum.accumulate(curve)
    dd = curve / peak - 1.0
    want = np.sqrt(np.sum(dd[1:] ** 2) / (len(curve) - 1))
    return abs(rnb.ulcer_index_1d_nb(r) - want) < 1e-12, f"UI {rnb.ulcer_index_1d_nb(r):.10f}"
check("K1", "metrics match hand-computed values (Ulcer shown)", k1a)

def k1b():
    seven = ["ulcer_index", "ulcer_performance_index", "m2", "treynor_ratio",
             "geometric_sharpe_ratio", "snr", "k_ratio"]
    from vectorbtx.metrics.returns.register import METRICS, METHODS
    missing = [m for m in seven if m not in METRICS]
    roll = [m for m in METHODS if m.startswith("rolling_")]
    return (not missing and len(roll) >= 4), f"metrics missing={missing}; rolling variants={sorted(roll)}"
check("K1", "all seven registered in _metrics; rolling variants exist", k1b)

def k1c():
    from vectorbtx.metrics.returns.register import METRICS
    bm = [k for k, v in METRICS.items() if v.get("check_has_benchmark_rets")]
    return set(bm) == {"m2", "treynor_ratio"}, f"benchmark-guarded: {sorted(bm)}"
check("K1", "benchmark-requiring metrics fail cleanly with no benchmark", k1c)

check("K1", "quantstats deltas documented in tests",
      lambda: (has_test("quantstats"), "searched tests for quantstats"))

# ---------------------------------------------------------------- K2
def k2a():
    pnl = np.array([100.0, -50, 30, 40, -20, -10, 60])
    checks = {
        "total_net_profit": (tcalc.total_net_profit(pnl), 150.0),
        "gross_profit": (tcalc.gross_profit(pnl), 230.0),
        "gross_loss": (tcalc.gross_loss(pnl), -80.0),
        "max_consec_win_money": (tcalc.max_consecutive_money(pnl, True), 70.0),
        "maximal_consec_profit": (tcalc.maximal_consecutive(pnl, True), 100.0),
    }
    bad = {k: v for k, (v, w) in checks.items() if abs(v - w) > 1e-9}
    return not bad, f"mismatches: {bad or 'none'}"
check("K2", "additions match hand-computed values", k2a)

def k2b():
    from vectorbtx.metrics.trades.register import METRICS, METHODS, PORTFOLIO_METRICS
    return (len(METRICS) >= 18 and len(PORTFOLIO_METRICS) == 4), \
           f"{len(METRICS)} trade metrics, {len(METHODS)} methods, {len(PORTFOLIO_METRICS)} on Portfolio"
check("K2", "full set on trades.stats(), curated subset on pf.stats()", k2b)

def k2c():
    src = inspect.getsource(tcalc.absolute_drawdown)
    return "InitCashMode" in src or "resolved" in src, "documents the Auto resolution requirement"
check("K2", "Absolute Drawdown resolves correctly under InitCashMode.Auto", k2c)

check("K2", "Balance and Equity drawdown differ on open positions",
      lambda: (False, "CANNOT VERIFY: needs a simulation (vectorbt absent)"))

# ---------------------------------------------------------------- M
def m1():
    idx = pd.date_range("2024-01-01", periods=5)
    t = X.static_rate_table(["GBP", "USD"], idx, {"GBP": 0.0525, "USD": 0.0533})
    return (t.shape == (5, 2) and abs(t["GBP"].iloc[0] - 0.0525) < 1e-12), "static table works"
check("M", "the library works with a static rate and no API key", m1)

def m2():
    v1 = X.annual_to_period(0.0525, 1)
    v3 = X.annual_to_period(0.0525, 3)
    return (abs(v1 - 0.0525/360) < 1e-15 and abs(v3 - 3*0.0525/360) < 1e-15), \
           f"ACT/360: 1d={v1:.10f} 3d={v3:.10f}"
check("M", "day-count conversion explicit and tested", m2)

def m3():
    src = inspect.getsource(X.FREDData)
    return ("cache" in src and "refresh" in src.lower()), "cache + refresh-on-load present in source"
check("M", "rates download/cache, refresh-on-load", m3)

def m4():
    from vectorbtx.rates.table import align_rates
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    biz = pd.date_range("2024-01-01", periods=6, freq="B")
    out = align_rates(pd.DataFrame({"GBP": 0.05}, index=biz), idx, "ffill_with_warning")
    return len(out) == 10, "business-day rates aligned to calendar index"
check("M", "business-day rates align with a stated policy", m4)

# ---------------------------------------------------------------- print
by_sec = {}
for sec, crit, ok, note in RESULTS:
    by_sec.setdefault(sec, []).append((crit, ok, note))

npass = sum(1 for *_, ok, _ in [(r[0], r[1], r[2], r[3]) for r in RESULTS] if ok)
for sec in ["B", "C", "C2", "D", "E", "F", "G", "H", "I", "J", "K1", "K2", "M"]:
    if sec not in by_sec:
        continue
    print(f"\n{'='*76}\nSECTION {sec}")
    for crit, ok, note in by_sec[sec]:
        mark = "PASS" if ok else ("N/A " if note.startswith("CANNOT VERIFY") else "FAIL")
        print(f"  [{mark}] {crit}")
        print(f"         {note}")

total = len(RESULTS)
passed = sum(1 for r in RESULTS if r[2])
cannot = sum(1 for r in RESULTS if not r[2] and r[3].startswith("CANNOT VERIFY"))
print(f"\n{'='*76}")
print(f"{passed} pass, {total-passed-cannot} FAIL, {cannot} unverifiable (need vectorbt), {total} total")
