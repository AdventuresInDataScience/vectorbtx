"""Run the strategy examples against stubbed data sources.

The strategy files are written for the real world: `import vectorbtX as vbt`,
`vbt.YFData.download(...)`, `vbt.x.FREDData.download_rates(...)`. This harness executes
everything in them that does not need vectorbt's simulation engine, substituting
deterministic synthetic prices for Yahoo and a static table for FRED.

Its purpose is to prove the *logic* of each strategy runs — sizing, quantization, FX
conversion, argument resolution, schedule construction — even where the final
`Portfolio.from_*` call cannot execute in this environment.
"""

import os
import sys
import types

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

try:
    import vectorbt  # noqa: F401

    HAS_VBT = True
except ImportError:
    HAS_VBT = False
    if "vectorbtx" not in sys.modules:
        pkg = types.ModuleType("vectorbtx")
        pkg.__path__ = [os.path.join(ROOT, "vectorbtx")]
        sys.modules["vectorbtx"] = pkg

import vectorbtx.x as _x

RNG = np.random.default_rng(20240817)
_SERIES_CACHE = {}


def _series(symbol, index, start_price, vol, drift=0.0003):
    """Deterministic synthetic price path, stable per symbol."""
    if symbol not in _SERIES_CACHE:
        steps = RNG.normal(drift, vol, len(index))
        _SERIES_CACHE[symbol] = pd.Series(
            start_price * np.exp(np.cumsum(steps)), index=index, name=symbol
        )
    return _SERIES_CACHE[symbol]


_PROFILES = {
    "EURUSD=X": (1.12, 0.005, 0.0),
    "GBPUSD=X": (1.27, 0.005, 0.0),
    "VOD.L": (75.0, 0.015, 0.0),
    "BP.L": (450.0, 0.014, 0.0002),
    "MSFT": (240.0, 0.018, 0.0006),
    "AAPL": (190.0, 0.017, 0.0005),
    "NVDA": (50.0, 0.030, 0.0012),
    "VWRL.L": (85.0, 0.010, 0.0004),
    "VUSA.L": (60.0, 0.011, 0.0005),
    "SPY": (300.0, 0.012, 0.0004),
    "AGG": (105.0, 0.004, 0.0001),
}


class _StubData:
    """Stands in for a vectorbt `Data` object."""

    def __init__(self, symbols, index):
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.index = index

    def get(self, field="Close", *args, **kwargs):
        cols = {}
        for sym in self.symbols:
            p0, vol, drift = _PROFILES.get(sym, (100.0, 0.015, 0.0003))
            base = _series(sym, self.index, p0, vol, drift)
            if field == "Close":
                cols[sym] = base
            elif field == "High":
                cols[sym] = base * 1.004
            elif field == "Low":
                cols[sym] = base * 0.996
            elif field == "Open":
                cols[sym] = base.shift(1).bfill()
            elif field == "Volume":
                cols[sym] = pd.Series(5e6, index=self.index)
            else:
                raise KeyError(field)
        frame = pd.DataFrame(cols)
        return frame[self.symbols[0]] if len(self.symbols) == 1 else frame


class _StubYFData:
    @staticmethod
    def download(symbols, start=None, end=None, **kwargs):
        index = pd.date_range(start or "2019-01-01", end or "2024-12-31", freq="B")
        return _StubData(symbols, index)


class _StubMA:
    def __init__(self, series, window):
        self.ma = series.rolling(window).mean()

    @classmethod
    def run(cls, series, window):
        return cls(series, window)

    def ma_crossed_above(self, other):
        return (self.ma > other.ma) & (self.ma.shift(1) <= other.ma.shift(1))

    def ma_crossed_below(self, other):
        return (self.ma < other.ma) & (self.ma.shift(1) >= other.ma.shift(1))


class _StubPortfolio:
    """Records the call instead of simulating, and reports what it was given."""

    last_call = None

    @classmethod
    def _record(cls, name, close, kwargs):
        cls.last_call = (name, kwargs)
        print(f"\n[harness] {name} would run on {np.shape(close)} with:")
        for key in sorted(kwargs):
            value = kwargs[key]
            if isinstance(value, pd.DataFrame):
                shown = f"<DataFrame {value.shape}>"
            elif isinstance(value, pd.Series):
                shown = f"<Series n={len(value)}>"
            elif isinstance(value, list):
                shown = str(value)
            else:
                shown = repr(value)
            print(f"           {key:18s} {shown}")
        return cls()

    @classmethod
    def from_signals(cls, close, **kwargs):
        return cls._record("Portfolio.from_signals", close, kwargs)

    @classmethod
    def from_weights(cls, close, **kwargs):
        return cls._record("Portfolio.from_weights", close, kwargs)

    def __getattr__(self, name):
        raise SystemExit(
            f"\n[harness] Logic verified up to the simulation call. `pf.{name}` needs "
            f"vectorbt's engine, which is not installed here."
        )


def build_vbt_stub():
    """A `vbt` module object exposing exactly what the strategies use.

    When vectorbt is importable the **real** vectorbtX is used — real `Portfolio`, real
    `SingleBacktest`/`PortfolioBacktest`, real cost and margin resolution — and only the
    two things that need a network are substituted: Yahoo prices and FRED rates. The
    strategies then run genuinely end to end, which is the only way these files are worth
    having as verification rather than as decoration.

    Without vectorbt, `Portfolio` falls back to a recorder that prints the call it would
    have made, so the argument-resolution logic is still exercised.
    """
    vbt = types.ModuleType("vectorbtX")
    vbt.x = _x
    vbt.YFData = _StubYFData
    vbt.MA = _StubMA
    vbt.Portfolio = _StubPortfolio

    # FRED needs a key and a network; substitute the documented static fallback.
    class _StubFRED:
        @staticmethod
        def download_rates(currencies, start=None, end=None, **kwargs):
            index = pd.date_range(start or "2019-01-01", end or "2024-12-31", freq="B")
            defaults = {"GBP": 0.0525, "USD": 0.0533, "EUR": 0.04, "JPY": 0.001, "CHF": 0.015}
            return _x.static_rate_table(list(currencies), index,
                                        {c: defaults[c] for c in currencies})

    if HAS_VBT:
        import vectorbtX as _real

        vbt.Portfolio = _real.Portfolio
        vbt.Instrument = _real.Instrument
        vbt.load_universe = _real.load_universe
        vbt.x = types.SimpleNamespace(
            **{n: getattr(_real.x, n) for n in _real.x.__all__}
        )
    else:
        vbt.x = types.SimpleNamespace(**{n: getattr(_x, n) for n in _x.__all__})

    vbt.x.FREDData = _StubFRED
    return vbt


def run(path):
    src = open(path).read()
    src = src.replace("pf.plot().show()", "pass  # [harness] plotting skipped")
    module_globals = {"__name__": "__main__", "vbt": build_vbt_stub()}
    # Strip the import; the stub is injected instead.
    src = src.replace("import vectorbtX as vbt", "# [harness] vbt injected")
    exec(compile(src, path, "exec"), module_globals)


if __name__ == "__main__":
    for name in sys.argv[1:] or sorted(
        f for f in os.listdir(os.path.dirname(os.path.abspath(__file__))) if f[0].isdigit()
    ):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
        print("\n" + "=" * 72)
        print(name)
        print("=" * 72)
        try:
            run(path)
        except SystemExit as exc:
            print(exc)
