"""Run the case files offline, against real vectorbtX.

The case files are written for the real world — `import vectorbtx as vbt`,
`vbt.YFData.download(...)`, `vbt.x.FREDData.download_rates(...)` — so they are genuinely
copy-pasteable. This harness runs them without a network by substituting **only** the two
things that need one:

* Yahoo prices -> deterministic synthetic OHLC, stable per symbol;
* FRED rates   -> the static table the docs recommend as the no-API-key fallback.

Everything else is the real library: the real `Portfolio`, the real front door, real cost,
margin and quantization resolution. That matters — running the strategy examples this way
found seven genuine bugs that parsing them would not have.

    python _harness.py             # every case
    python _harness.py C_spreadbet_fx_single.py
"""

import os
import sys
import types

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

# The case files print ⚠ and £; a Windows console stdout is cp1252 and cannot encode the
# first. Without this a case dies mid-output with UnicodeEncodeError. See _bootstrap.py.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):  # pragma: no cover - platform-dependent
        _stream.reconfigure(encoding="utf-8", errors="replace")

RNG = np.random.default_rng(20240818)
_CACHE = {}

#: start price, daily vol, daily drift
PROFILES = {
    "EURUSD=X": (1.12, 0.005, 0.00000), "GBPUSD=X": (1.27, 0.005, 0.00000),
    "USDJPY=X": (140.0, 0.005, 0.00005), "AUDUSD=X": (0.66, 0.006, 0.00000),
    "VOD.L": (75.0, 0.015, 0.00010), "BP.L": (450.0, 0.014, 0.00020),
    "HSBA.L": (600.0, 0.012, 0.00015), "SHEL.L": (2500.0, 0.013, 0.00025),
    "AZN.L": (10000.0, 0.012, 0.00035), "ULVR.L": (4200.0, 0.010, 0.00015),
    "MSFT": (240.0, 0.018, 0.00060), "AAPL": (190.0, 0.017, 0.00050),
    "NVDA": (50.0, 0.030, 0.00120), "SPY": (300.0, 0.012, 0.00040),
    "UKX": (7500.0, 0.010, 0.00020), "GER40": (15000.0, 0.011, 0.00025),
    "SAP.DE": (110.0, 0.014, 0.00030), "7203.T": (2000.0, 0.013, 0.00020),
}


def _paths(symbol, index):
    """Close and open for one symbol. The open carries an overnight gap on purpose —
    without one, next-open execution would be indistinguishable from close execution."""
    if (symbol, len(index)) not in _CACHE:
        p0, vol, drift = PROFILES.get(symbol, (100.0, 0.015, 0.0003))
        close = pd.Series(p0 * np.exp(np.cumsum(RNG.normal(drift, vol, len(index)))),
                          index=index, name=symbol)
        gap = RNG.normal(0.0, vol * 0.4, len(index))
        _CACHE[(symbol, len(index))] = (close, (close.shift(1) * np.exp(gap)).bfill())
    return _CACHE[(symbol, len(index))]


class _StubData:
    def __init__(self, symbols, index):
        self.symbols = [symbols] if isinstance(symbols, str) else list(symbols)
        self.index = index

    def get(self, field="Close", *args, **kwargs):
        cols = {}
        for sym in self.symbols:
            close, open_ = _paths(sym, self.index)
            cols[sym] = {
                "Close": close, "Open": open_,
                "High": pd.concat([close, open_], axis=1).max(axis=1) * 1.002,
                "Low": pd.concat([close, open_], axis=1).min(axis=1) * 0.998,
                "Volume": pd.Series(5e6, index=self.index),
            }[field]
        frame = pd.DataFrame(cols)
        if len(self.symbols) == 1:
            # ⚠️ Real vectorbt names a single-symbol Series after the FIELD ("Close"),
            # not after the symbol. See the matching note in strategies/_harness.py —
            # returning it symbol-named made this stub kinder than reality and hid a bug.
            return frame[self.symbols[0]].rename(field)
        return frame


class _StubYFData:
    @staticmethod
    def download(symbols, start=None, end=None, **kwargs):
        return _StubData(symbols, pd.date_range(start or "2019-01-01",
                                                end or "2024-12-31", freq="B"))


class _StubFRED:
    LEVELS = {"GBP": 0.0525, "USD": 0.0533, "EUR": 0.0400,
              "JPY": 0.0010, "AUD": 0.0435, "CHF": 0.0150}

    @classmethod
    def download_rates(cls, currencies, start=None, end=None, **kwargs):
        import vectorbtx as real

        index = pd.date_range(start or "2019-01-01", end or "2024-12-31", freq="B")
        return real.x.static_rate_table(list(currencies), index,
                                        {c: cls.LEVELS[c] for c in currencies})


class _StubMA:
    def __init__(self, series, window):
        self.ma = pd.DataFrame(series).rolling(window).mean()

    @classmethod
    def run(cls, series, window):
        return cls(series, window)

    def ma_crossed_above(self, other):
        return (self.ma > other.ma) & (self.ma.shift(1) <= other.ma.shift(1))

    def ma_crossed_below(self, other):
        return (self.ma < other.ma) & (self.ma.shift(1) >= other.ma.shift(1))


def build_vbt():
    """The real vectorbtX, with only the networked data sources replaced."""
    import vectorbtx as real

    vbt = types.ModuleType("vectorbtX")
    for name in dir(real):
        if not name.startswith("_"):
            setattr(vbt, name, getattr(real, name))
    vbt.x = types.SimpleNamespace(**{n: getattr(real.x, n) for n in real.x.__all__})
    vbt.YFData = _StubYFData
    vbt.MA = _StubMA
    vbt.x.FREDData = _StubFRED
    return vbt


def run(path):
    # encoding is explicit: the case files contain £, ⚠ and em-dashes, and Windows
    # defaults `open()` to cp1252, which cannot decode them.
    src = open(path, encoding="utf-8").read().replace(
        "import vectorbtx as vbt", "# [harness] vbt injected"
    )
    exec(compile(src, path, "exec"), {"__name__": "__main__", "vbt": build_vbt()})


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(f for f in os.listdir(HERE)
                                   if f[0].isupper() and f.endswith(".py"))
    failed = []
    for name in names:
        try:
            run(os.path.join(HERE, name))
        except Exception as exc:  # keep going; report at the end
            failed.append(name)
            import traceback

            print(f"\n!!! {name} FAILED")
            traceback.print_exc()
    print(f"\n{len(names) - len(failed)}/{len(names)} cases ran")
    if failed:
        print("failed:", failed)
        sys.exit(1)
