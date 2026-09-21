"""Shared bootstrap for the examples.

The examples are runnable in two environments:

* with vectorbt installed — `import vectorbtx as vbt` works and simulations run;
* without it — the pure-arithmetic parts still run, so every example that does not need a
  simulation is genuinely executable.

`x` is always the vectorbtX namespace. `HAS_VBT` says whether simulations are available.
"""

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The examples print ✓, ⚠ and £. Windows gives a fresh console cp1252 stdout, which cannot
# encode the first two, so the example dies with UnicodeEncodeError partway through its
# output — after doing the work, which makes it look like a library fault rather than a
# console one. Force UTF-8; `errors="replace"` keeps a terminal that genuinely cannot
# render a glyph from killing the run.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):  # pragma: no cover - platform-dependent
        _stream.reconfigure(encoding="utf-8", errors="replace")

try:
    import vectorbtx as vbt

    x = vbt.x
    HAS_VBT = True
except ImportError:
    if "vectorbtx" not in sys.modules:
        pkg = types.ModuleType("vectorbtx")
        pkg.__path__ = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vectorbtx")
        ]
        sys.modules["vectorbtx"] = pkg
    import vectorbtx.x as x

    vbt = None
    HAS_VBT = False


def header(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)
