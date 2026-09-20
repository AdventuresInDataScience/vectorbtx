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


def needs_vbt(what="this example"):
    if not HAS_VBT:
        print(f"[skipped] {what} needs vectorbt installed.")
        return True
    return False
