"""Sandbox test runner. Collects tests/test_*.py and runs test_* functions."""

import importlib.util
import inspect
import os
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, "/home/claude/_shim")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pytest  # the shim

# run conftest first
spec = importlib.util.spec_from_file_location("conftest", ROOT / "tests" / "conftest.py")
conftest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conftest)

passed = failed = skipped = 0
failures = []

for path in sorted((ROOT / "tests").glob("test_*.py")):
    name = path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        failed += 1
        failures.append((f"{name} [import]", traceback.format_exc()))
        continue

    fixtures = {
        n: f
        for n, f in vars(mod).items()
        if callable(f) and getattr(f, "__is_fixture__", False)
    }
    fixtures.update(
        {n: f for n, f in vars(conftest).items() if callable(f) and getattr(f, "__is_fixture__", False)}
    )

    for fname, fn in sorted(vars(mod).items()):
        if not (fname.startswith("test_") and callable(fn)):
            continue
        if getattr(fn, "__skip__", False):
            skipped += 1
            continue

        params = getattr(fn, "__params__", None)
        cases = []
        if params:
            names, values = params
            for v in values:
                v = v if isinstance(v, (tuple, list)) else (v,)
                cases.append(dict(zip(names, v)))
        else:
            cases.append({})

        for case in cases:
            kwargs = dict(case)
            for arg in inspect.signature(fn).parameters:
                if arg in kwargs:
                    continue
                if arg == "tmp_path":
                    kwargs[arg] = Path(tempfile.mkdtemp())
                elif arg in fixtures:
                    kwargs[arg] = fixtures[arg]()
                else:
                    kwargs[arg] = None
            try:
                fn(**kwargs)
                passed += 1
            except pytest._Skip:
                skipped += 1
            except Exception:
                failed += 1
                label = f"{name}::{fname}" + (f"[{case}]" if case else "")
                failures.append((label, traceback.format_exc()))

for label, tb in failures:
    print("=" * 70)
    print("FAIL", label)
    print(tb)

print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
sys.exit(1 if failed else 0)
