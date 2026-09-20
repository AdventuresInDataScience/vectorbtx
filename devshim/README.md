# devshim — running the suite without a compiler toolchain

Pure-Python stand-ins for vectorbt's uncompiled dependencies, so the whole suite —
including the simulation paths — runs in an environment with no network and no numba.

**This is a test harness, not a runtime dependency.** Nothing in `vectorbtx/` imports it.

    NUMBA_DISABLE_JIT=1 PYTHONPATH=devshim:/path/to/vectorbt-master python run_tests.py

`NUMBA_DISABLE_JIT=1` is vectorbt's own escape hatch: `checks.is_numba_func` honours it
for any function whose name ends `_nb`, which every kernel in this project does. Without
it, `from_order_func` refuses uncompiled callbacks and the Section G tests cannot run.

Stubbed: `numba`, `plotly`, `dill`, `tqdm`, `pytz`, `schedule`, `dateparser`,
`mypy_extensions`, plus a minimal `pytest` for the bundled runner.

Results are numerically identical to a real install for everything here, because all of
it is NumPy/pandas arithmetic — numba changes speed, not values. Do not use it to
benchmark, and do not treat a green run as a substitute for CI against real vectorbt.
