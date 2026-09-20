# Sandbox artefacts — what to strip for production

Written so the cleanup is a checklist rather than an archaeology exercise. The sandbox this
was built in had **no network and no compiler toolchain**, so vectorbt, numba, plotly and
pytest could not be installed. Several things exist purely to work around that.

**Nothing in `vectorbtx/` imports any of it.** The library itself is clean; everything
below is tooling, tests, or docs.

---

## 1. Delete outright

| Path | Size | What it is |
|---|---|---|
| `devshim/` | 124K, 23 modules | Pure-Python stand-ins for `numba`, `plotly`, `dill`, `tqdm`, `pytz`, `schedule`, `dateparser`, `mypy_extensions`, plus a minimal `pytest`. Lets the full suite run with no toolchain. |
| `run_tests.py` | 1 file | Offline test runner, written because `pytest` was not installable. `pytest` replaces it entirely. |
| `audit_plan.py` | 1 file | Dev tooling that cross-checks the implementation against `docs/dev/PLAN.md`. Useful during the build, not part of the product. |

After deleting, remove the references in `README.md` (§Testing) and `docs/EXAMPLES.md`
(§Verification) — they document the `NUMBA_DISABLE_JIT=1 PYTHONPATH=devshim` invocation.

---

## 2. Keep, but understand why they exist

| Path | Keep? | Note |
|---|---|---|
| `examples/strategies/_harness.py` | **Yes** | Substitutes deterministic synthetic prices for Yahoo and a static table for FRED, so the strategies run without network or API keys. Genuinely useful in CI — it found seven real bugs. Uses the **real** `Portfolio` when vectorbt is importable. |
| `examples/_bootstrap.py` | **Trim** | Has an import fallback for "vectorbtX installed vs. running from a checkout". The `HAS_VBT`/`needs_vbt` branches are dead once vectorbt is a hard dependency; the `sys.path` insert is still handy for running from a checkout. |
| `tests/test_holding_sim.py` | **Yes** | Has a `HAS_VECTORBT` guard so it skips rather than errors without vectorbt. Harmless once vectorbt is always present; the guard can go. |
| `vectorbX.py` | **Yes** | The case alias. Not a sandbox artefact — it is load-bearing on Linux. See its docstring. |

---

## 3. Conditional imports inside the library

Three, all legitimate, none sandbox-specific:

- `vectorbtx/core/_numba.py` — falls back to a no-op `njit` when numba is absent. Keep:
  it is what lets the pure-arithmetic layer be imported and tested standalone, which is a
  design goal (§1.4), not a workaround.
- `vectorbtx/holding/callback.py` — guards the `vectorbt.portfolio.nb` import so the
  module can be imported without vectorbt. Keep, same reason.
- `vectorbtx/resolve/load.py` — guards `openpyxl` for `.xlsx` universes, which is a
  genuine optional dependency declared in `pyproject.toml`.

---

## 4. First things to do on a networked machine

The suite has **never run against a real vectorbt install**. Everything here executed with
`NUMBA_DISABLE_JIT=1`, which means:

1. **Numerically identical, timing-wise untested.** numba changes speed, not values, so
   the arithmetic is trustworthy. But no kernel has ever been *compiled*.
2. ⚠️ **The JIT path can fail where pure Python passes.** numba's type inference rejects
   things the interpreter accepts — heterogeneous tuples, some `np.where` broadcasts,
   reflected lists. The three kernels in `holding/callback.py` are the exposure:
   `order_func_nb`, `pre_segment_func_nb`, `post_segment_func_nb`. Compile them first.
3. Run `pytest` proper, and diff against the recorded 225 passes.
4. Re-run all nine numbered examples and the three strategies against live Yahoo and FRED
   data, which will exercise index alignment, missing bars and holiday gaps that the
   synthetic harness does not.
