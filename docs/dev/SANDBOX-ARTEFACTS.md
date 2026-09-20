# Sandbox artefacts — what to strip for production

Written so the cleanup is a checklist rather than an archaeology exercise. The sandbox this
was built in had **no network and no compiler toolchain**, so vectorbt, numba, plotly and
pytest could not be installed. Several things exist purely to work around that.

**Nothing in `vectorbtx/` imports any of it.** The library itself is clean; everything
below is tooling, tests, or docs.

---

## 1. Deleted — done

All three were removed once the suite ran green against a real install. Recover any of
them from the initial commit if needed.

| Path | Was | Replaced by |
|---|---|---|
| `devshim/` | 124K, 23 modules: pure-Python stand-ins for `numba`, `plotly`, `dill`, `tqdm`, `pytz`, `schedule`, `dateparser`, `mypy_extensions` and a minimal `pytest`. | The real packages. Nothing outside `run_tests.py` ever imported it. |
| `run_tests.py` | Offline test runner, written because `pytest` was not installable. | `pytest`. |
| `audit_plan.py` | Cross-checked the implementation against `docs/dev/PLAN.md`. | `docs/dev/AUDIT.md`, now a static record of its final run. |

Note that `tests/conftest.py` keeps its own no-vectorbt fallback — it registers `vectorbtx`
as a namespace package so the pure-arithmetic modules still import standalone. That
capability did **not** depend on `devshim/` and survives its deletion.

---

## 2. Keep, but understand why they exist

| Path | Keep? | Note |
|---|---|---|
| `examples/strategies/_harness.py` | **Yes** | Substitutes deterministic synthetic prices for Yahoo and a static table for FRED, so the strategies run without network or API keys. Genuinely useful in CI — it found seven real bugs. Uses the **real** `Portfolio` when vectorbt is importable. |
| `examples/_bootstrap.py` | **Trim** | Has an import fallback for "vectorbtX installed vs. running from a checkout". The `HAS_VBT`/`needs_vbt` branches are dead once vectorbt is a hard dependency; the `sys.path` insert is still handy for running from a checkout. |
| `tests/test_holding_sim.py` | **Yes** | Has a `HAS_VECTORBT` guard so it skips rather than errors without vectorbt. Harmless once vectorbt is always present; the guard can go. |

> ~~`vectorbtX.py` — the case alias, load-bearing on Linux.~~ **Deleted.** It did not
> survive contact with a real editable install; see §4 item 4. The import name is lowercase.

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

## 4. The networked run — done, and what it found

**Status: the suite has now run against a real vectorbt 1.1.0 install.** 245 passed, 0
failed, 0 skipped, on Python 3.12 / Windows with numba 0.67.0, numpy 2.5.3, pandas 3.0.6.
This section records the outcome so nobody repeats the investigation.

Resolved, and the answers were better than feared:

1. ✅ **The kernels compile.** All three — `order_func_nb`, `pre_segment_func_nb`,
   `post_segment_func_nb` — acquire numba signatures on first call and pass type inference
   with no changes. The feared failures (heterogeneous tuples, `np.where` broadcasts,
   reflected lists) did not materialise. ~8.4s cold, ~0.02s warm.
2. ✅ **No upstream API drift.** Every `vectorbt.*` import site resolves against 1.1.0 —
   including the ones that looked most like 0.x holdovers (`base.reshape_fns`,
   `portfolio.dispatch`, `portfolio.nb.order_nb`) and `Portfolio.from_order_func`, which
   Section G depends on entirely.

Two things it *did* find:

3. ⚠️ **vectorbt 1.1.0 is uninstallable against current plotly.** It declares
   `plotly>=4.12.0` unbounded but registers a template containing `scattermapbox`
   (`vectorbt/_settings.py:96`), removed in plotly 7 — so `import vectorbt` raises
   `ValueError` at import. `pyproject.toml` now pins `plotly<7`. Remove that pin if
   upstream ever caps it.
4. ⚠️ **The `vectorbtX.py` case alias was broken and has been deleted.** setuptools'
   editable-install finder maps *both* `vectorbtX` and `vectorbtx` directly to the package
   directory, so the alias module never executed and the two spellings became two distinct
   module objects — precisely the fork its own docstring warned about. `apply_patches()`
   then ran twice and `attach()` raised `ShadowError` on `ulcer_index`. (The message blamed
   upstream; upstream has no `ulcer_index`. vectorbtX had attached it on the first pass.)
   The import name is now lowercase `vectorbtx` everywhere. The guard earned its keep.

Still outstanding:

5. Re-run all nine numbered examples and the three strategies against live Yahoo and FRED
   data, which will exercise index alignment, missing bars and holiday gaps that the
   synthetic harness does not.
