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
| `examples/_bootstrap.py` | **Yes — keep the fallback** | See the correction below. |
| `tests/test_holding_sim.py` | **Yes — keep the guard** | See the correction below. |

> ⚠️ **Correction.** This table previously said the `HAS_VBT`/`needs_vbt` branches in
> `_bootstrap.py` were "dead once vectorbt is a hard dependency", and that
> `test_holding_sim.py`'s `HAS_VECTORBT` guard "can go". **Both claims are wrong**, and
> acting on them would delete a working feature.
>
> Measured with `vectorbt` made unimportable: **all nine numbered examples still run**
> (exit 0), and the suite reports **217 passed, 39 skipped, 0 failed**. The guards are what
> deliver design goal §1.4 — the pure-arithmetic layer is importable and testable on its
> own — and they are doing it today, not hypothetically.
>
> Only two things really were dead, and both are now gone: `_bootstrap.needs_vbt()` (never
> called) and an unused `HAS_VBT` import in `07_holding_rates.py`.
>
> One gotcha when reproducing this: `@njit(cache=True)` leaves `.nbi`/`.nbc` files in
> `vectorbtx/holding/__pycache__/` whose index pickle references vectorbt types. Loading
> that cache re-imports vectorbt and fails confusingly. Delete the numba cache first.

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

**Status: the suite has now run against a real vectorbt 1.1.0 install**, and against live
Yahoo and live FRED. **256 passed, 0 failed, 0 skipped** on Python 3.12 / Windows with
numba 0.67.0, numpy 2.5.3, pandas 3.0.6, plotly 6.9.0. This section records the outcome so
nobody repeats the investigation.

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

5. ✅ **The live-data run is done, and it was worth doing.** All four strategies now run
   against live Yahoo (`4/4`). They did not before: **3 of the 4 were broken on real data
   and green under the harness.** FRED still needs an API key, so rates used the
   documented static fallback — the index-alignment risk lives in the price data, which
   was fully live.

   What only live data could surface:

   | Found | Fix |
   |---|---|
   | `Data.get("Close")` on a **single** symbol returns a Series named after the *field*, not the symbol. `as_fx_table` then could not identify the currency pair. | `fx/convert.py` now carries `Data.symbols` across the `.get()` |
   | Yahoo stamps daily bars at each market's **local** midnight (LSE 00:00 UTC, US 05:00 UTC), so a mixed-exchange `missing_index="drop"` intersects to **zero rows** with all columns intact | examples normalise to dates first; `resolve_universe` now rejects an empty frame by name instead of `IndexError: index 0 is out of bounds` from `costs/compose.py` |
   | The explicit route **caps leverage but never funds it**, so `appendix_explicit.py` placed *zero* orders and reported 0.00% as if it were a result — the exact bug `margin/buying_power.py` exists to fix | the appendix now calls `vbt.x.buying_power` and records the loan, as the front door does |
   | `pf.x.min_margin_level()` on a portfolio with no universe died as `TypeError: argument of type 'NoneType' is not iterable` | `margin_level` raises a message naming the argument and the fix |
   | **pandas 3.0 removed `Series.__float__`** — `float(one_element_series)` now raises | `.item()` in the appendix |

   ⚠️ **The root cause of three of those was an unfaithful stub.** Both harnesses returned
   a single-symbol Series named after the *symbol*, which real vectorbt never does. A stub
   more generous than the thing it replaces does not merely fail to catch bugs — it
   actively manufactures confidence. Both are now faithful, and they reproduce these
   failures offline.

6. ✅ **The live FRED run is done too, and it found the worst bug of the lot.** All four
   strategies now run against **live Yahoo and live FRED** together.

   | Found | Fix |
   |---|---|
   | ⚠️ **Yahoo is tz-aware, FRED is tz-naive, and their indexes share _zero_ timestamps.** Reindexing rates onto price bars gave NaN on every bar; `ffill`/`bfill` had nothing to carry. NaN rates → NaN carry → NaN equity — and **nothing raised**. `pf.stats()` still printed a believable −2.25% with the entire cost model contributing zero. | `holding/callback._rate_for` now refuses an all-NaN alignment and names the tz mismatch. Three tests in `test_holding.py` |
   | The EUR series ID `ECBESTRVOLWGTTRMDRT` does not exist at FRED (missing `MN`), so every EUR backtest died on HTTP 400 | corrected to `ECBESTRVOLWGTTRMDMNRT`, verified live |
   | ⚠️ **The API key was printed in tracebacks.** `raise_for_status()` embeds the request URL, `api_key=` included, so the key reached any log or bug report | `_fetch` reports FRED's own error text and never lets the URL surface |
   | `FREDData.download("SOFR")` iterated the **string**, requesting a series called `"S"`. `YFData.download("MSFT")` works, because upstream normalises and these overrides did not | `_as_symbol_list` |
   | `TONA` and `SARON` map to OECD **monthly** series, both **discontinued** (2023-12, 2024-03). Forward-filled, they charge recent backtests a stale constant | warn when a series ends before the backtest does; explain it when the window is empty |

   FRED has no daily JPY or CHF overnight benchmark — checked against the live search API,
   zero daily hits. The monthly series are the best available, so the fix is to say so
   loudly rather than to pretend otherwise.

Environment notes — not bugs in this library, but they cost a day to diagnose:

7. ⚠️ **HTTPS-scanning antivirus breaks the data providers silently.** On the machine this
   was run on, Avast Web/Mail Shield re-signs every TLS connection with its own root CA.
   Measured: the same request fails against `certifi`'s bundle and succeeds against the
   Windows store, whose issuer reads `Avast Web/Mail Shield`. `yfinance` does not raise —
   it logs `CertificateVerifyError`, then `possibly delisted; no price data found`, and
   returns an **empty frame**, so a backtest built on it runs and reports nonsense.

   `truststore` is not the fix: it patches Python's `ssl`, and `yfinance` ≥ 1.7 downloads
   via `curl_cffi`, which bypasses `ssl`. See README §"TLS interception and the data
   providers" for the bundle recipe and the three environment variables.

8. ⚠️ **uv's managed CPython cannot do Python-`ssl` networking on this machine.** Any
   `requests` call died with `OPENSSL_Uplink(...): no OPENSSL_Applink` — a
   `python-build-standalone` fault, not a certificate problem: it happened with no CA
   environment variables set at all. `uv` itself (Rust TLS) and `yfinance` (`curl_cffi`)
   were unaffected, which is why it only surfaced once FRED — which uses `requests` — was
   brought in.

   Fix: build the venv on a python.org interpreter rather than uv's managed one.

   ```bash
   uv venv --python "$(which python)" .venv     # or an explicit path to python.exe
   ```
