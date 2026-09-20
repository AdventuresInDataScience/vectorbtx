"""02 — Describing a whole universe, from a dict or a spreadsheet.

The mechanism (per-column broadcasting) already exists in vectorbt. This is ergonomics:
a safe way to *express* a universe, because hand-building parallel lists across 20 assets
means keeping five lists aligned in column order, and one misordered list attaches CFD
costs to an equity column while nothing errors.

Run: python examples/02_universe.py
"""

import pandas as pd

from _bootstrap import header, x

Instrument = x.Instrument


# --- 1. The dict front door ----------------------------------------------------------
header("1. A universe from a dict")

universe = x.load_universe({
    "UKX": Instrument(mode="spreadbet", point_size=1.0, asset_class="index"),
    "VOD": Instrument(mode="spreadbet", point_size=0.01, asset_class="stocks"),
    "MSFT": Instrument(mode="equity", point_size=0.01, quote_currency="USD",
                       cost_currency="USD", commission_flat=10.0),
})
print("columns:", universe.columns)
print("len    :", len(universe))
print("MSFT   :", universe["MSFT"].mode, universe["MSFT"].quote_currency)


# --- 2. Resolving to Order kwargs -----------------------------------------------------
header("2. Universe.resolve -> Order kwargs, ready to splat")

index = pd.date_range("2024-01-01", periods=10, freq="D")
close = pd.DataFrame(
    {"UKX": 7500.0, "VOD": 75.0, "MSFT": 400.0}, index=index
)

args = universe.resolve(close)
for key, value in args.items():
    shown = value if not hasattr(value, "shape") else f"<{type(value).__name__} {value.shape}>"
    print(f"  {key:18s} {shown}")

print()
print("Feed straight into any vectorbt constructor:")
print("  vbt.Portfolio.from_orders(close, size=size, **universe.resolve(close))")


# --- 3. Column order comes from the frame, not the dict --------------------------------
header("3. Column order comes from the WRAPPER, not dict insertion order")

reordered = close[["MSFT", "UKX", "VOD"]]
print("frame order :", list(reordered.columns))
print("specs order :", [s.mode for s in universe.order_for(reordered.columns)])
print("Resolution follows the frame. A misordered spreadsheet still resolves correctly.")


# --- 4. Mismatches are fatal in BOTH directions ----------------------------------------
header("4. A missing or extra instrument name raises — both directions")

from vectorbtx.resolve.load import UniverseError

try:
    universe.resolve(close[["UKX", "VOD"]].assign(TSLA=200.0))
except (UniverseError, KeyError) as exc:
    print("price column without a spec ->", type(exc).__name__, ":", exc)

print()
print("A warning would scroll past on a 20-asset backtest and one column would")
print("silently get defaults. So: error, both directions.")


# --- 5. The spreadsheet front door ------------------------------------------------------
header("5. The same universe from a CSV")

frame = universe.to_frame()
print("Exported schema has", len(frame.columns), "columns. First few:")
print(frame[["mode", "point_size", "increment", "min_size", "leverage"]].to_string())

import tempfile, pathlib

path = pathlib.Path(tempfile.mkdtemp()) / "universe.csv"
universe.to_csv(path)
reloaded = x.load_universe(path)
print()
print("round-trips equal:", reloaded == universe)
print("Both front doors converge BEFORE validation, so the spreadsheet path — precisely")
print("where bad data enters — cannot bypass any of Section C's checks.")


# --- 6. Strict, loud coercion -----------------------------------------------------------
header("6. Coercion is strict and loud")

from vectorbtx.resolve.coerce import coerce_bool, coerce_float, coerce_name, CoercionError

print("'FALSE' ->", coerce_bool("FALSE", "allow_fractional"))
print("'no'    ->", coerce_bool("no", "allow_fractional"))
print("'0.01'  ->", coerce_float("0.01", "increment"))
print("'1,000' ->", coerce_float("1,000", "adv"))
print("1234.0  ->", repr(coerce_name(1234.0)), "(a numeric ticker must not become '1234.0')")
try:
    coerce_bool("maybe", "allow_fractional", "row 3 (VOD)")
except CoercionError as exc:
    print("'maybe' ->", exc)


# --- 7. Blank is not NaN ------------------------------------------------------------------
header("7. A blank cell is NOT NaN")

from vectorbtx.resolve.coerce import coerce_field

print("blank increment coerces to:", repr(coerce_field("increment", "")))
print()
print("vectorbt uses NaN and inf MEANINGFULLY in Order fields:")
print("  size_granularity = NaN  means 'no quantization'")
print("  price            = inf  means 'use close'")
print("So a blank becoming NaN could accidentally *mean* something. Blanks become None,")
print("resolve against mode defaults, and become NaN only at emit.")


# --- 8. A typo'd keyword is fatal ----------------------------------------------------------
header("8. A typo'd keyword raises, naming the cell")

from vectorbtx.resolve.keywords import KeywordError, KEYWORDS

print("keyword registry:", dict(KEYWORDS))
print()
print("'auto' slippage  ->", coerce_field("slippage", "auto"))
print("0.002 slippage   ->", coerce_field("slippage", "0.002"))
try:
    coerce_field("slippage", "atuo", "row 7 (VOD)")
except KeywordError as exc:
    print("'atuo'           ->", exc)
print()
print("Numeric first, then the keyword set, then error. A typo falling through to a")
print("default is the exact silent-wrong-number failure this design exists to prevent.")
