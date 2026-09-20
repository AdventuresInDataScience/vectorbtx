"""03 — Modes and quantization.

The section that makes vectorbtX useful for derivatives: express a position in the units
you actually trade in, and round it to what a broker would fill.

Run: python examples/03_modes_quantize.py
"""

from _bootstrap import header, x

Instrument = x.Instrument


# --- 1. The cross-mode equivalence test ------------------------------------------------
header("1. The same position, four notations, identical P&L")

# One underlying at 500 (pence), moving 50 points to 505.
ENTRY, EXIT = 500.0, 505.0
TARGET_ENGINE = 100.0          # 100 shares-equivalent of exposure

specs = {
    "spread bet": Instrument(mode="spreadbet", point_size=0.01).filled(),
    "cfd":        Instrument(mode="cfd", point_size=0.01, contract_size=1.0).filled(),
    "equity":     Instrument(mode="equity", point_size=0.01).filled(),
}

print(f"{'mode':12s} {'user size':>12s} {'unit':14s} {'engine':>9s} {'P&L':>9s}")
for label, spec in specs.items():
    adapter = x.get_adapter(spec)
    user = adapter.from_engine_units(TARGET_ENGINE, spec, ENTRY)
    engine = adapter.to_engine_units(user, spec, ENTRY)
    pnl = x.pnl(user, ENTRY, EXIT, spec)
    print(f"{label:12s} {user:12.2f} {adapter.unit_name(spec):14s} {engine:9.2f} {pnl:9.2f}")

print()
print("£1/point on a pence-quoted stock IS 100 shares. The '100 contract size' is")
print("derived from stake / point_size, not configured anywhere.")


# --- 2. Round-tripping -----------------------------------------------------------------
header("2. Every adapter round-trips")

for label, spec in specs.items():
    adapter = x.get_adapter(spec)
    for value in (1.0, 2.5, 137.4829):
        back = adapter.from_engine_units(adapter.to_engine_units(value, spec, ENTRY), spec, ENTRY)
        assert abs(back - value) < 1e-9
    print(f"  {label:12s} from_engine_units(to_engine_units(x)) == x  ✓")


# --- 3. FX spot, and the fractional pip --------------------------------------------------
header("3. FX spot — the 5th decimal must survive")

eurusd = Instrument(mode="fx_spot", point_size=0.0001, contract_size=100_000,
                    quote_currency="USD", base_currency="EUR").filled()
price = 1.08532                       # five decimals, pip at the fourth
print("0.1 lots        ->", x.get_adapter(eurusd).to_engine_units(0.1, eurusd, price), "units")
print("value per pip   ->", x.value_per_point(0.1, eurusd))
print("price retained  ->", price, "(truncating to 1.0853 would discard real spread)")


# --- 4. Point stops become price fractions -------------------------------------------------
header("4. Point stops -> price fractions, for NATIVE sl_stop / tp_stop")

spec = specs["spread bet"]
for points in (20, 50, 100):
    frac = x.points_to_price_fraction(points, ENTRY, spec)
    back = x.price_fraction_to_points(frac, ENTRY, spec)
    print(f"  {points:3d} points at {ENTRY} -> sl_stop={frac:.6f}  (back: {back:.0f} points)")

print()
print("SL/TP is fully native in vectorbt and expressed as a FRACTION of price.")
print("Adapters convert; they do not reimplement stop logic.")
print("  pf = vbt.Portfolio.from_signals(close, entries, exits,")
print("           sl_stop=vbt.x.points_to_price_fraction(50, close, spec))")


# --- 5. Risk-based sizing — one formula, all four modes ---------------------------------------
header("5. Risking £100 over a 50-point stop, in every mode")

RISK, STOP_POINTS = 100.0, 50

for label, spec in specs.items():
    size = x.size_for_risk(RISK, STOP_POINTS, spec, ENTRY)
    stop_price = ENTRY - STOP_POINTS * spec.point_size
    realised = abs(x.pnl(size, ENTRY, stop_price, spec))
    print(f"  {label:12s} size={size:10.4f} {x.get_adapter(spec).unit_name(spec):14s} risk={realised:8.2f}")

print()
print("Identical monetary risk. Position sizing is mode-agnostic — same formula,")
print("different unit conversion — so it is built once, above the adapters.")


# --- 6. Quantization: floor, ceil, nearest -------------------------------------------------
header("6. Rounding to what a broker would actually fill")

shares = specs["equity"]
print("137.4829 shares:")
for direction in ("floor", "ceil", "nearest"):
    q, drift = x.quantize(137.4829, shares, direction)
    print(f"  {direction:8s} -> {q:8.2f}   drift {drift:+.4f}")

print()
print("Floor is the default: conservative, and it matches vectorbt's native")
print("size_granularity behaviour exactly, so it costs nothing.")


# --- 7. Sub-minimum sizes REJECT ------------------------------------------------------------
header("7. Below the minimum: reject, do not silently round to zero")

sb = specs["spread bet"]
cfd = specs["cfd"]
print(f"spread bet, £0.437/pt (min £1)   -> {x.quantize(0.437, sb)}")
print(f"cfd, 0.004 lots     (min 0.01)   -> {x.quantize(0.004, cfd)}")
print()
print("Note the drift is the FULL intended size, not zero. The order was rejected,")
print("not rounded away — and the report says so.")


# --- 8. Handing quantization to the engine ---------------------------------------------------
header("8. to_order_params — let the engine floor, quantize ourselves otherwise")

print("floor  ->", x.to_order_params(shares, "floor"))
print("ceil   ->", x.to_order_params(shares, "ceil"))
print()
print("For non-floor rounding we quantize BEFORE handing over and pass")
print("size_granularity=NaN. Do not try to make the engine round differently.")


# --- 9. A backtest that lies ------------------------------------------------------------------
header("9. Why this matters in a small account")

equity = 2_137.0
intended = equity * 0.05 / (ENTRY / 100)   # 5% position, in shares (price in pounds)
q, drift = x.quantize(intended, shares)
print(f"intended {intended:.4f} shares -> filled {q:.0f}, drift {drift:.4f}")
print(f"that is {drift / intended:.2%} of the intended position, unfilled")
print()
print("A backtest assuming 137.4829 shares is lying, and the lie compounds.")
