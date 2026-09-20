"""01 — Describing instruments.

Ten examples covering all four modes, defaults, and every validation rule that fires.
Run: python examples/01_instruments.py
"""

from _bootstrap import header, x

Instrument = x.Instrument


# --- 1. A UK spread bet on a stock quoted in pence ---------------------------------
header("1. Spread bet — the stake IS the size")

uk = Instrument(mode="spreadbet", point_size=0.01, asset_class="stocks")
print("point_size      :", uk.point_size)
print("contract_size   :", uk.contract_size, "(spread bets have none — by design)")
print("£1/pt in engine :", x.get_adapter(uk).to_engine_units(1.0, uk, 500.0), "shares-equivalent")
print("...the '100 contract size' for a pence-quoted UK stock is derived, not configured.")


# --- 2. A CFD ----------------------------------------------------------------------
header("2. CFD — contract_size alone describes any CFD")

cfd = Instrument(mode="cfd", point_size=0.01, contract_size=1.0, asset_class="stocks")
print("1 lot in engine units:", x.get_adapter(cfd).to_engine_units(1.0, cfd, 100.0))
print("min lot (filled)     :", cfd.filled().min_size)


# --- 3. FX spot, with fractional pips ----------------------------------------------
header("3. FX spot — 5-decimal quotes, pip at the 4th")

eurusd = Instrument(
    mode="fx_spot", point_size=0.0001, contract_size=100_000,
    quote_currency="USD", base_currency="EUR", asset_class="fx",
)
print("point_size (the pip) :", eurusd.point_size)
print("0.1 lots in units    :", x.get_adapter(eurusd).to_engine_units(0.1, eurusd, 1.08532))
print("price 1.08532 keeps its 5th digit — a real part of the spread lives there.")


# --- 4. Equity ----------------------------------------------------------------------
header("4. Equity — the model vectorbt natively assumes")

msft = Instrument(
    mode="equity", point_size=0.01, quote_currency="USD", cost_currency="USD",
    commission_flat=10.0,
)
filled = msft.filled()
print("increment   :", filled.increment, "(whole shares)")
print("price_field :", filled.price_field, "(adjusted close is correct for equity)")
print("max leverage:", filled.leverage)


# --- 5. What .filled() actually does -------------------------------------------------
header("5. Defaults are resolved by .filled(), not at construction")

bare = Instrument(mode="spreadbet", point_size=1.0)
print("before  increment:", bare.increment, " min_size:", bare.min_size)
f = bare.filled()
print("after   increment:", f.increment, " min_size:", f.min_size,
      " leverage:", f.leverage, " costs:", sorted(f.active_costs))
print("None means UNSPECIFIED throughout; NaN appears only at emit time.")


# --- 6. price_field differs by mode, and it matters ----------------------------------
header("6. price_field — adjusted close is WRONG for nominal-point modes")

print("equity    ->", Instrument(mode="equity", point_size=0.01).filled().price_field)
print("spreadbet ->", Instrument(mode="spreadbet", point_size=1.0).filled().price_field)
print("cfd       ->", Instrument(mode='cfd', point_size=1.0, contract_size=1.0).filled().price_field)
print()
print("Adjusting rescales the whole history on each dividend, so the number of POINTS")
print("between two dates changes depending on when you pulled the data. £1/point on")
print("adjusted data does not produce £1/point of real P&L.")


# --- 7..10. Every validation rule that fires -----------------------------------------
header("7-10. Validation rejects on construction")

from vectorbtx.instruments.validate import SpecError

cases = [
    ("contract_size on a spread bet",
     lambda: Instrument(mode="spreadbet", point_size=1.0, contract_size=100.0)),
    ("missing contract_size on a CFD",
     lambda: Instrument(mode="cfd", point_size=0.01)),
    ("increment <= 0",
     lambda: Instrument(mode="equity", point_size=0.01, increment=0.0)),
    ("min_size not a multiple of increment",
     lambda: Instrument(mode="equity", point_size=0.01, increment=1.0, min_size=0.5)),
]
for label, make in cases:
    try:
        make()
        print(f"  {label:38s} -> NO ERROR (unexpected!)")
    except SpecError as exc:
        print(f"  {label:38s} -> SpecError: {exc}")

print()
print("Leverage above the mode default WARNS rather than raises — brokers differ, and")
print("the Part 8 maxima are defaults, not laws:")
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    Instrument(mode="equity", point_size=0.01, leverage=50.0)
    print("  ->", caught[0].message)

print()
print("The cross-spec rule — portfolio_level with DIFFERING per-ticker leverage — is")
print("checked at the universe level, because one spec cannot see its siblings:")
try:
    x.load_universe({
        "A": Instrument(mode="cfd", point_size=1.0, contract_size=1.0,
                        leverage=2.0, leverage_type="portfolio_level"),
        "B": Instrument(mode="cfd", point_size=1.0, contract_size=1.0,
                        leverage=5.0, leverage_type="portfolio_level"),
    })
    print("  -> NO ERROR (unexpected!)")
except SpecError as exc:
    print("  -> SpecError:", exc)
