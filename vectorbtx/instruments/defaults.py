"""Section C — per-mode defaults.

Everything here is a *chosen* default from Part 8 of the plan, overridable per
instrument. Nothing in this module makes a decision at runtime; it is a lookup table with
one filling function.
"""

from __future__ import annotations

from dataclasses import fields, replace

MODES = ("spreadbet", "cfd", "fx_spot", "equity")
ASSET_CLASSES = ("stocks", "commodities", "fx", "index")

#: Sizing and quantization, per Part 8.
#:
#: * Spread bet — £1/point minimum (£0.10 at micro brokers), £0.01 increment, and
#:   **no contract size**: the stake is the size.
#: * CFD — 0.01 lots minimum (0.10 at some brokers); `contract_size` describes any CFD.
#: * FX spot — standard lot is 100,000 units of base; the pip sits at the 4th decimal
#:   while quotes carry 5.
#: * Equity — whole shares, no fractional by default.
MODE_DEFAULTS = {
    "spreadbet": dict(
        asset_class="stocks",
        point_size=0.01,
        contract_size=None,
        increment=0.01,
        min_size=1.0,
        allow_fractional=True,
        price_field="close",
        commission_pct=0.0,
        commission_flat=0.0,
        spread_points=1.0,
        triple_day="FRI",
    ),
    "cfd": dict(
        asset_class="stocks",
        point_size=0.01,
        contract_size=1.0,
        increment=0.01,
        min_size=0.01,
        allow_fractional=True,
        price_field="close",
        commission_pct=0.0,
        spread_points=1.0,
        triple_day="FRI",
    ),
    "fx_spot": dict(
        asset_class="fx",
        point_size=0.0001,
        contract_size=100_000.0,
        increment=0.01,
        min_size=0.01,
        allow_fractional=True,
        price_field="close",
        commission_pct=0.0,
        spread_points=1.0,
        triple_day="WED",
    ),
    "equity": dict(
        asset_class="stocks",
        point_size=0.01,
        contract_size=None,
        increment=1.0,
        min_size=1.0,
        allow_fractional=False,
        price_field="adj_close",
        commission_pct=0.0,
        spread_points=1.0,
        triple_day="FRI",
    ),
}

#: Quantization step for brokers offering fractional shares. Small enough to be
#: effectively continuous, large enough to keep floating-point rounding well behaved.
FRACTIONAL_INCREMENT = 1e-6

#: Maximum leverage, per Part 8. Tiered margin is deliberately not modelled — users set a
#: sensible maximum instead.
MAX_LEVERAGE = {
    ("equity", None): 2.0,
    ("fx_spot", None): 2.0,
    ("spreadbet", "stocks"): 5.0,
    ("spreadbet", "commodities"): 10.0,
    ("spreadbet", "index"): 10.0,
    ("spreadbet", "fx"): 30.0,
    ("cfd", "stocks"): 5.0,
    ("cfd", "commodities"): 10.0,
    ("cfd", "index"): 10.0,
    ("cfd", "fx"): 30.0,
}

#: Pip size for an FX pair, keyed on the **quote** currency (the second of the pair).
#:
#: The pip is a market convention, not a per-broker setting: it is the fourth decimal of the
#: quote for almost every pair, and the **second** for JPY-quoted ones, because a yen is worth
#: roughly two orders of magnitude less than a dollar. USDJPY at 150.25 moves one pip to
#: 150.26; EURUSD at 1.0925 moves one pip to 1.0926.
#:
#: ⚠️ Getting this wrong is silent and large. Using 0.0001 on USDJPY makes a 30-pip stop
#: 0.30 points wide instead of 30 — a stop 100x tighter than intended, which still runs,
#: still produces trades, and reports a risk figure that is wrong by two orders of magnitude.
#: That is why it is derived rather than left to the user to remember per pair.
FX_PIP_BY_QUOTE = {"JPY": 0.01, "KRW": 0.01, "HUF": 0.01, "CLP": 0.01, "IDR": 0.01}

#: Every other quote currency.
FX_PIP_DEFAULT = 0.0001


#: Which weekday carries the weekend roll, by asset class.
#:
#: ⚠️ **Two conventions, for two different reasons.** The plan set Wednesday globally.
#:
#: * **FX and spot metals — Wednesday.** Spot FX settles T+2, so a position rolled through
#:   Wednesday night settles the following Monday and is charged three days' carry. This
#:   is a *settlement* rule.
#: * **Shares, indices, commodities — Friday.** Nothing settles; the triple charge simply
#:   covers the calendar weekend, so it lands on the last trading day before it.
#:
#: Weekly totals are identical either way (seven funded days — see `day_multipliers`), so
#: this does not move a long backtest's aggregate cost much. It moves *per-trade* cost a
#: lot: a share position held Thursday to Friday is charged 1x under the Wednesday
#: convention and 3x under the correct one.
TRIPLE_DAY = {
    "fx": "WED",
    "stocks": "FRI",
    "index": "FRI",
    "commodities": "FRI",
}


#: What leverage financing is charged **on**, per mode.
#:
#: ⚠️ These are two genuinely different products and the plan conflated them.
#:
#: * **equity** — margin share dealing. You own the shares and borrow the shortfall, so
#:   interest is charged on the **debit balance**: a position fully covered by cash
#:   finances nothing. This is the rule §2.2 of the plan states, and it is correct here
#:   and only here.
#: * **spreadbet / cfd** — you own nothing and borrow nothing. The broker funds the
#:   **entire** position and you post margin as collateral, so financing is charged on
#:   the **full notional** every night it is held. Confirmed against IG, Capital.com and
#:   several independent references: "calculated on the full notional value of the
#:   position, not your margin".
#: * **fx_spot** — financing is not in the default stack at all; the cost of holding an
#:   FX position is carry, which already applies to the full notional regardless of
#:   leverage.
#:
#: Getting this wrong understates derivative financing by roughly `1 - 1/leverage` — at
#: 5x, four fifths of the true cost — and zeroes it entirely for an unlevered position
#: that a broker would still charge in full.
FINANCING_BASIS = {
    "equity": "borrowed",
    "spreadbet": "notional",
    "cfd": "notional",
    "fx_spot": "notional",
}


#: Default cost stack per mode and asset class (Section E).
#:
#: Spread bets carry no commission — the fee is baked into a wider spread. Carry applies
#: to anything with a currency pair underneath it, regardless of leverage.
#: ⚠️ **`carry` and `financing` are mutually exclusive.** An FX-underlying position has
#: exactly one overnight charge and it *is* the tom-next swap, which already embeds both
#: currencies' interest rates — that is what makes it a differential. Adding a
#: leverage-driven financing charge on top bills the same money twice. Everything with a
#: non-FX underlying works the other way: an interbank rate plus the broker's admin fee,
#: and no carry. `validate_spec` enforces the exclusion so it cannot be reintroduced by
#: hand.
#:
#: **`borrow` — the borrow fee on a short — is in the `equity` stack, and only there.**
#: Which positions pay it depends on how the instrument is traded, not on what it is.
#: Shorting something you trade directly (`equity`: a share, ETF, fund or commodity
#: product) means borrowing that security, and the lender charges for it. A CFD or spread
#: bet borrows nothing: its short side already pays (or earns) funding at
#: `benchmark - markup`, which is the whole cost of holding it, so a borrow fee there
#: would charge twice. (An index can only be traded through a CFD or spread bet, so it
#: never pays one.) The rate is the instrument's `borrow_rate`; with none set, nothing is
#: charged and a held short warns (G6: there is no default worth trusting).
DEFAULT_STACKS = {
    ("equity", None): ("commission", "spread", "slippage", "financing", "borrow"),
    ("fx_spot", None): ("commission", "spread", "slippage", "carry"),
    ("cfd", "fx"): ("commission", "spread", "slippage", "carry"),
    ("cfd", None): ("commission", "spread", "slippage", "financing"),
    ("spreadbet", "fx"): ("spread", "slippage", "carry"),
    ("spreadbet", None): ("spread", "slippage", "financing"),
}


def _lookup(table, mode: str, asset_class):
    """Look up `(mode, asset_class)`.

    An unspecified `asset_class` resolves to the mode's own default (equity and FX spot
    have none; spread bets and CFDs default to stocks) before falling back to the
    mode-wide `(mode, None)` entry.
    """
    if mode not in MODE_DEFAULTS:
        raise ValueError(f"Unknown mode {mode!r}. Expected one of {MODES}.")
    if (mode, asset_class) in table:
        return table[(mode, asset_class)]
    if asset_class is None:
        asset_class = MODE_DEFAULTS[mode].get("asset_class")
        if (mode, asset_class) in table:
            return table[(mode, asset_class)]
    if (mode, None) in table:
        return table[(mode, None)]
    raise KeyError(f"No entry for mode={mode!r}, asset_class={asset_class!r}")


def mode_defaults(mode: str) -> dict:
    """Default field values for a mode."""
    if mode not in MODE_DEFAULTS:
        raise ValueError(f"Unknown mode {mode!r}. Expected one of {MODES}.")
    return dict(MODE_DEFAULTS[mode])


def max_leverage(mode: str, asset_class=None) -> float:
    """Default maximum leverage for a mode and asset class."""
    return _lookup(MAX_LEVERAGE, mode, asset_class)


def fx_pip(quote_currency=None) -> float:
    """The pip size for a pair, from its quote currency. See `FX_PIP_BY_QUOTE`."""
    if quote_currency is None:
        return FX_PIP_DEFAULT
    return FX_PIP_BY_QUOTE.get(str(quote_currency).upper(), FX_PIP_DEFAULT)


def triple_day(asset_class=None, mode: str = None) -> str:
    """Which weekday carries the weekend roll. See `TRIPLE_DAY`."""
    if asset_class is None and mode is not None:
        asset_class = MODE_DEFAULTS.get(mode, {}).get("asset_class")
    return TRIPLE_DAY.get(asset_class, "FRI")


def financing_basis(mode: str) -> str:
    """What financing is charged on for a mode: `"borrowed"` or `"notional"`.

    See `FINANCING_BASIS` for why the two differ and why it matters.
    """
    if mode not in FINANCING_BASIS:
        raise ValueError(f"Unknown mode {mode!r}. Expected one of {MODES}.")
    return FINANCING_BASIS[mode]


def default_cost_stack(mode: str, asset_class=None) -> frozenset:
    """Default set of active cost modules for a mode and asset class."""
    return frozenset(_lookup(DEFAULT_STACKS, mode, asset_class))


def fill_defaults(spec):
    """Return a copy of `spec` with every unspecified field resolved.

    `None` means *unspecified* throughout vectorbtX and is only converted to NaN at emit
    time. That distinction matters: vectorbt uses NaN and inf meaningfully in `Order`
    fields (`size_granularity=NaN` means no quantization, `price=inf` means use close),
    so a blank cell becoming NaN could accidentally *mean* something.
    """
    d = mode_defaults(spec.mode)
    changes = {}

    asset_class = spec.asset_class or d["asset_class"]
    changes["asset_class"] = asset_class

    for name in ("point_size", "increment", "min_size", "price_field"):
        if getattr(spec, name) is None:
            changes[name] = d[name]

    # An FX pair's pip follows its QUOTE currency, so it need not be stated per instrument.
    # This is what lets a mixed book — EURUSD alongside USDJPY — share one set of terms.
    if spec.point_size is None and asset_class == "fx" and spec.quote_currency is not None:
        changes["point_size"] = fx_pip(spec.quote_currency)

    # The triple day follows the ASSET CLASS, not the mode: an FX spread bet rolls on
    # Wednesday (T+2 settlement) while an index spread bet rolls on Friday (calendar
    # weekend). Resolving it from the mode's default asset class alone would give every
    # FX spread bet the equity convention.
    if spec.triple_day is None:
        changes["triple_day"] = triple_day(asset_class)

    # `allow_fractional` has to change the fill increment or it does nothing at all.
    # Whole shares are the default; a broker offering fractional shares quantizes to a
    # small step instead, and its minimum drops with it.
    if spec.allow_fractional and spec.mode == "equity":
        if spec.increment is None:
            changes["increment"] = FRACTIONAL_INCREMENT
        if spec.min_size is None:
            changes["min_size"] = changes.get("increment", spec.increment)

    if spec.contract_size is None and d["contract_size"] is not None:
        changes["contract_size"] = d["contract_size"]

    for name in ("commission_pct", "commission_flat", "spread_points"):
        if getattr(spec, name) is None and name in d:
            changes[name] = d[name]

    if spec.cost_currency is None:
        changes["cost_currency"] = spec.quote_currency

    if spec.leverage is None:
        changes["leverage"] = max_leverage(spec.mode, asset_class)

    if not spec.active_costs:
        changes["active_costs"] = default_cost_stack(spec.mode, asset_class)

    # Financing is the backtest's benchmark (`rates=`) plus `financing_markup=`; the
    # per-instrument field is reserved (see `Instrument.financing`).
    if spec.financing is None:
        changes["financing"] = "auto"

    if spec.rounding is None:
        from ..core.settings import get

        changes["rounding"] = get("rounding", "floor")

    if spec.slippage_max is None:
        from ..core.settings import get

        changes["slippage_max"] = get("slippage_max", 0.005)
    if spec.slippage_coef is None:
        from ..core.settings import get

        changes["slippage_coef"] = get("slippage_coef", 1.0)

    return replace(spec, **changes)


def field_names() -> tuple:
    """Every `Instrument` field name, in declaration order. Used by the CSV schema."""
    from .spec import Instrument

    return tuple(f.name for f in fields(Instrument))
