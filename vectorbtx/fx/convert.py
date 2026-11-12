"""Section I — base-currency pre-conversion (D4).

Foreign-currency price series are converted **before** simulation, as a data-preparation
step. There is no intra-simulation currency handling.

This is exact, not an approximation. GBP P&L for a GBP investor holding MSFT is

    shares x (P_usd[t1] x fx[t1] - P_usd[t0] x fx[t0])

Pre-converting and simulating gives `shares x (P_gbp[t1] - P_gbp[t0])` — the same
expression. Cash is base currency throughout, so cash sharing works, cross-currency
rebalancing works, and FX moves correctly affect buying power mid-backtest, without
touching the engine.

What is lost: the converted series conflates asset return with FX return. Correct for
total return; decomposition needs the original, which is retained as `.native`.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

OHLC_FIELDS = ("open", "high", "low", "close", "adj_close", "adj close")


class ConvertedFrame(pd.DataFrame):
    """A converted price frame that remembers what it was converted from.

    `.native` holds the original quote-currency series, `.fx_rate` the rate used, and
    `.fx_markup` the broker markup that Section E folds into `slippage`.
    """

    _metadata = ["native", "fx_rate", "fx_markup", "base_currency"]

    @property
    def _constructor(self):
        return ConvertedFrame

    def decompose(self) -> dict:
        """Split total return into asset return and FX return.

        Returns:
            `{"total": ..., "asset": ..., "fx": ...}` of simple returns.
        """
        total = self.pct_change()
        asset = self.native.pct_change()
        fx = pd.DataFrame(self.fx_rate).pct_change()
        return {"total": total, "asset": asset, "fx": fx}


def _ohlc_columns(frame: pd.DataFrame) -> list:
    return [c for c in frame.columns if str(c).strip().lower() in OHLC_FIELDS]


def as_fx_table(fx, quote=None, base=None, index=None):
    """Coerce whatever the user has into an `FXTable`.

    Users almost never have an `FXTable`; they have what a data provider gave them — a
    `GBPUSD=X` close series, or a small dict of flat rates. Requiring the table to be
    hand-built first is the difference between a two-line multi-currency setup and a
    twenty-line one.

    Args:
        fx: an `FXTable` (returned unchanged), a `pd.Series` of one pair's rate, a
            single-column frame, a vectorbt `Data`, or a `dict[currency, rate_to_hub]`.
        quote: for a series — the currency the rate is quoted *in terms of one unit of*.
        base: for a series — the currency one unit of `quote` buys.
        index: index to build a constant table against, for the dict form.

    Returns:
        `FXTable`.

    Notes:
        A series `GBPUSD = 1.27` means one GBP buys 1.27 USD, i.e. `quote="GBP"`,
        `base="USD"`. That is the canonical direction; every reciprocal and cross is then
        **derived**, never fetched, so `value(A,B) x value(B,A) = 1` holds exactly and no
        round trip can manufacture phantom P&L.
    """
    from .table import FXTable

    if fx is None or isinstance(fx, FXTable):
        return fx

    if isinstance(fx, dict):
        if index is None:
            raise ValueError("Building an FXTable from a dict of rates needs an `index`.")
        return FXTable.constant(fx, index)

    if hasattr(fx, "get") and not isinstance(fx, (pd.Series, pd.DataFrame)):
        # ⚠️ `Data.get("Close")` names a SINGLE-symbol result after the field, not after
        # the symbol — so the one thing needed to infer the pair is dropped exactly here,
        # and the user sees "Cannot tell which currencies 'Close' refers to" for a
        # `GBPUSD=X` download that plainly says which. The Data object still knows its
        # symbol; carry it across. Multi-symbol results already come back symbol-named.
        symbols = getattr(fx, "symbols", None)
        fx = fx.get("Close")
        if isinstance(fx, pd.Series) and symbols is not None and len(symbols) == 1:
            fx = fx.rename(symbols[0])

    frame = fx.to_frame() if isinstance(fx, pd.Series) else pd.DataFrame(fx)
    if frame.shape[1] != 1:
        raise ValueError(
            f"Cannot infer a currency pair from a {frame.shape[1]}-column frame. Pass an "
            f"FXTable built with FXTable.from_canonical for multi-pair data."
        )
    series = frame.iloc[:, 0]

    if quote is None or base is None:
        guessed = _guess_pair(series.name if series.name is not None else frame.columns[0])
        if guessed is None:
            raise ValueError(
                f"Cannot tell which currencies {series.name!r} refers to. Pass "
                f"quote=... and base=..., e.g. as_fx_table(s, quote='GBP', base='USD') "
                f"for a GBPUSD series."
            )
        quote, base = guessed

    canonical = pd.DataFrame({quote: series.astype(float)}, index=series.index)
    return FXTable.from_canonical(canonical, hub=base)


def _guess_pair(name):
    """Pull a currency pair out of a ticker like `GBPUSD=X` or `EURUSD`."""
    text = "".join(ch for ch in str(name).upper() if ch.isalpha())
    if len(text) >= 6:
        head, tail = text[:3], text[3:6]
        if head.isalpha() and tail.isalpha():
            return head, tail
    return None


def to_base_currency(
    data,
    base: str,
    fx_table=None,
    quote_currency=None,
    instruments=None,
    fx_markup=None,
    fill_policy: str = "ffill_with_warning",
    validate: bool = True,
    fx=None,
):
    """Convert a price series or OHLC frame to `base` currency.

    Args:
        data: `pd.Series`, `pd.DataFrame` (OHLC or one column per asset), or a vectorbt
            `Data` object.
        base: the account currency.
        fx_table: an `FXTable`, or anything `as_fx_table` accepts — a rate series, a
            single-column frame, or a dict of flat rates.
        quote_currency: the currency `data` is priced in, or a `dict[column, currency]`.
        instruments: a `Universe`, a `dict[column, Instrument]`, or a `dict[column, str]`.
            An alternative to `quote_currency`: the universe already knows which columns
            are foreign, so restating it invites the two to disagree.
        fx_markup: the broker's worse-than-market conversion rate. None (the default)
            reads `settings["vectorbtx"]["fx_markup"]` (0.5% unless changed). Returned on
            `.fx_markup`, which the front door reads off the `open` frame and charges as
            slippage on every fill of a converted column — see Notes.
        fill_policy: how to align the rate series to the price index.
        validate: check that OHLC ordering survives conversion.
        fx: alias for `fx_table`, matching the front door's argument name.

    Returns:
        `ConvertedFrame` with `.native`, `.fx_rate`, `.fx_markup` and `.base_currency`.
        A Series input returns a `pd.Series`, which carries `fx_markup` and
        `base_currency` in its `.attrs` instead.

    Notes:
        **Quote-currency information arrives in one shape.** `quote_currency` and
        `instruments` are normalised to the same `dict[column, currency]` internally, so
        there is one signature rather than three overlapping ones.

        **One FX rate per bar, applied to all of OHLC.** Different rates for open, high,
        low and close break the `H >= C >= L` invariant, and native SL/TP then fires at
        prices that never existed. The same failure mode as D10's mixed-adjustment
        hazard, and the same validation check catches both.

        **The markup is a transaction cost, not a level shift.** Scaling a whole price
        series by `(1 - markup)` would cancel out of every return and cost nothing.
        Converting in at a worse rate and out at a worse rate is exactly a spread, so it
        is carried on `.fx_markup` and applied by Section E as additional `slippage` —
        adverse on both legs, by construction.

        **Convert the benchmark too**, or beta and alpha compare base-currency returns
        against foreign-currency returns and every risk-adjusted metric downstream is
        wrong.

    Examples:
        >>> gbp = to_base_currency(msft_usd, "GBP", table, quote_currency="USD")
        >>> gbp = to_base_currency(prices, "GBP", fx=gbpusd, instruments=universe)
        >>> gbp.decompose()["fx"]
    """
    if hasattr(data, "get") and not isinstance(data, (pd.Series, pd.DataFrame)):
        data = data.get()

    if fx_markup is None:
        from ..core.settings import get

        fx_markup = get("fx_markup", 0.005)

    is_series = isinstance(data, pd.Series)
    frame = data.to_frame() if is_series else pd.DataFrame(data)

    fx_table = as_fx_table(fx_table if fx_table is not None else fx, index=frame.index)
    if fx_table is None:
        raise ValueError("An fx table or rate series is required; pass fx_table= or fx=.")

    quotes = _quote_map(frame.columns, quote_currency, instruments)

    unique = set(quotes.values())
    if unique == {base}:
        warnings.warn(
            f"Every column is already quoted in {base}; no conversion applied.",
            stacklevel=2,
        )

    converted = {}
    rates = {}
    for col in frame.columns:
        quote = quotes[col]
        rate = fx_table.get(quote, base)
        rate = fx_table.align(rate, frame.index, fill_policy=fill_policy)
        rates[col] = rate
        converted[col] = frame[col].to_numpy(dtype=float) * rate.to_numpy(dtype=float)

    out = ConvertedFrame(converted, index=frame.index)
    out.native = frame
    out.fx_rate = pd.DataFrame(rates, index=frame.index)
    out.fx_markup = float(fx_markup)
    out.base_currency = base

    if validate:
        check_ohlc_ordering(out)

    if is_series:
        series = pd.Series(out.iloc[:, 0].to_numpy(), index=frame.index, name=data.name)
        # A plain Series has nowhere to keep ConvertedFrame's attributes, and dropping the
        # markup here meant a converted single-instrument backtest never paid it.
        series.attrs["fx_markup"] = out.fx_markup
        series.attrs["base_currency"] = base
        return series
    return out


def fx_markup_of(*frames):
    """The FX conversion markup a converted price frame carries, or None.

    Reads `ConvertedFrame.fx_markup`, or the `attrs` entry a converted Series carries,
    from the first frame that has one. Must be called **before** the frame is coerced
    with `pd.DataFrame(...)`, which drops both.
    """
    for frame in frames:
        if frame is None:
            continue
        # isinstance, not getattr: on a plain frame, attribute access would find a
        # column that happened to be called `fx_markup`.
        if isinstance(frame, ConvertedFrame):
            markup = getattr(frame, "fx_markup", None)
        else:
            markup = getattr(frame, "attrs", {}).get("fx_markup")
        if markup is not None:
            return float(markup)
    return None


def converted_currency_of(*frames):
    """The base currency a converted price frame was converted into, or None."""
    for frame in frames:
        if frame is None:
            continue
        if isinstance(frame, ConvertedFrame):
            currency = getattr(frame, "base_currency", None)
        else:
            currency = getattr(frame, "attrs", {}).get("base_currency")
        if currency is not None:
            return currency
    return None


def _quote_map(columns, quote_currency, instruments) -> dict:
    """Normalise every accepted shape of quote-currency information into one dict.

    The three shapes the earlier draft accepted at three different call sites — a plain
    string, a `dict[column, currency]`, a `Universe` or dict of `Instrument`s — converge
    here, once.
    """
    columns = list(columns)

    if quote_currency is not None and instruments is not None:
        raise ValueError(
            "Give either `quote_currency` or `instruments`, not both — two sources for "
            "one fact is how they end up disagreeing."
        )

    source = quote_currency if quote_currency is not None else instruments
    if source is None:
        raise ValueError(
            "quote_currency (or instruments) is required. Conversion is explicit, never "
            "automatic — silently auto-converting price data is too surprising."
        )

    if isinstance(source, str):
        return {c: source for c in columns}

    specs = getattr(source, "specs", source)  # a Universe, or a plain dict
    quotes = {}
    for col, value in specs.items():
        quotes[col] = value if isinstance(value, str) else getattr(value, "quote_currency", None)

    missing = [c for c in columns if quotes.get(c) is None]
    if missing:
        raise ValueError(
            f"No quote_currency for columns {missing}. Every price column needs one, or "
            f"the conversion silently leaves them in the wrong currency."
        )
    return {c: quotes[c] for c in columns}


def check_ohlc_ordering(frame: pd.DataFrame) -> None:
    """Assert `H >= max(O, C) >= min(O, C) >= L` after conversion or adjustment.

    One check covering two hazards: mismatched FX rates across OHLC (Section I gotcha 1)
    and an adjusted close alongside raw high and low (D10). Both break the invariant and
    both make native SL/TP fire at prices that never existed.
    """
    cols = {str(c).strip().lower(): c for c in frame.columns}
    if not {"high", "low"}.issubset(cols):
        return
    high = frame[cols["high"]]
    low = frame[cols["low"]]
    bad = high < low
    for name in ("open", "close", "adj_close"):
        if name in cols:
            bad = bad | (high < frame[cols[name]]) | (low > frame[cols[name]])
    n = int(bad.sum())
    if n:
        raise ValueError(
            f"OHLC ordering broken on {n} bars (H >= C >= L fails). This means the bar "
            f"was only partly converted or partly adjusted. Adjust the whole bar or none "
            f"of it, and use one FX rate per bar for all of OHLC."
        )


def convert_amount(amount, quote: str, base: str, fx_table, markup: float = 0.0, index=None):
    """Convert a cash amount, applying the markup **adversely**.

    Args:
        amount: money in `quote` currency.
        quote: currency the amount is in.
        base: currency you want.
        fx_table: an `FXTable`.
        markup: broker markup, applied so you always receive less than mid.
        index: optional index to align the rate to.

    Returns:
        Amount in `base` currency.

    Notes:
        Adverse in both directions by construction: converting `quote -> base` you
        receive `rate x (1 - markup)`; converting back you pay `1 / (rate x (1 + markup))`
        of a unit, so a round trip always loses roughly `2 x markup`. That is exactly what
        a broker's FX spread does.

    Examples:
        >>> convert_amount(100, "USD", "GBP", table, markup=0.005)
    """
    rate = fx_table.get(quote, base)
    if index is not None:
        rate = fx_table.align(rate, index)
    if np.ndim(amount) == 0 and index is None:
        rate_val = float(rate.iloc[-1])
        return float(amount) * rate_val * (1.0 - markup)
    return np.asarray(amount, dtype=float) * rate.to_numpy(dtype=float) * (1.0 - markup)


def convert_fixed_costs(amount, cost_currency: str, base: str, fx_table, index=None, markup=0.0):
    """Convert a fixed or per-unit cost quoted in a foreign currency.

    Percentage costs are FX-invariant and need no handling. Fixed and per-unit costs are
    not, and **must** be converted with the same rate series as the price conversion —
    `spec.cost_currency` exists for exactly this. It is the easiest thing in the plan to
    get silently wrong.
    """
    if amount is None:
        return None
    if cost_currency is None or cost_currency == base:
        return amount
    return convert_amount(amount, cost_currency, base, fx_table, markup=markup, index=index)
