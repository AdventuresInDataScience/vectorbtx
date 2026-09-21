"""Section C2 — specs to per-column `Order` arguments.

vectorbt already broadcasts a plain list per column at almost no memory cost via
`flex_select_auto_nb`, so the mechanism exists. This module is the ergonomics: it turns a
validated `Universe` into exactly the arrays that mechanism consumes.

Column order comes from the **wrapper's column index**, not dict insertion order or
spreadsheet row order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _columns_of(data):
    """Extract the resolved column order from whatever the caller passed."""
    if hasattr(data, "wrapper"):
        return list(data.wrapper.columns)
    if isinstance(data, pd.DataFrame):
        return list(data.columns)
    if isinstance(data, pd.Series):
        return [data.name]
    if hasattr(data, "get") and hasattr(data, "symbols"):  # vectorbt Data
        return list(data.symbols)
    raise TypeError(f"Cannot determine columns from {type(data).__name__}")


def _price_frame(data, columns):
    """Extract a price frame aligned to `columns`."""
    if isinstance(data, pd.DataFrame):
        return data[columns]
    if isinstance(data, pd.Series):
        return data.to_frame(name=columns[0])
    if hasattr(data, "get"):
        got = data.get("Close")
        return got[columns] if isinstance(got, pd.DataFrame) else got.to_frame(columns[0])
    raise TypeError(f"Cannot extract prices from {type(data).__name__}")


def resolve_universe(
    universe,
    data,
    rates=None,
    equity=None,
    fx_table=None,
    volume=None,
    base_currency=None,
    notional=None,
    size=None,
    include_costs: bool = True,
    include_quantization: bool = True,
    include_leverage: bool = True,
) -> dict:
    """Resolve a `Universe` into `Portfolio.from_orders` keyword arguments.

    Args:
        universe: the `Universe` to resolve.
        data: price frame or vectorbt object; supplies the resolved column order.
        rates: rate table for `financing='auto'` (Section M).
        equity: account equity, needed to turn leverage caps into `max_size`.
        fx_table: `FXTable` for converting foreign-currency fixed costs (Section I).
        volume: volume frame, required by `slippage='auto'`.
        base_currency: the account currency. Required alongside `fx_table` before any
            fixed cost quoted in `spec.cost_currency` can be converted — percentage costs
            are FX-invariant, fixed and per-unit ones are not.
        notional: order notional, required by `commission_min` so that
            `max(pct x notional, minimum)` can be pre-computed into `fixed_fees`.
        size: order size, required by `commission_per_unit`.
        include_costs: emit `fees` / `fixed_fees` / `slippage`.
        include_quantization: emit `size_granularity` / `min_size`.
        include_leverage: emit `max_size` (needs `equity`).

    Returns:
        dict of `Order` kwargs. Scalars stay scalars; per-column values become lists in
        wrapper column order.

    Raises:
        UniverseError: if specs and price columns do not match exactly.
        ResolveError: if an `'auto'` cell cannot be resolved.
    """
    columns = _columns_of(data)
    specs = universe.order_for(columns)
    price = _price_frame(data, columns)

    # ⚠️ An empty price frame is not a cost-resolution problem, but that is where it used
    # to surface — `IndexError: index 0 is out of bounds for size 0` from `arr.flat[0]`
    # deep inside `costs/compose.py`, which tells the reader nothing.
    #
    # Two quite different faults land here, and the column count tells them apart. With
    # several columns it is nearly always session misalignment. With one, nothing was
    # downloaded at all — and blaming misalignment there sends the reader hunting for a
    # timezone bug that does not exist.
    if len(price.index) == 0:
        from .load import UniverseError  # local: load.py imports this module

        if len(columns) == 1:
            cause = (
                "With a single column, the download itself returned nothing. Check the "
                "ticker is one the provider actually knows — Yahoo logs 'possibly "
                "delisted; no price data found' and hands back an empty frame rather "
                "than raising, so a wrong symbol looks like a library fault. Index "
                "names are not tickers: the FTSE 100 is '^FTSE', not 'UKX'."
            )
        else:
            cause = (
                "With several columns, the usual cause is aligning instruments from "
                "different exchanges: Yahoo timestamps daily bars at each market's own "
                "midnight (LSE 00:00 UTC, US 05:00 UTC), so an index intersection is "
                "empty even though every series has data. Normalise to dates before "
                "aligning — `frame.index = frame.index.tz_localize(None).normalize()` "
                "on each source, then combine."
            )
        raise UniverseError(
            f"The price data has {len(columns)} column(s) but no rows, so nothing can be "
            f"resolved. {cause} Columns seen: {list(columns)}."
        )

    out: dict = {}

    if include_quantization:
        from ..quantize.quantizer import to_order_params

        granularity, min_size = [], []
        for spec in specs:
            params = to_order_params(spec)
            granularity.append(params["size_granularity"])
            min_size.append(params["min_size"])
        out["size_granularity"] = _collapse(granularity)
        out["min_size"] = _collapse(min_size)

    if include_costs:
        from ..costs.compose import resolve_costs

        out.update(
            resolve_costs(
                universe,
                price,
                columns=columns,
                fx_table=fx_table,
                base_currency=base_currency,
                volume=volume,
                size=size,
                notional=notional,
                rates=rates,
            )
        )

    if include_leverage and equity is not None:
        from ..margin.cap import max_size_for

        out["max_size"] = max_size_for(specs, equity, price)

    lock = [spec.leverage is not None and spec.leverage > 1 for spec in specs]
    if any(lock):
        # The closest native analogue to "don't let one position borrow another's
        # collateral", and the right default for margined modes.
        out["lock_cash"] = _collapse(lock)

    return out


def _collapse(values):
    """Return a scalar if every column agrees, else the list. Keeps output readable."""
    vals = list(values)
    first = vals[0]
    for v in vals[1:]:
        same = (v == first) or (
            isinstance(v, float) and isinstance(first, float) and np.isnan(v) and np.isnan(first)
        )
        if not same:
            return vals
    return first


class ResolveError(ValueError):
    """Raised when an `'auto'` cell cannot be resolved.

    Always names the instrument and the column: `'auto'` slippage needs a volume series,
    `'auto'` financing needs Section M loaded and covering the backtest period. Failing
    here beats emitting NaNs that quietly zero a cost.
    """
