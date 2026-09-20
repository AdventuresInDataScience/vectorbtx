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
