"""Section E — composition.

Runs each active module and folds the partial dicts into `Order` kwargs.

The one rule worth stating twice: **spread and slippage both land on `slippage`, so they
must be summed before assignment.** Only one field exists. The same goes for the FX
markup from Section I, which is a spread on the exchange rate and belongs in exactly the
same place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .commission import COMMISSION
from .slippage import SLIPPAGE
from .spread import SPREAD

MODULES = {m.name: m for m in (COMMISSION, SPREAD, SLIPPAGE)}


def _as_column_value(value, n_rows):
    """Reduce a per-bar series to something vectorbt can broadcast per column."""
    if isinstance(value, pd.Series):
        return value.to_numpy(dtype=float)
    if isinstance(value, np.ndarray):
        return value.astype(float)
    return float(value)


def resolve_costs(
    universe,
    price,
    columns=None,
    fx_table=None,
    volume=None,
    rates=None,
    size=None,
    notional=None,
    base_currency=None,
    fx_markup=None,
) -> dict:
    """Compose every active cost module into `Order` keyword arguments.

    Args:
        universe: the `Universe`.
        price: price frame aligned to `columns`.
        columns: resolved column order. Defaults to `price.columns`.
        fx_table: `FXTable`, needed to convert foreign-currency fixed costs.
        volume: volume frame for `slippage='auto'`.
        rates: rate table for `financing='auto'` (consumed by Section G, not here).
        size: order sizes, needed by per-unit commission and auto slippage.
        notional: order notionals, needed by minimum commission.
        base_currency: the account currency.
        fx_markup: the FX conversion markup, a fraction added to `slippage`. A scalar
            applies to every column; a `dict[column, markup]` applies per column (the
            front door builds one so that columns already in the account currency, which
            were never converted, pay none). None falls back to `price.fx_markup`.

    Returns:
        `{"fees": ..., "fixed_fees": ..., "slippage": ...}`, each a scalar or a
        per-column array.

    Notes:
        Percentage costs are FX-invariant and need no handling. Fixed and per-unit costs
        quoted in a foreign currency must be converted with the same rate series as the
        price conversion — `spec.cost_currency` exists for this, and it is the most likely
        source of silent error in the section.

    Examples:
        >>> resolve_costs(universe, prices, fx_table=table, base_currency="GBP")
        {'fees': 0.001, 'fixed_fees': 0.0, 'slippage': [...]}
    """
    # Read the markup before coercion: pd.DataFrame(...) drops ConvertedFrame's attribute.
    if fx_markup is None:
        from ..fx.convert import fx_markup_of

        fx_markup = fx_markup_of(price)
    price = pd.DataFrame(price)
    columns = list(columns) if columns is not None else list(price.columns)
    specs = universe.order_for(columns)

    fees, fixed_fees, slippages = [], [], []

    for col, spec in zip(columns, specs):
        spec = spec.filled()
        col_price = price[col]
        ctx = dict(
            column=col,
            fx_table=fx_table,
            base_currency=base_currency,
            volume=None if volume is None else pd.DataFrame(volume)[col],
            size=None if size is None else _col_of(size, col),
            notional=None if notional is None else _col_of(notional, col),
            rates=rates,
        )

        fee, fixed, slip = 0.0, 0.0, 0.0
        for name in ("commission", "spread", "slippage"):
            module = MODULES[name]
            if not module.applies(spec):
                continue
            part = module.contribute(spec, col_price, ctx)
            if "fees" in part:
                fee = fee + np.asarray(part["fees"], dtype=float)
            if "fixed_fees" in part:
                fixed = fixed + np.asarray(part["fixed_fees"], dtype=float)
            if "slippage" in part:
                # Spread and slippage share one field — sum, never overwrite.
                slip = slip + _as_column_value(part["slippage"], len(price))

        markup = fx_markup.get(col) if isinstance(fx_markup, dict) else fx_markup
        if markup:
            # The FX markup is a spread on the exchange rate, so it behaves exactly like a
            # spread on price: adverse on entry and adverse again on exit.
            slip = slip + float(markup)

        fees.append(_scalarise(fee))
        fixed_fees.append(_scalarise(fixed))
        slippages.append(slip)

    return {
        "fees": _collapse(fees, price.index, columns),
        "fixed_fees": _collapse(fixed_fees, price.index, columns),
        "slippage": _stack(slippages, price.index, columns),
    }


def _col_of(obj, col):
    if isinstance(obj, pd.DataFrame):
        return obj[col]
    return obj


def _scalarise(value):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    if np.allclose(arr, arr.flat[0], equal_nan=True):
        return float(arr.flat[0])
    return arr


def _collapse(values, index=None, columns=None):
    """Reduce per-column cost values to the smallest form vectorbt broadcasts correctly.

    ⚠️ **Never return a bare 1-D per-bar array.** vectorbt broadcasts a length-`n` list or
    array across **columns**, not bars, so a per-bar fee series for a one-column portfolio
    silently widens it to `n_bars` columns — the simulation runs, and the order count comes
    back multiplied by the number of bars. A minimum commission (`max(pct x notional, min)`)
    is exactly such a per-bar series, so this is reachable from ordinary input.

    Scalars collapse to a float. Anything per-bar is stacked into a full DataFrame, whose
    shape is unambiguous.
    """
    first = values[0]
    if all(np.isscalar(v) and v == first for v in values):
        return float(first)
    if all(np.isscalar(v) for v in values):
        return list(values)          # one per column: unambiguous
    if index is None or columns is None:
        raise ValueError("Per-bar cost values need index and columns to stack safely.")
    return _stack(values, index, columns)


def _stack(slippages, index, columns):
    """Return a scalar, a per-column list, or a full DataFrame — whichever is smallest."""
    if all(np.isscalar(s) for s in slippages):
        return float(slippages[0]) if len(set(slippages)) == 1 else list(slippages)
    frame = pd.DataFrame(
        {c: (np.full(len(index), s) if np.isscalar(s) else np.asarray(s, dtype=float))
         for c, s in zip(columns, slippages)},
        index=index,
    )
    return frame
