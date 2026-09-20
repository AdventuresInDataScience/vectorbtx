"""Section E — commission.

Three shapes, plus a minimum:

* percentage of notional -> `fees`
* flat per trade         -> `fixed_fees`
* per unit               -> `fixed_fees` (needs size, so it needs the notional context)

A **minimum commission is pre-computed per order into `fixed_fees`**, with `fees` set to
zero for that instrument. `max(pct x notional, minimum)` is a conditional and genuinely
cannot be expressed as `fees` + `fixed_fees`, which are additive.

Spread-bet mode sets `fees = fixed_fees = 0` — the fee is baked into a wider spread.
"""

from __future__ import annotations

import numpy as np

from .base import CostError, _Base


class Commission(_Base):
    """Percentage, flat, per-unit and minimum commission."""

    name = "commission"

    def contribute(self, spec, price, ctx) -> dict:
        spec = spec.filled()
        if spec.mode == "spreadbet":
            return {"fees": 0.0, "fixed_fees": 0.0}

        pct = spec.commission_pct or 0.0
        flat = spec.commission_flat or 0.0
        per_unit = spec.commission_per_unit or 0.0
        minimum = spec.commission_min

        base = ctx.get("base_currency")
        fx_table = ctx.get("fx_table")
        if fx_table is not None and spec.cost_currency and base and spec.cost_currency != base:
            from ..fx.convert import convert_fixed_costs

            flat = convert_fixed_costs(flat, spec.cost_currency, base, fx_table, price.index)
            per_unit = convert_fixed_costs(per_unit, spec.cost_currency, base, fx_table, price.index)
            if minimum is not None:
                minimum = convert_fixed_costs(minimum, spec.cost_currency, base, fx_table, price.index)

        fixed = np.asarray(flat, dtype=float)
        per_unit = np.asarray(per_unit, dtype=float)
        # `per_unit` may be an array once FX-converted, so test it elementwise.
        if np.any(per_unit != 0.0):
            size = ctx.get("size")
            if size is None:
                raise CostError(
                    f"{ctx.get('column')}: commission_per_unit needs an order size. "
                    f"Pass size=... to resolve_costs, or use commission_flat/commission_pct."
                )
            fixed = fixed + np.abs(np.asarray(size, dtype=float)) * np.asarray(per_unit, dtype=float)

        if minimum is None:
            return {"fees": float(pct), "fixed_fees": fixed}

        notional = ctx.get("notional")
        if notional is None:
            raise CostError(
                f"{ctx.get('column')}: commission_min needs the order notional so that "
                f"max(pct x notional, minimum) can be pre-computed. Pass notional=... to "
                f"resolve_costs; it cannot be expressed with fees + fixed_fees alone."
            )
        pct_part = np.abs(np.asarray(notional, dtype=float)) * pct
        return {"fees": 0.0, "fixed_fees": np.maximum(pct_part, minimum) + fixed}


COMMISSION = Commission()
