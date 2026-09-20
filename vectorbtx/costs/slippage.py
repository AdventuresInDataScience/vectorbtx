"""Section E — slippage.

Either a fixed fraction of price, or the keyword `'auto'`:

    slippage = coef x sigma x sqrt(size / ADV),  clipped to [0, slippage_max]

Square-root market impact is the standard empirical result and beats a linear model.
`sigma` is 20-bar realised volatility as a fraction of price.

`'auto'` **requires a volume column**. Failing at resolve time is deliberate: silently
falling back to zero would quietly remove a cost the user explicitly asked to model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import CostError, _Base


def realised_volatility(price, window: int = 20) -> pd.Series:
    """Rolling realised volatility as a fraction of price."""
    series = pd.Series(np.asarray(price, dtype=float), index=getattr(price, "index", None))
    return series.pct_change().rolling(window, min_periods=2).std().bfill().fillna(0.0)


def auto_slippage(price, volume, size, coef=1.0, slippage_max=0.005, window=20, adv=None):
    """Volume- and volatility-scaled slippage.

    Args:
        price: price series.
        volume: volume series, or None if `adv` is supplied.
        size: order size in engine units.
        coef: impact coefficient.
        slippage_max: upper clip, default 50bp.
        window: volatility lookback, default 20 bars.
        adv: static average daily volume, used when no volume series exists.

    Returns:
        Series of slippage fractions, clipped to `[0, slippage_max]`.
    """
    sigma = realised_volatility(price, window)
    if volume is not None:
        advs = pd.Series(np.asarray(volume, dtype=float), index=sigma.index).rolling(
            window, min_periods=1
        ).mean()
    else:
        advs = pd.Series(float(adv), index=sigma.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.abs(np.asarray(size, dtype=float)) / advs.to_numpy(dtype=float)
    impact = coef * sigma.to_numpy(dtype=float) * np.sqrt(np.clip(ratio, 0.0, None))
    return pd.Series(np.clip(np.nan_to_num(impact, nan=0.0), 0.0, slippage_max), index=sigma.index)


class Slippage(_Base):
    """Fixed or auto slippage."""

    name = "slippage"

    def contribute(self, spec, price, ctx) -> dict:
        spec = spec.filled()
        value = spec.slippage
        if value is None:
            return {"slippage": 0.0}
        if not isinstance(value, str):
            return {"slippage": float(value)}
        if value != "auto":  # keywords are validated upstream; belt and braces
            raise CostError(f"{ctx.get('column')}: unknown slippage keyword {value!r}")

        volume = ctx.get("volume")
        size = ctx.get("size")
        if volume is None and spec.adv is None:
            raise CostError(
                f"{ctx.get('column')}: slippage='auto' needs a volume series or an `adv` "
                f"value on the spec. Pass volume=... to resolve_costs, set adv, or use a "
                f"fixed slippage number."
            )
        if size is None:
            raise CostError(
                f"{ctx.get('column')}: slippage='auto' needs an order size to scale "
                f"market impact. Pass size=... to resolve_costs."
            )
        return {
            "slippage": auto_slippage(
                price,
                volume,
                size,
                coef=spec.slippage_coef,
                slippage_max=spec.slippage_max,
                window=ctx.get("vol_window", 20),
                adv=spec.adv,
            )
        }


SLIPPAGE = Slippage()
