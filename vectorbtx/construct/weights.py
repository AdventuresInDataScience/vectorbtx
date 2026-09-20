"""Section J — weighting schemes.

Pure pandas, pre-simulation. Each returns a full-index DataFrame of target weights; the
schedule decides which of those rows actually reach the engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def equal_weight(columns, index) -> pd.DataFrame:
    """Equal weight across `columns`.

    Examples:
        >>> equal_weight(["A", "B", "C", "D"], index).iloc[0]
        A    0.25
        B    0.25
        C    0.25
        D    0.25
    """
    cols = list(columns)
    return pd.DataFrame(1.0 / len(cols), index=pd.Index(index), columns=cols)


def price_weight(prices) -> pd.DataFrame:
    """Price-weighted, Dow-style: each asset's share of total price."""
    frame = pd.DataFrame(prices).astype(float)
    total = frame.sum(axis=1)
    return frame.div(total, axis=0)


def cap_weight(market_caps, index=None, fx=None, base=None, fx_table=None) -> pd.DataFrame:
    """Market-capitalisation weighted, **recomputed at each rebalance**.

    Fixed-at-construction is not cap weighting, it is an initial-cap-weighted
    buy-and-hold — a materially different strategy.

    Args:
        market_caps: DataFrame of caps, one column per asset.
        index: optional index to reindex onto.
        fx: optional `dict[column, currency]` for cross-currency universes.
        base: base currency, required when `fx` is given.
        fx_table: `FXTable`, required when `fx` is given.

    Notes:
        Cap weighting needs the caps converted with the **same rate table** as the price
        conversion, or the weights are computed across mismatched currencies.
    """
    frame = pd.DataFrame(market_caps).astype(float)
    if fx is not None:
        if fx_table is None or base is None:
            raise ValueError("cap_weight with `fx` needs both `base` and `fx_table`")
        for col, ccy in fx.items():
            rate = fx_table.align(fx_table.get(ccy, base), frame.index)
            frame[col] = frame[col] * rate
    if index is not None:
        frame = frame.reindex(pd.Index(index)).ffill()
    total = frame.sum(axis=1)
    return frame.div(total, axis=0)


