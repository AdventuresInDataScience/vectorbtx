"""Section N — `SingleBacktest`, a thin validating forwarder.

**Two names, one implementation.** `SingleBacktest` validates that the price is 1-D,
rejects the portfolio-only arguments with a message pointing at `PortfolioBacktest`, and
forwards. There is no divergence risk because there is no second implementation — and
each name shows only the arguments that actually apply to it.
"""

from __future__ import annotations

import pandas as pd

from .backtest import PortfolioBacktest
from .prepare import BacktestError

#: Arguments that only mean something across several columns. Rejected here so a user who
#: reaches for them is told which class they want, rather than having them silently ignored.
_PORTFOLIO_ONLY = {
    "instruments": "describes several instruments; give the terms directly instead",
    "weights": "rebalancing needs more than one column",
    "schedule": "rebalancing needs more than one column",
    "min_trade": "only meaningful for rebalancing",
    "leverage_budget": "portfolio-level leverage needs a portfolio",
    "cash_sharing": "there is only one column to share cash between",
    "group_by": "there is only one column to group",
    "cash": "there is only one column, so per-instrument and shared cash are the same account",
    "entry_priority": "with one column no entries compete for cash",
}


def SingleBacktest(*, close=None, open=None, high=None, low=None, **kwargs):
    """Run a single-instrument backtest and return a `vectorbTX.Portfolio`.

    Takes every argument `PortfolioBacktest` does except the portfolio-only ones, which
    raise with a pointer to the other class.

    On an FX instrument (`asset_class="fx"` or `mode="fx_spot"`), `base_currency` is the
    pair's base currency, so `base_currency="EUR", quote_currency="USD"` describes EURUSD
    and carry is the EUR/USD differential. Name the account's currency with
    `account_currency=`.

    Args:
        close: a 1-D close series, or a single-column frame.
        open: the matching open series. **Required** — orders fill at the next bar's open.
        high, low: the matching intrabar extremes. **Required** — stops are checked against
            them and drawdown is measured against them.
        **kwargs: see `PortfolioBacktest`.

    Returns:
        `vectorbtx.Portfolio`.

    Raises:
        BacktestError: if `price` has more than one column, or a portfolio-only argument
            was given.

    Examples:
        >>> pf = vbt.x.SingleBacktest(
        ...     close=close, open=open_, high=high, low=low,
        ...     entries=entries, exits=exits,
        ...     mode="spreadbet", asset_class="fx", point_size=0.0001,
        ...     spread_points=0.9, risk_pct=0.01, stop_points=60,
        ...     init_cash=10_000, freq="D",
        ... )
        >>> pf.x.user_size().iloc[-1]      # "£1.66/pt", not 16600 engine units
    """
    offenders = {k: why for k, why in _PORTFOLIO_ONLY.items() if kwargs.get(k) is not None}
    if offenders:
        detail = "; ".join(f"`{k}` — {why}" for k, why in offenders.items())
        raise BacktestError(
            f"SingleBacktest does not take {sorted(offenders)}: {detail}. "
            f"Use vbt.x.PortfolioBacktest for anything spanning several instruments."
        )

    if close is None:
        raise BacktestError(
            "`close` is required. All four price series are passed by keyword:\n"
            "  vbt.x.SingleBacktest(close=..., open=..., high=..., low=..., ...)"
        )
    # A converted Series carries its FX markup in `.attrs`; read it before reshaping, so it
    # survives whatever the reshaping does to them.
    if kwargs.get("fx_markup") is None:
        from ..fx.convert import fx_markup_of

        kwargs["fx_markup"] = fx_markup_of(open, close)

    frame = close.to_frame() if isinstance(close, pd.Series) else pd.DataFrame(close)
    def _align(series):
        if series is None:
            return None
        out = series.to_frame() if isinstance(series, pd.Series) else pd.DataFrame(series)
        return out.set_axis(frame.columns, axis=1)

    open, high, low = _align(open), _align(high), _align(low)
    if frame.shape[1] != 1:
        raise BacktestError(
            f"SingleBacktest takes one instrument; `price` has {frame.shape[1]} columns "
            f"({list(frame.columns)[:4]}...). Use vbt.x.PortfolioBacktest."
        )

    for name in _PORTFOLIO_ONLY:
        kwargs.pop(name, None)
    return PortfolioBacktest(close=frame, open=open, high=high, low=low, **kwargs)
