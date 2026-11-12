"""Section B — the `.x` namespaces.

`.x` appears in two places, mirroring how vectorbt already splits accessors from module
functions:

* `pf.x.*`   — anything portfolio-bound (`PortfolioXAccessor`)
* `df.vbt.x.*` — anything data-bound (`XSRAccessor` / `XDFAccessor`)
* `vbt.x.*`  — constructors and loaders (see `vectorbtx.x`)
"""

from __future__ import annotations

import pandas as pd


class PortfolioXAccessor:
    """vectorbtX extras bound to a `Portfolio`. Reached as `pf.x`."""

    def __init__(self, pf):
        self._pf = pf

    @property
    def pf(self):
        """The wrapped portfolio."""
        return self._pf

    @property
    def universe(self):
        """The `Universe` the portfolio was built with, or None."""
        return getattr(self._pf, "_universe", None)

    def borrowed_portion(self, **kwargs):
        """See `vectorbtx.margin.exposure.borrowed_portion`."""
        from ..margin.exposure import borrowed_portion

        return borrowed_portion(self._pf, **kwargs)

    def margin_level(self, universe=None, **kwargs):
        """See `vectorbtx.margin.exposure.margin_level`."""
        from ..margin.exposure import margin_level

        return margin_level(self._pf, self._resolve_universe(universe), **kwargs)

    def min_margin_level(self, universe=None, **kwargs):
        """See `vectorbtx.margin.exposure.min_margin_level`."""
        from ..margin.exposure import min_margin_level

        return min_margin_level(self._pf, self._resolve_universe(universe), **kwargs)

    def _resolve_universe(self, universe):
        """Explicit argument first, then whatever the portfolio carries.

        `universe or self.universe` would be wrong for a `Universe` that defines
        `__len__`: an empty one is falsy and would be silently swapped for the
        portfolio's. Identity against None is the only safe test.
        """
        return universe if universe is not None else self.universe

    def drift(self):
        """Quantization drift recorded at construction, or None."""
        return getattr(self._pf, "_drift", None)

    def intrabar_value(self) -> pd.DataFrame:
        """The account's best and worst value inside each bar.

        Returns:
            DataFrame with columns `best` and `worst`, one row per bar: the whole
            account's value with every position at its most favourable extreme, and at
            its least favourable one. Summed over every column, so a portfolio of
            separate per-instrument pools reports their total.

        Notes:
            **Each side is bracketed by its own extremes.** A long is best at the high and
            worst at the low; a short is the other way round. Valuing every position at
            the low for the trough would report a spike against a short as a *gain*.

            ⚠️ **Cash is read at the portfolio's own grouping, never per column.** Under
            cash sharing, `pf.cash(group_by=False)` hands every column the group's whole
            starting cash, so summing it counts the shared pot once per column. That
            inflated the denominator and reported a drawdown *shallower* than the
            close-to-close one. `pf.cash()` reconciles exactly with `pf.value()`: it
            counts shared cash once, nets out the margin loan and includes holding costs.
        """
        if self._pf.low is None or self._pf.high is None:
            raise ValueError(
                "No intrabar highs/lows on this portfolio. Build it through "
                "vbt.x.PortfolioBacktest, which requires them."
            )
        assets = pd.DataFrame(self._pf.assets())
        cash = pd.DataFrame(self._pf.cash()).sum(axis=1)
        close = pd.DataFrame(self._pf.close).reindex(columns=assets.columns)
        # A missing extreme falls back to the close rather than valuing a held position
        # at zero, which would manufacture a drawdown.
        highs = pd.DataFrame(self._pf.high).reindex(columns=assets.columns).fillna(close)
        lows = pd.DataFrame(self._pf.low).reindex(columns=assets.columns).fillna(close)

        longs, shorts = assets.clip(lower=0.0), assets.clip(upper=0.0)
        best = (longs * highs + shorts * lows).sum(axis=1) + cash
        worst = (longs * lows + shorts * highs).sum(axis=1) + cash
        return pd.DataFrame({"best": best, "worst": worst})

    def intrabar_drawdown_series(self) -> pd.Series:
        """Drawdown at each bar, measured against the intrabar extremes.

        Returns:
            Series of positive fractions, one per bar: how far the bar's worst value sits
            below the best value seen so far. Its maximum is `intrabar_drawdown()`; this
            is the curve to plot beside `pf.drawdown()`.
        """
        values = self.intrabar_value()
        peak = values["best"].cummax()
        return (1.0 - values["worst"] / peak).rename("intrabar_drawdown")

    def intrabar_drawdown(self) -> float:
        """Maximum drawdown measured against intrabar extremes, not closes.

        The standard figure values the portfolio at each bar's close, so an excursion that
        happened and recovered within the bar is invisible to it. On daily data that
        routinely understates the worst moment of a backtest — and the worst moment is
        usually the number that decides whether a strategy is fundable.

        Returns:
            Float, the maximum peak-to-trough decline as a positive fraction.

        Notes:
            ⚠️ The **peak is taken from each bar's best value and the trough from its
            worst** (see `intrabar_value`). Valuing both ends at the same extreme would
            move the running peak by as much as the trough and report a drawdown
            *smaller* than the close-to-close figure — the opposite of the intent.

            This is the pessimistic bracket: it assumes every position hit its best
            simultaneously, then its worst simultaneously. The realised figure lies between
            this and `pf.max_drawdown()`, and the gap between them measures how much
            intrabar risk a strategy is carrying. Longs and shorts are each bracketed by
            their own side, so the figure holds for long, short and mixed books.
        """
        return float(self.intrabar_drawdown_series().max())

    def user_size(self, universe=None):
        """Position size in the units the user actually trades in, not engine units.

        The engine simulates one number; a spread bettor thinks in £/point, a CFD trader
        in lots, an equity trader in shares. This converts back, so a report reads
        "£1.66/pt" rather than "16600".

        Returns:
            DataFrame of per-bar user-unit sizes, one column per instrument.

        Raises:
            ValueError: if the portfolio carries no `Universe`, since without one there
                is no mode to convert through.
        """
        from ..modes.base import get_adapter

        universe = universe or self.universe
        if universe is None:
            raise ValueError(
                "No Universe on this portfolio, so there is no mode to convert into user "
                "units. Build it through vbt.x.PortfolioBacktest or Portfolio.from_weights."
            )
        assets = pd.DataFrame(self._pf.assets())
        out = {}
        for col in assets.columns:
            if col not in universe:
                continue
            spec = universe[col].filled()
            out[col] = get_adapter(spec.mode).from_engine_units(
                assets[col].to_numpy(dtype=float), spec
            )
        return pd.DataFrame(out, index=assets.index)

    def unit_names(self, universe=None) -> dict:
        """The unit each column is measured in — `£/point`, `lots` or `shares`."""
        from ..modes.base import get_adapter

        universe = universe or self.universe
        if universe is None:
            return {}
        return {
            col: get_adapter(spec.mode).unit_name(spec)
            for col, spec in universe.specs.items()
        }

    def holding_costs(self):
        """Holding-cost cash adjustments recorded at construction, or None."""
        return getattr(self._pf, "_holding_costs", None)


class _XAccessorBase:
    """Data-bound vectorbtX extras. Reached as `df.vbt.x`."""

    def __init__(self, obj, **kwargs):
        self._xobj = obj if isinstance(obj, (pd.Series, pd.DataFrame)) else getattr(obj, "obj", obj)

    @property
    def obj(self):
        """The wrapped pandas object."""
        return self._xobj

    def to_base_currency(self, base, fx_table, quote_currency=None, **kwargs):
        """See `vectorbtx.fx.convert.to_base_currency`."""
        from ..fx.convert import to_base_currency

        return to_base_currency(self._xobj, base, fx_table, quote_currency=quote_currency, **kwargs)

    def to_returns_metrics(self, **kwargs):
        """Shortcut to `obj.vbt.returns(**kwargs).stats()`."""
        return self._xobj.vbt.returns(**kwargs).stats()


def register_accessors() -> None:
    """Register `.x` on pandas' `vbt` accessor and on the vectorbtX `Portfolio`."""
    from vectorbt.root_accessors import (
        register_dataframe_vbt_accessor,
        register_series_vbt_accessor,
    )

    from .patch import ShadowError
    from vectorbt.root_accessors import Vbt_SRAccessor, Vbt_DFAccessor

    for parent in (Vbt_SRAccessor, Vbt_DFAccessor):
        if hasattr(parent, "x"):
            raise ShadowError(f"{parent.__name__}.x already exists upstream.")

    @register_series_vbt_accessor("x")
    class XSRAccessor(_XAccessorBase):
        """`pd.Series.vbt.x.*`"""

    @register_dataframe_vbt_accessor("x")
    class XDFAccessor(_XAccessorBase):
        """`pd.DataFrame.vbt.x.*`"""

    return XSRAccessor, XDFAccessor
