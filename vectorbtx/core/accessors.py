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

        return margin_level(self._pf, universe or self.universe, **kwargs)

    def min_margin_level(self, universe=None, **kwargs):
        """See `vectorbtx.margin.exposure.min_margin_level`."""
        from ..margin.exposure import min_margin_level

        return min_margin_level(self._pf, universe or self.universe, **kwargs)

    def drift(self):
        """Quantization drift recorded at construction, or None."""
        return getattr(self._pf, "_drift", None)

    def intrabar_drawdown(self):
        """Maximum drawdown measured against intrabar **lows**, not closes.

        The standard figure values the portfolio at each bar's close, so an excursion that
        happened and recovered within the bar is invisible to it. On daily data that
        routinely understates the worst moment of a backtest — and the worst moment is
        usually the number that decides whether a strategy is fundable.

        Returns:
            Float, the maximum peak-to-trough decline as a positive fraction.

        Notes:
            ⚠️ The **peak is taken from the highs and the trough from the lows**. Valuing
            both ends at the low would depress the running peak by the same amount as the
            trough and report a drawdown *smaller* than the close-to-close figure — the
            opposite of the intent.

            This is the pessimistic bracket: it assumes every position hit its best
            simultaneously, then its worst simultaneously. The realised figure lies between
            this and `pf.max_drawdown()`, and the gap between them measures how much
            intrabar risk a strategy is carrying.

            Long-only is the assumption. A short position is hurt by the **high**, so for a
            short book the two series should be swapped; this reads the long case, which is
            what the mandatory `high`/`low` are principally there to protect.
        """
        if self._pf.low is None or self._pf.high is None:
            raise ValueError(
                "No intrabar highs/lows on this portfolio. Build it through "
                "vbt.x.PortfolioBacktest, which requires them."
            )
        assets = pd.DataFrame(self._pf.assets())
        cash = pd.DataFrame(self._pf.cash(group_by=False)).sum(axis=1)
        highs = pd.DataFrame(self._pf.high).reindex(columns=assets.columns)
        lows = pd.DataFrame(self._pf.low).reindex(columns=assets.columns)

        best = (assets * highs).sum(axis=1) + cash
        worst = (assets * lows).sum(axis=1) + cash
        peak = best.cummax()
        return float((1.0 - worst / peak).max())

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
