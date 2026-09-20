"""Section I — the FX rate table.

Long form, keyed `(date, quote, base) -> value`, where `value` is what you **multiply a
`quote`-currency price by to get `base` currency**. Direction lives in the key, so there
is no multiply-or-divide decision left to get wrong.

Fetch one canonical direction and derive the reciprocals. Providers mostly quote majors
against USD, so the grid has to be filled by inversion and triangulation regardless.
Fetching both directions independently gives non-exact reciprocals — different
timestamps, mid versus bid/ask capture — and a round-trip conversion then produces
phantom P&L. That noise is **random in sign**, so it inflates measured volatility rather
than acting as a cost, quietly degrading every risk-adjusted metric with a standard
deviation in the denominator. Deriving from canonical guarantees
`value(A,B) x value(B,A) = 1` by construction.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


class FXError(ValueError):
    """Raised when a rate table is inconsistent or a pair cannot be resolved."""


class FXTable:
    """A validated grid of FX rates.

    Attributes:
        rates: wide DataFrame indexed by date, columns `(quote, base)` tuples.
        hub: the canonical currency everything was derived through.
    """

    def __init__(self, rates: pd.DataFrame, hub: str = "USD", validate: bool = True):
        self.rates = rates
        self.hub = hub
        if validate:
            self.validate()

    # ############# Construction ############# #

    @classmethod
    def from_canonical(cls, canonical, hub: str = "USD") -> "FXTable":
        """Build the full grid from one canonical direction per pair.

        Args:
            canonical: DataFrame indexed by date, one column per currency, holding the
                value of **one unit of that currency in `hub`**. E.g. column `GBP` = 1.27
                means 1 GBP buys 1.27 USD.
            hub: the pivot currency.

        Returns:
            `FXTable` with every ordered pair filled by inversion and triangulation.

        Examples:
            >>> canonical = pd.DataFrame({"GBP": [1.27], "EUR": [1.09]}, index=idx)
            >>> table = FXTable.from_canonical(canonical, hub="USD")
            >>> table.get("USD", "GBP")     # multiply a USD price by this to get GBP
        """
        canonical = pd.DataFrame(canonical).astype(float)
        currencies = list(canonical.columns)
        if hub not in currencies:
            canonical = canonical.copy()
            canonical[hub] = 1.0
            currencies.append(hub)

        out = {}
        for quote in currencies:
            for base in currencies:
                if quote == base:
                    out[(quote, base)] = pd.Series(1.0, index=canonical.index)
                else:
                    # price in `quote` -> hub -> `base`
                    out[(quote, base)] = canonical[quote] / canonical[base]
        frame = pd.DataFrame(out)
        frame.columns = pd.MultiIndex.from_tuples(frame.columns, names=["quote", "base"])
        return cls(frame, hub=hub)

    @classmethod
    def from_long(cls, long_frame: pd.DataFrame, hub: str = None) -> "FXTable":
        """Build from an explicit long frame with columns `date, quote, base, value`.

        The supplied rows need only cover one direction per currency. The rest of the
        grid — reciprocals and cross pairs — is **derived**, never taken from the input,
        so `value(A,B) x value(B,A) = 1` holds exactly by construction.

        Args:
            long_frame: columns `date`, `quote`, `base`, `value`.
            hub: pivot currency. Defaults to the most frequently occurring `base`.

        Returns:
            `FXTable` with every ordered pair present.

        Raises:
            FXError: if a currency cannot be connected to the hub.
        """
        frame = pd.DataFrame(long_frame)
        if hub is None:
            hub = frame["base"].value_counts().index[0]

        canonical = {}
        for (quote, base), grp in frame.groupby(["quote", "base"], sort=False):
            series = grp.set_index("date")["value"].astype(float)
            if base == hub:
                canonical.setdefault(quote, series)
            elif quote == hub:
                canonical.setdefault(base, 1.0 / series)

        if not canonical:
            raise FXError(
                f"No rows connect any currency to the hub {hub!r}. Supply at least one "
                f"row per currency quoted against it, or pass an explicit hub."
            )
        unreachable = ({*frame["quote"], *frame["base"]} - set(canonical) - {hub})
        if unreachable:
            raise FXError(
                f"Cannot connect {sorted(unreachable)} to the hub {hub!r}. Every currency "
                f"needs one row against the hub so the grid can be derived rather than "
                f"fetched in both directions."
            )
        return cls.from_canonical(pd.DataFrame(canonical), hub=hub)

    @classmethod
    def constant(cls, pairs: dict, index, hub: str = "USD") -> "FXTable":
        """A static table — one rate per pair, held flat across `index`.

        Useful for tests and for backtests where FX drift is not the subject.
        """
        canonical = pd.DataFrame({c: float(v) for c, v in pairs.items()}, index=pd.Index(index))
        return cls.from_canonical(canonical, hub=hub)

    # ############# Access ############# #

    @property
    def currencies(self) -> list:
        """Every currency present in the grid."""
        return sorted({q for q, _ in self.rates.columns})

    def get(self, quote: str, base: str, index=None) -> pd.Series:
        """The series you multiply a `quote`-currency price by to get `base`.

        Args:
            quote: currency the price is in.
            base: currency you want.
            index: optional index to align to.

        Returns:
            `pd.Series` of rates.

        Raises:
            FXError: if the pair is absent from the grid.
        """
        if quote == base:
            series = pd.Series(1.0, index=self.rates.index)
        elif (quote, base) in self.rates.columns:
            series = self.rates[(quote, base)]
        else:
            raise FXError(
                f"No rate for {quote}->{base}. Present: {sorted(set(self.rates.columns))[:8]}..."
            )
        if index is not None:
            series = self.align(series, index)
        return series

    def align(self, series: pd.Series, index, fill_policy: str = "ffill_with_warning") -> pd.Series:
        """Align a rate series to a backtest index under an explicit fill policy.

        FX trades roughly 24/5, equities keep exchange hours, and holidays differ by
        country, so mismatches are guaranteed. Silent forward-filling is not offered:
        the policy is an argument and the default warns.

        Args:
            series: the rate series.
            index: target index.
            fill_policy: `"ffill_with_warning"`, `"ffill"`, `"strict"`, or `"nan"`.
        """
        target = pd.Index(index)
        out = series.reindex(target)
        missing = int(out.isna().sum())
        if missing == 0 or fill_policy == "nan":
            return out
        if fill_policy == "strict":
            raise FXError(
                f"{missing} of {len(target)} bars have no FX rate. Supply a complete "
                f"table, or choose fill_policy='ffill' deliberately."
            )
        if fill_policy == "ffill_with_warning":
            warnings.warn(
                f"{missing} of {len(target)} bars have no FX rate and were forward-filled. "
                f"Calendars differ between the price series and the rate series; pass "
                f"fill_policy explicitly to silence this.",
                stacklevel=3,
            )
        combined = pd.concat([series, out[out.isna()]]).sort_index()
        combined = combined[~combined.index.duplicated(keep="first")].ffill().bfill()
        return combined.reindex(target)

    # ############# Validation ############# #

    def validate(self, tolerance: float = 1e-4) -> None:
        """Check reciprocal consistency and triangulation.

        Both checks are cheap once the full grid is held, and they catch stale rows, bad
        joins and mixed-source contamination. Divergence beyond a basis point warns.
        """
        currencies = self.currencies
        for a in currencies:
            for b in currencies:
                if a >= b or (a, b) not in self.rates.columns or (b, a) not in self.rates.columns:
                    continue
                product = self.rates[(a, b)] * self.rates[(b, a)]
                worst = float(np.nanmax(np.abs(product - 1.0)))
                if worst > tolerance:
                    warnings.warn(
                        f"Reciprocal check failed for {a}/{b}: "
                        f"value({a},{b}) x value({b},{a}) deviates from 1 by up to {worst:.2e}. "
                        f"The table was probably assembled from two independent fetches; "
                        f"derive from one canonical direction instead.",
                        stacklevel=3,
                    )
                    return
        for a in currencies:
            for b in currencies:
                for c in currencies:
                    if len({a, b, c}) < 3:
                        continue
                    if not all(k in self.rates.columns for k in ((a, c), (a, b), (b, c))):
                        continue
                    direct = self.rates[(a, c)]
                    via = self.rates[(a, b)] * self.rates[(b, c)]
                    worst = float(np.nanmax(np.abs(direct / via - 1.0)))
                    if worst > tolerance:
                        warnings.warn(
                            f"Triangulation check failed for {a}->{c} via {b}: "
                            f"deviation up to {worst:.2e}.",
                            stacklevel=3,
                        )
                        return

    def __repr__(self) -> str:
        return f"FXTable({len(self.currencies)} currencies, {len(self.rates)} bars, hub={self.hub})"
