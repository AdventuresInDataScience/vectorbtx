"""Section M — the currency-keyed rate table, shared by G and I.

Two conversions have to be explicit here, because getting either wrong is silent:

* **Rates are quoted annualised; financing accrues daily.** `annual_to_period` does that
  conversion and nothing else, so it is visible in one place.
* **Day count is ACT/360 for all currencies.** Chosen for simplicity, and because it
  slightly overstates cost — the safer direction. Note the inversion for carry: the same
  arithmetic overstates a *received* credit, so a positive-carry position looks marginally
  better than reality. "Overstates = conservative" is not universal here.

Rate series are published on business days while backtests may run on calendar days, so
alignment needs the same explicit fill policy discipline as Section I — and shares it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

#: Post-LIBOR overnight benchmarks, one per currency (Part 8).
DEFAULT_SERIES = {
    "GBP": "SONIA",
    "USD": "SOFR",
    "EUR": "ESTR",
    "JPY": "TONA",
    "CHF": "SARON",
}

#: FRED series IDs for the benchmarks above.
#:
#: ⚠️ **Verified against the live FRED API, and two of these are not what they look like.**
#: FRED publishes no daily JPY or CHF overnight benchmark, so `TONA` and `SARON` fall back
#: to OECD *monthly* "Immediate Rates" series — both of which have been **discontinued**
#: (last observations 2023-12 and 2024-03). They are the best FRED offers, and they are
#: forward-filled, so a recent backtest gets a stale constant rather than a measured rate.
#: `fred._warn_if_coverage_falls_short` says so at download time; for JPY or CHF financing
#: that actually matters, supply a table through `vbt.x.static_rate_table` instead.
#:
#: GBP, USD and EUR are genuine daily overnight benchmarks and are current.
FRED_SERIES = {
    "SONIA": "IUDSOIA",                    # daily, current
    "SOFR": "SOFR",                        # daily, current
    # Was "ECBESTRVOLWGTTRMDRT", which FRED answers with "The series does not exist" —
    # it is missing the "MN". Every EUR-denominated backtest raised HTTP 400.
    "ESTR": "ECBESTRVOLWGTTRMDMNRT",       # daily, current
    "TONA": "IRSTCB01JPM156N",             # ⚠️ monthly, discontinued 2023-12
    "SARON": "IRSTCI01CHM156N",            # ⚠️ monthly, discontinued 2024-03
}

#: Day-count divisor by the currency the market is **denominated in**.
#:
#: ⚠️ Not a house convention — it follows the market. IG uses a 365-day divisor for the
#: FTSE 100 and other GBP, SGD and ZAR denominated markets, and 360 for all others. The
#: plan's uniform ACT/360 overstates every GBP position's financing by 365/360 - 1 ~ 1.4%.
#: Small per night, systematic over a multi-year backtest, and it inverts for received
#: credits — so "overstating is the safe direction" does not hold here.
DAY_COUNT_BY_CURRENCY = {"GBP": 365.0, "SGD": 365.0, "ZAR": 365.0}

#: Fallback for every other currency.
DAY_COUNT = 360.0


def day_count_for(currency=None) -> float:
    """The ACT/360 or ACT/365 divisor for a market's denomination currency."""
    if currency is None:
        return DAY_COUNT
    return DAY_COUNT_BY_CURRENCY.get(str(currency).upper(), DAY_COUNT)


def annual_to_period(annual_rate, days: float = 1.0, day_count: float = DAY_COUNT):
    """Convert an annualised rate to the fraction accruing over `days`.

    ACT/360: `annual_rate * days / 360`. Explicit rather than assumed, because an
    accidental /365 or a missing division is a silent 1–2% error compounded daily.

    Args:
        annual_rate: annualised rate as a fraction (0.0525 = 5.25%).
        days: number of accrual days, including any triple-day multiplier.
        day_count: denominator, default 360.

    Returns:
        Fraction of notional to charge.

    Examples:
        >>> annual_to_period(0.0525, 1)      # one day at 5.25%
        0.00014583333333333335
        >>> annual_to_period(0.0525, 3)      # a Wednesday triple day
        0.0004375
    """
    return np.asarray(annual_rate, dtype=float) * np.asarray(days, dtype=float) / day_count


def static_rate_table(currencies, index, rate=0.05) -> pd.DataFrame:
    """A flat rate table — no API key, no network.

    The library must work with a static rate if the user does not want a FRED key, so
    this is a supported path rather than a test fixture.

    Args:
        currencies: iterable of currency codes, or a `dict[currency, rate]`.
        index: the backtest index.
        rate: a flat annualised rate for every currency, or a `dict[currency, rate]`.
    """
    if isinstance(rate, dict):
        lookup = {str(c): float(r) for c, r in rate.items()}
        keys = currencies.keys() if isinstance(currencies, dict) else currencies
        missing = [c for c in keys if str(c) not in lookup]
        if missing:
            raise KeyError(
                f"No static rate supplied for {missing}. Give a rate for every currency, "
                f"or pass a single number to apply to all of them."
            )
        data = {c: lookup[str(c)] for c in keys}
    elif isinstance(currencies, dict):
        data = {c: float(r) for c, r in currencies.items()}
    else:
        data = {c: float(rate) for c in currencies}
    return pd.DataFrame(data, index=pd.Index(index))


def rate_table(currencies, index, source: str = "fred", fill_policy: str = "ffill", **kwargs):
    """Benchmark rates aligned to the backtest index, one column per currency.

    Args:
        currencies: iterable of currency codes.
        index: the backtest index.
        source: `"fred"` or `"static"`.
        fill_policy: `"ffill"`, `"ffill_with_warning"`, or `"strict"`. Business-day rate
            series against a calendar-day index will always need one.
        **kwargs: passed to `FREDData.download`, or `rate=` for the static source.

    Returns:
        DataFrame of annualised rates as fractions.

    Notes:
        If FRED is unreachable, the cache is used with a warning; it fails only when
        there is no cache and no static fallback.
    """
    if source == "static":
        return static_rate_table(currencies, index, kwargs.get("rate", 0.05))

    from .fred import FREDData

    try:
        data = FREDData.download_rates(list(currencies), start=index[0], end=index[-1], **kwargs)
    except Exception as exc:
        fallback = kwargs.get("static_rate")
        if fallback is None:
            raise
        warnings.warn(
            f"Rate download failed ({exc}); falling back to the static rate {fallback}.",
            stacklevel=2,
        )
        return static_rate_table(currencies, index, fallback)

    return align_rates(data, index, fill_policy=fill_policy)


def align_rates(rates: pd.DataFrame, index, fill_policy: str = "ffill") -> pd.DataFrame:
    """Align a business-day rate frame to a possibly calendar-day backtest index."""
    target = pd.Index(index)
    out = rates.reindex(rates.index.union(target)).ffill().reindex(target)
    missing = int(out.isna().any(axis=1).sum())
    if missing:
        if fill_policy == "strict":
            raise ValueError(
                f"{missing} bars have no benchmark rate. Rate series are published on "
                f"business days; choose fill_policy='ffill' deliberately or extend the data."
            )
        if fill_policy == "ffill_with_warning":
            warnings.warn(
                f"{missing} bars had no benchmark rate and were back-filled.", stacklevel=2
            )
        out = out.bfill()
    return out
