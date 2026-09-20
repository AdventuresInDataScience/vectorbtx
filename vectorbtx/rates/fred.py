"""Section M — FRED rate data.

Follows the `YFData` / `BinanceData` / `AlpacaData` pattern in `vectorbt/data/custom.py`:
subclass `Data`, implement `download_symbol` and `update_symbol`, and let the base class
handle wrapping, alignment and scheduled refresh via `data/updater.py`.

A free FRED API key is needed. **Optional dependency**: the library works with a static
rate if the user does not want one — see `rates.table.static_rate_table`.

Caching lives in the platform cache directory, keyed by series and date range, and is
refreshed on load for any range extending past the cached end. If FRED is unreachable the
cache is used with a warning; it fails only when there is no cache and no static
fallback.

The broker markup is deliberately **not** here. M supplies benchmarks; G applies the
adverse markup.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import pandas as pd

from .table import DEFAULT_SERIES, FRED_SERIES

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"


def cache_dir() -> Path:
    """Platform cache directory for downloaded rate series."""
    from ..core.settings import get

    configured = get("cache_dir")
    if configured:
        path = Path(configured)
    else:
        base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
        path = Path(base) / "vectorbtx" / "rates"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_path(series_id: str) -> Path:
    return cache_dir() / f"{series_id}.csv"


def read_cache(series_id: str):
    """Read a cached series, or None."""
    path = _cache_path(series_id)
    if not path.exists():
        return None
    series = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    series.name = series_id
    return series


def write_cache(series_id: str, series: pd.Series) -> None:
    """Write a series to the cache, merging with anything already there."""
    existing = read_cache(series_id)
    if existing is not None:
        series = pd.concat([existing, series])
        series = series[~series.index.duplicated(keep="last")].sort_index()
    series.to_frame(series_id).to_csv(_cache_path(series_id))


def _fetch(series_id: str, start=None, end=None, api_key=None) -> pd.Series:
    """One HTTP call to FRED. Raises if `requests` or the key is missing."""
    import requests

    api_key = api_key or os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No FRED API key. Set FRED_API_KEY, pass api_key=..., or use "
            "vbt.x.static_rate_table / rate_table(source='static') instead — "
            "vectorbtX works fine without one."
        )
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
    if start is not None:
        params["observation_start"] = pd.Timestamp(start).strftime("%Y-%m-%d")
    if end is not None:
        params["observation_end"] = pd.Timestamp(end).strftime("%Y-%m-%d")
    response = requests.get(FRED_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = json.loads(response.text)
    rows = [
        (pd.Timestamp(o["date"]), float(o["value"]))
        for o in payload.get("observations", [])
        if o.get("value") not in (".", "", None)
    ]
    if not rows:
        raise RuntimeError(f"FRED returned no observations for {series_id}")
    index, values = zip(*rows)
    # FRED quotes these in percent; vectorbtX works in fractions throughout.
    return pd.Series([v / 100.0 for v in values], index=pd.DatetimeIndex(index), name=series_id)


def _lazy_data_base():
    """Import vectorbt's `Data` lazily so this module stays importable without it."""
    from vectorbt.data.base import Data

    return Data


class FREDData:
    """Benchmark interest rates from FRED.

    Mirrors the `Data` subclass API. Constructed through `download`, which caches and
    refreshes on load rather than re-downloading history.

    Examples:
        >>> rates = FREDData.download(["SOFR", "SONIA"], start="2020-01-01")
        >>> table = FREDData.download_rates(["GBP", "USD"], start="2020-01-01")
    """

    def __init__(self, data: dict):
        self.data = data

    # ############# Download ############# #

    @classmethod
    def download_symbol(cls, symbol, start=None, end=None, cache=True, api_key=None, **kwargs):
        """Download one series, using and refreshing the cache.

        Args:
            symbol: a benchmark name (`"SOFR"`) or a raw FRED series ID.
            start, end: date range.
            cache: read from and write to the platform cache.
            api_key: FRED key; falls back to `$FRED_API_KEY`.

        Returns:
            `pd.Series` of annualised rates as fractions.
        """
        series_id = FRED_SERIES.get(str(symbol).upper(), symbol)
        cached = read_cache(series_id) if cache else None

        need_start = start
        if cached is not None and len(cached):
            if end is None or pd.Timestamp(end) <= cached.index[-1]:
                if start is None or pd.Timestamp(start) >= cached.index[0]:
                    return cls._slice(cached, start, end)
            # Only fetch the tail beyond what we already hold.
            need_start = cached.index[-1]

        try:
            fresh = _fetch(series_id, need_start, end, api_key)
        except Exception as exc:
            if cached is not None:
                warnings.warn(
                    f"FRED unreachable for {series_id} ({exc}); using cached data.",
                    stacklevel=2,
                )
                return cls._slice(cached, start, end)
            raise

        if cache:
            write_cache(series_id, fresh)
            fresh = read_cache(series_id)
        return cls._slice(fresh, start, end)

    @staticmethod
    def _slice(series, start, end):
        if start is not None:
            series = series[series.index >= pd.Timestamp(start)]
        if end is not None:
            series = series[series.index <= pd.Timestamp(end)]
        return series

    @classmethod
    def download(cls, series_ids=None, base="GBP", start=None, end=None, cache=True, **kwargs):
        """Download several series into a `FREDData`.

        Args:
            series_ids: benchmark names or FRED IDs. None uses the Part 8 defaults.
            base: the account currency, retained for reporting.
            start, end: date range.
            cache: use the platform cache.
        """
        if series_ids is None:
            series_ids = list(DEFAULT_SERIES.values())
        data = {
            s: cls.download_symbol(s, start=start, end=end, cache=cache, **kwargs)
            for s in series_ids
        }
        obj = cls(data)
        obj.base = base
        return obj

    @classmethod
    def download_rates(cls, currencies, start=None, end=None, cache=True, **kwargs) -> pd.DataFrame:
        """Download one benchmark per currency and return a currency-keyed frame.

        Args:
            currencies: e.g. `["GBP", "USD"]`.

        Returns:
            DataFrame with one column per currency, annualised rates as fractions.
        """
        out = {}
        for ccy in currencies:
            name = DEFAULT_SERIES.get(str(ccy).upper())
            if name is None:
                raise ValueError(
                    f"No default benchmark for {ccy!r}. Known: {sorted(DEFAULT_SERIES)}. "
                    f"Pass an explicit series or use a static rate."
                )
            out[ccy] = cls.download_symbol(name, start=start, end=end, cache=cache, **kwargs)
        return pd.DataFrame(out).sort_index()

    def update_symbol(self, symbol, **kwargs):
        """Refresh one series from the last cached observation forward."""
        existing = self.data[symbol]
        kwargs["start"] = existing.index[-1]
        return self.download_symbol(symbol, **kwargs)

    def get(self, symbol=None):
        """Return one series, or the whole frame."""
        if symbol is not None:
            return self.data[symbol]
        return pd.DataFrame(self.data).sort_index()

    def to_vbt_data(self):
        """Wrap into a real vectorbt `Data` object, for use with `data/updater.py`."""
        Data = _lazy_data_base()
        return Data.from_data({k: v.to_frame(k) for k, v in self.data.items()})

    def __repr__(self) -> str:
        return f"FREDData({list(self.data)})"
