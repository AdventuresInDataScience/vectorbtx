"""Section B — `settings["vectorbtx"]`.

A single top-level key is added to vectorbt's settings. `settings["portfolio"]` is never
touched: mutating it would change behaviour for plain vectorbt code paths and break the
drop-in guarantee (§1.3).
"""

DEFAULTS = dict(
    # quantization
    rounding="floor",
    # costs
    financing_markup=0.03,
    # The broker's admin fee on the tom-next rate. Left at zero the FX carry number is
    # missing the broker's entire margin on it, which flatters carry strategies
    # specifically. IG quotes 0.5-1% on forex and spot metals.
    carry_markup=0.01,
    fx_markup=0.005,
    slippage_max=0.005,
    slippage_coef=1.0,
    vol_window=20,
    # accrual
    triple_day="WED",
    end_of_day="22:00",
    day_count=360.0,
    # construction
    min_trade=None,
    fill_policy="ffill_with_warning",
    # rates
    rate_series=dict(GBP="SONIA", USD="SOFR", EUR="ESTR", JPY="TONA", CHF="SARON"),
    static_rate=None,
    cache_dir=None,
    # validation
    triangulation_tolerance=1e-4,
)


def register_defaults(force: bool = True) -> None:
    """Register `settings["vectorbtx"]` on vectorbt's global settings.

    Idempotent: re-registering leaves user overrides intact.
    """
    from vectorbt._settings import settings

    if "vectorbtx" in settings:
        return
    settings.__setitem__("vectorbtx", dict(DEFAULTS), force=force)


def get(key: str, default=None):
    """Read one vectorbtX setting, falling back to the module default."""
    try:
        from vectorbt._settings import settings

        return settings["vectorbtx"].get(key, DEFAULTS.get(key, default))
    except Exception:  # vectorbt not importable (pure-arithmetic unit tests)
        return DEFAULTS.get(key, default)
