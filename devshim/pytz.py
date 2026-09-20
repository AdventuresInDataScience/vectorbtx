__version__ = "2024.1"
from datetime import timedelta, timezone, tzinfo  # noqa: F401

utc = timezone.utc
UTC = utc


class UnknownTimeZoneError(Exception):
    pass


class BaseTzInfo(tzinfo):
    pass


def timezone_(name):
    return timezone.utc


def timezone(name):  # noqa: F811
    from datetime import timezone as _tz
    if name in ("UTC", "utc"):
        return _tz.utc
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(name)
    except Exception as exc:
        raise UnknownTimeZoneError(name) from exc


all_timezones = ["UTC"]
common_timezones = ["UTC"]
