"""Resolution: dict | DataFrame -> validated specs -> numpy args (Section C2)."""

from .coerce import CoercionError
from .export import export_universe
from .keywords import KEYWORDS, KeywordError, check_keyword, is_keyword
from .load import Universe, UniverseError, load_universe
from .resolver import ResolveError, resolve_universe

__all__ = [
    "Universe",
    "load_universe",
    "UniverseError",
    "export_universe",
    "resolve_universe",
    "ResolveError",
    "KEYWORDS",
    "check_keyword",
    "is_keyword",
    "KeywordError",
    "CoercionError",
]
