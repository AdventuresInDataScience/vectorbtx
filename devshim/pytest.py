"""Minimal pytest shim for the sandbox (no network, no real pytest)."""
import math
import warnings as _warnings


class _Skip(Exception):
    pass


def skip(reason=""):
    raise _Skip(reason)


def fixture(fn=None, **kw):
    def deco(f):
        f.__is_fixture__ = True
        return f
    return deco(fn) if fn is not None else deco


class _Mark:
    @staticmethod
    def skipif(cond, reason=""):
        def deco(f):
            if cond:
                f.__skip__ = True
            return f
        return deco

    @staticmethod
    def parametrize(names, values):
        if isinstance(names, str):
            names = [n.strip() for n in names.split(",")]
        def deco(f):
            f.__params__ = (names, values)
            return f
        return deco

    def __getattr__(self, item):
        def deco(*a, **k):
            def inner(f):
                return f
            return inner if not (len(a) == 1 and callable(a[0]) and not k) else a[0]
        return deco


mark = _Mark()


class _Approx:
    def __init__(self, expected, rel=None, abs=None):
        self.expected = expected
        self.rel = rel
        self.abs = abs

    def _cmp(self, a, b):
        if isinstance(b, bool) or isinstance(a, bool):
            return a == b
        try:
            a = float(a); b = float(b)
        except (TypeError, ValueError):
            return a == b
        if math.isnan(a) and math.isnan(b):
            return True
        rel = self.rel if self.rel is not None else 1e-6
        abs_ = self.abs if self.abs is not None else 1e-12
        return math.isclose(a, b, rel_tol=rel, abs_tol=abs_)

    def __eq__(self, other):
        exp = self.expected
        if hasattr(exp, "__len__") and not isinstance(exp, (str, bytes)):
            try:
                if len(exp) != len(other):
                    return False
            except TypeError:
                return False
            return all(self._cmp(o, e) for o, e in zip(other, exp))
        return self._cmp(other, exp)

    def __req__(self, other):
        return self.__eq__(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return f"approx({self.expected!r})"


def approx(expected, rel=None, abs=None):
    return _Approx(expected, rel=rel, abs=abs)


class _RaisesCtx:
    def __init__(self, exc, match=None):
        self.exc = exc
        self.match = match
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError(f"DID NOT RAISE {self.exc}")
        if not issubclass(et, self.exc):
            return False
        self.value = ev
        if self.match is not None:
            import re
            if not re.search(self.match, str(ev)):
                raise AssertionError(f"pattern {self.match!r} not found in {str(ev)!r}")
        return True


def raises(exc, match=None):
    return _RaisesCtx(exc, match)


class _WarnsCtx:
    def __init__(self, category, match=None):
        self.category = category
        self.match = match
        self._cm = None
        self.list = []

    def __enter__(self):
        self._cm = _warnings.catch_warnings(record=True)
        self.list = self._cm.__enter__()
        _warnings.simplefilter("always")
        return self

    def __exit__(self, et, ev, tb):
        self._cm.__exit__(et, ev, tb)
        if et is not None:
            return False
        matched = [w for w in self.list if issubclass(w.category, self.category)]
        if not matched:
            raise AssertionError(f"DID NOT WARN {self.category}")
        if self.match is not None:
            import re
            if not any(re.search(self.match, str(w.message)) for w in matched):
                raise AssertionError(f"pattern {self.match!r} not found in warnings")
        return False


def warns(category, match=None):
    return _WarnsCtx(category, match)


def importorskip(name, *a, **k):
    try:
        return __import__(name)
    except ImportError:
        skip(f"no {name}")
