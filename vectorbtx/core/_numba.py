"""Numba shim.

vectorbtX's kernels are written in numba style so they read like upstream's, but the
pure-arithmetic modules must import without numba present (they are unit-tested in
isolation and numba is a heavy import). If numba is unavailable, `njit` degrades to a
no-op decorator and the kernels run as plain NumPy/Python.
"""

try:  # pragma: no cover - trivial branch
    from numba import njit

    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        """No-op stand-in for `numba.njit`."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator


__all__ = ["njit", "HAS_NUMBA"]
