"""Minimal numba stand-in: pure-Python execution, no compilation."""
import numpy as np

__version__ = "0.66.0-shim"


def njit(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    def deco(f):
        return f
    return deco


jit = njit
generated_jit = njit


def vectorize(*a, **k):
    def deco(f):
        return np.vectorize(f)
    return deco


guvectorize = vectorize
prange = range


class _Types:
    def __getattr__(self, item):
        return object


types = _Types()


class typeof:
    def __init__(self, *a, **k):
        pass


def objmode(*a, **k):
    raise NotImplementedError


config = type("cfg", (), {"DISABLE_JIT": True})()


def literal_unroll(x):
    return x


def literally(x):
    return x


int64 = np.int64
float64 = np.float64
boolean = np.bool_
b1 = np.bool_
i8 = np.int64
f8 = np.float64
