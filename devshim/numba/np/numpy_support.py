import numpy as np


def as_dtype(t):
    try:
        return np.dtype(t)
    except Exception:
        return np.dtype(object)


def from_dtype(d):
    return np.dtype(d)
