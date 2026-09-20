"""Test bootstrap.

Most of vectorbtX is pure NumPy/pandas arithmetic and is deliberately testable without a
simulation — and therefore without vectorbt or numba installed. When vectorbt *is*
absent, this shim registers `vectorbtx` as a namespace package so submodules import
directly, bypassing `vectorbtx/__init__.py` (which star-imports vectorbt).

When vectorbt is present nothing is stubbed and the full test suite runs.
"""

import os
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import vectorbt  # noqa: F401

    HAS_VECTORBT = True
except ImportError:  # pragma: no cover
    HAS_VECTORBT = False
    if "vectorbtx" not in sys.modules:
        pkg = types.ModuleType("vectorbtx")
        pkg.__path__ = [os.path.join(ROOT, "vectorbtx")]
        pkg.__version__ = "0.1.0"
        sys.modules["vectorbtx"] = pkg

needs_vectorbt = pytest.mark.skipif(not HAS_VECTORBT, reason="requires vectorbt")


@pytest.fixture
def has_vectorbt():
    """True when the real vectorbt is importable."""
    return HAS_VECTORBT
