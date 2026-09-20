"""Case alias for the `vectorbtx` package.

**Why this file exists.**

The distribution is named `vectorbtX` (capital X) and every line of documentation, the
plan, and the drop-in contract in §1.3 is written as::

    import vectorbtX as vbt

The importable package directory, however, is lowercase `vectorbtx/` — matching PEP 8 and
matching how the code refers to itself internally (`from vectorbtx.x import ...`).

On macOS and Windows those two spellings resolve to the same thing, because the
filesystem is case-insensitive. **On Linux they do not**, so `import vectorbtX` raises
`ModuleNotFoundError` there while working fine on a developer's laptop — a platform-
specific break in the project's single most-typed line, and one that CI on
`{ubuntu, windows, macos}` would report as failing on exactly one of the three.

This module closes that gap. It is a pure alias: it imports the real package and rebinds
this module object to it in `sys.modules`, so both spellings return **the same module
object**, not two copies.

That last point is what matters. A naive `from vectorbtx import *` would create a second,
distinct module, and the two would then disagree in ways that are tedious to debug:

* `apply_patches()` and `register_defaults()` are idempotent via module-level flags, so a
  second module object would hold a second set of flags and could re-run the patching.
* `isinstance(pf, vectorbtX.Portfolio)` would be `False` for a portfolio built through
  `vectorbtx.Portfolio`, because they would be different classes.
* The mode-adapter registry in `modes/base.py` is module-level state and would be
  populated twice.

Rebinding `sys.modules` avoids all of it: `vectorbtX is vectorbtx` is `True`.

There is nothing to maintain here. Anything added to `vectorbtx/__init__.py` is reachable
through this alias automatically, because there is only ever one module.
"""

import sys

import vectorbtx

# Both names must resolve to ONE module object, or the class identities, the patch-applied
# flag and the adapter registry all silently fork. Assigning into sys.modules replaces the
# half-initialised module Python created for this file with the real package.
sys.modules[__name__] = vectorbtx
