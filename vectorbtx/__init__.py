"""vectorbtX — a pure wrapper over vectorbt 1.1.0.

    import vectorbtx as vbt

is a drop-in replacement for `import vectorbt as vbt`. Everything upstream still works
unchanged; extras are additive and live under the `.x` namespace, except metrics, which
appear alongside native ones in `.stats()` (D5).

See README.md for the file-by-file map and docs/API.md for the reference.
"""

__version__ = "0.1.0"

# 1. Re-export the entire upstream namespace.
from vectorbt import *  # noqa: F401,F403
from vectorbt import settings, nb, plotting  # noqa: F401
from vectorbt import __version__ as vectorbt_version  # noqa: F401

# 2. Swap in our Portfolio subclass, shadowing vbt's.
from .core.portfolio import Portfolio  # noqa: F401

# 3. Patch accessors and record classes, and register settings.
from .core.patch import apply_patches
from .core.settings import register_defaults

apply_patches()
register_defaults()

# 4. The `vbt.x` namespace for constructors and loaders.
from . import x  # noqa: F401,E402

# 5. Public vectorbtX types, re-exported at top level for convenience.
from .instruments.spec import Instrument  # noqa: F401,E402
from .resolve.load import Universe, load_universe  # noqa: F401,E402
