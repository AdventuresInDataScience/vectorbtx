"""Section N — the front-door API.

A facade over the tested core of C–M. **It adds no new behaviour.** Everything here
normalises user-facing arguments into calls the existing layers already accept, so a
single-instrument spread bet no longer requires a `Universe` of one, a three-call sizing
sequence in one exact order, manual conversion into engine units, and an opaque
`**universe.resolve(...)` splat.

Two names, one implementation:

* `PortfolioBacktest` — the implementation.
* `SingleBacktest` — a thin validating forwarder that rejects the portfolio-only
  arguments and points at `PortfolioBacktest`.

Both return a `vectorbtx.Portfolio`, so `pf.stats()`, `pf.trades.stats()`,
`pf.returns_stats()` and every `pf.x.*` extra keep working with no result wrapper to
learn.
"""

from .backtest import PortfolioBacktest
from .prepare import BacktestError, prepare
from .single import SingleBacktest

__all__ = ["PortfolioBacktest", "SingleBacktest", "prepare", "BacktestError"]
