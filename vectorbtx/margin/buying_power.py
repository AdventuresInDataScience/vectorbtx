"""Section F — funded margin (F1b).

**The gap this closes.** `cap.py` computes `max_size` so a position cannot exceed
`leverage x equity`. That stops a broker refusing the trade — but nothing in vectorbt
*supplies* the leverage. `buy_nb` ends its cash-limit calculation with

    cash_limit = min(cash_limit, exec_state.cash)          # nb.py:113

so a long is bounded by cash, full stop, and no `Order` field changes that. A 30x spread
bet sized at £1.67/pt on a £10,000 account was therefore filled at about £1.01/pt: the
backtest ran, the equity curve looked plausible, and the leverage silently was not there.

**How a margin account actually works.** Equity is £10,000; buying power is £300,000. The
difference is a loan the broker extends and calls back when you close. So:

* give the **engine** the buying power, `E x L`, as its initial cash;
* record the loan, `E x (L - 1)`, on the portfolio;
* **report** initial cash as `E`, the real equity.

Reported cash is then `E + cumsum(cash_flow)`, so buying £18,333 of asset against £10,000
of equity gives cash of `-£8,333` and value of `-8,333 + 18,333 = £10,000` — correct
equity, and the negative cash *is* the drawn margin loan. Everything downstream
(`value`, `returns`, every metric) follows from `cash`, so all of it comes out right
without touching the engine.

Two things this buys beyond correct sizing:

* **`borrowed_portion` becomes exact and non-trivial.** `max(0, -cash)` is the drawn loan
  by construction. Previously a long could never exceed cash, so it was always ~zero and
  Section G's leverage financing barely fired.
* **`max_size` reverts to being what it was meant to be** — a cap that binds because the
  user set a leverage limit, not because they ran out of money.

**What is still not modelled** (unchanged by this, and deferred by D2): maintenance
margin, margin calls, and forced liquidation. A position can draw the whole loan and sit
there. `min_margin_level` reports how close it came.
"""

from __future__ import annotations

import warnings

import numpy as np


def leverage_of(universe, columns) -> np.ndarray:
    """Per-column maximum leverage, defaulting to 1 (unlevered)."""
    out = []
    for col in columns:
        spec = universe[col].filled() if col in universe else None
        out.append(float((spec.leverage if spec is not None else 1.0) or 1.0))
    return np.asarray(out, dtype=float)


def buying_power(universe, init_cash, columns, cash_sharing: bool = False):
    """Split account equity into the engine's buying power and the margin loan.

    Args:
        universe: the `Universe`, supplying each column's leverage.
        init_cash: account **equity** — what the user actually has.
        columns: resolved column order.
        cash_sharing: whether the columns share one cash balance.

    Returns:
        `(engine_init_cash, margin_loan)`. Both are expressed in the same basis vectorbt
        uses for `init_cash` — **per group** when `cash_sharing` is on, **per column**
        otherwise — so they can be handed straight to the constructor and to
        `Portfolio.get_init_cash`.

    Notes:
        With cash sharing there is one pot, so there is one leverage: the **maximum**
        across the group. Warned about when the columns disagree, because the per-column
        `max_size` caps still bind individually — the group can fund a 30x position in the
        one instrument allowed 30x, and the others remain capped where they were. That is
        the intended reading, but it is worth saying out loud.

        ⚠️ Aggregate exposure across a cash-sharing group is still only enforced exactly
        through Section J's weight-matrix route (`leverage_type="portfolio_level"`).
        A signal-driven strategy gets the per-column backstop plus this shared pot, so it
        can in principle run the whole group up to the largest single leverage. Unchanged
        from the plan's documented known gap.

    Examples:
        >>> buying_power(universe, 10_000, ["EURUSD"], cash_sharing=False)
        (array([300000.]), array([290000.]))
    """
    equity = np.asarray(init_cash, dtype=float)
    leverage = leverage_of(universe, columns)

    if equity.ndim == 0:
        per_col = np.full(len(columns), float(equity))
    elif len(equity) == len(columns):
        per_col = equity.astype(float)
    else:
        raise ValueError(
            f"init_cash has {len(equity)} entries but there are {len(columns)} columns."
        )

    if not cash_sharing:
        return per_col * leverage, per_col * (leverage - 1.0)

    # One pot, therefore one leverage.
    shared_equity = float(per_col[0]) if np.allclose(per_col, per_col[0]) else float(per_col.sum())
    lev = float(leverage.max())
    if not np.allclose(leverage, lev):
        warnings.warn(
            f"Columns share cash but declare different leverage ({dict(zip(columns, leverage))}). "
            f"The shared pot is funded at the maximum, {lev}x; each column's own max_size cap "
            f"still binds it individually. If you meant one portfolio-wide budget, set "
            f"leverage_type='portfolio_level' and use Portfolio.from_weights, where the "
            f"constraint is enforced exactly on the weight matrix.",
            stacklevel=3,
        )
    return (
        np.asarray([shared_equity * lev], dtype=float),
        np.asarray([shared_equity * (lev - 1.0)], dtype=float),
    )


def is_levered(loan) -> bool:
    """True if any margin loan was extended."""
    return loan is not None and bool(np.any(np.asarray(loan, dtype=float) > 0))
