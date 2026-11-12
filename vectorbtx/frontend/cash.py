"""Section N — the cash model: an account per instrument, or one shared account.

A portfolio of signal strategies is usually **one** account trading several instruments.
vectorbt's default replicates `init_cash` per column instead, so three symbols on 10,000
trade 30,000 and nothing says so. Both models are legitimate; the front door makes the
choice explicit:

* `cash="per_instrument"` — each column trades its own copy of `init_cash` (today's
  default, and what `cash=None` still means for a signal strategy).
* `cash="shared"` — one account: `cash_sharing=True, group_by=True`, funded as one pool.

A shared account needs a rule for **who gets filled when entries compete for cash**, and
that rule is simulation semantics, so it lives here rather than in any caller:

* `entry_priority` — `None` (column order), a list of columns (a static order), a
  DataFrame of per-bar scores (higher first; aligned with the signals and shifted with
  them), or `"random"` (a seeded permutation of the competing entries on each bar).
* `when_cash_short` — `"skip"` (the default under shared cash: a risk-sized position
  should not silently shrink) or `"partial"` (fill what the cash allows). Maps to
  vectorbt's `allow_partial`.

**Priority decides only between entries.** On each bar the orders that close or reduce a
position at the open go first, so the cash they free is there for the entries, then
entries in priority order. A plain vectorbt `call_seq` cannot say this: an array ranks
*every* order in the bar, exits included, and `call_seq="auto"` puts exits first but then
ranks entries by order value rather than by priority.

**Stops are timed, not just ordered.** With a stop or target active, each bar is simulated
as an open step and the rest of the bar (`vectorbtx.frontend.split_bars`). A stop that
gaps through at the open fires on the open step, before the entries, so its cash funds
them — it was freed at the open. A stop hit intrabar fires on the rest-of-bar step, after
every entry, so its cash never funds an entry at an open it came after. Neither bias, in
either direction.

Without split bars (a custom numba callback rules them out) a column's intrabar stop and
its bar's entries share one step, so columns with no signal are ordered after the
entries: the intrabar stop cannot fund them, at the cost of a gap stop's cash waiting a bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .prepare import BacktestError

CASH_MODELS = ("per_instrument", "shared")

#: `when_cash_short` -> vectorbt's `allow_partial`.
WHEN_CASH_SHORT = {"skip": False, "partial": True}


def resolve_cash_model(cash, strategy, sized, sim_kwargs, entry_priority=None,
                       when_cash_short=None):
    """Validate the cash arguments and write the vectorbt ones into `sim_kwargs`.

    Args:
        cash: `None`, `"per_instrument"` or `"shared"`.
        strategy: `"signals"`, `"weights"` or `"hold"`, from `strategy_of`.
        sized: whether a sizing argument was given (a hold with no size buys with all of
            its cash, which a shared account cannot split sensibly).
        sim_kwargs: the arguments bound for vectorbt. Mutated in place.
        entry_priority: see the module docstring.
        when_cash_short: `None`, `"skip"` or `"partial"`.

    Returns:
        The resolved model: `"shared"`, `"per_instrument"`, or None when the caller left
        it to raw vectorbt arguments (`cash_sharing=` / `group_by=`), as before.

    Raises:
        BacktestError: on an unknown value, or a combination with no coherent meaning.
    """
    if cash is not None and cash not in CASH_MODELS:
        raise BacktestError(f"`cash` must be one of {list(CASH_MODELS)} or None, got {cash!r}.")
    if when_cash_short is not None and when_cash_short not in WHEN_CASH_SHORT:
        raise BacktestError(
            f"`when_cash_short` must be one of {list(WHEN_CASH_SHORT)} or None, got "
            f"{when_cash_short!r}."
        )
    if when_cash_short is not None and "allow_partial" in sim_kwargs:
        raise BacktestError(
            "Give `when_cash_short` or `allow_partial`, not both: `when_cash_short='skip'` "
            "IS `allow_partial=False`, and two sources for one setting is how they end up "
            "disagreeing."
        )

    if cash == "shared":
        if sim_kwargs.get("cash_sharing") is False:
            raise BacktestError("`cash='shared'` contradicts `cash_sharing=False`. Drop one.")
        group_by = sim_kwargs.get("group_by")
        if group_by is not None and group_by is not True:
            raise BacktestError(
                "`cash='shared'` is ONE account across every column, so it takes no "
                "`group_by`. For several separate pools, pass `cash_sharing=True` and "
                "`group_by=` straight to vectorbt instead (and order them with `call_seq=`)."
            )
        if strategy == "hold" and not sized:
            raise BacktestError(
                "A buy-and-hold with no size buys with all the cash it has, so in a shared "
                "account the first column would take the whole pot. For an equal-weight "
                "book use `weights=`, or give a size."
            )
        sim_kwargs["cash_sharing"] = True
        sim_kwargs["group_by"] = True
        if when_cash_short is None:
            when_cash_short = "skip"
    elif cash == "per_instrument":
        if sim_kwargs.get("cash_sharing"):
            raise BacktestError("`cash='per_instrument'` contradicts `cash_sharing=True`. Drop one.")
        if strategy == "weights":
            raise BacktestError(
                "A rebalanced (`weights=`) portfolio is one account by construction: target "
                "weights are fractions of a single equity. Use `cash='shared'` or leave "
                "`cash` unset."
            )
    elif strategy == "weights":
        cash = "shared"

    if entry_priority is not None:
        if cash != "shared":
            raise BacktestError(
                "`entry_priority` decides who is filled when entries compete for one "
                "account's cash, so it needs `cash='shared'`. With a separate account per "
                "instrument nothing competes."
            )
        if strategy != "signals":
            raise BacktestError(
                f"`entry_priority` orders competing entry signals; a {strategy!r} "
                f"backtest has none."
            )
        if "call_seq" in sim_kwargs:
            raise BacktestError(
                "Give `entry_priority` or `call_seq`, not both: `entry_priority` builds the "
                "call sequence itself (closing orders first, then entries by priority)."
            )

    if when_cash_short is not None:
        sim_kwargs["allow_partial"] = WHEN_CASH_SHORT[when_cash_short]
    return cash


def shared_call_seq(index, columns, entries=None, exits=None, short_entries=None,
                    short_exits=None, entry_priority=None, seed=None, direction=None,
                    split_bars=False):
    """Per-bar call order for one shared account.

    Closing orders first, then entries by priority. Every signal is **already shifted**
    to the bar it fills on, and so is a DataFrame `entry_priority`.

    Where the columns with no signal go depends on `split_bars`:

    * **True** (`vectorbtx.frontend.split_bars`): this is the order of the OPEN step, where
      a column with no signal can only be stopped by a gap through its stop — at the
      open. That frees cash at the open, so it goes before the entries. Intrabar stops
      run on the rest-of-bar step, after every entry.
    * **False**: one step per bar, so a column with no signal may be stopped intrabar,
      after the open. It goes after the entries, so that cash cannot fund them.

    Args:
        index, columns: the backtest's index and column order.
        entries, exits, short_entries, short_exits: shifted boolean signals, or None.
        entry_priority: None, a list of columns, a DataFrame of scores, or `"random"`.
        seed: seed for `"random"`.
        direction: vectorbt's `direction`, which decides what `entries`/`exits` mean when
            no short arrays are given.
        split_bars: whether the bar is simulated as an open step plus the rest of the bar.

    Returns:
        `(n_bars, n_cols)` int64 array for `call_seq=`: row `i` lists the column offsets
        in the order they are processed on bar `i`.

    Notes:
        A column with any entry signal counts as entering, even if it also exits: a
        reversal closes and reopens, and the reopening is what competes for cash.
    """
    columns = list(columns)
    shape = (len(index), len(columns))

    def _mask(signal):
        if signal is None:
            return np.zeros(shape, dtype=bool)
        if isinstance(signal, pd.Series) or np.ndim(signal) == 1:
            row = pd.Series(np.asarray(signal).ravel(), index=index).fillna(False)
            return np.repeat(row.to_numpy(dtype=bool)[:, None], shape[1], axis=1)
        frame = pd.DataFrame(signal).reindex(index=index, columns=columns)
        return frame.fillna(False).to_numpy(dtype=bool)

    entries, exits = _mask(entries), _mask(exits)
    if short_entries is not None or short_exits is not None:
        # vectorbt's long/short mode: `entries`/`exits` are the long side, whatever
        # `direction` says.
        opening = entries | _mask(short_entries)
        closing = exits | _mask(short_exits)
    else:
        mode = _direction_mode(direction)
        if mode == "both":
            # `exits` are short entries here: each one opens (or reverses into) a short.
            opening, closing = entries | exits, np.zeros(shape, dtype=bool)
        else:
            # longonly: buy / sell. shortonly: short / cover. Either way, entries open.
            opening, closing = entries, exits

    entering, idle = (2, 1) if split_bars else (1, 2)
    group = np.where(opening, entering, np.where(closing, 0, idle))
    by_column = np.broadcast_to(np.arange(shape[1]), shape)
    rank = np.where(group == entering, _entry_rank(entry_priority, index, columns, seed), by_column)
    # Group first, then rank within it. Ranks are < 2 * n_cols, so this cannot overlap.
    key = group * (2 * shape[1] + 1) + rank
    return np.argsort(key, axis=1, kind="stable").astype(np.int64)


def _direction_mode(direction) -> str:
    """'longonly', 'shortonly' or 'both', from whatever vectorbt accepts."""
    if direction is None:
        try:
            from vectorbt._settings import settings

            direction = settings["portfolio"]["signal_direction"]
        except Exception:  # pragma: no cover - vectorbt is always present on this path
            direction = "longonly"
    if np.ndim(direction) != 0:
        raise BacktestError(
            "`cash='shared'` needs one `direction` for the whole portfolio to tell entries "
            "from exits. For per-column directions, pass `short_entries=`/`short_exits=` "
            "alongside `entries=`/`exits=` instead."
        )
    if isinstance(direction, str):
        name = direction.replace("_", "").lower()
    else:
        from vectorbt.portfolio.enums import Direction

        name = Direction._fields[int(direction)].lower()
    if name not in ("longonly", "shortonly", "both"):
        raise BacktestError(f"Unknown direction {direction!r}.")
    return name


def _entry_rank(entry_priority, index, columns, seed) -> np.ndarray:
    """Per-bar rank of each column among entries: 0 is filled first."""
    shape = (len(index), len(columns))
    by_column = np.broadcast_to(np.arange(shape[1]), shape)

    if entry_priority is None:
        return by_column

    if isinstance(entry_priority, str):
        if entry_priority != "random":
            raise BacktestError(
                f"`entry_priority` must be None, a list of columns, a DataFrame of scores or "
                f"'random', got {entry_priority!r}."
            )
        keys = np.random.default_rng(seed).random(shape)
        return _ranks(keys)

    if isinstance(entry_priority, pd.DataFrame):
        unknown = [c for c in entry_priority.columns if c not in columns]
        if unknown:
            raise BacktestError(f"`entry_priority` scores name columns not in the backtest: {unknown}.")
        scores = entry_priority.reindex(index=index, columns=columns).to_numpy(dtype=float)
        # Higher first; a missing score goes last; ties keep column order.
        keys = np.where(np.isnan(scores), np.inf, -scores)
        return _ranks(keys)

    if isinstance(entry_priority, (list, tuple, pd.Index)):
        order = list(entry_priority)
        unknown = [c for c in order if c not in columns]
        if unknown:
            raise BacktestError(f"`entry_priority` names columns not in the backtest: {unknown}.")
        if len(set(order)) != len(order):
            raise BacktestError(f"`entry_priority` lists a column twice: {order}.")
        # Listed columns in list order; any left out follow, in column order.
        static = np.array(
            [order.index(c) if c in order else len(order) + k for k, c in enumerate(columns)]
        )
        return np.broadcast_to(static, shape)

    raise BacktestError(
        f"`entry_priority` must be None, a list of columns, a DataFrame of scores or "
        f"'random', got {type(entry_priority).__name__}."
    )


def _ranks(keys: np.ndarray) -> np.ndarray:
    """Row-wise rank of `keys`, ascending, ties broken by column order."""
    order = np.argsort(keys, axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(keys.shape[1])[None, :].repeat(keys.shape[0], 0), axis=1)
    return ranks
