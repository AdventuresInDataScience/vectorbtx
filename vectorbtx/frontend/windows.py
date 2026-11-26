"""Section N — several windows of one history in one backtest (OpenStrategy Studio issue 42).

A backtest of one strategy over several windows of its history, each window its own account from
`init_cash` (cross-validation folds, walk-forward tests, random periods), took one front-door call
per window. Most of a call is fixed work around vectorbt's simulation, the re-sizing loop's passes
(`backtest._settle`: preparing the run, simulating it, reading the account off it), and it was paid
again for every window: on the Studio owner's levered, risk-sized daily strategy about 0.43 s of a
0.47 s call on one window, around a few milliseconds of vectorbt's own numba simulation.

`PortfolioBacktest(..., windows=[(start, end), ...])` makes them one call. Each window is a block
of columns of its own, over the bars from the first window's start to the last one's end:

* every per-column input is repeated for each block (`tile`): the prices, the signals, the stops
  and targets, the sizes, the volume, the conversion rate, the instruments, the starting cash and
  vectorbt's own per-bar and per-column arguments, the callbacks' argument arrays among them;
* a block's signals are kept on its window's signal bars only, from its first bar up to the one
  before its last (a signal on the window's last bar would fill after it, as one on the last bar of
  a run fills nowhere). So before its window a block holds nothing and its account is exactly
  `init_cash`, and after it no entry fills, so nothing after the window moves what the re-sizing
  loop works out per account (a size, a ceiling, the grown margin loan of issue 41, all read at
  entries). `cut` checks the first and refuses to return a block that traded before its window;
* the backtest is run once, the re-sizing loop settling every block at once: the blocks are
  separate accounts, so each settles exactly as it would alone, and the loop's passes are shared;
* each block is cut to its window's bars and given the call's own column names back (`cut`).

So each window's portfolio is the one the same call makes on that window's bars alone
(`.loc[start:end]` of every per-bar input): the same orders, ids included, the same value,
returns, trades and stats. `tests/test_windows.py` checks it on every route covered.

Three things are done per window, so they come out as a call on the window alone has them:

* **Shared cash.** Under `cash="shared"` each window is one shared account: the blocks are the cash
  groups, each with its own margin loan (`buying_power_by_window`) and its own call order, worked
  out on the window's own bars (`shared_call_seq_by_window`), so `entry_priority="random"` draws
  what a call on the window draws.
* **Automatic slippage** measures its volatility and average volume from the window's first bar
  (`first_bars`), not from the bars before it.
* **Per-bar stops, targets and levels** are blanked outside the window, so a column whose stop is
  missing throughout the window (an ATR still warming up) is a column with no stop, as it is alone.

A warning about one window's column names it as `<column> (window k)`, k counting from 1: inside
the call the columns carry those names (`window_name`).

Not covered, and refused: a rebalanced portfolio (`weights=`), whose cash fit and leverage budget
work across every column of a bar; a buy-and-hold, which orders on the run's first bar; a raw
`cash_sharing=`, `group_by=` or `call_seq=` (give `cash=`), whose grouping cannot be repeated per
window; and a custom `signal_func_nb`, which makes its own signals, so they cannot be kept to a
window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .prepare import BacktestError

#: vectorbt arguments a block cannot repeat: they group or order the call's columns as one.
_GROUPING_ARGS = ("cash_sharing", "group_by", "call_seq")


def window_name(column, k) -> str:
    """The name window `k`'s copy of `column` carries inside the call, and in its warnings."""
    return f"{column} (window {k + 1})"


def window_rows(windows, index) -> list:
    """Each window's first and last bar, as positions on `index`, both inclusive.

    A window is a `(start, end)` pair of labels, read as `.loc[start:end]` reads them: both
    inclusive. It must lie inside the index and hold at least two bars, one to signal on and one
    to fill on.
    """
    if isinstance(windows, (str, bytes, dict)) or not hasattr(windows, "__iter__"):
        raise BacktestError(
            "`windows` takes a list of (start, end) pairs, one per window, each two labels on the "
            f"index; got {type(windows).__name__}."
        )
    windows = list(windows)
    if not windows:
        raise BacktestError("`windows` is empty: give at least one (start, end) pair.")
    if not index.is_monotonic_increasing:
        raise BacktestError("`windows` needs the bars in time order: sort the price frames by their index.")
    rows = []
    for k, window in enumerate(windows):
        try:
            start, end = window
        except (TypeError, ValueError):
            raise BacktestError(
                f"Window {k + 1} is {window!r}: give each window as a (start, end) pair of labels on "
                f"the index."
            ) from None
        first, last = start, end
        if isinstance(index, pd.DatetimeIndex):
            first, last = pd.Timestamp(start), pd.Timestamp(end)
            if index.tz is not None:
                first = first.tz_localize(index.tz) if first.tz is None else first
                last = last.tz_localize(index.tz) if last.tz is None else last
        try:
            outside = first < index[0] or last > index[-1]
        except TypeError:
            raise BacktestError(
                f"Window {k + 1} ({start!r} to {end!r}) is not labelled like the index ({index[0]!r} ...)."
            ) from None
        if outside:
            raise BacktestError(
                f"Window {k + 1} ({start} to {end}) runs outside the bars, which run from {index[0]} to "
                f"{index[-1]}. Keep each window inside them."
            )
        a, b = index.slice_locs(first, last)
        if b - a < 2:
            raise BacktestError(
                f"Window {k + 1} ({start} to {end}) holds {max(int(b - a), 0)} bar(s); a window needs at "
                f"least 2, one to signal on and one to fill on."
            )
        rows.append((int(a), int(b) - 1))
    return rows


def tile(windows, *, close, open, high, low, point_fx, volume, signals, stops, sizing, init_cash,
         instruments, spec_kwargs, entry_priority, kwargs, passthrough, weights, schedule):
    """Every input of a `windows=` call as one block of columns per window.

    Args:
        windows: the `(start, end)` pairs (`window_rows`).
        close, open, high, low: the price frames, on the call's index and columns (coerced).
        point_fx: the conversion rate per bar and column (`fx_rate_of`, read before coercion), or None.
        volume: the volume, or None.
        signals: `{entries, exits, short_entries, short_exits}`, unshifted, as given.
        stops: `{stop_points, target_points, trail_points, stop_price, target_price, trail_price}`.
        sizing: `{risk_pct, risk_amount, stake, notional, size, notional_pct}`.
        init_cash, instruments, spec_kwargs, entry_priority: as given to the front door.
        kwargs: the arguments bound for vectorbt (vectorbt's own stops among them).
        passthrough: the `Order` fields passed straight through (`_ORDER_PASSTHROUGH`).
        weights, schedule: refused (see the module docstring).

    Returns:
        dict of the tiled inputs under the same names, `instruments` a `Universe` of the blocks'
        columns (so `spec_kwargs` is spent), and `blocks`: `{"index"` (the bars the call runs on),
        `"column_index"` (the call's own columns), `"windows"` (each `{"rows"`: its first and last
        bar on `index`, `"names"`: its columns' names, `"positions"`: their positions`}`), `"groups"`
        (each column's window, the cash groups under shared cash) and `"first_bars"` (each column's
        first bar, for automatic slippage)`}`.
    """
    from ..resolve.load import Universe
    from .prepare import strategy_of, to_universe

    if weights is not None or schedule is not None:
        raise BacktestError(
            "`windows` runs rule strategies (entries and exits) for now. A rebalanced portfolio fits its "
            "weights to the cash and the leverage budget across every column of a bar, which a block per "
            "window would mix. Run each window as a call of its own."
        )
    if strategy_of(signals["entries"], signals["exits"], None, None, signals["short_entries"],
                   signals["short_exits"]) != "signals":
        raise BacktestError(
            "`windows` runs rule strategies (entries and exits). A buy-and-hold orders on the run's first "
            "bar, not on its window's: run each window as a call of its own."
        )
    given = [name for name in _GROUPING_ARGS if kwargs.get(name) is not None]
    if given:
        raise BacktestError(
            f"`windows` cannot repeat {given} for each window: they group or order the call's columns as "
            f"one. Give `cash='shared'` (one account per window) or `cash='per_instrument'` instead."
        )
    if kwargs.get("signal_func_nb") is not None:
        raise BacktestError(
            "`windows` keeps each window's signals to its own bars, and a custom `signal_func_nb` makes its "
            "own signals, so they cannot be kept to a window. Pass the signals as arrays instead."
        )

    index = close.index
    columns = close.columns
    m = len(columns)
    rows = window_rows(windows, index)
    first = min(a for a, _ in rows)
    last = max(b for _, b in rows)
    bars = index[first:last + 1]
    blocks = {"index": bars, "column_index": columns, "windows": []}
    for k, (a, b) in enumerate(rows):
        blocks["windows"].append({"rows": (a - first, b - first),
                                  "names": [window_name(c, k) for c in columns],
                                  "positions": np.arange(k * m, (k + 1) * m)})
    count = len(rows)
    names = [name for window in blocks["windows"] for name in window["names"]]
    blocks["groups"] = np.repeat(np.arange(count), m)
    blocks["first_bars"] = {name: window["rows"][0] for window in blocks["windows"] for name in window["names"]}

    def on_bars(value):
        """A per-bar value (a frame, a Series on the bars, an array) as a frame on the call's bars."""
        if isinstance(value, pd.DataFrame):
            return value.reindex(index=index, columns=columns)
        if isinstance(value, pd.Series):
            values = value.reindex(index).to_numpy()
            return pd.DataFrame(np.repeat(values[:, None], m, axis=1), index=index, columns=columns)
        values = np.asarray(value)
        if values.ndim == 1:
            values = np.repeat(values[:, None], m, axis=1)
        return pd.DataFrame(values, index=index, columns=columns)

    def blocked(frame, keep=None, fill=None):
        """A frame on the call's bars and columns as the blocks' frame. `keep`: the bars each block keeps,
        `"signals"` (its window's first bar up to the one before its last) or `"window"` (all of it); the
        others are `fill`. None keeps every bar."""
        data = frame.to_numpy()[first:last + 1]
        parts = []
        for window in blocks["windows"]:
            part = data
            if keep is not None:
                s, e = window["rows"]
                part = data.copy()
                part[:s] = fill
                part[e if keep == "signals" else e + 1:] = fill
            parts.append(part)
        return pd.DataFrame(np.concatenate(parts, axis=1), index=bars, columns=names)

    def per_column(values, name):
        """`{column: value}` as the blocks' `{name: value}`."""
        unknown = [c for c in values if c not in columns]
        if unknown:
            raise BacktestError(f"`{name}` names columns not in the backtest: {unknown}.")
        return {window["names"][list(columns).index(c)]: value
                for window in blocks["windows"] for c, value in values.items()}

    def on_columns(value):
        """Whether a Series holds one value per column rather than one per bar."""
        return (not isinstance(value.index, pd.DatetimeIndex) and len(value) > 0
                and all(label in columns for label in value.index))

    def tiled_value(value, name, series, keep=None, as_float=False):
        """One argument in any of its shapes, for the blocks.

        `series`: what a Series means here, `"columns"` (a value per column when its labels are the
        columns, as `stop_values` reads it) or `"bars"` (a value per bar, as `_spread` reads a size).
        `keep`: as `blocked`, for a per-bar value. A shape that fits neither the columns nor the bars
        passes through, for the reader downstream to refuse with its own words.
        """
        if value is None:
            return None
        if isinstance(value, dict):
            return per_column(value, name)
        if isinstance(value, pd.DataFrame):
            frame = on_bars(value)
            return blocked(frame.astype(float) if as_float else frame, keep, np.nan)
        if isinstance(value, pd.Series):
            if series == "columns" and on_columns(value):
                return per_column(value.to_dict(), name)
            if keep is None and series == "bars":
                # A Series stays one: `_spread` repeats a per-bar Series across every column.
                return value.reindex(index).iloc[first:last + 1]
            frame = on_bars(value)
            return blocked(frame.astype(float) if as_float else frame, keep, np.nan)
        array = np.asarray(value)
        if array.ndim == 0:
            return value
        if array.ndim == 1 and len(array) == m:
            return np.tile(array, count)
        if array.ndim == 2 and array.shape == (len(index), m):
            frame = pd.DataFrame(array, index=index, columns=columns)
            return blocked(frame.astype(float) if as_float else frame, keep, np.nan)
        return value

    def tiled_vectorbt(value):
        """One of vectorbt's own arguments for the blocks: a per-bar frame or array (bars x columns) and a
        per-column array or Series repeated, a per-bar Series cut to the call's bars, a tuple (a callback's
        arguments) element by element, anything else as it is.

        ⚠️ A 1-D array as long as the columns is read as one value per column and a 2-D one of the call's
        shape as one per bar and column, as vectorbt broadcasts them; a callback indexing its own arrays by
        bar (`c.i`) then reads them on the call's bars, which the blocks run on.
        """
        if isinstance(value, tuple) and not hasattr(value, "_fields"):
            return tuple(tiled_vectorbt(item) for item in value)
        if isinstance(value, pd.DataFrame):
            if isinstance(value.index, pd.DatetimeIndex) or value.index.equals(index):
                return blocked(on_bars(value))
            return value
        if isinstance(value, pd.Series):
            if on_columns(value):
                return pd.Series(np.tile(value.reindex(columns).to_numpy(), count), index=names, name=value.name)
            if isinstance(value.index, pd.DatetimeIndex) or value.index.equals(index):
                return value.reindex(index).iloc[first:last + 1]
            return value
        if isinstance(value, np.ndarray):
            if value.ndim == 2 and value.shape == (len(index), m):
                return np.ascontiguousarray(np.concatenate([value[first:last + 1]] * count, axis=1))
            if value.ndim == 1 and len(value) == m:
                return np.tile(value, count)
        return value

    def signal(value):
        """A signal (a frame, a Series or 1-D array for every column, None) as the blocks' boolean frame."""
        if value is None:
            return None
        if isinstance(value, pd.DataFrame):
            frame = value.reindex(index=index, columns=columns).fillna(False).astype(bool)
        else:
            frame = on_bars(pd.Series(np.asarray(value).ravel(), index=index) if not isinstance(value, pd.Series)
                            else value).fillna(False).astype(bool)
        return blocked(frame, "signals", False)

    # The instruments, in whichever form they came, as one Universe of the blocks' columns.
    universe = to_universe(instruments, columns=list(columns), **spec_kwargs)
    specs = universe.order_for(list(columns))       # refuses a mismatch, as `prepare` would
    tiled_specs, meta = {}, {}
    for window in blocks["windows"]:
        for column, name, spec in zip(columns, window["names"], specs):
            tiled_specs[name] = spec
            if column in universe.meta:
                meta[name] = universe.meta[column]

    out = dict(
        close=blocked(close), open=blocked(open), high=blocked(high), low=blocked(low),
        point_fx=None if point_fx is None else blocked(on_bars(point_fx)),
        volume=None if volume is None else blocked(on_bars(volume)),
        signals={name: signal(value) for name, value in signals.items()},
        # A per-bar stop is blanked outside its window, so "a stop somewhere in the column" (`stop_values`'
        # named columns) is read on the window's bars, as a call on the window reads it.
        stops={name: tiled_value(value, name, "columns", keep="window", as_float=True)
               for name, value in stops.items()},
        sizing={name: tiled_value(value, name, "columns" if name == "notional_pct" else "bars")
                for name, value in sizing.items()},
        init_cash=init_cash if init_cash is None or np.ndim(init_cash) == 0
        else np.tile(np.asarray(init_cash, dtype=float).ravel(), count),
        instruments=Universe(tiled_specs, meta),
        # Kept on the call's own column names: each window's call order is worked out on its own bars and
        # columns (`shared_call_seq_by_window`).
        entry_priority=entry_priority.reindex(index).iloc[first:last + 1]
        if isinstance(entry_priority, pd.DataFrame) else entry_priority,
        passthrough={name: tiled_vectorbt(value) for name, value in passthrough.items()},
        blocks=blocks,
    )
    kw = {}
    for name, value in kwargs.items():
        if name in ("sl_stop", "tp_stop"):
            kw[name] = tiled_value(value, name, "columns", keep="window", as_float=True)
        elif name == "sl_trail":
            kw[name] = tiled_value(value, name, "columns")
        else:
            kw[name] = tiled_vectorbt(value)
    out["kwargs"] = kw
    return out


def inside(blocks) -> np.ndarray:
    """Each block's own window, as a boolean `(bars, columns)` mask of the call."""
    count = len(blocks["windows"])
    m = len(blocks["column_index"])
    mask = np.zeros((len(blocks["index"]), count * m), dtype=bool)
    for window in blocks["windows"]:
        s, e = window["rows"]
        mask[s:e + 1, window["positions"]] = True
    return mask


def buying_power_by_window(universe, init_cash, columns, blocks):
    """`buying_power` for a shared account per window: each window's starting cash and margin loan.

    Returns `(engine_init_cash, margin_loan)`, one each per window, the cash groups' basis.
    """
    from ..margin.buying_power import buying_power

    cash, loan = [], []
    for window in blocks["windows"]:
        own = [columns[p] for p in window["positions"]]
        money = init_cash if np.ndim(init_cash) == 0 else np.asarray(init_cash, dtype=float)[window["positions"]]
        engine, lent = buying_power(universe, money, own, cash_sharing=True)
        cash.append(np.atleast_1d(np.asarray(engine, dtype=float)))
        loan.append(np.atleast_1d(np.asarray(lent, dtype=float)))
    return np.concatenate(cash), np.concatenate(loan)


def shared_call_seq_by_window(blocks, index, signals, entry_priority=None, seed=None, direction=None,
                              split_bars=False):
    """The call order of a shared account per window, each worked out on its window's own bars and columns.

    `signals`: the shifted signals of the call (`{entries, exits, short_entries, short_exits}`, frames on
    the blocks' columns, or None); `entry_priority`: shifted, on the call's own column names. Each window's
    rows are `shared_call_seq` on that window alone, so a random priority draws what a call on the window
    draws; outside it a block places nothing it would order, and its columns go in column order.
    """
    from .cash import shared_call_seq

    m = len(blocks["column_index"])
    originals = list(blocks["column_index"])
    out = np.tile(np.arange(m, dtype=np.int64), (len(index), len(blocks["windows"])))
    for window in blocks["windows"]:
        s, e = window["rows"]
        positions = window["positions"]
        own = {}
        for name, frame in signals.items():
            if frame is None:
                own[name] = None
                continue
            part = pd.DataFrame(frame).iloc[s:e + 1, positions]
            own[name] = part.set_axis(originals, axis=1)
        priority = entry_priority
        if isinstance(entry_priority, pd.DataFrame):
            priority = entry_priority.iloc[s:e + 1]
        out[s:e + 1, positions] = shared_call_seq(index[s:e + 1], originals, **own, entry_priority=priority,
                                                  seed=seed, direction=direction, split_bars=split_bars)
    return out


def _leaked(k, what):
    # A BacktestError, so an optimiser records the trial rather than stopping (as `_settle`'s `unsettled`).
    return BacktestError(
        f"Window {k + 1} has {what} before its first bar, so its account did not start there from "
        f"`init_cash`. Please report this: a window's signals are kept to its own bars, so it should not."
    )


def cut(pf, blocks) -> list:
    """The call's portfolio as one per window: each block on its window's bars, with the call's columns.

    The order records are the block's on those bars, re-based on the window's first bar and numbered again
    in the order vectorbt made them (by account, then bar), which is how a call on the window numbers them;
    log records likewise, with their order ids. Starting cash, the margin loan and the call order are each
    block's own; the holding costs, highs and lows are cut to the window; the universe is renamed.

    Raises:
        BacktestError: when a block holds an order, a log or a holding cost before its window, which would
            mean its account did not start there from `init_cash` (not expected: see `tile`).
    """
    from vectorbt.base.reshape_fns import to_1d_array, to_2d_array

    from ..resolve.load import Universe

    shared = bool(pf.cash_sharing) and bool(pf.wrapper.grouper.is_grouped())
    index = pf.wrapper.index
    originals = blocks["column_index"]
    close = to_2d_array(pf.close)
    orders = pf.order_records
    logs = pf.log_records
    call_seq = None if pf._call_seq is None else np.asarray(pf._call_seq)
    init = pf._init_cash
    loan = pf.margin_loan
    costs = None if pf.holding_costs is None else pd.DataFrame(pf.holding_costs)
    high = None if pf.high is None else pd.DataFrame(pf.high)
    low = None if pf.low is None else pd.DataFrame(pf.low)
    drift = None if pf.drift is None else pd.DataFrame(pf.drift)

    def rows_of(frame, window):
        if frame is None:
            return None
        s, e = window["rows"]
        names = [name for name in window["names"] if name in frame.columns]
        part = frame[names].iloc[s:e + 1]
        return part.set_axis([originals[window["names"].index(name)] for name in names], axis=1)

    out = []
    for k, window in enumerate(blocks["windows"]):
        s, e = window["rows"]
        positions = window["positions"]
        lo, m = int(positions[0]), len(positions)

        mine = (orders["col"] >= lo) & (orders["col"] < lo + m)
        if (mine & (orders["idx"] < s)).any():
            raise _leaked(k, "an order")
        kept = orders[mine & (orders["idx"] <= e)].copy()
        renumbered = dict(zip(kept["id"].tolist(), range(len(kept))))
        kept["id"] = np.arange(len(kept), dtype=kept["id"].dtype)
        kept["col"] -= lo
        kept["idx"] -= s

        new_logs = logs
        if len(logs):
            own = (logs["col"] >= lo) & (logs["col"] < lo + m)
            if (own & (logs["idx"] < s)).any():
                raise _leaked(k, "a log record")
            new_logs = logs[own & (logs["idx"] <= e)].copy()
            new_logs["id"] = np.arange(len(new_logs), dtype=new_logs["id"].dtype)
            new_logs["col"] -= lo
            new_logs["idx"] -= s
            if "group" in new_logs.dtype.names:
                new_logs["group"] = 0 if shared else new_logs["col"]
            if "order_id" in new_logs.dtype.names:
                new_logs["order_id"] = [renumbered.get(int(old), -1) for old in new_logs["order_id"]]

        new_costs = None
        if costs is not None:
            names = [name for name in window["names"] if name in costs.columns]
            before = costs[names].iloc[:s].to_numpy(dtype=float)
            if np.any(before != 0.0):
                raise _leaked(k, "a holding cost")
            new_costs = rows_of(costs, window)

        wrapper = pf.wrapper.replace(index=index[s:e + 1], columns=originals, group_by=True if shared else None)
        accounts = [k] if shared else positions
        universe = None
        if pf.universe is not None:
            universe = Universe(
                {column: pf.universe.specs[name] for column, name in zip(originals, window["names"])},
                {column: pf.universe.meta[name] for column, name in zip(originals, window["names"])
                 if name in pf.universe.meta},
            )
        # Built afresh rather than with `pf.replace`, which copies the whole call's config (its frames over
        # every block) once per window; these are every field a portfolio holds (vectorbt's, and vectorbtX's
        # own: `Portfolio.__init__`).
        out.append(type(pf)(
            wrapper,
            wrapper.wrap(close[s:e + 1][:, positions], group_by=False),
            kept,
            new_logs,
            init if isinstance(init, int) else to_1d_array(init)[accounts],
            pf.cash_sharing,
            call_seq=None if call_seq is None else call_seq[s:e + 1][:, positions],
            fillna_close=pf.fillna_close,
            trades_type=pf.trades_type,
            engine=pf.engine,
            universe=universe,
            drift=rows_of(drift, window),
            holding_costs=new_costs,
            margin_loan=None if loan is None else np.atleast_1d(np.asarray(loan, dtype=float))[accounts],
            high=rows_of(high, window),
            low=rows_of(low, window),
        ))
    return out
