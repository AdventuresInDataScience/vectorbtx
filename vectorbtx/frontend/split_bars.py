"""Section N — stops on the entry bar, by simulating every bar in two steps.

⚠️ **The bias this removes.** vectorbt's `from_signals` places at most one order per column
per bar, and checks a position's stops *before* that bar's signal. A position opened at
bar `t`'s open therefore has its stops checked from bar `t+1`: whatever bar `t` does after
the open is invisible to it. A stop the entry bar's low trades straight through is never
hit, and the position survives to bar `t+1` as if the drop had not happened. The error is
always in the strategy's favour, and it is largest exactly where stops are tight relative
to the bar's range — short stops on daily data.

**The fix.** Each bar is simulated as two steps:

1. **The open step** — open, high, low and close all equal to the bar's open. Signals fill
   here, at the open. A stop that gaps through at the open fires here, at the open.
2. **The rest of the bar** — the bar's real open, high, low and close, and no signals.
   Every position's stops are checked against the whole range — including those of a
   position opened at this bar's open.

The order records are then mapped back onto the original bars, so the portfolio, every
metric and every per-bar series stay on the original index. A bar can now carry two
orders for one column (an entry and its stop), which vectorbt's flow accounting adds up.
A position opened and stopped on the same bar holds nothing overnight, so it is charged
no financing or carry — correct, since it was closed before the cut-off.

Two further consequences, both towards realism:

* **An exit signal beats an intrabar stop on the same bar.** The signal fills at the open
  step; the stop could only have been hit later in the bar. vectorbt on whole bars let the
  stop win, at the stop price, even though the open came first.
* **Stop orders inside one bar are unchanged otherwise:** a stop beats a target when both
  are in range (the pessimistic reading of an unknown intrabar path), and a trailing stop
  ratchets on the bar's high only after the stop has been checked.

Only used when a stop or target is active: without one, a whole-bar simulation is already
exact and the split would only double the work.

**Stop-adjustment callbacks** (`adjust_sl_func_nb`, `adjust_tp_func_nb`) work unchanged:
each is wrapped so it sees bars, not steps, and is called exactly as on whole bars — once
per bar, at the open, before that bar's stops are checked (see `bar_view`). A custom
`signal_func_nb`, or templates in a callback's arguments, cannot be split; the front door
falls back to whole bars for those and warns.

Entries fill on the open step, whose close is the open, so every `stop_entry_price` anchor
except `"fillprice"` resolves to the open there. The front door's default, `"price"`, is
the open anyway; an explicit `"close"` no longer reaches past the fill to the bar's close.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: The signal arrays: placed on the open step, absent from the rest of the bar.
SIGNAL_ARGS = ("entries", "exits", "short_entries", "short_exits")

#: Stop-adjustment callbacks, with their argument keyword and the context they receive.
ADJUST_CALLBACKS = {
    "adjust_sl_func_nb": ("adjust_sl_args", "sl"),
    "adjust_tp_func_nb": ("adjust_tp_args", "tp"),
}


# ############# Callbacks on split bars ############# #
#
# ⚠️ A stop-adjustment callback is written for whole bars: it indexes its own per-bar
# arrays with `c.i` and counts bars as `c.i - c.init_i`. Handed the split index
# unchanged, `c.i` runs to twice the number of bars, so it reads the wrong bar, and
# because numba does not bounds-check it WRITES PAST THE END of its own arrays and
# corrupts the heap (found while building this: the process died later, elsewhere).
#
# So each callback is wrapped in a bar view. The wrapper calls it only on a bar's OPEN
# step, with `i`, `init_i` and `curr_i` divided back to bars, and on the rest of the bar
# returns the stop unchanged. The callback therefore sees exactly the calls whole bars
# made — once per bar, at the open, before any of that bar's stops are checked, with
# bar indices — and its arguments pass through untouched. Only the stop check gains the
# rest of the entry bar.
#
# The wrapper is generated per callback with the callback as a module GLOBAL, which numba
# treats as a compile-time constant. A closure over it crashed numba's compiler, and
# passing it inside the argument tuple works but goes through numba's experimental
# first-class function types (and warns). Generated once per process per callback.

_BAR_VIEW_SOURCE = {
    "sl": '''
def bar_view(c{args}):
    if c.i % 2 == 1:
        return c.curr_stop, c.curr_trail
    bar = AdjustSLContext(
        i=c.i // 2, col=c.col, position_now=c.position_now, val_price_now=c.val_price_now,
        init_i=to_bar(c.init_i), init_price=c.init_price, curr_i=to_bar(c.curr_i),
        curr_price=c.curr_price, curr_stop=c.curr_stop, curr_trail=c.curr_trail,
    )
    return user_func(bar{args})
''',
    "tp": '''
def bar_view(c{args}):
    if c.i % 2 == 1:
        return c.curr_stop
    bar = AdjustTPContext(
        i=c.i // 2, col=c.col, position_now=c.position_now, val_price_now=c.val_price_now,
        init_i=to_bar(c.init_i), init_price=c.init_price, curr_stop=c.curr_stop,
    )
    return user_func(bar{args})
''',
}

_bar_views = {}


def bar_view(kind, user_func, n_args):
    """The bar-view wrapper of one stop-adjustment callback (see the note above)."""
    key = (kind, user_func, n_args)
    if key not in _bar_views:
        from numba import njit
        from vectorbt.portfolio.enums import AdjustSLContext, AdjustTPContext

        @njit
        def to_bar(i):
            # -1 means "no position" or "never updated"; it stays -1.
            return i // 2 if i >= 0 else i

        names = "".join(f", a{k}" for k in range(n_args))
        namespace = {"AdjustSLContext": AdjustSLContext, "AdjustTPContext": AdjustTPContext,
                     "to_bar": to_bar, "user_func": user_func}
        exec(_BAR_VIEW_SOURCE[kind].format(args=names), namespace)
        _bar_views[key] = njit(namespace["bar_view"])
    return _bar_views[key]


def stops_active(stops, sim_kwargs) -> bool:
    """True when any stop, target or trailing stop can fire."""
    if sim_kwargs.get("use_stops") is False:
        return False
    for name in ("sl_stop", "tp_stop"):
        value = stops.get(name, sim_kwargs.get(name))
        if value is None:
            continue
        arr = np.asarray(value, dtype=float)
        if np.any(~np.isnan(arr)):
            return True
    return False


def _custom(func):
    """True for a callback that is not one of vectorbt's own no-op defaults."""
    from vectorbt.portfolio import nb

    return func is not None and func not in (
        nb.no_signal_func_nb, nb.no_adjust_sl_func_nb, nb.no_adjust_tp_func_nb
    )


def can_split(sim_kwargs) -> bool:
    """False for the two cases a bar view cannot cover; stop-adjustment callbacks are fine.

    * A custom `signal_func_nb` generates the signals itself, step by step, so it would
      also fire on the rest of the bar. (It also bypasses the front door's next-open
      shift, so it is a lookahead risk in its own right.)
    * Templates (`vbt.Rep(...)` and friends) in a callback's arguments are filled from the
      broadcast inputs — on split bars, the two-step arrays — which the bar view would
      then index by bar.
    """
    from vectorbt.utils.template import has_templates

    if _custom(sim_kwargs.get("signal_func_nb")):
        return False
    for name, (args_name, _) in ADJUST_CALLBACKS.items():
        if _custom(sim_kwargs.get(name)) and has_templates(sim_kwargs.get(args_name, ())):
            return False
    return True


def simulate_split_bars(cls, close, open, high, low, price, signals, call_seq=None, **kwargs):
    """Run `cls.from_signals` on the two-step index and map the result back.

    Args:
        cls: the `Portfolio` class.
        close, open, high, low, price: frames on the original index and columns.
        signals: `{name: frame or None}` for `SIGNAL_ARGS`, already shifted to the bar
            they fill on.
        call_seq: an `(n_bars, n_cols)` array for the open steps, or anything vectorbt
            accepts. The rest of each bar contains only stop exits, which run in column
            order.
        **kwargs: everything else `from_signals` takes. Per-bar values (frames and
            series on the original index, `(n_bars, ...)` arrays) are repeated onto both
            steps; scalars and per-column values pass through.

    Returns:
        A `cls` on the original index.
    """
    index, columns = close.index, close.columns
    n = len(index)
    steps = pd.RangeIndex(2 * n)

    def two(first, second):
        """Open-step values on even rows, rest-of-bar values on odd rows."""
        a, b = np.asarray(first), np.asarray(second)
        out = np.empty((2 * n,) + a.shape[1:], dtype=np.result_type(a, b))
        out[0::2], out[1::2] = a, b
        if out.ndim == 1:
            return pd.Series(out, index=steps)
        return pd.DataFrame(out, index=steps, columns=columns)

    def repeat(value):
        """A per-bar value repeated onto both steps; anything else unchanged."""
        if isinstance(value, (pd.DataFrame, pd.Series)):
            if not value.index.equals(index):
                if not isinstance(value.index, pd.DatetimeIndex):
                    return value                # per column, not per bar
                value = value.reindex(index)
            data = np.repeat(value.to_numpy(), 2, axis=0)
            if isinstance(value, pd.Series):
                return pd.Series(data, index=steps, name=value.name)
            return pd.DataFrame(data, index=steps, columns=value.columns)
        if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[0] == n:
            return np.repeat(value, 2, axis=0)
        return value

    split_signals = {}
    for name in SIGNAL_ARGS:
        signal = signals.get(name)
        if signal is None:
            split_signals[name] = None
            continue
        on_open = np.asarray(signal, dtype=bool)
        split_signals[name] = two(on_open, np.zeros_like(on_open)).astype(bool)

    if isinstance(call_seq, np.ndarray) and call_seq.ndim == 2 and call_seq.shape[0] == n:
        split_seq = np.tile(np.arange(call_seq.shape[1], dtype=np.int64), (2 * n, 1))
        split_seq[0::2] = call_seq
        call_seq = split_seq
    if call_seq is not None:
        kwargs["call_seq"] = call_seq

    # Stop-adjustment callbacks see bars, not steps; their own arguments stay per bar, so
    # they are taken out of `kwargs` before the per-bar values are repeated.
    callbacks = {}
    for name, (args_name, kind) in ADJUST_CALLBACKS.items():
        func = kwargs.pop(name, None)
        user_args = tuple(kwargs.pop(args_name, ()))
        if _custom(func):
            callbacks[name] = bar_view(kind, func, len(user_args))
            callbacks[args_name] = user_args
        elif func is not None:
            callbacks[name] = func
            callbacks[args_name] = user_args

    open_ = open.to_numpy(dtype=float)
    split = cls.from_signals(
        two(open_, close.to_numpy(dtype=float)),
        **split_signals,
        open=two(open_, open_),
        high=two(open_, high.to_numpy(dtype=float)),
        low=two(open_, low.to_numpy(dtype=float)),
        # The rest of the bar has no signals, so its price is only a valuation price; the
        # close is the bar's last.
        price=two(price.to_numpy(dtype=float), close.to_numpy(dtype=float)),
        **{k: repeat(v) for k, v in kwargs.items()},
        **callbacks,
    )

    orders = split.order_records.copy()
    orders["idx"] //= 2
    logs = split.log_records.copy()
    if len(logs):
        logs["idx"] //= 2
    seq = split._call_seq
    if seq is not None:
        seq = np.asarray(seq)[0::2]
    return split.replace(
        wrapper=split.wrapper.replace(index=index),
        close=close,
        order_records=orders,
        log_records=logs,
        call_seq=seq,
    )
