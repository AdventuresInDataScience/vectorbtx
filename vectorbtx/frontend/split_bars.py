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

Entries fill on the open step, whose close is the open, so every `stop_entry_price` anchor
except `"fillprice"` resolves to the open there. The front door's default, `"price"`, is
the open anyway; an explicit `"close"` no longer reaches past the fill to the bar's close.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: The signal arrays: placed on the open step, absent from the rest of the bar.
SIGNAL_ARGS = ("entries", "exits", "short_entries", "short_exits")

#: Callbacks whose arguments are opaque, so they cannot be expanded to the split index.
CALLBACK_ARGS = ("signal_func_nb", "adjust_sl_func_nb", "adjust_tp_func_nb")


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


def can_split(sim_kwargs) -> bool:
    """False when a custom numba callback is given: its arguments cannot be re-indexed.

    vectorbt's own no-op defaults, passed explicitly, are not custom and do not count.
    """
    from vectorbt.portfolio import nb

    defaults = {nb.no_signal_func_nb, nb.no_adjust_sl_func_nb, nb.no_adjust_tp_func_nb}
    return not any(
        sim_kwargs.get(name) is not None and sim_kwargs.get(name) not in defaults
        for name in CALLBACK_ARGS
    )


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
