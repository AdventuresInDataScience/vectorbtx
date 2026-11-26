"""Section N — the re-sizing loop's passes without the pipeline around them.

OpenStrategy Studio issue 37. `_settle` (`backtest.py`) simulates a run again for each entry the account
changes (a share of the balance or equity, the leverage limit at the equity, an entry cut to the cash), so a
levered run that grows can take one simulation per entry: 188 on 2,970 daily bars of a 5x spread bet sized at
20% of its balance. Each pass prepared the whole run twice and built a whole `Portfolio`, about 30 ms of
pandas and vectorbt wrapping around about 1 ms of vectorbt's own numba simulation. So the passes between the
first run and the last are made here, on arrays:

* **The simulation** is vectorbt's own kernel, called again with the arguments `from_signals` handed it on
  the first run (`capture`, as it ran), with only `size` swapped (`replay`). The kernel's inputs are built
  from the sizes and from nothing that changes between passes, so this is the run `from_signals` would make.
* **The account** comes from the order records through the same vectorbt dispatch functions
  `Portfolio.value()` and `Portfolio.assets()` call, in the same order (`ledger_of_records`), so its numbers
  are the `Portfolio`'s, bit for bit.
* **The sizes** are `prepare`'s own sizing (`ctx["resize"]`, the same functions on the same inputs).

Nothing here decides anything. The loop is the same loop; the run it settles on is made, and its settling
checked, by the full pipeline as before. And each piece is checked against the first run before it is
trusted (`fast_runner`): the replay must give the first run's own orders and the ledger its own account, bit
for bit, or the loop runs as it did.

What is not covered, so runs as before: holding costs (charged after the simulation, from a `Portfolio`),
costs that follow the size (a commission per unit or with a minimum, automatic slippage: they change the
kernel's cost arrays too), a grouped portfolio without shared cash, and any run whose simulation is not one
call of one of the two kernels.
"""

from __future__ import annotations

import threading

import numpy as np
import pandas as pd

#: The two kernels `from_signals` simulates with: `dispatch.simulate_from_signals` with no callback, and
#: `nb.simulate_from_signal_func_nb` with a stop-adjustment callback (the trailing stops). Each is a module
#: attribute that `vectorbt.portfolio.base` looks up when it calls it.
_KERNELS = (("dispatch", "simulate_from_signals"), ("nb", "simulate_from_signal_func_nb"))

_local = threading.local()
_installed = []
_install_lock = threading.Lock()


def _copied(value):
    """Arrays copied, anywhere in tuples, lists and dicts; everything else as it is.

    ⚠️ The kernel writes into some of its arguments: a trailing stop's per-position state
    (`trail_level_nb`'s arrays), and the call sequence under automatic ordering. A replay must start from
    what the first call started from, so the arguments are copied before that call runs, and again for
    each replay.
    """
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, tuple) and not hasattr(value, "_fields"):
        return tuple(_copied(item) for item in value)
    if isinstance(value, list):
        return [_copied(item) for item in value]
    if isinstance(value, dict):
        return {key: _copied(item) for key, item in value.items()}
    return value


def _install():
    """Wrap the two kernels, once per process, so a thread can record the calls it makes.

    The wrappers stay in place and pass every call straight through; they record only for a thread that
    is inside `capture`, so other threads' runs, and every run outside the loop, are untouched. Installing
    once, rather than wrapping and restoring around each run, means two threads cannot restore each
    other's wrappers.
    """
    with _install_lock:
        if _installed:
            return
        from vectorbt.portfolio import base

        for owner_name, name in _KERNELS:
            owner = getattr(base, owner_name)
            real = getattr(owner, name)

            def wrapper(*args, _real=real, **kwargs):
                calls = getattr(_local, "calls", None)
                if calls is not None:
                    calls.append({"kernel": _real, "args": _copied(args), "kwargs": _copied(kwargs)})
                return _real(*args, **kwargs)

            setattr(owner, name, wrapper)
        _installed.append(True)


def capture(run):
    """`run()`, and the one kernel call it made: `(result, call)`, the call None unless it was exactly one."""
    _install()
    outer = getattr(_local, "calls", None)
    _local.calls = []
    try:
        result = run()
        calls = _local.calls
    finally:
        _local.calls = outer
    return result, (calls[0] if len(calls) == 1 else None)


def replay(call, size, init_cash=None):
    """The kernel's order records for the captured call with `size` in place of its own, and `init_cash` in
    place of its starting cash when given (the margin loan the loop has grown, OpenStrategy Studio issue 41).

    Both kernels take the starting cash third (`target_shape, group_lens, init_cash, call_seq, ...`), one
    per cash group, as `from_signals` broadcast it.
    """
    args = _copied(call["args"])
    if init_cash is not None:
        args = list(args)
        args[2] = np.require(np.broadcast_to(np.asarray(init_cash, dtype=float), np.shape(args[2])),
                             dtype=np.asarray(args[2]).dtype).copy()
        args = tuple(args)
    kwargs = _copied(call["kwargs"])
    kwargs["size"] = size
    order_records, _ = call["kernel"](*args, **kwargs)
    return order_records


def statics_of(pf, charged=None, universe=None):
    """What a ledger needs from the first run that no pass changes, or None for a portfolio not covered.

    `charged(universe, index, columns, borrowed, notional, position)`: the holding costs positions pay,
    as the run charges them (`backtest.PortfolioBacktest`), for a run that pays any.
    """
    from vectorbt.base.reshape_fns import to_1d_array, to_2d_array

    if isinstance(pf._init_cash, int) or (pf.holding_costs is not None and charged is None):
        return None
    grouper = pf.wrapper.grouper
    grouped = bool(grouper.is_grouped())
    if grouped and not pf.cash_sharing:
        return None
    close = to_2d_array(pf.close)
    loan = pf.margin_loan
    return dict(
        index=pf.wrapper.index, columns=list(pf.wrapper.columns), shape=close.shape, close=close,
        filled=to_2d_array(pf.get_filled_close()) if pf.fillna_close else close,
        init=to_1d_array(pf._init_cash), loan=None if loan is None else np.atleast_1d(np.asarray(loan, dtype=float)),
        group_lens=grouper.get_group_lens(), shared=bool(pf.cash_sharing) and grouped,
        cash_sharing=bool(pf.cash_sharing), groups=np.asarray(grouper.get_groups()) if grouped else None,
        engine=pf.engine,
        charge=None if pf.holding_costs is None else (lambda borrowed, notional, position: charged(
            universe, pf.wrapper.index, list(pf.wrapper.columns), borrowed, notional, position)),
    )


def ledger_of(pf):
    """What the account basis is worked out from, read off a `Portfolio` (`backtest._basis`).

    `value` is per account: per group under shared cash (`pf.value()`), else per column; `start` is the
    starting equity of each, the margin loan taken out (`get_init_cash`).
    """
    from vectorbt.base.reshape_fns import to_2d_array

    shared = bool(pf.cash_sharing) and bool(pf.wrapper.grouper.is_grouped())
    out = dict(index=pf.wrapper.index, columns=list(pf.wrapper.columns), records=pf.orders.values,
               assets=to_2d_array(pf.assets()), close=to_2d_array(pf.close), shared=bool(pf.cash_sharing))
    if pf.cash_sharing:
        out["groups"] = np.asarray(pf.wrapper.grouper.get_groups())
        out["value"] = to_2d_array(pf.value())
        out["start"] = np.atleast_1d(np.asarray(pf.get_init_cash(), dtype=float))
    else:
        out["groups"] = None
        out["value"] = to_2d_array(pf.value(group_by=False))
        out["start"] = np.broadcast_to(np.asarray(pf.get_init_cash(group_by=False), dtype=float),
                                       (len(out["columns"]),)).copy()
    out["grouped"] = shared
    return out


def ledger_of_records(records, statics):
    """`ledger_of` for order records on the first run's bars, with vectorbt's own dispatch functions.

    Mirrors `Portfolio.assets`, `asset_value`, `cash_flow`, `cash`, `value` and vectorbtX's
    `get_init_cash` (the loan taken out), call for call, so the numbers are the `Portfolio`'s.
    """
    from vectorbt.portfolio import dispatch
    from vectorbt.portfolio.enums import Direction
    from vectorbt.records import dispatch as records_dispatch

    s = statics
    n, m = s["shape"]
    engine = s["engine"]
    col_map = records_dispatch.col_map(records["col"], m, engine=engine)
    assets = dispatch.assets(dispatch.asset_flow((n, m), records, col_map, Direction.Both, engine=engine),
                             engine=engine)
    cash_flow = dispatch.cash_flow((n, m), records, col_map, False, engine=engine)
    close = s["filled"].copy()
    close[assets == 0] = 0.0                     # as `asset_value` does, for a price of NaN
    asset_value = dispatch.asset_value(close, assets, engine=engine)
    lens = s["group_lens"]
    if s["cash_sharing"]:
        init = np.asarray(dispatch.init_cash_grouped(s["init"], lens, True, engine=engine), dtype=float)
        if s["loan"] is not None:
            init = init - np.asarray(dispatch.init_cash_grouped(s["loan"], lens, True, engine=engine), dtype=float)

        def cash_of(flow):
            return dispatch.cash_grouped((n, m), dispatch.cash_flow_grouped(flow, lens, engine=engine), lens,
                                         init, engine=engine)
    else:
        init = np.asarray(dispatch.init_cash(s["init"], lens, False, engine=engine), dtype=float)
        if s["loan"] is not None:
            init = init - np.asarray(dispatch.init_cash(s["loan"], lens, False, engine=engine), dtype=float)

        def cash_of(flow):
            return dispatch.cash(flow, init, engine=engine)

    if s["charge"] is not None:
        # The holding costs, as `apply_holding_costs` charges the run: on its borrowing before any
        # charge (negative cash, per account; a shared account's split by long value,
        # `margin.exposure.borrowed_portion`), its notional and its positions; then folded into the
        # cash flow per column, as vectorbtX's `Portfolio.cash_flow` folds them.
        index, columns = s["index"], s["columns"]
        debit = pd.DataFrame(-cash_of(cash_flow)).clip(lower=0.0)
        if s["cash_sharing"]:
            from ..margin.exposure import split_by_long_value

            debit = split_by_long_value(debit.to_numpy(), asset_value, s["groups"])
        charges = pd.DataFrame(s["charge"](
            pd.DataFrame(np.asarray(debit, dtype=float), index=index, columns=columns),
            pd.DataFrame(asset_value, index=index, columns=columns).abs(),
            pd.DataFrame(assets, index=index, columns=columns)))
        flow = np.array(cash_flow, dtype=float, copy=True)
        for k, col in enumerate(columns):
            if col in charges.columns:
                flow[:, k] += charges[col].to_numpy(dtype=float)
        cash_flow = flow
    cash = cash_of(cash_flow)
    if s["cash_sharing"]:
        value = dispatch.value(cash, dispatch.asset_value_grouped(asset_value, lens, engine=engine), engine=engine)
    else:
        value = dispatch.value(cash, asset_value, engine=engine)
    return dict(index=s["index"], columns=s["columns"], records=records, assets=assets, close=s["close"],
                shared=s["cash_sharing"], groups=s["groups"], value=value, start=init, grouped=s["shared"])


def _same_ledger(a, b) -> bool:
    for key in ("assets", "value", "start"):
        x, y = np.asarray(a[key], dtype=float), np.asarray(b[key], dtype=float)
        if x.shape != y.shape or not np.array_equal(x, y, equal_nan=True):
            return False
    return _same_records(a["records"], b["records"])


def _same_records(a, b) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    for name in a.dtype.names:
        if not np.array_equal(a[name], b[name], equal_nan=np.issubdtype(a.dtype[name], np.floating)):
            return False
    return True


def fast_runner(call, pf, sizes, charged=None, universe=None):
    """The loop's fast passes for this run, or None when one is not covered or does not check out.

    Args:
        call: the kernel call the first run made (`capture`).
        pf: the first run's portfolio.
        sizes: the sizes it was simulated with (`ctx["size"]`, per bar and column).
        charged, universe: how the run charges holding costs, and its universe (`statics_of`).

    Returns:
        `{"simulate": sizes -> records, "ledger": records -> ledger}`, or None.
    """
    if call is None:
        return None
    statics = statics_of(pf, charged, universe)
    if statics is None:
        return None
    n, m = statics["shape"]
    steps = call["args"][0][0] if call["args"] else call["kwargs"].get("target_shape", (None,))[0]
    if steps not in (n, 2 * n):
        return None
    split = steps == 2 * n
    own = call["kwargs"].get("size")
    if not isinstance(own, np.ndarray) or own.shape != (steps, m):
        return None

    def kernel_size(new):
        values = pd.DataFrame(new).to_numpy(dtype=float)
        if split:
            values = np.repeat(values, 2, axis=0)      # the open step and the rest of the bar, as split_bars does
        out = np.asarray(values, dtype=own.dtype)
        return np.asfortranarray(out) if own.flags.f_contiguous and not own.flags.c_contiguous else out

    def simulate(new, loan=None):
        # `loan`: the margin loan to fund the run with (issue 41), in place of the first run's; the
        # starting cash moves by the difference, as `PortfolioBacktest`'s `simulated(ctx, loan)` moves it.
        init_cash = None
        if loan is not None:
            init_cash = statics["init"] - statics["loan"] + np.asarray(loan, dtype=float)
        records = replay(call, kernel_size(new), init_cash).copy()
        if split:
            records["idx"] //= 2                       # back onto the bars, as split_bars does
        return records

    # Trusted only when it gives the first run back, bit for bit: its sizes, its orders and its account.
    if not np.array_equal(kernel_size(sizes), own, equal_nan=True):
        return None
    own_records = pf.orders.values
    if not _same_records(simulate(sizes), own_records):
        return None
    if not _same_ledger(ledger_of_records(own_records, statics), ledger_of(pf)):
        return None
    return {"simulate": simulate, "ledger": lambda records: ledger_of_records(records, statics)}
