"""Section G — the level-3 callback.

**The only level-3 section** after D2 and D4, and therefore the only part of vectorbtX
that depends on `portfolio/nb.py` internals. Everything about how this file is written
serves keeping that dependency small: all arithmetic lives in `financing.py`, `carry.py`
and `accrual.py` as pure functions, and the kernels here do nothing but index into
pre-computed arrays.

Four things to know before reading further. The first two are the plan's own warnings;
the third and fourth were found by auditing the 1.1.0 source and contradict the sketch in
plan §5.3.

* **`call_post_segment` defaults to `False`.** It must be `True`, or the callback never
  fires on bars where no order was placed — exactly where financing accrues. The output
  looks plausible and is wrong.
* **Accrual lands on the next bar (D6).** `last_value` and `last_return` are recomputed
  from `last_cash` immediately *before* the callback runs, so a cash mutation shows up in
  the following bar's value. This shifts *when* the charge appears in the equity curve,
  not the total. To reverse it, update `last_value` and `last_return` in the same
  callback after mutating `last_cash`.
* ⚠️ **`last_cash` is indexed by GROUP under cash sharing, not by column.** It is
  allocated as `init_cash.copy()` (`nb.py:3383`), and `init_cash` has one entry per group
  when `cash_sharing=True` and one per column otherwise — the read sites at `nb.py:3521`
  and `3549` branch on exactly that. Plan §5.3 sketches `c.last_cash[col] -= ...`, which
  is only correct without cash sharing; with it, the write lands in the wrong group or
  runs off the end of the array. `post_segment_func_nb` below branches accordingly.
* ⚠️ **There is no `nb.order_func_nb` upstream.** vectorbt ships only `no_order_func_nb`
  (a placeholder returning `NoOrder`); `from_orders` uses `simulate_from_orders_nb`, a
  different entry point entirely. Replaying orders through `from_order_func` means
  supplying an order function, which is what `order_func_nb` here is for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core._numba import njit

try:  # pragma: no cover - vectorbt is absent in the pure-arithmetic test path
    from vectorbt.portfolio.nb import order_nb, order_nothing_nb

    HAS_VECTORBT = True
except ImportError:  # pragma: no cover
    HAS_VECTORBT = False
    order_nb = order_nothing_nb = None


# ############# Kernels ############# #


@njit(cache=True)
def post_segment_func_nb(c, charges):
    """Mutate `last_cash` by the pre-computed charge for this bar and this segment.

    `charges` is a `(n_bars, n_cols)` float array built by `build_charges`. `last_cash`
    is a numpy array passed **by reference** into every context, so mutating it
    propagates through the rest of the simulation.

    Under cash sharing the group holds one cash balance, so every column's charge is
    summed into `last_cash[c.group]`. Without it each column has its own balance and the
    charge goes straight to `last_cash[col]`. Getting this branch wrong is silent in the
    ungrouped case and out-of-bounds in the grouped one.

    The body is deliberately trivial. When upstream changes the callback signature, this
    is the only thing that needs rewriting.
    """
    if c.cash_sharing:
        total = 0.0
        for col in range(c.from_col, c.to_col):
            total += charges[c.i, col]
        c.last_cash[c.group] += total
    else:
        for col in range(c.from_col, c.to_col):
            c.last_cash[col] += charges[c.i, col]
    return None


@njit(cache=True)
def pre_segment_func_nb(c, val_price):
    """Set the valuation price for the segment.

    `from_order_func` only refreshes `last_val_price` *after* a segment, so without this
    the first bar values every position at NaN and any `TargetPercent` sizing there is
    undefined. `from_orders` does the same thing internally; this makes the replay match.
    """
    for col in range(c.from_col, c.to_col):
        p = val_price[c.i, col]
        if not np.isnan(p):
            c.last_val_price[col] = p
    return ()


@njit(cache=True)
def order_func_nb(
    c,
    size,
    price,
    size_type,
    direction,
    fees,
    fixed_fees,
    slippage,
    min_size,
    max_size,
    size_granularity,
    lock_cash,
    allow_partial,
    raise_reject,
):
    """Emit one order per column per bar from fully materialised 2-D arrays.

    Every argument is a `(n_bars, n_cols)` array, broadcast in Python by
    `_broadcast_2d` rather than relying on vectorbt's flexible indexing. That costs a
    little memory on a path that is already the slow one, and buys immunity to the
    `flex_2d` semantics — which are the exact kind of upstream internal §5.5 ranks as
    fragile.
    """
    i, col = c.i, c.col
    s = size[i, col]
    if np.isnan(s):
        return order_nothing_nb()
    return order_nb(
        size=s,
        price=price[i, col],
        size_type=size_type[i, col],
        direction=direction[i, col],
        fees=fees[i, col],
        fixed_fees=fixed_fees[i, col],
        slippage=slippage[i, col],
        min_size=min_size[i, col],
        max_size=max_size[i, col],
        size_granularity=size_granularity[i, col],
        lock_cash=lock_cash[i, col],
        allow_partial=allow_partial[i, col],
        raise_reject=raise_reject[i, col],
    )


# ############# Charge construction ############# #


def build_charges(
    universe,
    index,
    borrowed=None,
    notional=None,
    position=None,
    rates=None,
    markup: float = 0.03,
    carry_markup: float = None,
    holidays=None,
    columns=None,
) -> pd.DataFrame:
    """Pre-compute the per-bar, per-column cash adjustment.

    Args:
        universe: the `Universe`.
        index: the backtest index.
        borrowed: DataFrame of borrowed notional per bar and column (Section F's
            `borrowed_portion`). Financing is charged on this and nothing else.
        notional: DataFrame of full position notional, for carry.
        position: **signed** position size per bar and column. Supplies the direction
            each charge is computed in. Without it everything is treated as long, which
            silently loses the adverse-markup asymmetry the whole module exists for.
        rates: DataFrame of annualised benchmark rates, one column per currency.
        markup: broker financing markup, applied adversely. Default 3%.
        carry_markup: markup on the carry differential, also adverse.
        holidays: dates carrying no charge.
        columns: resolved column order.

    Returns:
        DataFrame of signed cash adjustments — negative is a charge.

    Notes:
        Financing consumes only the **borrowed** portion; carry consumes the **full**
        notional. Separate inputs, separate multiplier tables, deliberately not merged.

        Direction is resolved **per bar**, not once for the whole column. A column that
        is long for six months and short for six is charged `benchmark + markup` on the
        first half and credited `benchmark - markup` on the second, which is what a
        broker does. Both branches are computed with the same tested pure functions and
        selected by sign, so the asymmetry cannot drift out of step with them.
    """
    from ..instruments.defaults import financing_basis
    from ..rates.table import day_count_for
    from .accrual import day_multipliers
    from .carry import carry_series
    from .financing import financing_series

    if carry_markup is None:
        from ..core.settings import get

        carry_markup = get("carry_markup", 0.01)

    idx = pd.Index(index)
    cols = list(columns) if columns is not None else list(universe.columns)
    charges = pd.DataFrame(0.0, index=idx, columns=cols)
    no_pair = []

    for col in cols:
        if col not in universe:
            continue
        spec = universe[col].filled()
        mult = day_multipliers(
            idx, triple_day=spec.triple_day, end_of_day=spec.end_of_day, holidays=holidays
        )
        is_long = _direction_mask(position, col, len(idx))
        # The divisor follows the market's denomination, not a house convention: 365 for
        # GBP/SGD/ZAR markets, 360 elsewhere.
        day_count = day_count_for(spec.quote_currency)

        if "financing" in spec.active_costs:
            # ⚠️ The basis is mode-dependent and the two are NOT interchangeable.
            # Margin share dealing finances the debit balance; a spread bet or CFD is
            # funded by the broker in full and finances the whole notional. Charging a
            # CFD on the borrowed portion understates it by 1 - 1/leverage, and drops it
            # to zero for an unlevered position a broker would still bill in full.
            basis = financing_basis(spec.mode)
            source = borrowed if basis == "borrowed" else notional
            if source is None:
                continue
            b = np.abs(_col_series(source, col, len(idx)))
            bench = _rate_for(rates, spec.quote_currency, idx)
            charges[col] += np.where(
                is_long,
                financing_series(b, bench, markup, True, mult, day_count),
                financing_series(b, bench, markup, False, mult, day_count),
            )

        if "carry" in spec.active_costs and notional is not None:
            # ⚠️ Carry is a differential, so it needs BOTH legs of the pair. A missing leg
            # used to be priced at 0% by `_rate_for`, so an FX instrument with no
            # `base_currency` paid (or earned) the whole quote-currency rate, while the
            # construction warning said carry would resolve to zero. Skip it instead, as
            # that warning promises, and say so again here, where it bites.
            if spec.base_currency is None or spec.quote_currency is None:
                no_pair.append(col)
                continue
            n = np.abs(_col_series(notional, col, len(idx)))
            base_rate = _rate_for(rates, spec.base_currency, idx)
            quote_rate = _rate_for(rates, spec.quote_currency, idx)
            charges[col] += np.where(
                is_long,
                carry_series(n, base_rate, quote_rate, mult, carry_markup, True, day_count),
                carry_series(n, base_rate, quote_rate, mult, carry_markup, False, day_count),
            )

    if no_pair and rates is not None:
        import warnings

        warnings.warn(
            f"Carry skipped for {no_pair}: carry is active but the instrument does not "
            f"name both currencies of its pair (base_currency and quote_currency), so the "
            f"tom-next differential cannot be computed. These columns are charged NO "
            f"carry, which is also wrong. Set both currencies on the instrument.",
            stacklevel=3,
        )
    return charges


def _col_series(frame, col, n) -> np.ndarray:
    """One column's series, tolerating group-level reporting.

    Under cash sharing vectorbt reports cash **per group**, so a per-column lookup fails
    on every column but the first. A single reported column is the group's, which every
    member shares; anything else with no match contributes nothing.
    """
    frame = pd.DataFrame(frame)
    if col in frame.columns:
        return frame[col].to_numpy(dtype=float)
    if frame.shape[1] == 1:
        return frame.iloc[:, 0].to_numpy(dtype=float)
    return np.zeros(n)


def _direction_mask(position, col, n) -> np.ndarray:
    """Per-bar long/short mask for a column, defaulting to long when unknown."""
    if position is None:
        return np.ones(n, dtype=bool)
    return _col_series(position, col, n) >= 0


def _rate_for(rates, currency, index):
    """Look up a currency's rate series, or zero if none is supplied."""
    if rates is None or currency is None:
        return np.zeros(len(index))
    frame = pd.DataFrame(rates)
    if currency not in frame.columns:
        raise ValueError(
            f"No benchmark rate for {currency!r}. Section M must be loaded and cover the "
            f"backtest period. Present: {list(frame.columns)}"
        )
    aligned = frame[currency].reindex(index).ffill().bfill()

    # ⚠️ All-NaN here is the library's worst failure mode, because nothing downstream
    # complains: NaN rates give NaN financing and carry, which give a NaN equity curve,
    # and `pf.stats()` still prints a plausible-looking total return while the cost model
    # has contributed exactly nothing.
    #
    # It has one overwhelmingly common cause. Yahoo returns a **tz-aware** index (UTC);
    # FRED returns a **tz-naive** one. Their intersection is empty, so every bar reindexes
    # to NaN, and `ffill`/`bfill` cannot help because there is no value anywhere to carry.
    if len(aligned) and bool(aligned.isna().all()):
        idx_tz = getattr(index, "tz", None)
        rate_tz = getattr(frame.index, "tz", None)
        hint = ""
        if (idx_tz is None) != (rate_tz is None):
            hint = (
                f" The price index is {'tz-aware (' + str(idx_tz) + ')' if idx_tz is not None else 'tz-naive'} "
                f"but the rate index is {'tz-aware (' + str(rate_tz) + ')' if rate_tz is not None else 'tz-naive'}, "
                f"so they share no timestamps at all — this is the usual cause. Normalise "
                f"one to match the other, e.g. "
                f"`price.index = price.index.tz_localize(None).normalize()`."
            )
        raise ValueError(
            f"The {currency!r} rate series does not overlap the backtest index, so every "
            f"rate resolves to NaN and all financing and carry would silently be NaN."
            f"{hint} Rate range: {frame.index.min()} to {frame.index.max()}; backtest "
            f"range: {index.min()} to {index.max()}."
        )
    return aligned.to_numpy(dtype=float)


# ############# Order-argument materialisation ############# #

#: `Order` fields the replay understands, with the default each falls back to.
#: Mirrors `vectorbt.portfolio.enums.Order`; anything absent here is not replayable and
#: is rejected loudly rather than dropped.
_ORDER_DEFAULTS = {
    "price": np.inf,
    "size_type": 0,          # SizeType.Amount
    "direction": 2,          # Direction.Both
    "fees": 0.0,
    "fixed_fees": 0.0,
    "slippage": 0.0,
    "min_size": 0.0,
    "max_size": np.inf,
    "size_granularity": np.nan,
    "lock_cash": False,
    "allow_partial": True,
    "raise_reject": False,
}

_ORDER_ORDER = tuple(_ORDER_DEFAULTS)


def _broadcast_2d(value, shape, columns, dtype=float):
    """Materialise a scalar, per-column list, Series or DataFrame to `(n_bars, n_cols)`.

    vectorbt would broadcast most of these itself via `flex_select_auto_nb`, but the
    replay path passes arrays straight into a kernel, so the broadcasting has to happen
    here — and doing it in one place means the per-column-list case (a plain list is
    per *column*, not per row) is handled once rather than at every call site.
    """
    n_rows, n_cols = shape
    if isinstance(value, pd.DataFrame):
        return np.ascontiguousarray(value[list(columns)].to_numpy(dtype=dtype))
    if isinstance(value, pd.Series):
        return np.broadcast_to(
            value.to_numpy(dtype=dtype)[:, None], shape
        ).astype(dtype, copy=True)
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim == 0:
        return np.full(shape, arr.item(), dtype=dtype)
    if arr.ndim == 1:
        if len(arr) == n_cols:
            return np.broadcast_to(arr[None, :], shape).astype(dtype, copy=True)
        if len(arr) == n_rows:
            return np.broadcast_to(arr[:, None], shape).astype(dtype, copy=True)
        raise ValueError(
            f"Cannot broadcast a length-{len(arr)} array to {shape}: it matches neither "
            f"the {n_rows} bars nor the {n_cols} columns."
        )
    if arr.shape == shape:
        return np.ascontiguousarray(arr, dtype=dtype)
    raise ValueError(f"Cannot broadcast shape {arr.shape} to {shape}")


def _order_arrays(order_kwargs, shape, columns):
    """Turn resolved `Order` kwargs into the positional array tuple `order_func_nb` wants."""
    from vectorbt.portfolio.enums import Direction, SizeType
    from vectorbt.utils.enum_ import map_enum_fields

    unknown = sorted(set(order_kwargs) - set(_ORDER_ORDER))
    if unknown:
        raise ValueError(
            f"Cannot replay these Order fields through the holding-cost path: {unknown}. "
            f"Pass 2 materialises each field into the kernel explicitly, so a field it "
            f"does not know about would be silently dropped and the costs applied to a "
            f"portfolio that never existed. Replayable fields: {list(_ORDER_ORDER)}."
        )

    out = []
    for name in _ORDER_ORDER:
        value = order_kwargs.get(name, _ORDER_DEFAULTS[name])
        if value is None:
            value = _ORDER_DEFAULTS[name]
        if name == "size_type" and isinstance(value, str):
            value = map_enum_fields(value, SizeType)
        if name == "direction" and isinstance(value, str):
            value = map_enum_fields(value, Direction)
        dtype = bool if isinstance(_ORDER_DEFAULTS[name], bool) else (
            np.int64 if name in ("size_type", "direction") else float
        )
        out.append(_broadcast_2d(value, shape, columns, dtype=dtype))
    return tuple(out)


# ############# Entry point ############# #


def simulate_with_holding_costs(
    cls,
    close,
    size,
    universe,
    rates=None,
    markup=None,
    carry_markup=None,
    holidays=None,
    order_kwargs=None,
    **kwargs,
):
    """Run a simulation with financing and carry charged per bar.

    Two passes. A plain `from_orders` run establishes positions and therefore the
    borrowed portion — Section F derives it post-simulation from `cash(free=True)` and
    `asset_value`, which is why F2 being deferred does not block this. `from_order_func`
    then replays the *same* orders with the charges applied through the cash hook.

    Args:
        cls: the `Portfolio` class.
        close: price frame.
        size: order sizes.
        universe: the `Universe`.
        rates: benchmark rate table from Section M.
        markup: broker financing markup. Defaults to `settings["vectorbtx"]`.
        carry_markup: markup on the carry differential. None (the default) reads
            `settings["vectorbtx"]["carry_markup"]`; pass 0.0 to charge none.
        holidays: dates carrying no charge.
        order_kwargs: explicit `Order` field overrides, merged over the universe's.
        **kwargs: simulation arguments (`init_cash`, `cash_sharing`, `group_by`, `freq`,
            `call_seq`, `seed`, ...), passed to **both** passes so they agree.

    Returns:
        `Portfolio`, with `pf.holding_costs` carrying the per-bar charges.

    Notes:
        ⚠️ `call_post_segment=True` is set here and must never be removed — see the
        module docstring.

        **Both passes get the same order arguments.** Resolving the universe for pass 1
        and running pass 2 bare would replay a portfolio with no costs, no quantization
        and no size, so the charges would be applied to positions that never existed.

        **The borrowed portion is taken from pass 1.** Pass 2's cash differs from pass
        1's by the charges themselves, so the borrowed amounts it implies differ very
        slightly. It is a one-iteration approximation, and a deliberate one: the error is
        second-order (a charge on a charge) while iterating to a fixed point would double
        the cost of the slowest path in the library for a difference far below the
        uncertainty in the markup itself.
    """
    from ..core.settings import get
    from ..margin.exposure import borrowed_portion

    markup = get("financing_markup", 0.03) if markup is None else markup
    close = pd.DataFrame(close)
    columns = list(close.columns)

    resolved = universe.resolve(close) if universe is not None else {}
    if order_kwargs:
        resolved.update(order_kwargs)

    # Simulation arguments belong to both passes; Order fields belong to the order func.
    sim_kwargs = {k: v for k, v in kwargs.items() if k not in _ORDER_DEFAULTS}
    resolved.update({k: v for k, v in kwargs.items() if k in _ORDER_DEFAULTS})

    # --- Pass 1: establish positions, and therefore the borrowed portion. ---
    base = cls.from_orders(close, size=size, **resolved, **sim_kwargs)

    borrowed = borrowed_portion(base)
    position = pd.DataFrame(base.assets())
    notional = pd.DataFrame(base.asset_value()).abs()

    charges = build_charges(
        universe,
        close.index,
        borrowed=borrowed,
        notional=notional,
        position=position,
        rates=rates,
        markup=markup,
        carry_markup=carry_markup,
        holidays=holidays,
        columns=columns,
    )

    # --- Pass 2: replay the same orders with the charges applied via the cash hook. ---
    shape = (len(close.index), len(columns))
    size_arr = _broadcast_2d(size, shape, columns)
    val_price = _broadcast_2d(close, shape, columns)
    order_args = _order_arrays(resolved, shape, columns)

    pf = cls.from_order_func(
        close,
        order_func_nb,
        size_arr,
        *order_args,
        pre_segment_func_nb=pre_segment_func_nb,
        pre_segment_args=(val_price,),
        post_segment_func_nb=post_segment_func_nb,
        post_segment_args=(np.ascontiguousarray(charges.to_numpy(dtype=float)),),
        call_post_segment=True,  # never remove: financing accrues on order-less bars
        **sim_kwargs,
    )
    return pf.replace(universe=universe, holding_costs=charges)


def apply_holding_costs(
    pf,
    universe,
    rates=None,
    markup=None,
    carry_markup=None,
    holidays=None,
):
    """Charge financing and carry on an **already-simulated** portfolio.

    Args:
        pf: a simulated `Portfolio` of any kind — `from_signals`, `from_orders`,
            `from_weights`.
        universe: the `Universe`.
        rates: benchmark rate table from Section M.
        markup: broker financing markup, adverse. Defaults to `settings["vectorbtx"]`.
        carry_markup: admin fee on the tom-next differential, also adverse.
        holidays: dates carrying no accrual.

    Returns:
        The same portfolio with `holding_costs` attached, so `cash`, `value`, `returns`
        and every metric include them.

    Notes:
        **Why this exists alongside `simulate_with_holding_costs`.** That function replays
        orders through `from_order_func` so the simulation's own cash reflects the
        charges as it runs. It buys one thing — later sizing decisions see the reduced
        balance — at a high price: it is the only level-3 path in the library, it is the
        most fragile thing here (§5.5 ranks `portfolio/nb.py` internals first), and it
        **cannot express a signal-driven strategy**, because it replays explicit sizes and
        has no notion of entries and exits.

        This function takes the other trade-off. Costs are computed from the realised
        positions and folded into reported cash through `Portfolio.cash_flow`, so equity,
        returns and every metric are correct for **any** strategy type, with no engine
        callback involved. What is lost is the feedback loop: a charge does not shrink the
        cash available to the *next* order within the same run.

        ⚠️ That difference is second-order for most strategies and material for one case —
        a strategy sized as a fraction of live equity while running close to fully
        invested, where the charges would have throttled later positions. Use
        `Portfolio.from_holding_costs` there, and accept that it needs explicit sizes.
    """
    charges = build_charges(
        universe,
        pf.wrapper.index,
        borrowed=pd.DataFrame(pf.x.borrowed_portion()),
        notional=pd.DataFrame(pf.asset_value()).abs(),
        position=pd.DataFrame(pf.assets()),
        rates=rates,
        markup=_default_markup(markup),
        carry_markup=carry_markup,
        holidays=holidays,
        columns=list(pf.wrapper.columns),
    )
    return pf.replace(holding_costs=charges)


def _default_markup(markup):
    if markup is not None:
        return markup
    from ..core.settings import get

    return get("financing_markup", 0.03)
