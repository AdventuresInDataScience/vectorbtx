"""Section C2 — the front doors.

Both paths converge on `dict[column, Instrument]` *before* validation:

    dict[str, Instrument]  ─┐
                            ├─► dict[column, Instrument] ─► validate ─► resolve ─► numpy
    DataFrame (CSV/Excel) ──┘         (Section C)

Converging before validating is the point: the DataFrame path would otherwise bypass
every check in Section C — and it is precisely the path where bad data enters.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..instruments.defaults import field_names
from ..instruments.spec import Instrument
from ..instruments.validate import validate_universe
from .coerce import coerce_field, coerce_name, is_blank

NAME_COLUMNS = ("name", "instrument", "symbol", "ticker", "column")


class UniverseError(ValueError):
    """Raised when a universe cannot be built or joined to price data."""


class Universe:
    """A validated collection of `Instrument`s, keyed by price-frame column.

    Attributes:
        specs: `dict[column, Instrument]`, defaults already filled.
        meta: `dict[column, dict]` of pass-through spreadsheet columns.
    """

    def __init__(self, specs: dict, meta: dict | None = None):
        self.specs = {k: v.filled() for k, v in specs.items()}
        self.meta = meta or {}
        validate_universe(self.specs)

    # ############# Dunder ############# #

    def __getitem__(self, col) -> Instrument:
        return self.specs[col]

    def __contains__(self, col) -> bool:
        return col in self.specs

    def __iter__(self):
        return iter(self.specs)

    def __len__(self) -> int:
        return len(self.specs)

    def __repr__(self) -> str:
        modes = ", ".join(f"{c}:{s.mode}" for c, s in self.specs.items())
        return f"Universe({len(self.specs)} instruments: {modes})"

    def __eq__(self, other) -> bool:
        return isinstance(other, Universe) and self.specs == other.specs

    # ############# Views ############# #

    @property
    def columns(self) -> list:
        """Spec keys, in insertion order. **Not** the order resolution uses."""
        return list(self.specs)

    def subset(self, columns) -> "Universe":
        """Restrict to `columns`. Used by `Portfolio.indexing_func`."""
        keep = [c for c in columns if c in self.specs]
        return Universe(
            {c: self.specs[c] for c in keep},
            {c: self.meta[c] for c in keep if c in self.meta},
        )

    def order_for(self, columns) -> list:
        """Return specs in the order of `columns`, erroring on any mismatch.

        The instrument-name column is a join key and mismatches are fatal in **both**
        directions: every price column needs a spec and every spec needs a price column.
        A warning would scroll past on a 20-asset backtest and one column would silently
        get defaults.
        """
        cols = list(columns)
        missing = [c for c in cols if c not in self.specs]
        extra = [c for c in self.specs if c not in cols]
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"price columns with no spec: {missing}")
            if extra:
                parts.append(f"specs with no price column: {extra}")
            raise UniverseError("Universe does not match the price data — " + "; ".join(parts))
        return [self.specs[c] for c in cols]

    # ############# Resolution ############# #

    def resolve(
        self,
        data,
        rates=None,
        equity=None,
        fx_table=None,
        base_currency=None,
        **kwargs,
    ) -> dict:
        """Per-column `Order` kwargs, in the wrapper's column order.

        See `vectorbtx.resolve.resolver.resolve_universe` for every argument.
        """
        from .resolver import resolve_universe

        return resolve_universe(
            self,
            data,
            rates=rates,
            equity=equity,
            fx_table=fx_table,
            base_currency=base_currency,
            **kwargs,
        )

    def to_frame(self) -> pd.DataFrame:
        """The fully-resolved universe, including filled defaults, as a DataFrame.

        For eyeballing, and the best available defence against silent misalignment.
        """
        rows = []
        for col, spec in self.specs.items():
            row = {"name": col}
            row.update(spec.to_dict())
            row.pop("meta", None)
            row.update(self.meta.get(col, {}))
            rows.append(row)
        return pd.DataFrame(rows).set_index("name")

    def to_csv(self, path) -> None:
        """Round-trip export. See `vectorbtx.resolve.export.export_universe`."""
        from .export import export_universe

        export_universe(self, path)


# ############# Front doors ############# #


def _from_dict(source: dict) -> Universe:
    specs = {}
    for col, value in source.items():
        if isinstance(value, Instrument):
            specs[col] = value
        elif isinstance(value, dict):
            specs[col] = Instrument(**value)
        else:
            raise UniverseError(
                f"{col!r}: expected an Instrument or a dict of its fields, got {type(value).__name__}"
            )
    return Universe(specs)


def _from_frame(df: pd.DataFrame) -> Universe:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    name_col = next((c for c in NAME_COLUMNS if c in df.columns), None)
    if name_col is None:
        if df.index.name is not None:
            df = df.reset_index()
            df.columns = [str(c).strip().lower() for c in df.columns]
            name_col = next((c for c in NAME_COLUMNS if c in df.columns), None)
    if name_col is None:
        raise UniverseError(
            f"No instrument-name column found. Expected one of {NAME_COLUMNS}."
        )

    known = set(field_names())
    specs, meta = {}, {}
    for i, row in df.iterrows():
        where = f"row {i}"
        name = coerce_name(row[name_col], where)
        where = f"row {i} ({name})"
        kwargs, extras = {}, {}
        for field, raw in row.items():
            if field == name_col:
                continue
            if field in known and field != "meta":
                value = coerce_field(field, raw, where)
                if value is not None:
                    kwargs[field] = value
            elif not is_blank(raw):
                extras[field] = raw
        if "mode" not in kwargs:
            raise UniverseError(f"{where}: 'mode' is required")
        if name in specs:
            raise UniverseError(f"duplicate instrument name {name!r} at {where}")
        specs[name] = Instrument(**kwargs)
        if extras:
            meta[name] = extras
    return Universe(specs, meta)


def load_universe(source) -> Universe:
    """Build a `Universe` from a dict, a DataFrame, or a CSV/Excel path.

    Args:
        source: `dict[column, Instrument | dict]`, a `pd.DataFrame`, or a path to a
            `.csv`, `.xlsx` or `.xls` file.

    Returns:
        `Universe` — validated, with defaults filled.

    Raises:
        UniverseError: on a missing name column, a duplicate name, or a missing `mode`.
        CoercionError: on any cell that cannot be read unambiguously.
        SpecError: on any instrument a broker would refuse.

    Examples:
        >>> universe = load_universe({
        ...     "UKX": Instrument(mode="spreadbet", point_size=1.0, asset_class="index"),
        ...     "MSFT": Instrument(mode="equity", quote_currency="USD"),
        ... })
        >>> universe = load_universe("instruments.csv")
    """
    if isinstance(source, Universe):
        return source
    if isinstance(source, dict):
        return _from_dict(source)
    if isinstance(source, pd.DataFrame):
        return _from_frame(source)
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower() in (".xlsx", ".xls", ".xlsm"):
            try:
                df = pd.read_excel(path)
            except ImportError as exc:  # pragma: no cover
                raise UniverseError(
                    "Excel support needs the optional 'openpyxl' extra: "
                    "pip install vectorbtX[excel]"
                ) from exc
        else:
            df = pd.read_csv(path)
        return _from_frame(df)
    raise UniverseError(f"Cannot build a Universe from {type(source).__name__}")
