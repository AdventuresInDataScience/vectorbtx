"""Section C2 — round-trip export.

If it reads a DataFrame it writes one, showing the fully resolved universe including
filled defaults. The best available defence against silent misalignment: a user can
export, eyeball, and re-import to confirm what the library actually believes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_universe(universe, path=None) -> pd.DataFrame:
    """Export a `Universe` to a DataFrame, and optionally to CSV or Excel.

    Args:
        universe: the `Universe` to export.
        path: optional `.csv`/`.xlsx` destination. None returns the frame only.

    Returns:
        `pd.DataFrame` indexed by instrument name, including pass-through metadata.

    Examples:
        >>> export_universe(universe, "resolved.csv")
        >>> load_universe("resolved.csv") == universe
        True
    """
    frame = universe.to_frame()
    if path is not None:
        path = Path(path)
        if path.suffix.lower() in (".xlsx", ".xls", ".xlsm"):
            frame.to_excel(path)
        else:
            frame.to_csv(path)
    return frame
