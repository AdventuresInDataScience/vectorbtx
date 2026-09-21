"""Section C2 — both front doors, coercion, keywords and round-trip export."""

import numpy as np
import pandas as pd
import pytest

from vectorbtx.instruments.spec import Instrument
from vectorbtx.resolve.coerce import CoercionError, coerce_bool, coerce_name, is_blank
from vectorbtx.resolve.keywords import KeywordError, check_keyword
from vectorbtx.resolve.load import UniverseError, load_universe

DICT_SOURCE = {
    "UKX": Instrument(mode="spreadbet", point_size=1.0, asset_class="index"),
    "MSFT": Instrument(mode="equity", point_size=0.01, quote_currency="USD"),
}

FRAME_SOURCE = pd.DataFrame(
    [
        {"name": "UKX", "mode": "spreadbet", "point_size": 1.0, "asset_class": "index"},
        {"name": "MSFT", "mode": "equity", "point_size": 0.01, "quote_currency": "USD"},
    ]
)


def test_both_front_doors_produce_identical_universes():
    assert load_universe(DICT_SOURCE) == load_universe(FRAME_SOURCE)


def test_misordered_spreadsheet_still_resolves_correctly():
    reversed_frame = FRAME_SOURCE.iloc[::-1].reset_index(drop=True)
    universe = load_universe(reversed_frame)
    assert universe["UKX"].mode == "spreadbet"
    assert universe["MSFT"].mode == "equity"


def test_missing_instrument_raises_in_both_directions():
    universe = load_universe(DICT_SOURCE)
    with pytest.raises(UniverseError, match="no spec"):
        universe.order_for(["UKX", "MSFT", "AAPL"])
    with pytest.raises(UniverseError, match="no price column"):
        universe.order_for(["UKX"])


def test_column_order_comes_from_the_price_data():
    universe = load_universe(DICT_SOURCE)
    specs = universe.order_for(["MSFT", "UKX"])
    assert [s.mode for s in specs] == ["equity", "spreadbet"]


def test_typo_keyword_raises_naming_the_cell():
    frame = FRAME_SOURCE.copy()
    frame.loc[0, "slippage"] = "atuo"
    with pytest.raises(KeywordError, match="not a recognised keyword"):
        load_universe(frame)


def test_keyword_valid_for_another_parameter_raises():
    check_keyword("slippage", "auto")
    check_keyword("financing", "auto")
    with pytest.raises(KeywordError):
        check_keyword("point_size", "auto")


def test_blank_cell_resolves_to_the_mode_default_not_nan():
    frame = FRAME_SOURCE.copy()
    frame.loc[0, "increment"] = np.nan
    universe = load_universe(frame)
    assert universe["UKX"].increment == 0.01
    assert not np.isnan(universe["UKX"].increment)


def test_blank_detection():
    assert is_blank(None) and is_blank(float("nan")) and is_blank("  ")
    assert not is_blank(0.0)


def test_boolean_coercion_accepts_spreadsheet_spellings():
    for truthy in ("TRUE", "yes", "Y", "1", 1, True):
        assert coerce_bool(truthy, "allow_fractional") is True
    for falsy in ("FALSE", "no", "N", "0", 0, False):
        assert coerce_bool(falsy, "allow_fractional") is False
    with pytest.raises(CoercionError):
        coerce_bool("maybe", "allow_fractional")


def test_numeric_looking_name_survives_the_reader():
    assert coerce_name(1234.0) == "1234"
    assert coerce_name("  BP.L ") == "BP.L"


def test_string_numbers_and_whitespace_are_coerced():
    frame = pd.DataFrame([{"name": "X", "mode": " EQUITY ", "increment": "1", "min_size": "1"}])
    universe = load_universe(frame)
    assert universe["X"].mode == "equity"
    assert universe["X"].increment == 1.0


def test_extra_columns_are_kept_as_metadata():
    frame = FRAME_SOURCE.copy()
    frame["broker"] = ["IG", "IBKR"]
    universe = load_universe(frame)
    assert universe.meta["UKX"]["broker"] == "IG"
    assert "broker" in universe.to_frame().columns


def test_round_trip_export_reproduces_an_equivalent_universe(tmp_path):
    universe = load_universe(FRAME_SOURCE)
    path = tmp_path / "resolved.csv"
    universe.to_csv(path)
    assert load_universe(path) == universe


def test_duplicate_name_raises():
    frame = pd.concat([FRAME_SOURCE, FRAME_SOURCE.iloc[[0]]], ignore_index=True)
    with pytest.raises(UniverseError, match="duplicate"):
        load_universe(frame)


def test_missing_mode_raises():
    with pytest.raises(UniverseError, match="'mode' is required"):
        load_universe(pd.DataFrame([{"name": "X", "point_size": 1.0}]))


def test_subset_keeps_only_requested_columns():
    universe = load_universe(DICT_SOURCE)
    assert list(universe.subset(["MSFT"]).specs) == ["MSFT"]


def test_an_empty_price_frame_names_the_session_alignment_cause():
    """Regression: this used to die as `IndexError: index 0 is out of bounds for size 0`.

    The IndexError came from `arr.flat[0]` deep in `costs/compose.py`, several frames
    below anything the caller wrote, and said nothing about the actual problem.

    Real cause, every time: Yahoo timestamps daily bars at each market's LOCAL midnight,
    so LSE lands on 00:00 UTC and US on 05:00 UTC. Downloading both with
    `missing_index="drop"` intersects to zero rows while keeping every column, so the
    frame still looks structurally fine. That is what this message has to explain.
    """
    universe = load_universe(DICT_SOURCE)
    empty = pd.DataFrame(columns=["UKX", "MSFT"], dtype=float)
    empty.index = pd.DatetimeIndex([], tz="UTC")

    with pytest.raises(UniverseError, match="no rows"):
        universe.resolve(empty)
    with pytest.raises(UniverseError, match="each market's own midnight"):
        universe.resolve(empty)


def test_a_single_empty_column_blames_the_download_not_the_timezones():
    """One column cannot be a *misalignment* — there is nothing to misalign against.

    Found by running `examples/cases/A_cfd_single.py` live: its ticker "UKX" is not a
    Yahoo symbol (the FTSE 100 is "^FTSE"), Yahoo returned an empty frame, and the
    message above confidently blamed cross-exchange session times. A wrong-but-confident
    diagnosis is worse than a vague one.
    """
    universe = load_universe(DICT_SOURCE).subset(["UKX"])
    empty = pd.DataFrame(columns=["UKX"], dtype=float)
    empty.index = pd.DatetimeIndex([], tz="UTC")

    with pytest.raises(UniverseError, match="the download itself returned nothing"):
        universe.resolve(empty)
    # ...and it must NOT repeat the multi-exchange story here.
    try:
        universe.resolve(empty)
    except UniverseError as exc:
        assert "different exchanges" not in str(exc)
