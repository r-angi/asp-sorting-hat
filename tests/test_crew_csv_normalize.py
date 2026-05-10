"""Tests for crews CSV normalization."""

import polars as pl

from src.crew_csv_normalize import (
    ADULT_DETAIL_COLUMNS,
    OUTPUT_COLUMNS,
    normalize_crews_dataframe,
    proper_person_name,
)


def test_proper_person_name_initials_and_mc() -> None:
    assert proper_person_name("TJ BERGERON") == "TJ Bergeron"
    assert proper_person_name("JIM McMURRAY") == "Jim McMurray"
    assert proper_person_name("DEB AHLE") == "Deb Ahle"
    assert proper_person_name("  sal  bagliavio ") == "Sal Bagliavio"


def test_normalize_crews_renames_and_keeps_adult_metadata() -> None:
    df = pl.DataFrame(
        {
            "Name": ["PAT DOE", "Sam Youth"],
            "Center": [1, 1],
            "Crew": [None, None],
            "Role": ["Adult", "Youth"],
            "Gender": ["Male", "M"],
            "New/Vet": ["V", "N"],
            "Parent": ["YES", "NO"],
        }
    )
    out = normalize_crews_dataframe(df)
    assert out.columns == list(OUTPUT_COLUMNS)
    adult = out.row(0, named=True)
    assert adult == {
        "name": "Pat Doe",
        "Center": "1",
        "Crew": "",
        "role": "Adult",
        "history": "V",
        "gender": "Male",
        "parent": "Yes",
    }
    youth = out.row(1, named=True)
    assert youth["name"] == "Sam Youth"
    assert youth["role"] == "Youth"
    for col in ADULT_DETAIL_COLUMNS:
        assert youth[col] == ""


def test_normalize_crews_defaults_missing_meta_columns() -> None:
    df = pl.DataFrame(
        {
            "Name": ["PAT DOE"],
            "Center": [1],
            "Crew": [None],
            "Role": ["Adult"],
        }
    )
    out = normalize_crews_dataframe(df)
    row = out.row(0, named=True)
    assert row["history"] == ""
    assert row["gender"] == ""
    assert row["parent"] == ""


def test_normalize_from_first_last_and_adult_ya_column() -> None:
    df = pl.DataFrame(
        {
            "First Name": ["PAT", "Riley"],
            "Last Name": ["DOE", "Young"],
            "Center": [2, 2],
            "Crew": ["K01", ""],
            "Adult/YA": ["Adult", "YA"],
            "new/vet": ["N", ""],
            "gender": ["", ""],
            "parent": ["", ""],
        }
    )
    out = normalize_crews_dataframe(df)
    assert out.row(0, named=True)["name"] == "Pat Doe"
    assert out.row(1, named=True)["role"] == "Young Adult"
