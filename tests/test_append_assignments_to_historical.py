"""Tests for merging assignment results into historical_crews format."""

import io
from pathlib import Path

import polars as pl

from src.data_loaders import get_historical_youth_leaders
from src.historical import (
    append_assignments_final_to_historical,
    assignments_final_to_historical_df,
    default_versioned_assignments_path,
)


def test_default_versioned_assignments_path_v1_overrides_2025_v2(tmp_path: Path) -> None:
    base = tmp_path / 'results'
    assert default_versioned_assignments_path(2024, results_root=base) == base / '2024' / 'v1' / 'assignments_2024.csv'
    assert default_versioned_assignments_path(2026, results_root=base) == base / '2026' / 'v1' / 'assignments_2026.csv'
    assert default_versioned_assignments_path(2025, results_root=base) == base / '2025' / 'v2' / 'assignments_2025.csv'


def test_assignments_final_to_historical_df_roles_and_skip_blank_crew() -> None:
    df = pl.read_csv(
        io.StringIO(
            "Center,Crew,Name,Role\n"
            "X,C01,Alex You,Youth\n"
            "X,C01,B Lee,Adult\n"
            "X,,Skip Me,Adult\n"
            "X,C02,C Ya,Young Adult\n"
            "X,C02,Dee Adult,Adult\n"
        )
    )
    out = assignments_final_to_historical_df(df, year=2030)

    crew_years = sorted(out.get_column("crew_year").unique().sort().to_list())
    assert crew_years == ["C01 2030", "C02 2030"]

    adults = out.filter(pl.col("is_leader"))
    assert adults.height == 3
    assert set(adults.get_column("name").to_list()) == {"B Lee", "Dee Adult", "C Ya"}

    youth_like = out.filter(~pl.col("is_leader"))
    assert youth_like.height == 1
    assert "Skip Me" not in youth_like.get_column("name").to_list()


def test_assignments_merge_idempotent(tmp_path: Path) -> None:
    assign = tmp_path / "assignments_2031_final.csv"
    assign.write_text(
        "Center,Crew,Name,Role\n"
        "X,C09,Pat Kid,Youth\n"
        "X,C09,Chris Lead,Adult\n"
    )

    hist = tmp_path / "historical_crews.csv"

    merged1 = append_assignments_final_to_historical(
        2031,
        assignments_path=assign,
        historical_path=hist,
        dry_run=False,
    )

    merged2 = append_assignments_final_to_historical(
        2031,
        assignments_path=assign,
        historical_path=hist,
        dry_run=False,
    )

    key = ["name", "crew_year", "is_leader"]
    assert merged1.sort(key).equals(merged2.sort(key))
    assert merged1.height == 2
    reread = pl.read_csv(hist)
    assert reread.sort(key).equals(merged1.sort(key))


def test_append_assignments_dry_run_does_not_write(tmp_path: Path) -> None:
    assign = tmp_path / "a.csv"
    assign.write_text("Center,Crew,Name,Role\nX,C01,A,Youth\n")

    hist = tmp_path / "historical_crews.csv"
    hist.write_text("name,crew_year,is_leader\nBob,C99 2020,false\n")
    unchanged = hist.read_text()

    append_assignments_final_to_historical(
        2040,
        assignments_path=assign,
        historical_path=hist,
        dry_run=True,
    )

    assert hist.read_text() == unchanged


def test_get_historical_youth_leaders_from_mapped_assignments() -> None:
    df = pl.read_csv(
        io.StringIO(
            "Center,Crew,Name,Role\n"
            "Z,Z01,Jamie Teen,Youth\n"
            "Z,Z01,Morgan Lead,Adult\n"
        )
    )
    hist = assignments_final_to_historical_df(df, year=2099)
    leaders_map = get_historical_youth_leaders(hist)
    assert leaders_map["Jamie Teen"] == ["Morgan Lead"]
