"""Tests for clean CSV validation."""

import textwrap
from pathlib import Path

from src.validate_clean_data import validate_clean_data


def _write(p: Path, content: str) -> None:
    p.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


def test_validate_clean_data_passes_minimal(tmp_path: Path) -> None:
    year = 2099
    crews = tmp_path / f"crews_{year}.csv"
    buddies = tmp_path / f"buddies_{year}.csv"
    _write(
        crews,
        """
        name,Center,Crew,role
        Pat Doe,Fayette,F01,Adult
        """,
    )
    _write(
        buddies,
        """
        name,history,gender,year,first_choice,second_choice,third_choice,siblings,parent_name
        Alex Youth,V,M,Fr,,,,,""
        """,
    )
    r = validate_clean_data(year, data_dir=tmp_path)
    assert r.ok


def test_duplicate_youth_name_fails(tmp_path: Path) -> None:
    year = 2098
    _write(
        tmp_path / f"crews_{year}.csv",
        """
        name,Center,Crew,role
        Pat Doe,Fayette,F01,Adult
        """,
    )
    _write(
        tmp_path / f"buddies_{year}.csv",
        """
        name,history,gender,year,first_choice,second_choice,third_choice,siblings,parent_name
        Sam Same,V,M,Jr,,,,,""
        Sam Same,N,F,Sr,,,,,""
        """,
    )
    r = validate_clean_data(year, data_dir=tmp_path)
    assert not r.ok
    assert any("duplicate name rows" in e for e in r.errors)


def test_invalid_friend_with_near_miss_hint(tmp_path: Path) -> None:
    year = 2097
    _write(
        tmp_path / f"crews_{year}.csv",
        """
        name,Center,Crew,role
        Pat Doe,Fayette,F01,Adult
        """,
    )
    _write(
        tmp_path / f"buddies_{year}.csv",
        """
        name,history,gender,year,first_choice,second_choice,third_choice,siblings,parent_name
        Jordan Smith,V,M,Jr,Jordon Smith,,,,""
        Jordan Smithe,N,F,Sr,"","",,,""
        """,
    )
    r = validate_clean_data(year, data_dir=tmp_path)
    assert not r.ok
    assert any("did you mean" in e for e in r.errors)


def test_whitespace_warning(tmp_path: Path) -> None:
    year = 2096
    _write(
        tmp_path / f"crews_{year}.csv",
        """
        name,Center,Crew,role
        "  Pat  Doe ",Fayette,F01,Adult
        """,
    )
    _write(
        tmp_path / f"buddies_{year}.csv",
        """
        name,history,gender,year,first_choice,second_choice,third_choice,siblings,parent_name
        Alex Youth,V,M,Fr,,,,,""
        """,
    )
    r = validate_clean_data(year, data_dir=tmp_path)
    assert r.ok
    assert any("spaces" in w.lower() for w in r.warnings)


def test_preflight_rejects_crew_without_driver_adult(tmp_path: Path) -> None:
    """A pre-assigned crew with only Young Adults (no Adult) must fail validation."""
    year = 2095
    _write(
        tmp_path / f"crews_{year}.csv",
        """
        name,Center,Crew,role,history,gender
        Yara YA,Fayette,F01,Young Adult,V,F
        Zane YA,Fayette,F01,Young Adult,V,M
        """,
    )
    _write(
        tmp_path / f"buddies_{year}.csv",
        """
        name,history,gender,year,first_choice,second_choice,third_choice,siblings,parent_name
        Yara YA,V,F,Sr,,,,,""
        Zane YA,V,M,Sr,,,,,""
        """,
    )
    r = validate_clean_data(year, data_dir=tmp_path)
    assert not r.ok
    assert any("no Adult (driver)" in e for e in r.errors)


def test_preflight_rejects_new_only_pre_assignment(tmp_path: Path) -> None:
    """If a pre-assigned crew has any New leader and zero Vet leaders, it fails."""
    year = 2094
    _write(
        tmp_path / f"crews_{year}.csv",
        """
        name,Center,Crew,role,history,gender
        Adult New,Fayette,F01,Adult,N,M
        Adult Also New,Fayette,F01,Adult,N,F
        """,
    )
    _write(
        tmp_path / f"buddies_{year}.csv",
        """
        name,history,gender,year,first_choice,second_choice,third_choice,siblings,parent_name
        Test Youth,V,M,Sr,,,,,""
        """,
    )
    r = validate_clean_data(year, data_dir=tmp_path)
    assert not r.ok
    assert any("New leader" in e and "Vet" in e for e in r.errors)


def test_preflight_passes_when_new_paired_with_vet(tmp_path: Path) -> None:
    """A pre-assigned crew with a New AND a Vet (Adult or YA) is fine."""
    year = 2093
    _write(
        tmp_path / f"crews_{year}.csv",
        """
        name,Center,Crew,role,history,gender
        Adult Vet,Fayette,F01,Adult,V,M
        Adult New,Fayette,F01,Adult,N,F
        """,
    )
    _write(
        tmp_path / f"buddies_{year}.csv",
        """
        name,history,gender,year,first_choice,second_choice,third_choice,siblings,parent_name
        Test Youth,V,M,Sr,,,,,""
        """,
    )
    r = validate_clean_data(year, data_dir=tmp_path)
    assert r.ok


def test_preflight_skips_center_only_adults(tmp_path: Path) -> None:
    """Adults with empty Crew (center-only) must not trigger pre-flight errors —
    the solver places them into a specific crew, so the rule applies post-solve, not here."""
    year = 2092
    _write(
        tmp_path / f"crews_{year}.csv",
        """
        name,Center,Crew,role,history,gender
        Center Only New,Fayette,,Adult,N,M
        Center Only Vet,Fayette,,Adult,V,F
        """,
    )
    _write(
        tmp_path / f"buddies_{year}.csv",
        """
        name,history,gender,year,first_choice,second_choice,third_choice,siblings,parent_name
        Test Youth,V,M,Sr,,,,,""
        """,
    )
    r = validate_clean_data(year, data_dir=tmp_path)
    assert r.ok


def test_validate_clean_data_friend_may_reference_crew_leader(tmp_path: Path) -> None:
    year = 2088
    _write(
        tmp_path / f"crews_{year}.csv",
        """
        name,Center,Crew,role
        Pat Doe,Fayette,F01,Adult
        Hero Leader,Fayette,F01,Adult
        """,
    )
    _write(
        tmp_path / f"buddies_{year}.csv",
        """
        name,history,gender,year,first_choice,second_choice,third_choice,siblings,parent_name
        Alex Youth,V,M,Fr,Hero Leader,,,,""
        """,
    )
    r = validate_clean_data(year, data_dir=tmp_path)
    assert r.ok


def test_invalid_friend_hint_checks_youth_and_leaders(tmp_path: Path) -> None:
    year = 2087
    _write(
        tmp_path / f"crews_{year}.csv",
        """
        name,Center,Crew,role
        Pat Doe,Fayette,F01,Adult
        Sam Similar,Fayette,F01,Adult
        """,
    )
    _write(
        tmp_path / f"buddies_{year}.csv",
        """
        name,history,gender,year,first_choice,second_choice,third_choice,siblings,parent_name
        Alex Youth,V,M,Fr,Sam Simlar,,,,""
        Jordan Smith,V,M,Jr,,,,,""
        """,
    )
    r = validate_clean_data(year, data_dir=tmp_path)
    assert not r.ok
    assert any("did you mean" in e and "Similar" in e for e in r.errors)


def test_sibling_cannot_reference_adult_leader(tmp_path: Path) -> None:
    year = 2086
    _write(
        tmp_path / f"crews_{year}.csv",
        """
        name,Center,Crew,role
        Pat Doe,Fayette,F01,Adult
        Sis Name,Fayette,F01,Adult
        """,
    )
    _write(
        tmp_path / f"buddies_{year}.csv",
        """
        name,history,gender,year,first_choice,second_choice,third_choice,siblings,parent_name
        Alex Youth,V,M,Fr,,,,Sis Name,""
        """,
    )
    r = validate_clean_data(year, data_dir=tmp_path)
    assert not r.ok
    assert any("sibling" in e and "Sis Name" in e for e in r.errors)


def test_unassigned_adult_with_blank_center_is_accepted(tmp_path: Path) -> None:
    """Unassigned adult rows (Center and Crew both blank) must pass validation,
    matching the runtime contract supported by ``data_loaders.get_centers_from_adults_df``.
    """
    year = 2090
    _write(
        tmp_path / f"crews_{year}.csv",
        """
        name,Center,Crew,role,history,gender
        Drift Adult,,,Adult,V,M
        Anchor Adult,Fayette,F01,Adult,V,F
        """,
    )
    _write(
        tmp_path / f"buddies_{year}.csv",
        """
        name,history,gender,year,first_choice,second_choice,third_choice,siblings,parent_name
        Test Youth,V,M,Sr,,,,,""
        """,
    )
    r = validate_clean_data(year, data_dir=tmp_path)
    assert r.ok, r.errors


def test_center_only_ya_is_accepted(tmp_path: Path) -> None:
    """A Young Adult with Center set but Crew blank passes validation; the solver picks the crew."""
    year = 2083
    _write(
        tmp_path / f"crews_{year}.csv",
        """
        name,Center,Crew,role,history,gender
        Anchor Adult,Fayette,F01,Adult,V,F
        Roving YA,Fayette,,Young Adult,V,M
        """,
    )
    _write(
        tmp_path / f"buddies_{year}.csv",
        """
        name,history,gender,year,first_choice,second_choice,third_choice,siblings,parent_name
        Test Youth,V,M,Sr,,,,,""
        """,
    )
    r = validate_clean_data(year, data_dir=tmp_path)
    assert r.ok, r.errors


def test_unassigned_ya_with_blank_center_is_accepted(tmp_path: Path) -> None:
    """A Young Adult fully unassigned (Center and Crew blank) passes validation."""
    year = 2082
    _write(
        tmp_path / f"crews_{year}.csv",
        """
        name,Center,Crew,role,history,gender
        Anchor Adult,Fayette,F01,Adult,V,F
        Floater YA,,,Young Adult,V,M
        """,
    )
    _write(
        tmp_path / f"buddies_{year}.csv",
        """
        name,history,gender,year,first_choice,second_choice,third_choice,siblings,parent_name
        Test Youth,V,M,Sr,,,,,""
        """,
    )
    r = validate_clean_data(year, data_dir=tmp_path)
    assert r.ok, r.errors


def test_blank_center_with_assigned_crew_is_rejected(tmp_path: Path) -> None:
    """A blank Center is only legal when Crew is also blank (the unassigned-adult case)."""
    year = 2089
    _write(
        tmp_path / f"crews_{year}.csv",
        """
        name,Center,Crew,role,history,gender
        Bad Adult,,F01,Adult,V,M
        """,
    )
    _write(
        tmp_path / f"buddies_{year}.csv",
        """
        name,history,gender,year,first_choice,second_choice,third_choice,siblings,parent_name
        Test Youth,V,M,Sr,,,,,""
        """,
    )
    r = validate_clean_data(year, data_dir=tmp_path)
    assert not r.ok
    assert any("empty Center" in e for e in r.errors)


def test_duplicate_leader_name_in_crews_is_rejected(tmp_path: Path) -> None:
    """Crew leader names must be unique across centers/crews so parent-center
    mapping stays deterministic.
    """
    year = 2085
    _write(
        tmp_path / f"crews_{year}.csv",
        """
        name,Center,Crew,role,history,gender
        Same Name,Fayette,F01,Adult,V,M
        Same Name,Kanawha,K01,Adult,V,F
        """,
    )
    _write(
        tmp_path / f"buddies_{year}.csv",
        """
        name,history,gender,year,first_choice,second_choice,third_choice,siblings,parent_name
        Test Youth,V,M,Sr,,,,,""
        """,
    )
    r = validate_clean_data(year, data_dir=tmp_path)
    assert not r.ok
    assert any("duplicate leader names" in e for e in r.errors)


def test_duplicate_friend_choice_in_buddies_is_rejected(tmp_path: Path) -> None:
    """Same friend in two of first/second/third choice slots must be flagged."""
    year = 2084
    _write(
        tmp_path / f"crews_{year}.csv",
        """
        name,Center,Crew,role
        Pat Doe,Fayette,F01,Adult
        """,
    )
    _write(
        tmp_path / f"buddies_{year}.csv",
        """
        name,history,gender,year,first_choice,second_choice,third_choice,siblings,parent_name
        Alex Youth,V,M,Fr,Pat Doe,Pat Doe,,,""
        Other Youth,V,F,Sr,,,,,""
        """,
    )
    r = validate_clean_data(year, data_dir=tmp_path)
    assert not r.ok
    assert any("duplicate friend pick" in e and "Pat Doe" in e for e in r.errors)


def test_crews_invalid_long_form_gender_value_rejected(tmp_path: Path) -> None:
    """Gender on Adult/YA must be one of M/F/Male/Female; ``Other`` is rejected."""
    year = 2091
    _write(
        tmp_path / f"crews_{year}.csv",
        """
        name,Center,Crew,role,history,gender
        Adult X,Fayette,F01,Adult,V,Other
        Adult Y,Fayette,F01,Adult,V,M
        """,
    )
    _write(
        tmp_path / f"buddies_{year}.csv",
        """
        name,history,gender,year,first_choice,second_choice,third_choice,siblings,parent_name
        Test Youth,V,M,Sr,,,,,""
        """,
    )
    r = validate_clean_data(year, data_dir=tmp_path)
    assert not r.ok
    assert any("invalid gender" in e for e in r.errors)
