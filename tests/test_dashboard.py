"""Tests for the per-center summary dashboard.

Covers the pure aggregation in :func:`compute_center_summary` (alphabetic
center order, adult exclusion from diversity stats, dense per-category
counts) and smoke tests for the rendered PNG to catch regressions in plotting
code (matplotlib raising during draw, optional buddy-match panel toggling
mosaic shape).
"""

from pathlib import Path

import polars as pl
import pytest

from src.dashboard import (
    GENDER_ORDER,
    HISTORY_ORDER,
    YEAR_ORDER,
    compute_center_summary,
    render_center_dashboard,
)


@pytest.fixture
def sample_assignments_df() -> pl.DataFrame:
    """Two-center, four-crew synthetic roster covering all role types."""
    return pl.DataFrame([
        {'Center': '1', 'Crew': '101', 'Name': 'Anna',  'Role': 'Youth',       'Gender': 'F', 'Year': 'Fr', 'History': 'N'},
        {'Center': '1', 'Crew': '101', 'Name': 'Bob',   'Role': 'Youth',       'Gender': 'M', 'Year': 'So', 'History': 'V'},
        {'Center': '1', 'Crew': '101', 'Name': 'Cara',  'Role': 'Youth',       'Gender': 'F', 'Year': 'Jr', 'History': 'V'},
        {'Center': '1', 'Crew': '101', 'Name': 'Coach', 'Role': 'Adult',       'Gender': 'M', 'Year': '',   'History': 'V'},
        {'Center': '1', 'Crew': '102', 'Name': 'Dan',   'Role': 'Youth',       'Gender': 'M', 'Year': 'Sr', 'History': 'N'},
        {'Center': '1', 'Crew': '102', 'Name': 'Elle',  'Role': 'Youth',       'Gender': 'F', 'Year': 'So', 'History': 'V'},
        {'Center': '2', 'Crew': '201', 'Name': 'Finn',  'Role': 'Youth',       'Gender': 'M', 'Year': 'Fr', 'History': 'N'},
        {'Center': '2', 'Crew': '201', 'Name': 'YALi',  'Role': 'Young Adult', 'Gender': 'F', 'Year': '',   'History': 'V'},
        {'Center': '2', 'Crew': '201', 'Name': 'Mom',   'Role': 'Adult',       'Gender': 'F', 'Year': '',   'History': 'N'},
    ])


def test_compute_center_summary_sorts_centers_alphabetically(
    sample_assignments_df: pl.DataFrame,
) -> None:
    summary = compute_center_summary(sample_assignments_df)
    assert summary.centers == ['1', '2']
    assert summary.youth_counts == {'1': 5, '2': 1}
    assert summary.adult_counts == {'1': 1, '2': 1}
    assert summary.young_adult_counts == {'1': 0, '2': 1}


def test_compute_center_summary_sort_is_lexicographic_not_youth_count() -> None:
    """Smaller-roster centers come first when their name sorts earlier."""
    df = pl.DataFrame([
        {'Center': 'Beta',  'Crew': 'B1', 'Name': 'a', 'Role': 'Youth', 'Gender': 'F', 'Year': 'Fr', 'History': 'N'},
        {'Center': 'Beta',  'Crew': 'B1', 'Name': 'b', 'Role': 'Youth', 'Gender': 'M', 'Year': 'Fr', 'History': 'N'},
        {'Center': 'Beta',  'Crew': 'B1', 'Name': 'c', 'Role': 'Youth', 'Gender': 'F', 'Year': 'Fr', 'History': 'N'},
        {'Center': 'Alpha', 'Crew': 'A1', 'Name': 'd', 'Role': 'Youth', 'Gender': 'M', 'Year': 'Fr', 'History': 'N'},
    ])
    summary = compute_center_summary(df)
    assert summary.centers == ['Alpha', 'Beta']


def test_compute_center_summary_year_breakdown_is_youth_only(
    sample_assignments_df: pl.DataFrame,
) -> None:
    summary = compute_center_summary(sample_assignments_df)
    assert summary.year_counts['1'] == {'Fr': 1, 'So': 2, 'Jr': 1, 'Sr': 1}
    assert summary.year_counts['2'] == {'Fr': 1, 'So': 0, 'Jr': 0, 'Sr': 0}


def test_compute_center_summary_gender_and_history_breakdowns(
    sample_assignments_df: pl.DataFrame,
) -> None:
    summary = compute_center_summary(sample_assignments_df)
    assert summary.gender_counts == {
        '1': {'F': 3, 'M': 2},
        '2': {'F': 0, 'M': 1},
    }
    assert summary.history_counts == {
        '1': {'V': 3, 'N': 2},
        '2': {'V': 0, 'N': 1},
    }


def test_compute_center_summary_handles_center_with_no_youth() -> None:
    df = pl.DataFrame([
        {'Center': '1', 'Crew': '101', 'Name': 'Mom', 'Role': 'Adult', 'Gender': 'F', 'Year': '', 'History': 'V'},
    ])
    summary = compute_center_summary(df)
    assert summary.centers == ['1']
    assert summary.youth_counts == {'1': 0}
    assert summary.adult_counts == {'1': 1}
    assert summary.young_adult_counts == {'1': 0}
    assert summary.year_counts == {'1': dict.fromkeys(YEAR_ORDER, 0)}
    assert summary.gender_counts == {'1': dict.fromkeys(GENDER_ORDER, 0)}
    assert summary.history_counts == {'1': dict.fromkeys(HISTORY_ORDER, 0)}


def test_compute_center_summary_drops_unknown_categories() -> None:
    """Categories outside the canonical sets (typos / blanks) shouldn't crash or pollute counts."""
    df = pl.DataFrame([
        {'Center': '1', 'Crew': '101', 'Name': 'Anna', 'Role': 'Youth', 'Gender': 'F',  'Year': 'Fr', 'History': 'N'},
        {'Center': '1', 'Crew': '101', 'Name': 'Bug',  'Role': 'Youth', 'Gender': 'NB', 'Year': 'XX', 'History': 'YY'},
    ])
    summary = compute_center_summary(df)
    assert summary.year_counts == {'1': {'Fr': 1, 'So': 0, 'Jr': 0, 'Sr': 0}}
    assert summary.gender_counts == {'1': {'F': 1, 'M': 0}}
    assert summary.history_counts == {'1': {'V': 0, 'N': 1}}


def test_render_center_dashboard_writes_png_with_friend_scores(
    tmp_path: Path, sample_assignments_df: pl.DataFrame,
) -> None:
    csv_path = tmp_path / 'assignments_2099_final.csv'
    sample_assignments_df.write_csv(csv_path)
    out_path = tmp_path / 'center_dashboard_2099.png'

    render_center_dashboard(
        assignments_csv=csv_path,
        output_path=out_path,
        year=2099,
        friend_scores={'1': 1.6, '2': 0.8},
    )

    assert out_path.is_file()
    assert out_path.stat().st_size > 0


def test_render_center_dashboard_writes_png_with_buddy_match_counts(
    tmp_path: Path, sample_assignments_df: pl.DataFrame,
) -> None:
    """When buddy match counts are passed, the renderer adds a 4th panel without crashing."""
    csv_path = tmp_path / 'assignments_2099_final.csv'
    sample_assignments_df.write_csv(csv_path)
    out_path = tmp_path / 'center_dashboard_2099.png'

    pl.DataFrame(
        [
            {
                'name': 'Anna', 'history': 'N', 'gender': 'F', 'year': 'Fr',
                'first_choice': 'Bob', 'second_choice': '', 'third_choice': '',
                'siblings': '', 'parent_name': '', 'supervision_group': '', 'anti_buddy': '',
            },
            {
                'name': 'Bob', 'history': 'V', 'gender': 'M', 'year': 'So',
                'first_choice': 'Anna', 'second_choice': '', 'third_choice': '',
                'siblings': '', 'parent_name': '', 'supervision_group': '', 'anti_buddy': '',
            },
            {
                'name': 'Cara', 'history': 'V', 'gender': 'F', 'year': 'Jr',
                'first_choice': 'Dan', 'second_choice': '', 'third_choice': '',
                'siblings': '', 'parent_name': '', 'supervision_group': '', 'anti_buddy': '',
            },
            {
                'name': 'Dan', 'history': 'N', 'gender': 'M', 'year': 'Sr',
                'first_choice': 'Elle', 'second_choice': '', 'third_choice': '',
                'siblings': '', 'parent_name': '', 'supervision_group': '', 'anti_buddy': '',
            },
            {
                'name': 'Elle', 'history': 'V', 'gender': 'F', 'year': 'So',
                'first_choice': '', 'second_choice': '', 'third_choice': '',
                'siblings': '', 'parent_name': '', 'supervision_group': '', 'anti_buddy': '',
            },
            {
                'name': 'Finn', 'history': 'N', 'gender': 'M', 'year': 'Fr',
                'first_choice': 'Elle', 'second_choice': '', 'third_choice': '',
                'siblings': '', 'parent_name': '', 'supervision_group': '', 'anti_buddy': '',
            },
        ],
    ).write_csv(tmp_path / 'buddies_2099.csv')

    render_center_dashboard(
        assignments_csv=csv_path,
        output_path=out_path,
        year=2099,
        friend_scores={'1': 1.6, '2': 0.8},
        buddy_match_counts={
            '1': {0: 1, 1: 2, 2: 1, 3: 1},
            '2': {0: 0, 1: 1, 2: 0, 3: 0},
        },
    )

    assert out_path.is_file()
    assert out_path.stat().st_size > 0


def test_render_center_dashboard_handles_missing_friend_scores(
    tmp_path: Path, sample_assignments_df: pl.DataFrame,
) -> None:
    """No buddy data → renderer still produces a PNG with a placeholder panel."""
    csv_path = tmp_path / 'assignments_2099_final.csv'
    sample_assignments_df.write_csv(csv_path)
    out_path = tmp_path / 'center_dashboard_2099.png'

    render_center_dashboard(
        assignments_csv=csv_path,
        output_path=out_path,
        year=2099,
        friend_scores=None,
    )

    assert out_path.is_file()


def test_render_center_dashboard_skips_empty_csv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    csv_path = tmp_path / 'empty.csv'
    pl.DataFrame(schema={
        'Center': pl.Utf8, 'Crew': pl.Utf8, 'Name': pl.Utf8, 'Role': pl.Utf8,
        'Gender': pl.Utf8, 'Year': pl.Utf8, 'History': pl.Utf8,
    }).write_csv(csv_path)
    out_path = tmp_path / 'center_dashboard.png'

    render_center_dashboard(
        assignments_csv=csv_path,
        output_path=out_path,
        year=2099,
        friend_scores=None,
    )

    assert not out_path.exists()
    assert 'no centers found' in capsys.readouterr().out


def test_render_center_dashboard_raises_on_missing_csv(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        render_center_dashboard(
            assignments_csv=tmp_path / 'does_not_exist.csv',
            output_path=tmp_path / 'out.png',
            year=2099,
            friend_scores=None,
        )
