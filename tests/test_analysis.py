"""Edge-case tests for analysis scoring helpers.

The full integration tests in ``test_crew_assignment.py`` cover normal use; this
file isolates the empty-roster zero-safety contract so it stays explicit, plus
the ``--no-reassignment`` path's behavior when the crews scaffold is empty.
"""

from pathlib import Path

import polars as pl
import pytest

import main as main_mod
from main import load_existing_assignments
from src.analysis import (
    calculate_friend_choice_stats,
    calculate_friend_scores,
    synthesize_centers_from_assignments,
)
from src.clustering import analyze_clusters
from src.models import Center, Crew, Youth


class _FakeSolver:
    def Value(self, var: int) -> int:  # pragma: no cover - trivial
        return var


def test_calculate_friend_scores_empty_youth_returns_zeros() -> None:
    centers = [Center(name='Fayette', crews=[Crew(name='F01')])]
    scores, avg = calculate_friend_scores(_FakeSolver(), {}, [], centers)
    assert scores == {'Fayette': 0.0}
    assert avg == 0.0


def test_calculate_friend_choice_stats_empty_youth_returns_zero_pcts() -> None:
    centers = [Center(name='Fayette', crews=[Crew(name='F01')])]
    stats = calculate_friend_choice_stats(_FakeSolver(), {}, [], centers)
    assert stats == {
        'first_choice_pct': 0.0,
        'second_choice_pct': 0.0,
        'third_choice_pct': 0.0,
        'multiple_friends_pct': 0.0,
    }


def test_synthesize_centers_from_assignments_groups_by_center_and_crew() -> None:
    person_crew: dict[tuple[str, str, str], int] = {
        ('Anna', '1', '101'): 1,
        ('Bob', '1', '101'): 1,
        ('Cara', '1', '102'): 1,
        ('Dan', '2', '201'): 1,
    }
    centers = synthesize_centers_from_assignments(person_crew)
    assert [c.name for c in centers] == ['1', '2']
    assert [crew.name for crew in centers[0].crews] == ['101', '102']
    assert [crew.name for crew in centers[1].crews] == ['201']
    assert all(crew.adults == [] for c in centers for crew in c.crews)


def test_synthesize_centers_from_assignments_skips_zero_and_empty() -> None:
    person_crew: dict[tuple[str, str, str], int] = {
        ('Anna', '1', '101'): 1,
        ('Bob', '1', '101'): 0,
        ('Cara', '', '102'): 1,
        ('Dan', '2', ''): 1,
    }
    centers = synthesize_centers_from_assignments(person_crew)
    assert [c.name for c in centers] == ['1']
    assert [crew.name for crew in centers[0].crews] == ['101']


def test_synthesize_centers_from_assignments_empty_input_returns_empty_list() -> None:
    assert synthesize_centers_from_assignments({}) == []


def _write_final_csv(results_dir: Path, year: int, rows: list[dict[str, str]]) -> Path:
    path = results_dir / f'assignments_{year}_final.csv'
    pl.DataFrame(rows).write_csv(path)
    return path


def test_load_existing_assignments_loads_when_centers_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no crews scaffold, the finalized CSV alone drives placements."""
    monkeypatch.setattr(main_mod, 'RESULTS_DIR', tmp_path)
    _write_final_csv(
        tmp_path,
        2099,
        [
            {'Center': '1', 'Crew': '101', 'Name': 'Anna', 'Role': 'Youth'},
            {'Center': '1', 'Crew': '102', 'Name': 'Bob', 'Role': 'Youth'},
            {'Center': '2', 'Crew': '201', 'Name': 'Cara', 'Role': 'Youth'},
            {'Center': '2', 'Crew': '201', 'Name': 'Coach', 'Role': 'Adult'},
        ],
    )
    youth_list = [
        Youth(name='Anna', year='Fr', gender='F', history='N'),
        Youth(name='Bob', year='So', gender='M', history='V'),
        Youth(name='Cara', year='Jr', gender='F', history='V'),
    ]

    assigned = load_existing_assignments(2099, youth_list, centers=[])

    assert assigned == {
        ('Anna', '1', '101'): 1,
        ('Bob', '1', '102'): 1,
        ('Cara', '2', '201'): 1,
    }


def test_load_existing_assignments_filters_to_scaffold_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-empty crews scaffold acts as a typo guard against the finalized CSV."""
    monkeypatch.setattr(main_mod, 'RESULTS_DIR', tmp_path)
    _write_final_csv(
        tmp_path,
        2099,
        [
            {'Center': '1', 'Crew': '101', 'Name': 'Anna', 'Role': 'Youth'},
            {'Center': '1', 'Crew': '999', 'Name': 'Bob', 'Role': 'Youth'},
        ],
    )
    centers = [Center(name='1', crews=[Crew(name='101')])]
    youth_list = [
        Youth(name='Anna', year='Fr', gender='F', history='N'),
        Youth(name='Bob', year='So', gender='M', history='V'),
    ]

    assigned = load_existing_assignments(2099, youth_list, centers)

    assert assigned == {('Anna', '1', '101'): 1}


def test_analyze_clusters_handles_empty_centers(tmp_path: Path) -> None:
    """analyze_clusters should print stats and skip viz instead of crashing."""
    youth_list = [
        Youth(
            name='Anna', year='Fr', gender='F', history='N',
            first_choice='Bob', second_choice='Cara', third_choice=None,
        ),
        Youth(
            name='Bob', year='So', gender='M', history='V',
            first_choice='Anna', second_choice=None, third_choice=None,
        ),
        Youth(
            name='Cara', year='Jr', gender='F', history='V',
            first_choice='Anna', second_choice=None, third_choice=None,
        ),
    ]
    person_crew: dict[tuple[str, str, str], int] = {}

    result = analyze_clusters(
        youth_list, _FakeSolver(), person_crew, centers=[],
        year=2099, output_dir=str(tmp_path),
    )

    assert result['num_clusters'] >= 1
    assert not list(tmp_path.glob('cluster_analysis_*.png'))
