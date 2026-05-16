"""Edge-case tests for analysis scoring helpers.

The full integration tests in ``test_crew_assignment.py`` cover normal use; this
file isolates the empty-roster zero-safety contract so it stays explicit, plus
the ``--cluster-analysis-only`` path's behavior when the crews scaffold is empty.
"""

from pathlib import Path

import polars as pl
import pytest

from main import (
    allocate_next_versioned_run_dir,
    load_assignments_from_csv,
    normalize_run_version_label,
)
from src.analysis import (
    calculate_first_choice_same_center_pct_by_center,
    calculate_friend_choice_stats,
    calculate_friend_match_buckets,
    calculate_friend_scores,
    calculate_youth_buddy_weights_by_name,
    calculate_youth_total_buddy_weight_samples,
    synthesize_centers_from_assignments,
)
from src.clustering import analyze_clusters, merge_friend_clusters_into_assignments_csv
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


def test_calculate_friend_match_buckets_empty_inputs_return_zero_buckets() -> None:
    centers = [Center(name='Fayette', crews=[Crew(name='F01')])]
    buckets = calculate_friend_match_buckets(_FakeSolver(), {}, [], centers)
    assert buckets == {'Fayette': {0: 0, 1: 0, 2: 0, 3: 0}}


def test_calculate_friend_match_buckets_counts_same_center_friend_picks() -> None:
    """A 3-youth crew at one center: anchor matches all three picks; friends match 1 pick each."""
    centers = [Center(name='Fayette', crews=[Crew(name='F01')])]
    youth_list = [
        Youth(name='Anna', year='Fr', gender='F', history='N',
              first_choice='Bob',  second_choice='Cara', third_choice='Dan'),
        Youth(name='Bob',  year='So', gender='M', history='V',
              first_choice='Anna', second_choice=None,   third_choice=None),
        Youth(name='Cara', year='Jr', gender='F', history='V',
              first_choice='Anna', second_choice=None,   third_choice=None),
        Youth(name='Dan',  year='Sr', gender='M', history='N',
              first_choice=None,   second_choice=None,   third_choice=None),
    ]
    person_crew: dict[tuple[str, str, str], int] = {
        ('Anna', 'Fayette', 'F01'): 1,
        ('Bob',  'Fayette', 'F01'): 1,
        ('Cara', 'Fayette', 'F01'): 1,
        ('Dan',  'Fayette', 'F01'): 1,
    }

    buckets = calculate_friend_match_buckets(_FakeSolver(), person_crew, youth_list, centers)

    assert buckets == {'Fayette': {0: 1, 1: 2, 2: 0, 3: 1}}


def test_calculate_friend_match_buckets_ignores_cross_center_picks() -> None:
    """Friend picks landing at a different center don't count as same-center matches."""
    centers = [
        Center(name='A', crews=[Crew(name='A1')]),
        Center(name='B', crews=[Crew(name='B1')]),
    ]
    youth_list = [
        Youth(name='Anna', year='Fr', gender='F', history='N',
              first_choice='Bob', second_choice=None, third_choice=None),
        Youth(name='Bob',  year='So', gender='M', history='V',
              first_choice='Anna', second_choice=None, third_choice=None),
    ]
    person_crew: dict[tuple[str, str, str], int] = {
        ('Anna', 'A', 'A1'): 1,
        ('Bob',  'B', 'B1'): 1,
    }

    buckets = calculate_friend_match_buckets(_FakeSolver(), person_crew, youth_list, centers)

    assert buckets == {
        'A': {0: 1, 1: 0, 2: 0, 3: 0},
        'B': {0: 1, 1: 0, 2: 0, 3: 0},
    }


def test_first_choice_same_center_pct_by_center_matches_bucket_fixture() -> None:
    centers = [Center(name='Fayette', crews=[Crew(name='F01')])]
    youth_list = [
        Youth(name='Anna', year='Fr', gender='F', history='N',
              first_choice='Bob',  second_choice='Cara', third_choice='Dan'),
        Youth(name='Bob',  year='So', gender='M', history='V',
              first_choice='Anna', second_choice=None,   third_choice=None),
        Youth(name='Cara', year='Jr', gender='F', history='V',
              first_choice='Anna', second_choice=None,   third_choice=None),
        Youth(name='Dan',  year='Sr', gender='M', history='N',
              first_choice=None,   second_choice=None,   third_choice=None),
    ]
    person_crew: dict[tuple[str, str, str], int] = {
        ('Anna', 'Fayette', 'F01'): 1,
        ('Bob',  'Fayette', 'F01'): 1,
        ('Cara', 'Fayette', 'F01'): 1,
        ('Dan',  'Fayette', 'F01'): 1,
    }
    pct, cohort = calculate_first_choice_same_center_pct_by_center(_FakeSolver(), person_crew, youth_list, centers)
    assert pct == {'Fayette': 75.0}
    assert cohort == 75.0


def test_total_buddy_weight_samples_match_expected_weights() -> None:
    centers = [Center(name='Fayette', crews=[Crew(name='F01')])]
    youth_list = [
        Youth(name='Anna', year='Fr', gender='F', history='N',
              first_choice='Bob', second_choice=None, third_choice=None),
        Youth(name='Bob',  year='So', gender='M', history='V',
              first_choice='Anna', second_choice=None, third_choice=None),
    ]
    person_crew: dict[tuple[str, str, str], int] = {
        ('Anna', 'Fayette', 'F01'): 1,
        ('Bob',  'Fayette', 'F01'): 1,
    }
    per_center, overall = calculate_youth_total_buddy_weight_samples(_FakeSolver(), person_crew, youth_list, centers)
    assert sorted(overall) == sorted([4.0, 4.0])
    assert sorted(per_center['Fayette']) == sorted([4.0, 4.0])


def test_calculate_youth_buddy_weights_by_name_is_per_person_flat_dict() -> None:
    centers = [Center(name='Fayette', crews=[Crew(name='F01')])]
    youth_list = [
        Youth(name='Anna', year='Fr', gender='F', history='N',
              first_choice='Bob',  second_choice='Cara', third_choice='Dan'),
        Youth(name='Bob',  year='So', gender='M', history='V',
              first_choice='Anna', second_choice=None, third_choice=None),
        Youth(name='Cara', year='Jr', gender='F', history='V',
              first_choice='Anna', second_choice=None, third_choice=None),
        Youth(name='Dan',  year='Sr', gender='M', history='N',
              first_choice=None,   second_choice=None, third_choice=None),
    ]
    person_crew: dict[tuple[str, str, str], int] = {
        ('Anna', 'Fayette', 'F01'): 1,
        ('Bob',  'Fayette', 'F01'): 1,
        ('Cara', 'Fayette', 'F01'): 1,
        ('Dan',  'Fayette', 'F01'): 1,
    }
    weights = calculate_youth_buddy_weights_by_name(_FakeSolver(), person_crew, youth_list, centers)
    assert weights['Anna'] == 7.0
    assert weights['Bob'] == 4.0
    assert weights['Cara'] == 4.0
    assert weights['Dan'] == 0.0


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


def test_normalize_run_version_label_accepts_digit_and_prefix_forms() -> None:
    assert normalize_run_version_label('v1') == 'v1'
    assert normalize_run_version_label('1') == 'v1'
    assert normalize_run_version_label('01') == 'v1'
    assert normalize_run_version_label('V12') == 'v12'


def test_normalize_run_version_label_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        normalize_run_version_label('')
    with pytest.raises(ValueError):
        normalize_run_version_label('snap')


def _write_assignments_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(path)
    return path


def test_load_assignments_from_csv_loads_when_centers_empty(
    tmp_path: Path,
) -> None:
    """With no crews scaffold, the workbook alone drives placements."""
    csv_path = _write_assignments_csv(
        tmp_path / 'assignments_2099.csv',
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

    assigned = load_assignments_from_csv(csv_path, youth_list, centers=[])

    assert assigned == {
        ('Anna', '1', '101'): 1,
        ('Bob', '1', '102'): 1,
        ('Cara', '2', '201'): 1,
    }


def test_load_assignments_from_csv_filters_to_scaffold_when_present(
    tmp_path: Path,
) -> None:
    """A non-empty crews scaffold acts as a typo guard against the workbook."""
    csv_path = _write_assignments_csv(
        tmp_path / 'some.csv',
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

    assigned = load_assignments_from_csv(csv_path, youth_list, centers)

    assert assigned == {('Anna', '1', '101'): 1}


def test_allocate_next_versioned_run_dir_starts_at_v1_and_increments(tmp_path: Path) -> None:
    d1 = allocate_next_versioned_run_dir(tmp_path, 2026)
    assert d1 == tmp_path / '2026' / 'v1'
    assert d1.is_dir()

    d2 = allocate_next_versioned_run_dir(tmp_path, 2026)
    assert d2 == tmp_path / '2026' / 'v2'


def test_allocate_next_versioned_run_dir_only_considers_matching_vdirs(tmp_path: Path) -> None:
    """Unrelated dirs under ``<year>/`` do not affect the allocated version suffix."""
    (tmp_path / '2026').mkdir(parents=True)
    (tmp_path / '2026' / 'draft').mkdir()
    (tmp_path / '2026' / 'v2').mkdir()
    (tmp_path / '2026' / 'v3').mkdir()

    d = allocate_next_versioned_run_dir(tmp_path, 2026)
    assert d == tmp_path / '2026' / 'v4'


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


def test_merge_friend_clusters_into_assignments_csv_adds_columns(
    tmp_path: Path,
) -> None:
    """Youth rows get ``FriendCluster`` / ``FriendClusterId``; leaders stay blank."""
    path = tmp_path / 'assignments_2099.csv'
    rows = [
        {
            'Center': 'North', 'Crew': 'N1', 'Name': 'Alice', 'Role': 'Youth',
            'Gender': 'F', 'Year': 'Fr', 'History': 'N',
        },
        {
            'Center': 'North', 'Crew': 'N1', 'Name': 'Bob', 'Role': 'Youth',
            'Gender': 'M', 'Year': 'So', 'History': 'V',
        },
        {
            'Center': 'North', 'Crew': 'N1', 'Name': 'Coach Ray', 'Role': 'Adult',
            'Gender': '', 'Year': '', 'History': '',
        },
    ]
    pl.DataFrame(rows).write_csv(path)

    cohesion = {'cluster_0': {'size': 2, 'center_distribution': {'North': 2}, 'cohesion_score': 1.0}}
    merge_friend_clusters_into_assignments_csv(path, {'Alice': 0, 'Bob': 0}, cohesion)

    restored = pl.read_csv(path)
    assert 'BuddyWeight' not in restored.columns
    alice = restored.filter(pl.col('Name') == 'Alice').row(0, named=True)
    bob = restored.filter(pl.col('Name') == 'Bob').row(0, named=True)
    coach = restored.filter(pl.col('Name') == 'Coach Ray').row(0, named=True)

    assert alice['FriendCluster'] == 'C1' and str(alice['FriendClusterId']) == '0'
    assert bob['FriendCluster'] == 'C1' and str(bob['FriendClusterId']) == '0'
    assert coach['FriendCluster'] == '' and coach['FriendClusterId'] == ''


def test_merge_friend_clusters_writes_buddy_weight_when_mapping_passed(tmp_path: Path) -> None:
    path = tmp_path / 'z.csv'
    pl.DataFrame(
        [
            {
                'Center': 'North', 'Crew': 'N1', 'Name': 'Zed', 'Role': 'Youth',
                'Gender': 'M', 'Year': 'Sr', 'History': 'V',
            },
        ],
    ).write_csv(path)
    cohesion = {'cluster_0': {'size': 1, 'center_distribution': {'North': 1}, 'cohesion_score': 1.0}}
    merge_friend_clusters_into_assignments_csv(
        path,
        {'Zed': 0},
        cohesion,
        buddy_weights_by_name={'Zed': 7.0},
    )
    row = pl.read_csv(path).row(0, named=True)
    assert row['BuddyWeight'] in ('7', 7)


def test_merge_friend_clusters_orders_display_columns_by_cluster_size(tmp_path: Path) -> None:
    path = tmp_path / 'workbook.csv'
    pl.DataFrame(
        [
            {
                'Center': 'South', 'Crew': 'S9', 'Name': 'Cara', 'Role': 'Youth',
                'Gender': '', 'Year': '', 'History': '',
            },
            {
                'Center': 'North', 'Crew': 'N1', 'Name': 'Alice', 'Role': 'Youth',
                'Gender': '', 'Year': '', 'History': '',
            },
            {
                'Center': 'North', 'Crew': 'N1', 'Name': 'Bob', 'Role': 'Youth',
                'Gender': '', 'Year': '', 'History': '',
            },
        ],
    ).write_csv(path)

    cohesion = {
        'cluster_0': {'size': 2, 'center_distribution': {'North': 2}, 'cohesion_score': 1.0},
        'cluster_1': {'size': 1, 'center_distribution': {'South': 1}, 'cohesion_score': 1.0},
    }
    merge_friend_clusters_into_assignments_csv(path, {'Alice': 0, 'Bob': 0, 'Cara': 1}, cohesion)
    restored = pl.read_csv(path)

    assert 'BuddyWeight' not in restored.columns
    alice = restored.filter(pl.col('Name') == 'Alice').row(0, named=True)
    cara = restored.filter(pl.col('Name') == 'Cara').row(0, named=True)

    assert alice['FriendCluster'] == 'C1'
    assert cara['FriendCluster'] == 'C2'


def test_merge_friend_clusters_into_assignments_csv_replaces_prior_columns(tmp_path: Path) -> None:
    path = tmp_path / 'a.csv'
    pl.DataFrame(
        [
            {
                'Center': 'North', 'Crew': 'N1', 'Name': 'Ace', 'Role': 'Youth',
                'Gender': '', 'Year': '', 'History': '',
                'FriendCluster': 'junk', 'FriendClusterId': 'x',
                'BuddyWeight': '99',
            },
        ],
    ).write_csv(path)
    cohesion = {'cluster_0': {'size': 1, 'center_distribution': {'North': 1}, 'cohesion_score': 1.0}}
    merge_friend_clusters_into_assignments_csv(
        path,
        {'Ace': 0},
        cohesion,
        buddy_weights_by_name={'Ace': 6.0},
    )
    df = pl.read_csv(path)
    row = df.row(0, named=True)
    assert row['FriendCluster'] == 'C1' and str(row['FriendClusterId']) == '0'
    assert row['BuddyWeight'] in ('6', 6)


def test_merge_friend_clusters_no_op_when_cohesion_empty(tmp_path: Path) -> None:
    path = tmp_path / 'b.csv'
    body = (
        'Center,Crew,Name,Role,Gender,Year,History\n'
        'North,N1,Wren,Youth,F,Fr,N\n'
    )
    path.write_text(body)
    merge_friend_clusters_into_assignments_csv(path, {}, {})

    assert 'FriendCluster' not in path.read_text()
