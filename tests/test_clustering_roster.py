"""Smoke tests for the standalone cluster roster PNG."""

from pathlib import Path

from src.clustering import render_cluster_roster_table
from src.models import Adult, Center, Crew, Youth


class _FakeSolver:
    """Adapter matching :class:`AssignmentsLookup`; ``Value(var) -> var``."""

    def Value(self, var: int) -> int:  # pragma: no cover - trivial
        return var


def test_render_cluster_roster_writes_nonempty_png(tmp_path: Path) -> None:
    """Build a toy assignment and assert roster export produces bytes on disk."""
    adult_a = Adult(name='Pat Parent')
    crew_a = Crew(name='crew_a', adults=[adult_a])

    alice = Youth(
        name='Alice',
        year='Fr',
        gender='F',
        history='N',
        parent_name='Pat Parent',
        siblings='Bob',
        first_choice='Bob',
        second_choice='Charlie',
        third_choice=None,
    )
    bob = Youth(
        name='Bob',
        year='So',
        gender='M',
        history='N',
        parent_name=None,
        first_choice=None,
        second_choice=None,
        third_choice=None,
    )
    charlie = Youth(
        name='Charlie',
        year='Jr',
        gender='M',
        history='V',
        parent_name=None,
        siblings='Alice|Bob',
        first_choice=None,
        second_choice=None,
        third_choice=None,
    )
    youth_list = [alice, bob, charlie]

    west = Center(name='West', crews=[crew_a, Crew(name='crew_b')])
    east = Center(name='East', crews=[Crew(name='crew_e')])
    centers = [east, west]

    person_crew: dict[tuple[str, str, str], int] = {
        ('Alice', 'West', 'crew_a'): 1,
        ('Bob', 'West', 'crew_b'): 1,
        ('Charlie', 'East', 'crew_e'): 1,
    }

    cohesion = {
        'cluster_0': {
            'size': 3,
            'center_distribution': {'East': 1, 'West': 2},
            'cohesion_score': 2 / 3,
        },
    }
    clusters = {'Alice': 0, 'Bob': 0, 'Charlie': 0}

    out_file = tmp_path / 'cluster_roster_test.png'
    render_cluster_roster_table(
        clusters,
        cohesion,
        centers,
        _FakeSolver(),
        person_crew,
        youth_list,
        str(out_file),
        buddy_weights={'Alice': 4.0, 'Bob': 0.0, 'Charlie': 0.0},
    )

    assert out_file.is_file()
    assert out_file.stat().st_size > 2000

