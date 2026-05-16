"""Solver behavior when a youth's parent is fully UNASSIGNED on the crews CSV.

Covers the bug where ``_compute_eligibility`` raised ``Parent '…' not found in
any center`` for parents with empty ``Center`` and ``Crew``. The solver now ties
the youth's center to the parent's solver-chosen center while the existing
``enforce_parent_crew_separation_constraint`` keeps them on different crews.
"""

from typing import Any

import polars as pl
import pytest
from ortools.sat.python import cp_model

from src.config import Config
from src.data_loaders import (
    get_centers_from_adults_df,
    get_youth_from_buddy_form_df,
)
from src.linear_program.lp_model import create_crew_assignment_model
from src.models import PlacementMode

PARENT_NAME = 'Pat Parent'
YOUTH_NAME = 'Kit Parker'


def _crews_df() -> pl.DataFrame:
    return pl.DataFrame([
        {'name': 'Anne Adams', 'Center': 'Alpha', 'Crew': 'A01', 'role': 'Adult', 'history': 'V', 'gender': 'F'},
        {'name': 'Andy Allen', 'Center': 'Alpha', 'Crew': 'A02', 'role': 'Adult', 'history': 'V', 'gender': 'M'},
        {'name': 'Bea Brown', 'Center': 'Beta', 'Crew': 'B01', 'role': 'Adult', 'history': 'V', 'gender': 'F'},
        {'name': 'Bob Baker', 'Center': 'Beta', 'Crew': 'B02', 'role': 'Adult', 'history': 'V', 'gender': 'M'},
        {'name': PARENT_NAME, 'Center': '', 'Crew': '', 'role': 'Adult', 'history': 'V', 'gender': 'M'},
    ])


def _buddies_df() -> pl.DataFrame:
    blanks = {'first_choice': '', 'second_choice': '', 'third_choice': '', 'siblings': '', 'supervision_group': '', 'anti_buddy': ''}
    youths = [
        {'name': YOUTH_NAME, 'history': 'N', 'gender': 'M', 'year': 'Fr', 'parent_name': PARENT_NAME},
        {'name': 'Yara Young', 'history': 'V', 'gender': 'F', 'year': 'So', 'parent_name': ''},
        {'name': 'Zane Zhao', 'history': 'V', 'gender': 'M', 'year': 'Jr', 'parent_name': ''},
        {'name': 'Mia Moore', 'history': 'N', 'gender': 'F', 'year': 'Sr', 'parent_name': ''},
    ]
    return pl.DataFrame([{**y, **blanks} for y in youths])


@pytest.fixture(scope='module')
def unassigned_parent_solution() -> dict[str, Any]:
    """Build and solve a minimal 2-center / 2-crew model with one fully-unassigned parent."""
    youth_list = get_youth_from_buddy_form_df(_buddies_df())
    centers, center_only_adults, unassigned_adults = get_centers_from_adults_df(
        _crews_df().filter(pl.col('role') != 'Youth'), center_configs=None
    )
    parent = next(leader for leader in unassigned_adults if leader.name == PARENT_NAME)
    assert parent.placement == PlacementMode.UNASSIGNED

    cfg = Config(
        min_crew_size=2, max_crew_size=4,
        min_adults_per_crew=1, max_adults_per_crew=2,
        solver_max_time_seconds=10.0, solver_log_progress=False,
    )
    model, person_crew, adult_crew = create_crew_assignment_model(
        cfg, youth_list, centers, center_only_adults, unassigned_adults
    )

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 4
    solver.parameters.log_search_progress = False
    status = solver.Solve(model)

    youth_placement: tuple[str, str] | None = None
    parent_placement: tuple[str, str] | None = None
    for center in centers:
        for crew in center.crews:
            if solver.Value(person_crew[(YOUTH_NAME, center.name, crew.name)]) == 1:
                youth_placement = (center.name, crew.name)
            if solver.Value(adult_crew[(PARENT_NAME, center.name, crew.name)]) == 1:
                parent_placement = (center.name, crew.name)

    return {'status': status, 'youth': youth_placement, 'parent': parent_placement}


def test_unassigned_parent_model_builds(unassigned_parent_solution: dict[str, Any]) -> None:
    """Model construction no longer raises and the solver finds a placement."""
    assert unassigned_parent_solution['status'] in (cp_model.OPTIMAL, cp_model.FEASIBLE)


def test_unassigned_parent_same_center(unassigned_parent_solution: dict[str, Any]) -> None:
    """Youth lands at the same center the solver picks for the unassigned parent."""
    youth = unassigned_parent_solution['youth']
    parent = unassigned_parent_solution['parent']
    assert youth is not None and parent is not None
    assert youth[0] == parent[0], f'Youth at {youth[0]!r}, parent at {parent[0]!r}'


def test_unassigned_parent_different_crew(unassigned_parent_solution: dict[str, Any]) -> None:
    """Youth and parent share a center but never the exact crew."""
    youth = unassigned_parent_solution['youth']
    parent = unassigned_parent_solution['parent']
    assert youth is not None and parent is not None
    assert youth != parent, f'Youth and parent share crew {youth!r}'
