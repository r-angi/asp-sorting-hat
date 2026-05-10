"""Tests for adult-side constraints and objectives in the crew assignment model.

Exercises driver / new+vet hard constraints, flexible-adult crew-size accounting,
and adult diversity objective behavior. Each test builds the smallest possible
Center / Crew / Youth setup so failures point at one mechanism rather than the
whole solver.
"""

from typing import Literal

from ortools.sat.python import cp_model

from src.config import Config
from src.linear_program.lp_model import create_crew_assignment_model
from src.models import Adult, Center, Crew, Leader, PlacementMode, Youth, YoungAdult

GenderLit = Literal["M", "F"]
HistoryLit = Literal["V", "N"]


def _solve(model: cp_model.CpModel) -> tuple[cp_model.CpSolver, int]:
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 4
    solver.parameters.log_search_progress = False
    return solver, solver.Solve(model)


def _make_youth(
    name: str,
    *,
    history: HistoryLit = "V",
    gender: GenderLit = "M",
    year: str = "Sr",
) -> Youth:
    return Youth(
        name=name,
        history=history,
        gender=gender,
        year=year,
        first_choice=None,
        second_choice=None,
        third_choice=None,
        siblings="",
        parent_name="",
        supervision_group=None,
        anti_buddy=None,
    )


def _make_ya(
    name: str,
    *,
    placement: PlacementMode = PlacementMode.FIXED,
    fixed_center: str | None = None,
    fixed_crew: str | None = None,
    history: HistoryLit = "V",
    gender: GenderLit = "M",
    year: str = "Sr",
) -> YoungAdult:
    return YoungAdult(
        name=name,
        placement=placement,
        fixed_center=fixed_center,
        fixed_crew=fixed_crew,
        gender=gender,
        history=history,
        year=year,
    )


def _make_adult(
    name: str,
    *,
    placement: PlacementMode = PlacementMode.FIXED,
    fixed_center: str | None = None,
    fixed_crew: str | None = None,
    gender: GenderLit = "M",
    history: HistoryLit = "V",
) -> Adult:
    return Adult(
        name=name,
        placement=placement,
        fixed_center=fixed_center,
        fixed_crew=fixed_crew,
        gender=gender,
        history=history,
    )


def _make_center(name: str, crews: list[Crew]) -> Center:
    return Center(name=name, crews=crews)


def test_driver_rule_blocks_two_ya_crew() -> None:
    """A crew with two pre-assigned YAs and zero pre-assigned Adults must be infeasible.

    ``enforce_driver_per_crew`` requires at least one ``role=='Adult'`` leader
    (YAs cannot drive). With no flexible Adults to route in, the model reports
    INFEASIBLE.
    """
    youth = [_make_youth(f"Y{i}", year="Fr") for i in range(1, 4)]
    crew = Crew(
        name="C01",
        adults=[
            _make_ya("YA1", fixed_center="CenterA", fixed_crew="C01", gender="M"),
            _make_ya("YA2", fixed_center="CenterA", fixed_crew="C01", gender="F"),
        ],
    )
    centers = [_make_center("CenterA", [crew])]

    cfg = Config(min_crew_size=5, max_crew_size=7, min_adults_per_crew=2, max_adults_per_crew=3)
    model, _person, _adult = create_crew_assignment_model(cfg, youth, centers)
    _solver, status = _solve(model)
    assert status == cp_model.INFEASIBLE


def test_driver_rule_satisfied_by_unassigned_adult() -> None:
    """Same all-YA pre-assignment, but an unassigned Adult is available — must be feasible."""
    youth = [_make_youth(f"Y{i}", year="Fr") for i in range(1, 4)]
    crew = Crew(
        name="C01",
        adults=[
            _make_ya("YA1", fixed_center="CenterA", fixed_crew="C01", gender="M"),
            _make_ya("YA2", fixed_center="CenterA", fixed_crew="C01", gender="F"),
        ],
    )
    centers = [_make_center("CenterA", [crew])]
    unassigned = [_make_adult("Floater", placement=PlacementMode.UNASSIGNED, gender="F", history="V")]

    cfg = Config(min_crew_size=5, max_crew_size=7, min_adults_per_crew=2, max_adults_per_crew=3)
    model, _person, adult_crew = create_crew_assignment_model(
        cfg, youth, centers, unassigned_adults=unassigned,
    )
    solver, status = _solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert solver.Value(adult_crew["Floater", "CenterA", "C01"]) == 1


def test_new_requires_vet_routes_vet_to_new_only_crew() -> None:
    """A New unassigned adult plus a Vet unassigned adult must land on the same crew
    when there are no other Vet leaders available — exercises the reified new->vet rule.
    """
    youth = [_make_youth(f"Y{i}", year="Fr") for i in range(1, 6)]

    crew_a = Crew(
        name="A01",
        adults=[
            _make_adult("FixedAdult", fixed_center="CenterA", fixed_crew="A01", gender="M", history="V"),
            _make_ya("YA_V", fixed_center="CenterA", fixed_crew="A01", gender="F", history="V"),
        ],
    )
    crew_b = Crew(
        name="A02",
        adults=[
            _make_ya("YA_N", fixed_center="CenterA", fixed_crew="A02", gender="F", history="N"),
        ],
    )
    centers = [_make_center("CenterA", [crew_a, crew_b])]
    unassigned = [
        _make_adult("FloaterNew", placement=PlacementMode.UNASSIGNED, gender="M", history="N"),
        _make_adult("FloaterVet", placement=PlacementMode.UNASSIGNED, gender="F", history="V"),
    ]

    cfg = Config(min_crew_size=4, max_crew_size=8, min_adults_per_crew=2, max_adults_per_crew=4)
    model, _person, adult_crew = create_crew_assignment_model(
        cfg, youth, centers, unassigned_adults=unassigned,
    )
    solver, status = _solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    new_on_a02 = solver.Value(adult_crew["FloaterNew", "CenterA", "A02"])
    vet_on_a02 = solver.Value(adult_crew["FloaterVet", "CenterA", "A02"])
    if new_on_a02 == 1:
        assert vet_on_a02 == 1


def test_ya_vet_satisfies_new_requires_vet() -> None:
    """A Young Adult with history='V' counts as the vet partner for a New Adult."""
    youth = [_make_youth(f"Y{i}", year="Fr") for i in range(1, 5)]
    crew = Crew(
        name="C01",
        adults=[
            _make_adult("FixedAdult", fixed_center="CenterA", fixed_crew="C01", gender="M", history="V"),
            _make_ya("YA_V", fixed_center="CenterA", fixed_crew="C01", gender="F", history="V"),
        ],
    )
    centers = [_make_center("CenterA", [crew])]
    unassigned = [_make_adult("NewAdult", placement=PlacementMode.UNASSIGNED, gender="F", history="N")]

    cfg = Config(min_crew_size=4, max_crew_size=8, min_adults_per_crew=2, max_adults_per_crew=4)
    model, _person, adult_crew = create_crew_assignment_model(
        cfg, youth, centers, unassigned_adults=unassigned,
    )
    solver, status = _solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert solver.Value(adult_crew["NewAdult", "CenterA", "C01"]) == 1


def test_max_crew_size_includes_flexible_adults() -> None:
    """``enforce_crew_headcount`` must count flexible adults the solver places."""
    youth = [_make_youth(f"Y{i}", year="Fr") for i in range(1, 5)]
    crew = Crew(
        name="C01",
        adults=[
            _make_adult("FixedA", fixed_center="CenterA", fixed_crew="C01", gender="M", history="V"),
            _make_adult("FixedB", fixed_center="CenterA", fixed_crew="C01", gender="F", history="N"),
        ],
    )
    centers = [_make_center("CenterA", [crew])]
    unassigned = [_make_adult("Floater", placement=PlacementMode.UNASSIGNED, gender="F", history="V")]

    cfg = Config(min_crew_size=5, max_crew_size=6, min_adults_per_crew=2, max_adults_per_crew=3)
    model, _person, _adult = create_crew_assignment_model(
        cfg, youth, centers, unassigned_adults=unassigned,
    )
    _solver, status = _solve(model)
    # 4 youth + 2 fixed + 1 flex = 7 > max_crew_size=6 → INFEASIBLE.
    assert status == cp_model.INFEASIBLE


def test_adult_diversity_objective_prefers_balanced_placement() -> None:
    """With a choice between two equally-feasible placements, solver should prefer
    the one that increases adult M/F balance (objective term)."""
    youth = [_make_youth(f"Y{i}", year="Fr") for i in range(1, 9)]
    crew_a = Crew(
        name="A01",
        adults=[_make_adult("FixedM", fixed_center="CenterA", fixed_crew="A01", gender="M", history="V")],
    )
    crew_b = Crew(
        name="A02",
        adults=[_make_adult("FixedF", fixed_center="CenterA", fixed_crew="A02", gender="F", history="V")],
    )
    centers = [_make_center("CenterA", [crew_a, crew_b])]
    unassigned = [
        _make_adult("FloaterM", placement=PlacementMode.UNASSIGNED, gender="M", history="V"),
        _make_adult("FloaterF", placement=PlacementMode.UNASSIGNED, gender="F", history="V"),
    ]

    cfg = Config(
        min_crew_size=4, max_crew_size=6,
        min_adults_per_crew=2, max_adults_per_crew=2,
        adult_gender_weight=10,
    )
    model, _person, adult_crew = create_crew_assignment_model(
        cfg, youth, centers, unassigned_adults=unassigned,
    )
    solver, status = _solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    # Optimal: FloaterF on A01, FloaterM on A02 — balanced both crews.
    assert solver.Value(adult_crew["FloaterF", "CenterA", "A01"]) == 1
    assert solver.Value(adult_crew["FloaterM", "CenterA", "A02"]) == 1


def test_two_ya_crew_without_driver_is_infeasible() -> None:
    """Replacement for the old ``test_legacy_call_without_leader_info_skips_adult_rules``.

    The driver rule is now always enforced (no more legacy skip when leader_info is
    absent). A crew with only Young Adults and no flexible Adults must fail.
    """
    youth = [_make_youth(f"Y{i}", year="Fr") for i in range(1, 4)]
    crew = Crew(
        name="C01",
        adults=[
            _make_ya("YA1", fixed_center="CenterA", fixed_crew="C01"),
            _make_ya("YA2", fixed_center="CenterA", fixed_crew="C01"),
        ],
    )
    centers = [_make_center("CenterA", [crew])]

    cfg = Config(min_crew_size=4, max_crew_size=6, min_adults_per_crew=2, max_adults_per_crew=3)
    model, _person, _adult = create_crew_assignment_model(cfg, youth, centers)
    _solver, status = _solve(model)
    assert status == cp_model.INFEASIBLE


def test_symmetry_break_orders_interchangeable_unassigned_adults() -> None:
    """Two unassigned adults sharing role/gender/history land in non-decreasing
    flat-crew-index order. Names ``A`` and ``B`` should not swap places."""
    youth = [_make_youth(f"Y{i}", year="Fr") for i in range(1, 9)]

    crew_a = Crew(
        name="A01",
        adults=[_make_adult("FixedA", fixed_center="CenterA", fixed_crew="A01", gender="M", history="V")],
    )
    crew_b = Crew(
        name="A02",
        adults=[_make_adult("FixedB", fixed_center="CenterA", fixed_crew="A02", gender="F", history="V")],
    )
    centers = [_make_center("CenterA", [crew_a, crew_b])]
    unassigned = [
        _make_adult("AdultA", placement=PlacementMode.UNASSIGNED, gender="M", history="V"),
        _make_adult("AdultB", placement=PlacementMode.UNASSIGNED, gender="M", history="V"),
    ]

    cfg = Config(
        min_crew_size=4, max_crew_size=6,
        min_adults_per_crew=2, max_adults_per_crew=2,
    )
    model, _person, adult_crew = create_crew_assignment_model(
        cfg, youth, centers, unassigned_adults=unassigned,
    )
    solver, status = _solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    a_on_a02 = solver.Value(adult_crew["AdultA", "CenterA", "A02"])
    b_on_a01 = solver.Value(adult_crew["AdultB", "CenterA", "A01"])
    assert not (a_on_a02 == 1 and b_on_a01 == 1)


def test_center_only_ya_lands_on_one_crew_in_fixed_center() -> None:
    """A Young Adult with only Center set is assigned by the solver to one crew in that center."""
    youth = [_make_youth(f"Y{i}", year="Fr") for i in range(1, 9)]
    crew_a = Crew(
        name="A01",
        adults=[_make_adult("DriverA", fixed_center="CenterA", fixed_crew="A01", gender="M", history="V")],
    )
    crew_b = Crew(
        name="A02",
        adults=[_make_adult("DriverB", fixed_center="CenterA", fixed_crew="A02", gender="F", history="V")],
    )
    centers = [_make_center("CenterA", [crew_a, crew_b])]
    center_only: list[Leader] = [
        _make_ya("RovingYA", placement=PlacementMode.CENTER_ONLY, fixed_center="CenterA", gender="F", history="V"),
        _make_adult("RovingAdult", placement=PlacementMode.CENTER_ONLY, fixed_center="CenterA", gender="M", history="V"),
    ]

    cfg = Config(min_crew_size=4, max_crew_size=6, min_adults_per_crew=2, max_adults_per_crew=2)
    model, _person, adult_crew = create_crew_assignment_model(
        cfg, youth, centers, center_only_adults=center_only,
    )
    solver, status = _solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    placements = [
        solver.Value(adult_crew["RovingYA", "CenterA", c.name]) for c in [crew_a, crew_b]
    ]
    assert sum(placements) == 1


def test_unassigned_ya_lands_on_one_crew_globally() -> None:
    """A Young Adult fully unassigned (no Center, no Crew) lands on exactly one crew in any center."""
    youth = [_make_youth(f"Y{i}", year="Fr") for i in range(1, 9)]
    crew_a = Crew(
        name="A01",
        adults=[_make_adult("DriverA", fixed_center="CenterA", fixed_crew="A01", gender="M", history="V")],
    )
    crew_b = Crew(
        name="B01",
        adults=[_make_adult("DriverB", fixed_center="CenterB", fixed_crew="B01", gender="F", history="V")],
    )
    centers = [_make_center("CenterA", [crew_a]), _make_center("CenterB", [crew_b])]
    unassigned: list[Leader] = [
        _make_ya("FloaterYA", placement=PlacementMode.UNASSIGNED, gender="F", history="V"),
        _make_adult("FloaterAdult", placement=PlacementMode.UNASSIGNED, gender="M", history="V"),
    ]

    cfg = Config(min_crew_size=4, max_crew_size=6, min_adults_per_crew=2, max_adults_per_crew=2)
    model, _person, adult_crew = create_crew_assignment_model(
        cfg, youth, centers, unassigned_adults=unassigned,
    )
    solver, status = _solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    total_placements = sum(
        solver.Value(adult_crew["FloaterYA", c.name, k.name])
        for c in centers for k in c.crews
    )
    assert total_placements == 1


def test_flexible_ya_does_not_satisfy_driver_minimum() -> None:
    """A flexible (unassigned) Young Adult cannot count toward the per-crew driver minimum.

    A crew with only fixed YAs and a single floating YA must fail — solver
    needs an actual ``role == 'Adult'`` somewhere.
    """
    youth = [_make_youth(f"Y{i}", year="Fr") for i in range(1, 4)]
    crew = Crew(
        name="C01",
        adults=[
            _make_ya("YA1", fixed_center="CenterA", fixed_crew="C01", gender="M"),
            _make_ya("YA2", fixed_center="CenterA", fixed_crew="C01", gender="F"),
        ],
    )
    centers = [_make_center("CenterA", [crew])]
    unassigned: list[YoungAdult] = [
        _make_ya("FloaterYA", placement=PlacementMode.UNASSIGNED, gender="F", history="V"),
    ]

    cfg = Config(min_crew_size=4, max_crew_size=6, min_adults_per_crew=2, max_adults_per_crew=3)
    model, _person, _adult_crew = create_crew_assignment_model(
        cfg, youth, centers, unassigned_adults=unassigned,
    )
    _solver, status = _solve(model)
    assert status == cp_model.INFEASIBLE
