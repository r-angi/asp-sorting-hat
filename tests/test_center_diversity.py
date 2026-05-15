"""Center-level proportional diversity objectives."""

from collections.abc import Mapping

import pytest
from ortools.sat.python import cp_model

from src.config import Config
from src.linear_program.lp_model import create_crew_assignment_model
from src.linear_program.objectives import YEARS, proportional_integer_targets
from src.models import Adult, Center, Crew, PlacementMode, Youth


def _solve(model: cp_model.CpModel, *, seed: int | None = None) -> tuple[cp_model.CpSolver, int]:
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 4
    solver.parameters.log_search_progress = False
    if seed is not None:
        solver.parameters.random_seed = seed
    return solver, solver.Solve(model)


def _crew(center_name: str, crew_code: str) -> Crew:
    return Crew(
        name=crew_code,
        adults=[
            Adult(
                name=f'{crew_code}_A',
                placement=PlacementMode.FIXED,
                fixed_center=center_name,
                fixed_crew=crew_code,
                gender='M',
                history='V',
            ),
            Adult(
                name=f'{crew_code}_B',
                placement=PlacementMode.FIXED,
                fixed_center=center_name,
                fixed_crew=crew_code,
                gender='F',
                history='V',
            ),
        ],
    )


def _two_centers_fixture() -> tuple[list[Youth], list[Center]]:
    """Six crews (three per center); two Adults per crew; thirty youth (~5 youth / crew avg)."""
    center_a = Center(
        name='Alpha',
        crews=[_crew('Alpha', f'A{i:02d}') for i in range(1, 4)],
    )
    center_b = Center(
        name='Beta',
        crews=[_crew('Beta', f'B{i:02d}') for i in range(1, 4)],
    )
    centers = [center_a, center_b]

    # 18 distinct (gender, year) pairs × 30 not possible cleanly; reuse a 15-pair slate twice.
    pair_cycle: tuple[tuple[str, str], ...] = (
        ('M', 'Fr'),
        ('M', 'So'),
        ('M', 'Jr'),
        ('M', 'Sr'),
        ('F', 'Fr'),
        ('F', 'So'),
        ('F', 'Jr'),
        ('F', 'Sr'),
        ('M', 'Fr'),
        ('F', 'So'),
        ('M', 'Jr'),
        ('F', 'Sr'),
        ('M', 'So'),
        ('F', 'Jr'),
        ('M', 'Sr'),
    )
    youths = [
        Youth(
            name=f'Y{i}',
            gender=g,
            year=y,
            history='V' if i % 2 == 0 else 'N',
            parent_name=None,
            siblings=None,
            first_choice=None,
            second_choice=None,
            third_choice=None,
            supervision_group=None,
            anti_buddy=None,
        )
        for i, (g, y) in enumerate(pair_cycle + pair_cycle, start=1)
    ]
    assert len(youths) == 30
    assert len(center_a.crews) == len(center_b.crews) == 3
    return youths, centers


def _tiny_two_crew_fixture() -> tuple[list[Youth], list[Center]]:
    """Two centers, one crew each, ten youth (five M / five F) — fast model builds."""
    centers = [
        Center(name='Alpha', crews=[_crew('Alpha', 'A01')]),
        Center(name='Beta', crews=[_crew('Beta', 'B01')]),
    ]
    youths = [
        Youth(
            name=f'M{i}',
            gender='M',
            year='Fr',
            history='V',
            parent_name=None,
            siblings=None,
            first_choice=None,
            second_choice=None,
            third_choice=None,
            supervision_group=None,
            anti_buddy=None,
        )
        for i in range(1, 6)
    ] + [
        Youth(
            name=f'F{i}',
            gender='F',
            year='Fr',
            history='V',
            parent_name=None,
            siblings=None,
            first_choice=None,
            second_choice=None,
            third_choice=None,
            supervision_group=None,
            anti_buddy=None,
        )
        for i in range(6, 11)
    ]
    assert len(youths) == 10
    return youths, centers


def _isolate_cfg(
    *,
    center_gender_weight: int = 0,
    center_year_weight: int = 0,
    center_history_weight: int = 0,
) -> Config:
    return Config(
        min_crew_size=5,
        max_crew_size=7,
        min_adults_per_crew=2,
        max_adults_per_crew=3,
        friend_weight=0,
        adult_friend_weight=0,
        gender_weight=0,
        year_weight=0,
        history_weight=0,
        adult_gender_weight=0,
        adult_history_weight=0,
        center_gender_weight=center_gender_weight,
        center_year_weight=center_year_weight,
        center_history_weight=center_history_weight,
    )


def _scaled_linear_objective_size(model: cp_model.CpModel) -> int:
    """Scalar mass of CP-SAT's stored linear minimization objective (coeffs + offset).

    Matches are maximized internally by minimizing `-expr`; we only compare magnitudes across
    deterministic builds sharing the same center-proportional skeleton.
    """
    obj = model.Proto().objective
    return sum(abs(int(c)) for c in obj.coeffs) + abs(int(obj.offset))


def _assignments_center_bucket_counts(
    solver: cp_model.CpSolver,
    person_crew: Mapping[tuple[str, str, str], cp_model.LinearExpr],
    regular_youth: list[Youth],
    centers: list[Center],
    *,
    attribute: str,
    bucket: str,
) -> dict[str, int]:
    counts: dict[str, int] = {c.name: 0 for c in centers}
    for y in regular_youth:
        if getattr(y, attribute) != bucket:
            continue
        placed = False
        for center in centers:
            for crew in center.crews:
                key = (y.name, center.name, crew.name)
                if key in person_crew and solver.Value(person_crew[key]) == 1:
                    counts[center.name] += 1
                    placed = True
                    break
            if placed:
                break
        assert placed, f'youth {y.name} not assigned'
    return counts


def _l1_demographic_deviation(
    regular_youth: list[Youth],
    centers: list[Center],
    *,
    attribute: str,
    buckets: tuple[str, ...],
    solver: cp_model.CpSolver,
    person_crew: Mapping[tuple[str, str, str], cp_model.LinearExpr],
) -> int:
    center_crew_counts = [len(c.crews) for c in centers]
    deviation = 0
    for bucket in buckets:
        total_bucket = sum(1 for y in regular_youth if getattr(y, attribute) == bucket)
        if total_bucket == 0:
            continue
        targets = proportional_integer_targets(total_bucket, center_crew_counts)
        for center, target in zip(centers, targets, strict=True):
            counts = _assignments_center_bucket_counts(
                solver,
                person_crew,
                regular_youth,
                centers,
                attribute=attribute,
                bucket=bucket,
            )
            deviation += abs(counts[center.name] - target)
    return deviation


def test_proportional_integer_targets_sums_total() -> None:
    for n in range(41):
        out = proportional_integer_targets(n, [3, 3, 3])
        assert sum(out) == n
        assert len(out) == 3


def test_proportional_integer_targets_two_equal_centers_odd_total() -> None:
    targets = proportional_integer_targets(15, [3, 3])
    assert sum(targets) == 15
    assert abs(targets[0] - targets[1]) <= 1


def test_center_gender_balance_prefers_proportional_split() -> None:
    youth_list, centers = _two_centers_fixture()
    cfg0 = _isolate_cfg()
    cfg1 = _isolate_cfg(center_gender_weight=80)

    m0, p0, _ = create_crew_assignment_model(cfg0, youth_list, centers)
    s0, st0 = _solve(m0)
    assert st0 in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    m1, p1, _ = create_crew_assignment_model(cfg1, youth_list, centers)
    s1, st1 = _solve(m1)
    assert st1 in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    dev0 = _l1_demographic_deviation(youth_list, centers, attribute='gender', buckets=('M', 'F'), solver=s0, person_crew=p0)
    dev1 = _l1_demographic_deviation(youth_list, centers, attribute='gender', buckets=('M', 'F'), solver=s1, person_crew=p1)
    assert dev1 <= dev0


@pytest.mark.parametrize(('attribute', 'buckets'), [('year', YEARS), ('history', ('V', 'N'))])
def test_center_year_history_balance(attribute: str, buckets: tuple[str, ...]) -> None:
    youth_list, centers = _two_centers_fixture()

    cfg0 = _isolate_cfg()
    cfg1 = _isolate_cfg(
        center_year_weight=80 if attribute == 'year' else 0,
        center_history_weight=80 if attribute == 'history' else 0,
    )

    m0, p0, _ = create_crew_assignment_model(cfg0, youth_list, centers)
    s0, st0 = _solve(m0)
    assert st0 in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    m1, p1, _ = create_crew_assignment_model(cfg1, youth_list, centers)
    s1, st1 = _solve(m1)
    assert st1 in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    dev0 = _l1_demographic_deviation(youth_list, centers, attribute=attribute, buckets=buckets, solver=s0, person_crew=p0)
    dev1 = _l1_demographic_deviation(youth_list, centers, attribute=attribute, buckets=buckets, solver=s1, person_crew=p1)
    assert dev1 <= dev0


def test_center_balance_softness_scales_non_center_objective_mass() -> None:
    youth_list, centers = _tiny_two_crew_fixture()
    ratio_hi = 7
    cfg_lo = Config(
        center_balance_softness=1,
        min_crew_size=5,
        max_crew_size=7,
        min_adults_per_crew=2,
        max_adults_per_crew=3,
        friend_weight=0,
        adult_friend_weight=0,
        gender_weight=2,
        year_weight=3,
        history_weight=0,
        adult_gender_weight=5,
        adult_history_weight=0,
        center_gender_weight=0,
        center_year_weight=0,
        center_history_weight=0,
    )
    cfg_hi = Config(
        center_balance_softness=ratio_hi,
        min_crew_size=5,
        max_crew_size=7,
        min_adults_per_crew=2,
        max_adults_per_crew=3,
        friend_weight=0,
        adult_friend_weight=0,
        gender_weight=2,
        year_weight=3,
        history_weight=0,
        adult_gender_weight=5,
        adult_history_weight=0,
        center_gender_weight=0,
        center_year_weight=0,
        center_history_weight=0,
    )
    model_lo, _, _ = create_crew_assignment_model(cfg_lo, youth_list, centers)
    model_hi, _, _ = create_crew_assignment_model(cfg_hi, youth_list, centers)

    mass_lo = _scaled_linear_objective_size(model_lo)
    mass_hi = _scaled_linear_objective_size(model_hi)
    assert mass_hi == ratio_hi * mass_lo


def test_center_balance_softness_leaves_pure_center_penalty_mass_unchanged() -> None:
    youth_list, centers = _tiny_two_crew_fixture()
    m_a, _, _ = create_crew_assignment_model(
        Config(
            center_balance_softness=1,
            min_crew_size=5,
            max_crew_size=7,
            min_adults_per_crew=2,
            max_adults_per_crew=3,
            friend_weight=0,
            adult_friend_weight=0,
            gender_weight=0,
            year_weight=0,
            history_weight=0,
            adult_gender_weight=0,
            adult_history_weight=0,
            center_gender_weight=11,
            center_year_weight=0,
            center_history_weight=0,
        ),
        youth_list,
        centers,
    )
    m_b, _, _ = create_crew_assignment_model(
        Config(
            center_balance_softness=99,
            min_crew_size=5,
            max_crew_size=7,
            min_adults_per_crew=2,
            max_adults_per_crew=3,
            friend_weight=0,
            adult_friend_weight=0,
            gender_weight=0,
            year_weight=0,
            history_weight=0,
            adult_gender_weight=0,
            adult_history_weight=0,
            center_gender_weight=11,
            center_year_weight=0,
            center_history_weight=0,
        ),
        youth_list,
        centers,
    )
    assert _scaled_linear_objective_size(m_a) == _scaled_linear_objective_size(m_b)
