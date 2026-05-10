"""Adult / Young Adult names as buddy preferences (hard same-center, soft same-rank)."""

import polars as pl
import pytest
from ortools.sat.python import cp_model

from src.data_loaders import get_centers_from_adults_df, get_youth_from_buddy_form_df
from src.config import Config
from src.linear_program.lp_model import create_crew_assignment_model
from src.models import Center, Youth

from src.models import Adult

type CrewPlacementKey = tuple[str, str, str]
type CrewPlacementVars = dict[CrewPlacementKey, cp_model.IntVar]
type SolveResult = tuple[
    cp_model.CpSolver,
    int,
    CrewPlacementVars,
    list[Youth],
    list[Center],
    list[Adult],
    list[Adult],
    CrewPlacementVars,
]


def _trivial_cfg() -> Config:
    """Zero-out secondary objectives; large adult-buddy weight for a clean signal."""
    return Config(
        min_crew_size=2,
        max_crew_size=12,
        min_adults_per_crew=2,
        max_adults_per_crew=6,
        friend_weight=0,
        adult_friend_weight=100,
        gender_weight=0,
        year_weight=0,
        history_weight=0,
        adult_gender_weight=0,
        adult_history_weight=0,
    )


def _solve(
    buddies_df: pl.DataFrame,
    crews_df: pl.DataFrame,
    *,
    cfg: Config | None = None,
    youth_list_override: list[Youth] | None = None,
) -> SolveResult:
    cfg = cfg or _trivial_cfg()
    adults = crews_df.filter(pl.col("role") != "Youth")
    youth_list = youth_list_override or get_youth_from_buddy_form_df(buddies_df)
    centers, center_only_adults, unassigned_adults = get_centers_from_adults_df(
        adults,
        center_configs=None,
    )
    model, person_crew, adult_crew = create_crew_assignment_model(
        cfg,
        youth_list,
        centers,
        center_only_adults,
        unassigned_adults,
    )
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.log_search_progress = False
    status = solver.Solve(model)
    return solver, status, person_crew, youth_list, centers, center_only_adults, unassigned_adults, adult_crew


def _youth_center(
    solver: cp_model.CpSolver,
    person_crew: CrewPlacementVars,
    name: str,
    centers: list[Center],
) -> str:
    for center in centers:
        for crew in center.crews:
            key = (name, center.name, crew.name)
            if key in person_crew and solver.Value(person_crew[key]) == 1:
                return center.name
    raise AssertionError(f"{name} not assigned")


def _youth_crew(
    solver: cp_model.CpSolver,
    person_crew: CrewPlacementVars,
    name: str,
    centers: list[Center],
) -> str:
    for center in centers:
        for crew in center.crews:
            key = (name, center.name, crew.name)
            if key in person_crew and solver.Value(person_crew[key]) == 1:
                return crew.name
    raise AssertionError(f"{name} has no crew assignment")


def _adult_crew_from_solution(
    solver: cp_model.CpSolver,
    adult_crew: CrewPlacementVars,
    name: str,
    *,
    centers: list[Center],
) -> str | None:
    """Return the crew name for a flexible adult, or ``None`` for pre-assigned (caller knows)."""
    for center in centers:
        for crew in center.crews:
            key = (name, center.name, crew.name)
            if key in adult_crew and solver.Value(adult_crew[key]) == 1:
                return crew.name
    return None


def _adult_center_from_solution(
    solver: cp_model.CpSolver,
    adult_crew: CrewPlacementVars,
    name: str,
    *,
    center_only: list[Adult],
    unassigned: list[Adult],
    crews_df: pl.DataFrame,
    centers: list[Center],
) -> str:
    pre = crews_df.filter(
        (pl.col("name") == name)
        & pl.col("Crew").is_not_null()
        & (pl.col("Crew").cast(pl.Utf8).str.strip_chars() != "")
    )
    if len(pre) >= 1:
        return str(pre["Center"][0])

    co_names = {a.name: a.fixed_center for a in center_only}
    if name in co_names:
        assert co_names[name] is not None
        return str(co_names[name])

    assert any(a.name == name for a in unassigned)
    for center in centers:
        for crew in center.crews:
            key = (name, center.name, crew.name)
            if key in adult_crew and solver.Value(adult_crew[key]) == 1:
                return center.name
    raise AssertionError(f"could not locate adult {name!r}")


@pytest.fixture()
def buddies_row_preassigned() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "name": ["Solo Teen"],
            "history": ["V"],
            "gender": ["M"],
            "year": ["Jr"],
            "first_choice": ["Hero Adult"],
            "second_choice": [""],
            "third_choice": [""],
            "siblings": [""],
            "parent_name": [""],
            "supervision_group": [""],
            "anti_buddy": [""],
        },
    )


def test_same_center_bonus_preassigned_adult_leader(
    buddies_row_preassigned: pl.DataFrame,
) -> None:
    crews_df = pl.DataFrame(
        {
            "name": ["A1", "A2", "Hero Adult", "H2", "B1", "B2", "B3"],
            "Center": ["Fayette", "Fayette", "Fayette", "Fayette", "Kanawha", "Kanawha", "Kanawha"],
            "Crew": ["F01", "F01", "F02", "F02", "K01", "K01", "K01"],
            "role": ["Adult"] * 7,
        }
    )
    solver, status, person_crew, _, centers, _, _, _ = _solve(
        buddies_row_preassigned,
        crews_df,
    )
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert _youth_center(solver, person_crew, "Solo Teen", centers) == "Fayette"
    assert ("Solo Teen", "Fayette", "F02") not in person_crew or solver.Value(person_crew["Solo Teen", "Fayette", "F02"]) == 0


def test_same_center_bonus_center_only_adult(buddies_row_preassigned: pl.DataFrame) -> None:
    crews_df = pl.DataFrame(
        {
            "name": ["A1", "A2", "C1", "C2", "B1", "B2", "B3", "Hero Adult"],
            "Center": [
                "Fayette", "Fayette",
                "Fayette", "Fayette",
                "Kanawha", "Kanawha", "Kanawha",
                "Fayette",
            ],
            "Crew": ["F01", "F01", "F02", "F02", "K01", "K01", "K01", ""],
            "role": ["Adult"] * 8,
        }
    )
    solver, status, person_crew, _, centers, co, un, adult_crew = _solve(
        buddies_row_preassigned,
        crews_df,
    )
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    teen_c = _youth_center(solver, person_crew, "Solo Teen", centers)
    hero_c = _adult_center_from_solution(
        solver,
        adult_crew,
        "Hero Adult",
        centers=centers,
        center_only=co,
        unassigned=un,
        crews_df=crews_df,
    )
    assert teen_c == hero_c == "Fayette"
    teen_crew = _youth_crew(solver, person_crew, "Solo Teen", centers)
    hero_crew = _adult_crew_from_solution(solver, adult_crew, "Hero Adult", centers=centers)
    assert hero_crew is not None
    assert teen_crew != hero_crew


def test_same_center_bonus_unassigned_adult(buddies_row_preassigned: pl.DataFrame) -> None:
    crews_df = pl.DataFrame(
        {
            "name": ["A1", "A2", "C1", "C2", "B1", "B2", "B3", "Hero Adult"],
            "Center": [
                "Fayette", "Fayette",
                "Fayette", "Fayette",
                "Kanawha", "Kanawha", "Kanawha",
                "",
            ],
            "Crew": ["F01", "F01", "F02", "F02", "K01", "K01", "K01", ""],
            "role": ["Adult"] * 8,
        }
    )
    solver, status, person_crew, _, centers, co, un, adult_crew = _solve(
        buddies_row_preassigned,
        crews_df,
    )
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    teen_c = _youth_center(solver, person_crew, "Solo Teen", centers)
    hero_c = _adult_center_from_solution(
        solver,
        adult_crew,
        "Hero Adult",
        centers=centers,
        center_only=co,
        unassigned=un,
        crews_df=crews_df,
    )
    assert teen_c == hero_c
    teen_crew = _youth_crew(solver, person_crew, "Solo Teen", centers)
    hero_crew = _adult_crew_from_solution(solver, adult_crew, "Hero Adult", centers=centers)
    assert hero_crew is not None
    assert teen_crew != hero_crew


@pytest.fixture()
def buddies_parent_pick() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "name": ["Teen Kid"],
            "history": ["V"],
            "gender": ["F"],
            "year": ["So"],
            "first_choice": ["Mom Leader"],
            "second_choice": [""],
            "third_choice": [""],
            "siblings": [""],
            "parent_name": ["Mom Leader"],
            "supervision_group": [""],
            "anti_buddy": [""],
        },
    )


def test_parent_buddy_redundant_but_same_center(buddies_parent_pick: pl.DataFrame) -> None:
    crews_df = pl.DataFrame(
        {
            "name": ["Mom Leader", "Mom Co", "D1", "D2", "X1", "X2"],
            "Center": ["Fayette", "Fayette", "Fayette", "Fayette", "Kanawha", "Kanawha"],
            "Crew": ["F01", "F01", "F02", "F02", "K01", "K01"],
            "role": ["Adult"] * 6,
        }
    )
    solver, status, person_crew, _, centers, *_ = _solve(
        buddies_parent_pick,
        crews_df,
    )
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert _youth_center(solver, person_crew, "Teen Kid", centers) == "Fayette"
    assert ("Teen Kid", "Fayette", "F01") not in person_crew or solver.Value(person_crew["Teen Kid", "Fayette", "F01"]) == 0


@pytest.fixture()
def buddies_past_leader_pick() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "name": ["Teen PL"],
            "history": ["V"],
            "gender": ["M"],
            "year": ["Sr"],
            "first_choice": ["Old Leader"],
            "second_choice": [""],
            "third_choice": [""],
            "siblings": [""],
            "parent_name": [""],
            "supervision_group": [""],
            "anti_buddy": [""],
        },
    )


def test_past_leader_as_buddy_same_center_diff_crew(
    buddies_past_leader_pick: pl.DataFrame,
) -> None:
    crews_df = pl.DataFrame(
        {
            "name": ["Old Leader", "Leader Co", "L2", "L3", "O1", "O2"],
            "Center": ["Fayette", "Fayette", "Fayette", "Fayette", "Kanawha", "Kanawha"],
            "Crew": ["F01", "F01", "F02", "F02", "K01", "K01"],
            "role": ["Adult"] * 6,
        }
    )
    youth_list = get_youth_from_buddy_form_df(buddies_past_leader_pick)
    youth_list[0].past_leaders = ["Old Leader"]
    cfg = _trivial_cfg()
    adults = crews_df.filter(pl.col("role") != "Youth")
    centers, center_only_adults, unassigned_adults = get_centers_from_adults_df(
        adults,
        center_configs=None,
    )
    model, person_crew, adult_crew = create_crew_assignment_model(
        cfg,
        youth_list,
        centers,
        center_only_adults,
        unassigned_adults,
    )
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.log_search_progress = False
    status = solver.Solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert _youth_center(solver, person_crew, "Teen PL", centers) == "Fayette"
    assert ("Teen PL", "Fayette", "F01") not in person_crew or solver.Value(person_crew["Teen PL", "Fayette", "F01"]) == 0


def _zero_weight_cfg() -> Config:
    """All objective weights zero — verifies hard same-center constraint, not the reward."""
    return Config(
        min_crew_size=2,
        max_crew_size=12,
        min_adults_per_crew=2,
        max_adults_per_crew=6,
        friend_weight=0,
        adult_friend_weight=0,
        gender_weight=0,
        year_weight=0,
        history_weight=0,
        adult_gender_weight=0,
        adult_history_weight=0,
    )


def test_hard_same_center_for_preassigned_adult_with_zero_weights(
    buddies_row_preassigned: pl.DataFrame,
) -> None:
    """With every objective weight at zero the youth must still land at the picked
    pre-assigned adult's center — proves the rule is a hard constraint, not a reward.
    """
    crews_df = pl.DataFrame(
        {
            "name": ["A1", "A2", "Hero Adult", "H2", "B1", "B2", "B3"],
            "Center": ["Fayette", "Fayette", "Fayette", "Fayette", "Kanawha", "Kanawha", "Kanawha"],
            "Crew": ["F01", "F01", "F02", "F02", "K01", "K01", "K01"],
            "role": ["Adult"] * 7,
        }
    )
    solver, status, person_crew, _, centers, _, _, _ = _solve(
        buddies_row_preassigned,
        crews_df,
        cfg=_zero_weight_cfg(),
    )
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert _youth_center(solver, person_crew, "Solo Teen", centers) == "Fayette"


def test_hard_same_center_for_unassigned_adult_with_zero_weights(
    buddies_row_preassigned: pl.DataFrame,
) -> None:
    crews_df = pl.DataFrame(
        {
            "name": ["A1", "A2", "C1", "C2", "B1", "B2", "B3", "Hero Adult"],
            "Center": [
                "Fayette", "Fayette",
                "Fayette", "Fayette",
                "Kanawha", "Kanawha", "Kanawha",
                "",
            ],
            "Crew": ["F01", "F01", "F02", "F02", "K01", "K01", "K01", ""],
            "role": ["Adult"] * 8,
        }
    )
    solver, status, person_crew, _, centers, co, un, adult_crew = _solve(
        buddies_row_preassigned,
        crews_df,
        cfg=_zero_weight_cfg(),
    )
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    teen_c = _youth_center(solver, person_crew, "Solo Teen", centers)
    hero_c = _adult_center_from_solution(
        solver,
        adult_crew,
        "Hero Adult",
        centers=centers,
        center_only=co,
        unassigned=un,
        crews_df=crews_df,
    )
    assert teen_c == hero_c


def test_hard_same_center_infeasible_when_picked_adult_unreachable() -> None:
    """If the only buddy pick is a pre-assigned adult at center A but the youth's parent
    is at center B, no feasible assignment exists — the same-center friend rule
    contradicts the same-center parent rule. This documents that the hard rule is
    intentional and applies to leader picks too.
    """
    buddies_df = pl.DataFrame(
        {
            "name": ["Trapped Teen"],
            "history": ["V"],
            "gender": ["M"],
            "year": ["Jr"],
            "first_choice": ["Hero Adult"],
            "second_choice": [""],
            "third_choice": [""],
            "siblings": [""],
            "parent_name": ["Parent At Other"],
            "supervision_group": [""],
            "anti_buddy": [""],
        },
    )
    crews_df = pl.DataFrame(
        {
            "name": ["Parent At Other", "Co Parent", "Hero Adult", "H2"],
            "Center": ["Fayette", "Fayette", "Kanawha", "Kanawha"],
            "Crew": ["F01", "F01", "K01", "K01"],
            "role": ["Adult"] * 4,
        }
    )
    _, status, *_ = _solve(buddies_df, crews_df)
    assert status == cp_model.INFEASIBLE
