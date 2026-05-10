"""Comprehensive test suite for crew assignment system."""
import polars as pl
import pytest
from ortools.sat.python import cp_model
from pathlib import Path
from typing import Any

from src.data_loaders import (
    get_centers_from_adults_df,
    get_historical_youth_leaders,
    get_youth_from_buddy_form_df,
)
from src.config import Config
from src.linear_program.lp_model import create_crew_assignment_model
from src.models import Youth, Center


# Fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def test_data() -> dict[str, Any]:
    """Load all test data and solve the model once."""
    # Load test data
    adult_crew_df = pl.read_csv(FIXTURES_DIR / "crews_test.csv").filter(
        pl.col("role") != "Youth"
    )
    youth_df = pl.read_csv(FIXTURES_DIR / "buddies_test.csv")
    historical_pairings_df = pl.read_csv(FIXTURES_DIR / "historical_crews_test.csv")
    
    # Process data
    youth_list = get_youth_from_buddy_form_df(youth_df)
    centers, center_only_adults, unassigned_adults = get_centers_from_adults_df(
        adult_crew_df, center_configs=None
    )
    
    # Update youth list with past leaders
    historical_youth_leaders = get_historical_youth_leaders(historical_pairings_df)
    for youth in youth_list:
        if youth.name in historical_youth_leaders:
            youth.past_leaders = historical_youth_leaders[youth.name]
    
    # Create and solve model
    cfg = Config.default()
    model, person_crew, adult_crew = create_crew_assignment_model(
        cfg, youth_list, centers, center_only_adults, unassigned_adults
    )
    
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 8
    solver.parameters.log_search_progress = False
    status = solver.Solve(model)
    
    return {
        "status": status,
        "solver": solver,
        "person_crew": person_crew,
        "adult_crew": adult_crew,
        "youth_list": youth_list,
        "centers": centers,
        "center_only_adults": center_only_adults,
        "unassigned_adults": unassigned_adults,
        "config": cfg,
        "youth_df": youth_df,
    }


@pytest.fixture(scope="module")
def assignments(test_data: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """Extract youth assignments (name -> (center, crew))."""
    solver = test_data["solver"]
    person_crew = test_data["person_crew"]
    centers = test_data["centers"]
    youth_list = test_data["youth_list"]
    
    result = {}
    for youth in youth_list:
        for center in centers:
            for crew in center.crews:
                key = (youth.name, center.name, crew.name)
                if key in person_crew and solver.Value(person_crew[key]) == 1:
                    result[youth.name] = (center.name, crew.name)
                    break
    return result


@pytest.fixture(scope="module")
def adult_assignments(test_data: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """Extract adult assignments for center-only and unassigned adults."""
    solver = test_data["solver"]
    adult_crew = test_data["adult_crew"]
    centers = test_data["centers"]
    center_only_adults = test_data["center_only_adults"]
    unassigned_adults = test_data["unassigned_adults"]
    
    result = {}
    centers_by_name = {c.name: c for c in centers}

    for adult in center_only_adults:
        assert adult.fixed_center is not None
        center = centers_by_name[adult.fixed_center]
        for crew in center.crews:
            if solver.Value(adult_crew[adult.name, center.name, crew.name]) == 1:
                result[adult.name] = (center.name, crew.name)
                break

    for adult in unassigned_adults:
        for center in centers:
            for crew in center.crews:
                if solver.Value(adult_crew[adult.name, center.name, crew.name]) == 1:
                    result[adult.name] = (center.name, crew.name)
                    break

    return result


def test_solution_found(test_data: dict[str, Any]) -> None:
    """Verify that the model finds a feasible or optimal solution."""
    status = test_data["status"]
    assert status in (
        cp_model.OPTIMAL,
        cp_model.FEASIBLE,
    ), f"Expected OPTIMAL or FEASIBLE, got status {status}"


def test_basic_assignments(
    test_data: dict[str, Any], assignments: dict[str, tuple[str, str]]
) -> None:
    """Verify each youth is assigned to exactly one crew."""
    youth_list = test_data["youth_list"]
    regular_youth = [y for y in youth_list if y.role == "Youth"]
    
    # Check all regular youth are assigned
    for youth in regular_youth:
        assert youth.name in assignments, f"Youth {youth.name} not assigned"
        center, crew = assignments[youth.name]
        assert center is not None, f"Youth {youth.name} has no center"
        assert crew is not None, f"Youth {youth.name} has no crew"


def test_young_adults_in_correct_crews(
    test_data: dict[str, Any], assignments: dict[str, tuple[str, str]]
) -> None:
    """Verify young adults are assigned to their pre-assigned crews."""
    youth_list = test_data["youth_list"]
    centers = test_data["centers"]
    young_adults = [y for y in youth_list if y.role == "Young Adult"]
    
    for ya in young_adults:
        assigned_center, assigned_crew = assignments[ya.name]
        
        # Find the crew they should be in
        expected_crew = None
        expected_center = None
        for center in centers:
            for crew in center.crews:
                if ya.name in crew.adult_names:
                    expected_center = center.name
                    expected_crew = crew.name
                    break
        
        assert (
            expected_center is not None
        ), f"Young adult {ya.name} not found in any crew"
        assert (
            assigned_center == expected_center
        ), f"Young adult {ya.name} in wrong center"
        assert assigned_crew == expected_crew, f"Young adult {ya.name} in wrong crew"


def test_crew_sizes(
    test_data: dict[str, Any],
    assignments: dict[str, tuple[str, str]],
    adult_assignments: dict[str, tuple[str, str]],
) -> None:
    """Verify all crews are within min and max size constraints.

    Headcount per crew counts youth + pre-assigned adults (and YAs in crew.adults)
    + flexible adults the solver placed via adult_crew.
    """
    centers = test_data["centers"]
    config = test_data["config"]
    youth_list = test_data["youth_list"]

    crew_counts: dict[tuple[str, str], int] = {}
    for youth in youth_list:
        if youth.role != "Youth":
            continue
        center, crew = assignments[youth.name]
        crew_counts[(center, crew)] = crew_counts.get((center, crew), 0) + 1

    flex_counts: dict[tuple[str, str], int] = {}
    for _name, (center, crew) in adult_assignments.items():
        flex_counts[(center, crew)] = flex_counts.get((center, crew), 0) + 1

    for center in centers:
        for crew in center.crews:
            youth_count = crew_counts.get((center.name, crew.name), 0)
            preassigned_count = len(crew.adults)
            flex_count = flex_counts.get((center.name, crew.name), 0)
            total_size = youth_count + preassigned_count + flex_count

            assert (
                total_size >= config.min_crew_size
            ), f"Crew {center.name}/{crew.name} too small: {total_size} < {config.min_crew_size}"
            assert (
                total_size <= config.max_crew_size
            ), f"Crew {center.name}/{crew.name} too large: {total_size} > {config.max_crew_size}"


def test_adult_counts(
    test_data: dict[str, Any], adult_assignments: dict[str, tuple[str, str]]
) -> None:
    """Verify all crews have 2-3 adults."""
    centers = test_data["centers"]
    config = test_data["config"]
    center_only_adults = test_data["center_only_adults"]
    unassigned_adults = test_data["unassigned_adults"]
    
    # Count adults per crew
    adult_counts: dict[tuple[str, str], int] = {}
    
    # Count pre-assigned adults
    for center in centers:
        for crew in center.crews:
            adult_counts[(center.name, crew.name)] = len(crew.adults)
    
    # Add center-only and unassigned adults
    for adult in [*center_only_adults, *unassigned_adults]:
        if adult.name in adult_assignments:
            center, crew = adult_assignments[adult.name]
            adult_counts[(center, crew)] = adult_counts.get((center, crew), 0) + 1
    
    # Check constraints
    for center in centers:
        for crew in center.crews:
            count = adult_counts[(center.name, crew.name)]
            assert (
                count >= config.min_adults_per_crew
            ), f"Crew {center.name}/{crew.name} has too few adults: {count}"
            assert (
                count <= config.max_adults_per_crew
            ), f"Crew {center.name}/{crew.name} has too many adults: {count}"


def test_parent_constraints(
    test_data: dict[str, Any], assignments: dict[str, tuple[str, str]]
) -> None:
    """Verify youth with parents are in same center but different crew."""
    youth_list = test_data["youth_list"]
    centers = test_data["centers"]
    
    # Build parent location mapping
    parent_locations: dict[str, tuple[str, str]] = {}
    for center in centers:
        for crew in center.crews:
            for adult in crew.adults:
                parent_locations[adult.name] = (center.name, crew.name)
    
    # Check youth with parents
    for youth in youth_list:
        if youth.parent_names_list:
            youth_center, youth_crew = assignments[youth.name]
            
            for parent_name in youth.parent_names_list:
                assert (
                    parent_name in parent_locations
                ), f"Parent {parent_name} not found"
                parent_center, parent_crew = parent_locations[parent_name]
                
                # Same center
                assert (
                    youth_center == parent_center
                ), f"Youth {youth.name} in different center from parent {parent_name}"
                
                # Different crew
                assert (
                    youth_crew != parent_crew
                ), f"Youth {youth.name} in same crew as parent {parent_name}"


def test_sibling_constraints(
    test_data: dict[str, Any], assignments: dict[str, tuple[str, str]]
) -> None:
    """Verify siblings are in same center but different crew."""
    youth_list = test_data["youth_list"]
    youth_dict = {y.name: y for y in youth_list}
    
    # Track checked pairs to avoid duplicates
    checked_pairs: set[tuple[str, str]] = set()
    
    for youth in youth_list:
        for sibling in youth.siblings_list:
            if sibling in youth_dict:
                pair = tuple(sorted([youth.name, sibling]))
                if pair not in checked_pairs:
                    checked_pairs.add(pair)
                    
                    youth_center, youth_crew = assignments[youth.name]
                    sib_center, sib_crew = assignments[sibling]
                    
                    # Same center
                    assert (
                        youth_center == sib_center
                    ), f"Siblings {youth.name} and {sibling} in different centers"
                    
                    # Different crew
                    assert (
                        youth_crew != sib_crew
                    ), f"Siblings {youth.name} and {sibling} in same crew"


def test_friend_constraints(
    test_data: dict[str, Any], assignments: dict[str, tuple[str, str]]
) -> None:
    """Verify friends are in different crews but at least one is in same center."""
    youth_list = test_data["youth_list"]
    youth_dict = {y.name: y for y in youth_list}
    
    # Track checked pairs
    checked_pairs: set[tuple[str, str]] = set()
    
    for youth in youth_list:
        friend_choices = [youth.first_choice, youth.second_choice, youth.third_choice]
        valid_choices = [f for f in friend_choices if f and f in youth_dict]
        
        # Check at least one friend in same center
        if valid_choices:
            youth_center, _ = assignments[youth.name]
            friend_centers = [assignments[f][0] for f in valid_choices]
            assert (
                youth_center in friend_centers
            ), f"Youth {youth.name} has no friends in center {youth_center}"
        
        # Check friends not in same crew
        for friend in valid_choices:
            pair = tuple(sorted([youth.name, friend]))
            if pair not in checked_pairs:
                checked_pairs.add(pair)
                
                youth_center, youth_crew = assignments[youth.name]
                friend_center, friend_crew = assignments[friend]
                
                # Cannot be in same crew
                if youth_center == friend_center:
                    assert (
                        youth_crew != friend_crew
                    ), f"Friends {youth.name} and {friend} in same crew"


def test_anti_buddy_constraints(
    test_data: dict[str, Any], assignments: dict[str, tuple[str, str]]
) -> None:
    """Verify anti-buddies are not at the same center."""
    youth_list = test_data["youth_list"]
    youth_dict = {y.name: y for y in youth_list}
    
    checked_pairs: set[tuple[str, str]] = set()
    
    for youth in youth_list:
        for anti_buddy in youth.anti_buddy_list:
            if anti_buddy in youth_dict:
                pair = tuple(sorted([youth.name, anti_buddy]))
                if pair not in checked_pairs:
                    checked_pairs.add(pair)
                    
                    youth_center, _ = assignments[youth.name]
                    anti_center, _ = assignments[anti_buddy]
                    
                    assert (
                        youth_center != anti_center
                    ), f"Anti-buddies {youth.name} and {anti_buddy} in same center"


def test_past_leader_constraints(
    test_data: dict[str, Any], assignments: dict[str, tuple[str, str]]
) -> None:
    """Verify youth are not assigned to crews with their past leaders."""
    youth_list = test_data["youth_list"]
    centers = test_data["centers"]
    
    # Build leader location mapping
    leader_crews: dict[str, list[tuple[str, str]]] = {}
    for center in centers:
        for crew in center.crews:
            for adult in crew.adults:
                if adult.name not in leader_crews:
                    leader_crews[adult.name] = []
                leader_crews[adult.name].append((center.name, crew.name))
    
    # Check youth with past leaders
    for youth in youth_list:
        if youth.past_leaders:
            youth_center, youth_crew = assignments[youth.name]
            
            for leader in youth.past_leaders:
                if leader in leader_crews:
                    leader_locations = leader_crews[leader]
                    assert (
                        youth_center,
                        youth_crew,
                    ) not in leader_locations, (
                        f"Youth {youth.name} assigned to crew with past leader {leader}"
                    )


def test_supervision_groups(
    test_data: dict[str, Any], assignments: dict[str, tuple[str, str]]
) -> None:
    """Verify max 2 youth per supervision group per center."""
    youth_list = test_data["youth_list"]
    centers = test_data["centers"]
    
    # Group youth by supervision group
    groups: dict[str, list[Youth]] = {}
    for youth in youth_list:
        if youth.supervision_group:
            if youth.supervision_group not in groups:
                groups[youth.supervision_group] = []
            groups[youth.supervision_group].append(youth)
    
    # Check each group at each center
    for group_name, group_youth in groups.items():
        for center in centers:
            count = sum(
                1 for y in group_youth if assignments[y.name][0] == center.name
            )
            assert (
                count <= 2
            ), f"Group {group_name} has {count} > 2 youth at center {center.name}"


def test_center_only_adults(
    test_data: dict[str, Any], adult_assignments: dict[str, tuple[str, str]]
) -> None:
    """Verify center-only adults are assigned to their correct center."""
    center_only_adults = test_data["center_only_adults"]
    
    for adult in center_only_adults:
        assert (
            adult.name in adult_assignments
        ), f"Center-only adult {adult.name} not assigned"
        actual_center, _ = adult_assignments[adult.name]
        assert (
            actual_center == adult.fixed_center
        ), f"Adult {adult.name} assigned to wrong center: {actual_center} != {adult.fixed_center}"


def test_unassigned_adults(
    test_data: dict[str, Any], adult_assignments: dict[str, tuple[str, str]]
) -> None:
    """Verify unassigned adults are assigned to exactly one crew."""
    unassigned_adults = test_data["unassigned_adults"]
    
    for adult in unassigned_adults:
        assert (
            adult.name in adult_assignments
        ), f"Unassigned adult {adult.name} not assigned"
        center, crew = adult_assignments[adult.name]
        assert center is not None, f"Adult {adult.name} has no center"
        assert crew is not None, f"Adult {adult.name} has no crew"


def test_objectives_scored(
    test_data: dict[str, Any], assignments: dict[str, tuple[str, str]]
) -> None:
    """Verify that friend preferences and diversity metrics are positive."""
    youth_list = test_data["youth_list"]
    youth_dict = {y.name: y for y in youth_list}
    centers = test_data["centers"]
    
    # Count friend preferences satisfied
    friend_score = 0
    for youth in youth_list:
        youth_center, _ = assignments[youth.name]
        
        if youth.first_choice and youth.first_choice in youth_dict:
            if assignments[youth.first_choice][0] == youth_center:
                friend_score += 3
        if youth.second_choice and youth.second_choice in youth_dict:
            if assignments[youth.second_choice][0] == youth_center:
                friend_score += 2
        if youth.third_choice and youth.third_choice in youth_dict:
            if assignments[youth.third_choice][0] == youth_center:
                friend_score += 1
    
    assert friend_score > 0, "No friend preferences satisfied"
    
    # Check gender diversity exists
    crew_genders: dict[tuple[str, str], dict[str, int]] = {}
    for youth in youth_list:
        if youth.role == "Youth":
            center, crew = assignments[youth.name]
            if (center, crew) not in crew_genders:
                crew_genders[(center, crew)] = {"M": 0, "F": 0}
            crew_genders[(center, crew)][youth.gender] += 1
    
    diverse_crews = sum(
        1 for counts in crew_genders.values() if counts["M"] > 0 and counts["F"] > 0
    )
    assert diverse_crews > 0, "No crews have gender diversity"
    
    # Check year diversity exists
    crew_years: dict[tuple[str, str], set[str]] = {}
    for youth in youth_list:
        if youth.role == "Youth":
            center, crew = assignments[youth.name]
            if (center, crew) not in crew_years:
                crew_years[(center, crew)] = set()
            crew_years[(center, crew)].add(youth.year)
    
    diverse_year_crews = sum(1 for years in crew_years.values() if len(years) > 1)
    assert diverse_year_crews > 0, "No crews have year diversity"
    
    # Check history diversity exists
    crew_history: dict[tuple[str, str], dict[str, int]] = {}
    for youth in youth_list:
        if youth.role == "Youth":
            center, crew = assignments[youth.name]
            if (center, crew) not in crew_history:
                crew_history[(center, crew)] = {"V": 0, "N": 0}
            crew_history[(center, crew)][youth.history] += 1
    
    diverse_history_crews = sum(
        1
        for counts in crew_history.values()
        if counts["V"] > 0 and counts["N"] > 0
    )
    assert diverse_history_crews > 0, "No crews have history diversity"

