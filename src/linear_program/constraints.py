from ortools.sat.python import cp_model
from src.models import Youth, Center
from typing import cast
from src.config import Config


def add_one_crew_per_youth(model: cp_model.CpModel, person_crew: dict, youth_list: list[Youth], centers: list[Center]):
    """
    Ensures each youth is assigned to exactly one crew.

    This is a fundamental constraint that prevents a youth from being:
    - Unassigned (must be in at least one crew)
    - Double-assigned (cannot be in multiple crews)
    """
    for youth in youth_list:
        if youth.role == 'Youth':
            model.Add(
                sum(person_crew[youth.name, center.name, crew.name] for center in centers for crew in center.crews) == 1
            )
        else:  # Young Adult
            for center in centers:
                for crew in center.crews:
                    # If young adult is in this crew's adults list, force assignment
                    if youth.name in crew.adults:
                        model.Add(person_crew[youth.name, center.name, crew.name] == 1)
                    else:
                        model.Add(person_crew[youth.name, center.name, crew.name] == 0)


def enforce_parent_center_constraint(
    model: cp_model.CpModel,
    person_crew: dict,
    youth_list: list[Youth],
    centers: list[Center],
):
    """
    Ensures youth are assigned to the same center as their parent(s), but not the same crew.

    This constraint:
    1. Forces youth to be in the same center as their parent(s)
    2. Prevents youth from being in their parent's crew
    3. Raises an error if a parent is not found in any center
    4. Assumes all parents of the same child are at the same center (guaranteed by data)
    """
    # Pre-compute adult names and their center mappings for efficiency
    adult_names = set()
    adult_to_center = {}
    for center in centers:
        for crew in center.crews:
            for adult in crew.adults:
                adult_names.add(adult)
                adult_to_center[adult] = center

    for youth in youth_list:
        if youth.parent_names_list:
            # Check that all parents exist in adult crews
            for parent_name in youth.parent_names_list:
                if parent_name not in adult_names:
                    raise ValueError(f'Parent {parent_name} not found in any center for {youth.name}')

            # Find the center where parents are located (use pre-computed mapping)
            parent_center = adult_to_center.get(youth.parent_names_list[0])

            if parent_center:
                # Youth must be assigned to the parent's center (computed from crew assignments)
                youth_at_parent_center = sum(
                    person_crew[youth.name, parent_center.name, crew.name] 
                    for crew in parent_center.crews
                )
                model.Add(youth_at_parent_center == 1)

                # Prevent youth from being in same crew as any parent
                for crew in parent_center.crews:
                    for parent_name in youth.parent_names_list:
                        if parent_name in crew.adults:
                            model.Add(person_crew[youth.name, parent_center.name, crew.name] == 0)


def enforce_sibling_center_constraint(
    model: cp_model.CpModel, person_crew: dict, youth_list: list[Youth], centers: list[Center], youth_dict: dict
):
    """
    Ensures siblings are assigned to the same center.

    This keeps families together at the same worksite while still allowing
    siblings to be in different crews within that center.
    """
    for youth in youth_list:
        for sibling in youth.siblings_list:
            if sibling in youth_dict:
                for center in centers:
                    # Sum of crew assignments equals 1 if person is at center, 0 otherwise
                    youth_at_center = sum(
                        person_crew[youth.name, center.name, crew.name] 
                        for crew in center.crews
                    )
                    sibling_at_center = sum(
                        person_crew[sibling, center.name, crew.name] 
                        for crew in center.crews
                    )
                    model.Add(youth_at_center == sibling_at_center)


def enforce_sibling_crew_separation_constraint(
    model: cp_model.CpModel, person_crew: dict, youth_list: list[Youth], centers: list[Center], youth_dict: dict
):
    """
    Prevents siblings from being assigned to the same crew.

    This ensures that while siblings are at the same center (enforced by
    enforce_sibling_center_constraint), they are placed in different crews.

    Optimized to avoid duplicate constraints by only processing each sibling pair once.
    """
    processed_pairs = set()

    for youth in youth_list:
        for sibling in youth.siblings_list:
            if sibling in youth_dict:
                # Create a canonical pair representation to avoid duplicates
                pair = tuple(sorted([youth.name, sibling]))
                if pair not in processed_pairs:
                    processed_pairs.add(pair)

                    for center in centers:
                        for crew in center.crews:
                            model.Add(
                                person_crew[youth.name, center.name, crew.name]
                                + person_crew[sibling, center.name, crew.name]
                                <= 1
                            )


def enforce_friend_separation_constraint(
    model: cp_model.CpModel, person_crew: dict, youth_list: list[Youth], centers: list[Center], youth_dict: dict
):
    """
    Prevents friends from being assigned to the same crew.

    This encourages youth to meet new people and prevents cliques from forming.
    It applies to all friend choices (first, second, and third choices).

    Optimized to avoid duplicate constraints by only processing each friend pair once.
    """
    processed_pairs = set()

    for youth in youth_list:
        choices = [youth.first_choice, youth.second_choice, youth.third_choice]
        choices = [c for c in choices if c is not None]
        for friend in choices:
            if friend in youth_dict:
                # Create a canonical pair representation to avoid duplicates
                pair = tuple(sorted([youth.name, cast(str, friend)]))
                if pair not in processed_pairs:
                    processed_pairs.add(pair)

                    for center in centers:
                        for crew in center.crews:
                            model.Add(
                                person_crew[youth.name, center.name, crew.name]
                                + person_crew[friend, center.name, crew.name]
                                <= 1
                            )


def enforce_friend_center_constraint(
    model: cp_model.CpModel, person_crew: dict, youth_list: list[Youth], centers: list[Center], youth_dict: dict
):
    """
    Ensures youth are assigned to centers with at least one of their friend choices.

    This balances the friend separation constraint by guaranteeing that while
    friends can't be in the same crew, they will at least be at the same worksite
    and can interact during non-work times.
    """
    for youth in youth_list:
        choices = [youth.first_choice, youth.second_choice, youth.third_choice]
        valid_choices = [c for c in choices if c is not None and c in youth_dict]
        if valid_choices:
            for center in centers:
                # Youth at center (computed from crew assignments)
                youth_at_center = sum(
                    person_crew[youth.name, center.name, crew.name]
                    for crew in center.crews
                )
                # Friends at center (computed from crew assignments)
                friends_at_center = sum(
                    person_crew[friend, center.name, crew.name]
                    for friend in valid_choices
                    for crew in center.crews
                )
                model.Add(youth_at_center <= friends_at_center)


def enforce_crew_size_constraints(
    model: cp_model.CpModel,
    person_crew: dict,
    youth_list: list[Youth],
    centers: list[Center],
    config: Config,
):
    """
    Enforces minimum and maximum crew size constraints.

    This ensures:
    1. Each crew has enough people to be effective (min_crew_size)
    2. No crew is too large to manage (max_crew_size)
    3. Counts both youth and existing adults in the size calculations
    4. Links crew assignments to center assignments for consistency
    """
    for center in centers:
        for crew in center.crews:
            # Count regular youth in crew (youth_list is already filtered to regular youth)
            youth_in_crew = sum(person_crew[youth.name, center.name, crew.name] for youth in youth_list)

            # Count all adults (including young adults) in crew
            current_adult_count = len(crew.adults)

            # Size constraints including adults
            model.Add(youth_in_crew + current_adult_count >= config.min_crew_size)
            model.Add(youth_in_crew + current_adult_count <= config.max_crew_size)


def enforce_past_leader_constraint(
    model: cp_model.CpModel,
    person_crew: dict,
    youth_list: list[Youth],
    centers: list[Center],
):
    """
    Prevents youth from being assigned to crews led by their past leaders.

    This constraint ensures youth don't repeat experiences with the same adult leaders,
    encouraging them to work with different adults each year.
    """
    for youth in youth_list:
        if youth.past_leaders:  # Only apply if youth has past leaders
            for center in centers:
                for crew in center.crews:
                    # Check if any of youth's past leaders are in this crew
                    if any(leader in crew.adults for leader in youth.past_leaders):
                        model.Add(person_crew[youth.name, center.name, crew.name] == 0)


def enforce_supervision_group_limit(
    model: cp_model.CpModel,
    person_crew: dict,
    youth_list: list[Youth],
    centers: list[Center],
    max_per_center: int = 2,
):
    """Limit members of each supervision group to max_per_center per center.
    
    Each supervision group (A, B, C, etc.) is constrained independently.
    Example: With max_per_center=2 and groups A, B:
      - Center Fayette can have at most 2 from group A AND at most 2 from group B
    """
    # Group youth by supervision_group
    groups: dict[str, list[Youth]] = {}
    for youth in youth_list:
        if youth.supervision_group:
            groups.setdefault(youth.supervision_group, []).append(youth)
    
    # Add constraint for each group/center combination
    for group_name, group_youth in groups.items():
        for center in centers:
            # Count group members at this center (computed from crew assignments)
            group_at_center = sum(
                person_crew[y.name, center.name, crew.name]
                for y in group_youth
                for crew in center.crews
            )
            model.Add(group_at_center <= max_per_center)


def enforce_anti_buddy_constraint(
    model: cp_model.CpModel,
    person_crew: dict,
    youth_list: list[Youth],
    centers: list[Center],
    youth_dict: dict[str, Youth],
):
    """Prevent anti-buddies from being at the same center."""
    processed_pairs: set[tuple[str, str]] = set()
    
    for youth in youth_list:
        for anti_buddy in youth.anti_buddy_list:
            if anti_buddy in youth_dict:
                pair = tuple(sorted([youth.name, anti_buddy]))
                if pair not in processed_pairs:
                    processed_pairs.add(pair)
                    for center in centers:
                        # Youth at center (computed from crew assignments)
                        youth_at_center = sum(
                            person_crew[youth.name, center.name, crew.name]
                            for crew in center.crews
                        )
                        anti_buddy_at_center = sum(
                            person_crew[anti_buddy, center.name, crew.name]
                            for crew in center.crews
                        )
                        model.Add(youth_at_center + anti_buddy_at_center <= 1)


def assign_center_only_adults(
    model: cp_model.CpModel,
    adult_crew: dict,
    center_only_adults: list[tuple[str, str]],  # (name, center)
    centers: list[Center],
):
    """Assign center-only adults to exactly one crew within their center.
    
    These are adults who are pre-assigned to a center but not a specific crew.
    The algorithm will assign them to exactly one crew in their center.
    """
    for adult_name, center_name in center_only_adults:
        center = next(c for c in centers if c.name == center_name)
        # Must be assigned to exactly one crew in their center
        model.Add(
            sum(adult_crew[adult_name, center_name, crew.name] 
                for crew in center.crews) == 1
        )


def assign_unassigned_adults(
    model: cp_model.CpModel,
    adult_crew: dict,
    unassigned_adults: list[str],
    centers: list[Center],
):
    """Assign unassigned adults to exactly one crew across all centers.
    
    These are adults with no center or crew assignment.
    The algorithm will assign them to any crew in any center.
    """
    for adult_name in unassigned_adults:
        # Must be assigned to exactly one crew across all centers
        model.Add(
            sum(adult_crew[adult_name, center.name, crew.name]
                for center in centers
                for crew in center.crews) == 1
        )


def enforce_adult_count_constraints(
    model: cp_model.CpModel,
    adult_crew: dict,
    center_only_adults: list[tuple[str, str]],
    unassigned_adults: list[str],
    centers: list[Center],
    config: Config,
):
    """Enforce minimum and maximum adult count per crew.
    
    Ensures each crew has adequate supervision (min) without too many leaders (max).
    Counts pre-assigned adults, center-only adults, and unassigned adults.
    """
    for center in centers:
        for crew in center.crews:
            # Count pre-assigned adults in this crew
            pre_assigned_count = len(crew.adults)
            
            # Count center-only adults that could be assigned to this crew
            center_only_in_crew = sum(
                adult_crew[adult_name, center.name, crew.name]
                for adult_name, center_name in center_only_adults
                if center_name == center.name
            )
            
            # Count unassigned adults that could be assigned to this crew
            unassigned_in_crew = sum(
                adult_crew[adult_name, center.name, crew.name]
                for adult_name in unassigned_adults
            )
            
            total_adults = pre_assigned_count + center_only_in_crew + unassigned_in_crew
            
            # Enforce min/max adult constraints
            model.Add(total_adults >= config.min_adults_per_crew)
            model.Add(total_adults <= config.max_adults_per_crew)
