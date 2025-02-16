from ortools.sat.python import cp_model
from src.models import Youth, Center


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


def link_crew_and_center_vars(
    model: cp_model.CpModel, person_crew: dict, person_center: dict, youth_list: list[Youth], centers: list[Center]
):
    """
    Links the crew and center assignment variables.

    If a youth is assigned to a crew in a center, they must be marked as assigned to that center.
    This maintains consistency between crew and center assignments and simplifies other constraints.
    """
    for youth in youth_list:
        for center in centers:
            model.Add(
                person_center[youth.name, center.name]
                == sum(person_crew[youth.name, center.name, crew.name] for crew in center.crews)
            )


def enforce_parent_center_constraint(
    model: cp_model.CpModel,
    person_crew: dict,
    person_center: dict,
    youth_list: list[Youth],
    centers: list[Center],
):
    """
    Ensures youth are assigned to the same center as their parent, but not the same crew.

    This constraint:
    1. Forces youth to be in the same center as their parent
    2. Prevents youth from being in their parent's crew
    3. Raises an error if a parent is not found in any center
    """
    adult_names = [adult for center in centers for crew in center.crews for adult in crew.adults]
    for youth in youth_list:
        if youth.parent_name:
            if youth.parent_name not in adult_names:
                raise ValueError(f'Parent {youth.parent_name} not found in any center for {youth.name}')
            for center in centers:
                parent_in_center = any(youth.parent_name in crew.adults for crew in center.crews)
                if parent_in_center:
                    model.Add(person_center[youth.name, center.name] == 1)
                    # Prevent youth from being in parent's crew
                    for crew in center.crews:
                        if youth.parent_name in crew.adults:
                            model.Add(person_crew[youth.name, center.name, crew.name] == 0)
                else:
                    model.Add(person_center[youth.name, center.name] == 0)


def enforce_sibling_center_constraint(
    model: cp_model.CpModel, person_center: dict, youth_list: list[Youth], centers: list[Center]
):
    """
    Ensures siblings are assigned to the same center.

    This keeps families together at the same worksite while still allowing
    siblings to be in different crews within that center.
    """
    youth_dict = {youth.name: youth for youth in youth_list}
    for youth in youth_list:
        for sibling in youth.siblings_list:
            if sibling in youth_dict:
                for center in centers:
                    model.Add(person_center[youth.name, center.name] == person_center[sibling, center.name])


def enforce_friend_separation_constraint(
    model: cp_model.CpModel, person_crew: dict, youth_list: list[Youth], centers: list[Center]
):
    """
    Prevents friends from being assigned to the same crew.

    This encourages youth to meet new people and prevents cliques from forming.
    It applies to all friend choices (first, second, and third choices).
    """
    youth_dict = {youth.name: youth for youth in youth_list}
    for youth in youth_list:
        choices = [youth.first_choice, youth.second_choice, youth.third_choice]
        choices = [c for c in choices if c is not None]
        for friend in choices:
            if friend in youth_dict:
                for center in centers:
                    for crew in center.crews:
                        model.Add(
                            person_crew[youth.name, center.name, crew.name]
                            + person_crew[friend, center.name, crew.name]
                            <= 1
                        )


def enforce_friend_center_constraint(
    model: cp_model.CpModel, person_center: dict, youth_list: list[Youth], centers: list[Center]
):
    """
    Ensures youth are assigned to centers with at least one of their friend choices.

    This balances the friend separation constraint by guaranteeing that while
    friends can't be in the same crew, they will at least be at the same worksite
    and can interact during non-work times.
    """
    youth_dict = {youth.name: youth for youth in youth_list}
    for youth in youth_list:
        choices = [youth.first_choice, youth.second_choice, youth.third_choice]
        valid_choices = [c for c in choices if c is not None and c in youth_dict]
        if valid_choices:
            for center in centers:
                friend_vars = [person_center[friend, center.name] for friend in valid_choices]
                model.Add(person_center[youth.name, center.name] <= sum(friend_vars))


def enforce_crew_size_constraints(
    model: cp_model.CpModel,
    person_crew: dict,
    person_center: dict,
    youth_list: list[Youth],
    centers: list[Center],
    config,
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
            # Count regular youth in crew (excluding young adults)
            youth_in_crew = sum(
                person_crew[youth.name, center.name, crew.name] for youth in youth_list if youth.role == 'Youth'
            )

            # Count all adults (including young adults) in crew
            current_adult_count = len(crew.adults)

            # Size constraints including adults
            model.Add(youth_in_crew + current_adult_count >= config.min_crew_size)
            model.Add(youth_in_crew + current_adult_count <= config.max_crew_size)

            # Ensure crew assignments are consistent with center assignments
            for youth in youth_list:
                if youth.role == 'Youth':
                    model.Add(person_crew[youth.name, center.name, crew.name] <= person_center[youth.name, center.name])


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
