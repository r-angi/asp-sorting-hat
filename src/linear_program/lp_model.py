from ortools.sat.python import cp_model
from src.models import Center, Youth
from src.config import Config
from src.linear_program.constraints import (
    add_one_crew_per_youth,
    enforce_parent_center_constraint,
    enforce_sibling_center_constraint,
    enforce_sibling_crew_separation_constraint,
    enforce_friend_separation_constraint,
    enforce_friend_center_constraint,
    enforce_crew_size_constraints,
    enforce_past_leader_constraint,
    enforce_supervision_group_limit,
    enforce_anti_buddy_constraint,
    assign_center_only_adults,
    assign_unassigned_adults,
    enforce_adult_count_constraints,
)
from src.linear_program.objectives import (
    add_friend_preference_objectives,
    add_gender_diversity_objectives,
    add_year_diversity_objectives,
    add_history_diversity_objectives,
)


def create_crew_assignment_model(
    cfg: Config,
    youth_list: list[Youth],
    centers: list[Center],
    center_only_adults: list[tuple[str, str]] | None = None,
    unassigned_adults: list[str] | None = None,
) -> tuple[cp_model.CpModel, dict, dict]:
    print(f'Youth count: {len(youth_list)}')
    print(f'Centers: {[c.name for c in centers]}')
    print(f'Total crews: {sum(len(c.crews) for c in centers)}')
    if center_only_adults:
        print(f'Center-only adults: {len(center_only_adults)}')
    if unassigned_adults:
        print(f'Unassigned adults: {len(unassigned_adults)}')

    model = cp_model.CpModel()

    # Create crew variables for each center
    # person_crew[i, c, k] = 1 if person i is assigned to crew k in center c
    # Note: We don't create separate person_center variables as they're redundant.
    # A person is at a center if they're in any crew at that center.
    person_crew = {}
    for center in centers:
        for crew in center.crews:
            for youth in youth_list:
                person_crew[(youth.name, center.name, crew.name)] = model.NewBoolVar(
                    f'person_{youth.name}_center_{center.name}_crew_{crew.name}'
                )

    # Create variables for center-only adults if any
    adult_crew = {}
    if center_only_adults:
        for adult_name, center_name in center_only_adults:
            center = next(c for c in centers if c.name == center_name)
            for crew in center.crews:
                adult_crew[(adult_name, center_name, crew.name)] = model.NewBoolVar(
                    f'adult_{adult_name}_center_{center_name}_crew_{crew.name}'
                )
    
    # Create variables for unassigned adults (can go to any center/crew)
    if unassigned_adults:
        for adult_name in unassigned_adults:
            for center in centers:
                for crew in center.crews:
                    adult_crew[(adult_name, center.name, crew.name)] = model.NewBoolVar(
                        f'adult_{adult_name}_center_{center.name}_crew_{crew.name}'
                    )

    # Pre-compute youth dictionary and filter by role for efficiency
    youth_dict = {youth.name: youth for youth in youth_list}
    regular_youth = [youth for youth in youth_list if youth.role == 'Youth']

    # Add constraints
    add_one_crew_per_youth(model, person_crew, youth_list, centers)
    enforce_parent_center_constraint(model, person_crew, youth_list, centers)
    enforce_sibling_center_constraint(model, person_crew, youth_list, centers, youth_dict)
    enforce_sibling_crew_separation_constraint(model, person_crew, youth_list, centers, youth_dict)
    enforce_friend_separation_constraint(model, person_crew, youth_list, centers, youth_dict)
    enforce_friend_center_constraint(model, person_crew, youth_list, centers, youth_dict)
    enforce_crew_size_constraints(model, person_crew, regular_youth, centers, cfg)
    enforce_past_leader_constraint(model, person_crew, youth_list, centers)
    enforce_supervision_group_limit(model, person_crew, youth_list, centers)
    enforce_anti_buddy_constraint(model, person_crew, youth_list, centers, youth_dict)
    
    # Handle center-only adults
    if center_only_adults:
        assign_center_only_adults(model, adult_crew, center_only_adults, centers)
    
    # Handle unassigned adults
    if unassigned_adults:
        assign_unassigned_adults(model, adult_crew, unassigned_adults, centers)
    
    # Enforce adult count constraints (always, even if no flexible adults)
    enforce_adult_count_constraints(
        model, 
        adult_crew, 
        center_only_adults or [], 
        unassigned_adults or [], 
        centers, 
        cfg
    )

    # Combine all objective terms
    objective_terms = []
    objective_terms.extend(add_friend_preference_objectives(model, person_crew, youth_list, centers, cfg, youth_dict))
    objective_terms.extend(add_gender_diversity_objectives(model, person_crew, regular_youth, centers, cfg))
    objective_terms.extend(add_year_diversity_objectives(model, person_crew, regular_youth, centers, cfg))
    objective_terms.extend(add_history_diversity_objectives(model, person_crew, regular_youth, centers, cfg))

    model.Maximize(sum(objective_terms))

    return model, person_crew, adult_crew
