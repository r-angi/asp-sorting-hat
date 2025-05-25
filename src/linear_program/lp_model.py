from ortools.sat.python import cp_model
from src.models import Center, Youth
from src.config import Config
from src.linear_program.constraints import (
    add_one_crew_per_youth,
    link_crew_and_center_vars,
    enforce_parent_center_constraint,
    enforce_sibling_center_constraint,
    enforce_sibling_crew_separation_constraint,
    enforce_friend_separation_constraint,
    enforce_friend_center_constraint,
    enforce_crew_size_constraints,
    enforce_past_leader_constraint,
)
from src.linear_program.objectives import (
    add_friend_preference_objectives,
    add_gender_diversity_objectives,
    add_year_diversity_objectives,
    add_history_diversity_objectives,
)


def create_crew_assignment_model(
    cfg: Config, youth_list: list[Youth], centers: list[Center]
) -> tuple[cp_model.CpModel, dict, dict]:
    print(f'Youth count: {len(youth_list)}')
    print(f'Centers: {[c.name for c in centers]}')
    print(f'Total crews: {sum(len(c.crews) for c in centers)}')

    model = cp_model.CpModel()

    # Create variables
    # person_center[i, c] = 1 if person i is assigned to center c
    person_center = {
        (youth.name, center.name): model.NewBoolVar(f'person_{youth.name}_center_{center.name}')
        for youth in youth_list
        for center in centers
    }

    # Create crew variables for each center
    # person_crew[i, c, k] = 1 if person i is assigned to crew k in center c
    person_crew = {}
    for center in centers:
        for crew in center.crews:
            for youth in youth_list:
                person_crew[(youth.name, center.name, crew.name)] = model.NewBoolVar(
                    f'person_{youth.name}_center_{center.name}_crew_{crew.name}'
                )

    # Pre-compute youth dictionary and filter by role for efficiency
    youth_dict = {youth.name: youth for youth in youth_list}
    regular_youth = [youth for youth in youth_list if youth.role == 'Youth']

    # Add constraints
    add_one_crew_per_youth(model, person_crew, youth_list, centers)
    link_crew_and_center_vars(model, person_crew, person_center, youth_list, centers)
    enforce_parent_center_constraint(model, person_crew, person_center, youth_list, centers)
    enforce_sibling_center_constraint(model, person_center, youth_list, centers, youth_dict)
    enforce_sibling_crew_separation_constraint(model, person_crew, youth_list, centers, youth_dict)
    enforce_friend_separation_constraint(model, person_crew, youth_list, centers, youth_dict)
    enforce_friend_center_constraint(model, person_center, youth_list, centers, youth_dict)
    enforce_crew_size_constraints(model, person_crew, regular_youth, centers, cfg)
    enforce_past_leader_constraint(model, person_crew, youth_list, centers)

    # Combine all objective terms
    objective_terms = []
    objective_terms.extend(add_friend_preference_objectives(model, person_center, youth_list, centers, cfg, youth_dict))
    objective_terms.extend(add_gender_diversity_objectives(model, person_crew, regular_youth, centers, cfg))
    objective_terms.extend(add_year_diversity_objectives(model, person_crew, regular_youth, centers, cfg))
    objective_terms.extend(add_history_diversity_objectives(model, person_crew, regular_youth, centers, cfg))

    model.Maximize(sum(objective_terms))

    return model, person_center, person_crew
