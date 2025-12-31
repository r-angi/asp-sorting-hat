from ortools.sat.python import cp_model
from src.models import Youth, Center
from src.config import Config


def add_friend_preference_objectives(
    model: cp_model.CpModel,
    person_crew: dict,
    youth_list: list[Youth],
    centers: list[Center],
    cfg: Config,
    youth_dict: dict,
) -> list:
    """
    Rewards youth being in the same center as their friend choices.
    Young adults' friend preferences are considered but their assignments are fixed.

    Weights are:
    - First choice: 3 points
    - Second choice: 2 points
    - Third choice: 1 point

    Each point is multiplied by the friend_weight from config.
    """
    objective_terms = []
    for youth in youth_list:
        # Include friend preferences for both youth and young adults
        friend_choices = {
            youth.first_choice: 3,
            youth.second_choice: 2,
            youth.third_choice: 1,
        }
        for friend, weight in friend_choices.items():
            if friend is not None and friend in youth_dict:
                for center in centers:
                    # For young adults, their center assignment is fixed but still contributes to objective
                    if youth.role == 'Young Adult':
                        # Only add objective term if young adult is actually in this center
                        is_ya_in_center = any(youth.name in crew.adults for crew in center.crews)
                        if is_ya_in_center:
                            # Friend at center (computed from crew assignments)
                            friend_at_center = sum(
                                person_crew[friend, center.name, crew.name]
                                for crew in center.crews
                            )
                            objective_terms.append(cfg.friend_weight * weight * friend_at_center)
                    else:
                        # Reward when both youth and friend are in the same center
                        # Compute center assignment from crew assignments
                        youth_at_center = sum(
                            person_crew[youth.name, center.name, crew.name]
                            for crew in center.crews
                        )
                        friend_at_center = sum(
                            person_crew[friend, center.name, crew.name]
                            for crew in center.crews
                        )
                        
                        # Create boolean for both being at same center (logical AND)
                        same_center = model.NewBoolVar(f'same_center_{youth.name}_{friend}_{center.name}')
                        model.Add(same_center <= youth_at_center)
                        model.Add(same_center <= friend_at_center)
                        model.Add(same_center >= youth_at_center + friend_at_center - 1)
                        objective_terms.append(cfg.friend_weight * weight * same_center)

    return objective_terms


def add_gender_diversity_objectives(
    model: cp_model.CpModel,
    person_crew: dict,
    youth_list: list[Youth],
    centers: list[Center],
    cfg: Config,
) -> list:
    """
    Rewards crews that have a good balance of male and female youth.

    Creates a gender_balance variable for each crew that represents the
    minimum of males and females in that crew. This encourages having
    similar numbers of each gender.
    """
    objective_terms = []

    for center in centers:
        for crew in center.crews:
            females_in_crew = sum(
                person_crew[youth.name, center.name, crew.name] for youth in youth_list if youth.gender == 'F'
            )
            males_in_crew = sum(
                person_crew[youth.name, center.name, crew.name] for youth in youth_list if youth.gender == 'M'
            )

            gender_balance = model.NewIntVar(0, cfg.max_crew_size, f'gender_balance_{center.name}_{crew.name}')
            model.Add(gender_balance <= females_in_crew)
            model.Add(gender_balance <= males_in_crew)
            objective_terms.append(cfg.gender_weight * gender_balance)

    return objective_terms


def add_year_diversity_objectives(
    model: cp_model.CpModel,
    person_crew: dict,
    youth_list: list[Youth],
    centers: list[Center],
    cfg: Config,
) -> list:
    """
    Rewards crews that have youth from multiple grade levels.

    Adds a point for each year (Fr/So/Jr/Sr) that is represented in the crew,
    encouraging a mix of ages rather than grouping by grade level.
    
    Optimized: Creates one IntVar per crew that counts total years present,
    rather than 4 separate objective terms per crew.
    """
    objective_terms = []
    years = ['Fr', 'So', 'Jr', 'Sr']

    for center in centers:
        for crew in center.crews:
            # Track which years are present with booleans
            year_booleans = []
            
            for year in years:
                year_count = sum(
                    person_crew[youth.name, center.name, crew.name] 
                    for youth in youth_list if youth.year == year
                )
                # Boolean: is this year present in the crew?
                has_year = model.NewBoolVar(f'has_{year}_{center.name}_{crew.name}')
                model.Add(year_count >= 1).OnlyEnforceIf(has_year)
                model.Add(year_count == 0).OnlyEnforceIf(has_year.Not())
                year_booleans.append(has_year)
            
            # Create single IntVar that counts how many different years are present (0-4)
            years_present = model.NewIntVar(0, 4, f'years_present_{center.name}_{crew.name}')
            model.Add(years_present == sum(year_booleans))
            
            # Add single objective term per crew (instead of 4)
            objective_terms.append(cfg.year_weight * years_present)

    return objective_terms


def add_history_diversity_objectives(
    model: cp_model.CpModel,
    person_crew: dict,
    youth_list: list[Youth],
    centers: list[Center],
    cfg: Config,
) -> list:
    """
    Rewards crews that have a mix of veterans and new participants.

    Creates a history_balance variable for each crew that represents the
    minimum of veterans and new participants, encouraging a balanced mix
    of experience levels.
    """
    objective_terms = []

    for center in centers:
        for crew in center.crews:
            vets_in_crew = sum(
                person_crew[youth.name, center.name, crew.name] for youth in youth_list if youth.history == 'V'
            )
            new_in_crew = sum(
                person_crew[youth.name, center.name, crew.name] for youth in youth_list if youth.history == 'N'
            )

            history_balance = model.NewIntVar(0, cfg.max_crew_size, f'history_balance_{center.name}_{crew.name}')
            model.Add(history_balance <= vets_in_crew)
            model.Add(history_balance <= new_in_crew)
            objective_terms.append(cfg.history_weight * history_balance)

    return objective_terms
