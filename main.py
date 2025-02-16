import polars as pl
from ortools.sat.python import cp_model
from src.analysis import calculate_friend_scores, calculate_historical_friend_scores
from src.cleaning import (
    get_centers_from_adults_df,
    get_youth_from_buddy_form_df,
    all_parents_are_valid,
    all_friends_are_valid,
    get_historical_youth_leaders,
)
from src.config import Config
from src.writer import write_results_to_csv
from src.analysis import print_crew_assignments, status_to_string
from src.linear_program.lp_model import create_crew_assignment_model


def main():
    # Read and process data
    adult_crew_df = pl.read_csv('./data/clean/crews_2024.csv').filter(pl.col('role') != 'Youth')
    youth_df = pl.read_csv('./data/clean/buddies_2024.csv')
    historical_pairings_df = pl.read_csv('./data/clean/historical_crews_2023.csv')
    youth_list = get_youth_from_buddy_form_df(youth_df)
    centers = get_centers_from_adults_df(adult_crew_df)

    # Update youth list with past leaders
    historical_youth_leaders = get_historical_youth_leaders(historical_pairings_df)
    for youth in youth_list:
        if youth.name in historical_youth_leaders:
            youth.past_leaders = historical_youth_leaders[youth.name]

    all_parents_are_valid(youth_df, adult_crew_df)
    all_friends_are_valid(youth_list)

    # Print initial data stats
    print('\nInitial Data:')
    print(f'Total youth: {len(youth_list)}')
    print(f'Youth with parents: {len([y for y in youth_list if y.parent_name])}')
    print(f'Youth with siblings: {len([y for y in youth_list if y.siblings])}')
    print(f'Centers: {[(c.name, len(c.crews)) for c in centers]}')

    cfg = Config.default()

    # Create and solve model
    model, person_center, person_crew = create_crew_assignment_model(cfg, youth_list, centers)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f'Solution found! Status: {status_to_string(status)}')
        print_crew_assignments(solver, person_crew, person_center, youth_list, centers)
        write_results_to_csv(solver, person_crew, youth_list, centers, year=2024)
    else:
        print(f'No solution found. Status: {status_to_string(status)}')
        # Print some stats about the failed solve
        print('Statistics:')
        print(solver.ResponseStats())

    center_scores, avg_score = calculate_friend_scores(solver, person_center, youth_list, centers)
    print('=' * 50)
    print('Algorithm Friend Scores:')
    print(f'Center scores: {center_scores}')
    print(f'Average score: {avg_score}')


if __name__ == '__main__':
    main()
