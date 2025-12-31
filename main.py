import argparse

import polars as pl
from ortools.sat.python import cp_model

from src.analysis import calculate_friend_scores, print_crew_assignments, status_to_string
from src.cleaning import (
    all_friends_are_valid,
    all_parents_are_valid,
    get_centers_from_adults_df,
    get_historical_youth_leaders,
    get_youth_from_buddy_form_df,
)
from src.clustering import analyze_clusters
from src.config import CenterConfig, Config
from src.linear_program.lp_model import create_crew_assignment_model
from src.models import Center, Youth
from src.writer import write_results_to_csv


class MockSolver:
    """Mock solver that returns values directly, compatible with cp_model.CpSolver interface.

    In the optimization path, person_crew contains CP-SAT variables and solver.Value() extracts their values.
    In the mock path, person_crew contains integers directly, so Value() just returns them as-is.
    """

    def Value(self, var: int) -> int:
        """Return the value as-is since person_crew already contains integers, not variables."""
        return var


def load_existing_assignments(year: int, youth_list: list[Youth], centers: list[Center]) -> dict[tuple[str, str, str], int]:
    """Load existing assignments from CSV and create lookup dict compatible with solver interface."""
    assignments_df = pl.read_csv(f'./data/results/assignments_{year}_final.csv')

    # Create a lookup dict: (person_name, center_name, crew_name) -> 1 or 0
    person_crew = {}

    for center in centers:
        for crew in center.crews:
            for youth in youth_list:
                # Check if this person is assigned to this center/crew
                match = assignments_df.filter((pl.col('Name') == youth.name) & (pl.col('Center') == center.name) & (pl.col('Crew') == crew.name))
                person_crew[youth.name, center.name, crew.name] = 1 if len(match) > 0 else 0

    return person_crew


def analyze_existing_assignments(year: int, youth_list: list[Youth], centers: list[Center]) -> None:
    """Analyze existing assignments without re-running optimization."""
    print('\n' + '=' * 50)
    print('ANALYZING EXISTING ASSIGNMENTS (--no-reassignment)')
    print('=' * 50)

    # Load existing assignments from CSV
    person_crew = load_existing_assignments(year, youth_list, centers)
    solver = MockSolver()

    # Print assignments and calculate friend scores
    print_crew_assignments(solver, person_crew, youth_list, centers)

    # Always run cluster analysis when analyzing existing assignments
    regular_youth = [y for y in youth_list if y.role == 'Youth']
    analyze_clusters(regular_youth, solver, person_crew, centers, year=year, output_dir='./data/results')

    # Calculate and display friend scores
    center_scores, avg_score = calculate_friend_scores(solver, person_crew, youth_list, centers)
    print('=' * 50)
    print('Algorithm Friend Scores:')
    print(f'Center scores: {center_scores}')
    print(f'Average score: {avg_score}')


def run_optimization(
    year: int,
    youth_list: list[Youth],
    centers: list[Center],
    center_only_adults: list,
    unassigned_adults: list,
    analyze_clusters_flag: bool,
) -> None:
    """Run the optimization and optionally analyze clusters."""
    cfg = Config.default()

    # Create and solve model
    model, person_crew, adult_crew = create_crew_assignment_model(cfg, youth_list, centers, center_only_adults, unassigned_adults)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 300.0
    solver.parameters.num_search_workers = 8
    solver.parameters.log_search_progress = True
    solver.parameters.relative_gap_limit = 0.005  # Stop within 0.5% of optimal
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f'Solution found! Status: {status_to_string(status)}')
        print_crew_assignments(solver, person_crew, youth_list, centers)
        write_results_to_csv(
            solver,
            person_crew,
            youth_list,
            centers,
            year=year,
            adult_crew=adult_crew,
            unassigned_adults=unassigned_adults,
            center_only_adults=center_only_adults,
        )

        # Run cluster analysis if requested
        if analyze_clusters_flag:
            regular_youth = [y for y in youth_list if y.role == 'Youth']
            analyze_clusters(regular_youth, solver, person_crew, centers, year=year, output_dir='./data/results')
    else:
        print(f'No solution found. Status: {status_to_string(status)}')
        print('Statistics:')
        print(solver.ResponseStats())

    # Calculate and display friend scores
    center_scores, avg_score = calculate_friend_scores(solver, person_crew, youth_list, centers)
    print('=' * 50)
    print('Algorithm Friend Scores:')
    print(f'Center scores: {center_scores}')
    print(f'Average score: {avg_score}')


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run crew assignment optimization')
    parser.add_argument('-y', '--year', type=int, required=True, help='Year for the crew assignments')
    parser.add_argument(
        '--centers', nargs='*', help='Center specifications in format "CenterName:CrewCount" or "CenterName" (e.g., Fayette:11 Kanawha:12)'
    )
    parser.add_argument('--analyze-clusters', action='store_true', help='Run friend cluster analysis and generate visualization')
    parser.add_argument('--no-reassignment', action='store_true', help='Skip optimization and analyze existing assignments from CSV')
    args = parser.parse_args()

    year = args.year
    center_specs = args.centers
    analyze_clusters_flag = args.analyze_clusters
    no_reassignment = args.no_reassignment

    # Parse center configurations
    center_configs = None
    if center_specs:
        center_configs = [CenterConfig.parse(s) for s in center_specs]
        print(f'Center configurations: {[(c.name, c.crew_count) for c in center_configs]}')

    # Read and process data
    adult_crew_df = pl.read_csv(f'./data/clean/crews_{year}.csv').filter(pl.col('role') != 'Youth')
    youth_df = pl.read_csv(f'./data/clean/buddies_{year}.csv')
    historical_pairings_df = pl.read_csv('./data/clean/historical_crews.csv')
    youth_list = get_youth_from_buddy_form_df(youth_df)
    centers, center_only_adults, unassigned_adults = get_centers_from_adults_df(adult_crew_df, center_configs)

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
    print(f'Youth with siblings: {len([y for y in youth_list if y.siblings_list])}')
    print(f'Centers: {[(c.name, len(c.crews)) for c in centers]}')
    if center_only_adults:
        print(f'Center-only adults (algorithm assigns crew): {len(center_only_adults)}')
    if unassigned_adults:
        print(f'Unassigned adults (algorithm assigns center & crew): {len(unassigned_adults)}')

    # Route to appropriate workflow
    if no_reassignment:
        analyze_existing_assignments(year, youth_list, centers)
    else:
        run_optimization(year, youth_list, centers, center_only_adults, unassigned_adults, analyze_clusters_flag)


if __name__ == '__main__':
    main()
