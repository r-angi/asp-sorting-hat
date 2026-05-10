"""Crew assignment optimizer entry point.

Two paths:

1. ``python main.py -y 2026`` — build the CP-SAT model and solve.
2. ``python main.py -y 2026 --no-reassignment`` — skip the solver and re-score
   an existing ``data/results/assignments_<year>_final.csv`` for analysis.

The re-analysis path uses a plain ``dict[(name, center, crew), int]`` rather
than a ``CpSolver``; downstream analysis / writer / clustering helpers accept
either via a thin :class:`AssignmentsLookup` wrapper.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import polars as pl
from ortools.sat.python import cp_model

from src.analysis import (
    PersonCrew,
    SolverLike,
    calculate_friend_scores,
    print_crew_assignments,
    status_to_string,
    synthesize_centers_from_assignments,
)
from src.clustering import analyze_clusters
from src.config import CenterConfig, Config
from src.data_loaders import (
    all_friends_are_valid,
    all_parents_are_valid,
    get_centers_from_adults_df,
    get_historical_youth_leaders,
    get_youth_from_buddy_form_df,
)
from src.linear_program.lp_model import create_crew_assignment_model
from src.models import Center, Leader, Youth
from src.writer import write_results_to_csv

CLEAN_DATA_DIR: Path = Path('./data/clean')
RESULTS_DIR: Path = Path('./data/results')


class AssignmentsLookup:
    """Adapter that exposes ``Value(var)`` over a precomputed assignments dict.

    ``person_crew`` already holds raw 0/1 ints for the re-analysis path, so
    ``Value(var)`` returns ``var`` unchanged (mirroring ``cp_model.CpSolver.Value``
    for the optimization path where ``var`` is an ``IntVar``).
    """

    def Value(self, var: object) -> int:
        if not isinstance(var, int):
            raise TypeError(
                'AssignmentsLookup.Value only accepts ints from the precomputed assignments dict'
            )
        return var


def load_existing_assignments(
    year: int,
    youth_list: list[Youth],
    centers: list[Center],
) -> dict[tuple[str, str, str], int]:
    """Load `assignments_{year}_final.csv` into a sparse `(name, center, crew) -> 1` dict.

    The finalized CSV is the source of truth for youth placements. When the
    crews scaffold (``centers``) is non-empty its ``(Center, Crew)`` pairs are
    used as a typo guard against the modeled topology; an empty scaffold (e.g.
    all leaders are ``CENTER_ONLY`` so :func:`get_centers_from_adults_df`
    returns ``[]``) skips that gate so the finalized roster still loads.
    """
    final_path = RESULTS_DIR / f'assignments_{year}_final.csv'
    if not final_path.is_file():
        raise FileNotFoundError(
            f'Required finalized assignments file not found: {final_path}. '
            'Run a full solve first or copy a finalized roster to that path.'
        )
    assignments_df = pl.read_csv(final_path).with_columns(
        pl.col('Center').cast(pl.Utf8, strict=False),
        pl.col('Crew').cast(pl.Utf8, strict=False),
    )
    youth_names = {y.name for y in youth_list}
    valid_pairs = {
        (center.name, crew.name) for center in centers for crew in center.crews
    }

    assigned: dict[tuple[str, str, str], int] = {}
    for row in assignments_df.iter_rows(named=True):
        name, center, crew = row['Name'], row['Center'], row['Crew']
        if name not in youth_names or not center or not crew:
            continue
        if valid_pairs and (center, crew) not in valid_pairs:
            continue
        assigned[(name, center, crew)] = 1
    return assigned


def _print_friend_scores(
    solver: SolverLike,
    person_crew: PersonCrew,
    youth_list: list[Youth],
    centers: list[Center],
) -> None:
    center_scores, avg_score = calculate_friend_scores(solver, person_crew, youth_list, centers)
    print('=' * 50)
    print('Algorithm Friend Scores:')
    print(f'Center scores: {center_scores}')
    print(f'Average score: {avg_score}')


def analyze_existing_assignments(year: int, youth_list: list[Youth], centers: list[Center]) -> None:
    """Re-score existing assignments without re-solving."""
    print('\n' + '=' * 50)
    print('ANALYZING EXISTING ASSIGNMENTS (--no-reassignment)')
    print('=' * 50)

    person_crew = load_existing_assignments(year, youth_list, centers)
    solver = AssignmentsLookup()

    effective_centers = centers if centers else synthesize_centers_from_assignments(person_crew)
    if not centers:
        synthesized = [(c.name, [crew.name for crew in c.crews]) for c in effective_centers]
        print(f'Centers (synthesized from finalized assignments): {synthesized}')

    print_crew_assignments(solver, person_crew, youth_list, effective_centers)

    regular_youth = [y for y in youth_list if y.role == 'Youth']
    analyze_clusters(regular_youth, solver, person_crew, effective_centers, year=year, output_dir=str(RESULTS_DIR))

    _print_friend_scores(solver, person_crew, youth_list, effective_centers)


def _configure_solver(cfg: Config) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = cfg.solver_max_time_seconds
    solver.parameters.num_search_workers = cfg.solver_num_workers
    solver.parameters.log_search_progress = cfg.solver_log_progress
    solver.parameters.relative_gap_limit = cfg.solver_relative_gap_limit
    return solver


def run_optimization(
    year: int,
    youth_list: list[Youth],
    centers: list[Center],
    center_only_adults: Sequence[Leader],
    unassigned_adults: Sequence[Leader],
    analyze_clusters_flag: bool,
    cfg: Config | None = None,
) -> None:
    """Run the optimization and optionally analyze clusters."""
    cfg = cfg or Config.default()

    model, person_crew, adult_crew = create_crew_assignment_model(
        cfg, youth_list, centers, center_only_adults, unassigned_adults,
    )

    solver = _configure_solver(cfg)
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f'No solution found. Status: {status_to_string(status)}')
        print('Statistics:')
        print(solver.ResponseStats())
        return

    print(f'Solution found! Status: {status_to_string(status)}')
    print_crew_assignments(
        solver,
        person_crew,
        youth_list,
        centers,
        adult_crew=adult_crew,
        center_only_adults=center_only_adults,
        unassigned_adults=unassigned_adults,
    )
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

    if analyze_clusters_flag:
        regular_youth = [y for y in youth_list if y.role == 'Youth']
        analyze_clusters(regular_youth, solver, person_crew, centers, year=year, output_dir=str(RESULTS_DIR))

    _print_friend_scores(solver, person_crew, youth_list, centers)


def main() -> None:
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

    center_configs = None
    if center_specs:
        center_configs = [CenterConfig.parse(s) for s in center_specs]
        print(f'Center configurations: {[(c.name, c.crew_count) for c in center_configs]}')

    adult_crew_df = pl.read_csv(CLEAN_DATA_DIR / f'crews_{year}.csv').filter(pl.col('role') != 'Youth')
    youth_df = pl.read_csv(CLEAN_DATA_DIR / f'buddies_{year}.csv')
    youth_list = get_youth_from_buddy_form_df(youth_df)
    centers, center_only_adults, unassigned_adults = get_centers_from_adults_df(adult_crew_df, center_configs)

    historical_path = CLEAN_DATA_DIR / 'historical_crews.csv'
    if historical_path.is_file():
        historical_youth_leaders = get_historical_youth_leaders(pl.read_csv(historical_path))
        for youth in youth_list:
            if youth.name in historical_youth_leaders:
                youth.past_leaders = historical_youth_leaders[youth.name]
    else:
        print(f'No historical crews file at {historical_path}; skipping past-leader exclusions.')

    all_parents_are_valid(youth_df, adult_crew_df)
    leader_names_raw = adult_crew_df['name'].drop_nulls().to_list()
    leader_names_unique = sorted({str(x) for x in leader_names_raw})
    all_friends_are_valid(youth_list, leader_names_unique)

    youth_name_set = {y.name for y in youth_list}
    leader_set = set(leader_names_unique)
    adult_only_buddy_picks = sum(
        1
        for y in youth_list
        for choice in (y.first_choice, y.second_choice, y.third_choice)
        if choice and choice in leader_set and choice not in youth_name_set
    )

    print('\nInitial Data:')
    print(f'Buddy picks referencing Adult/YA leaders (not buddy-roster youths): {adult_only_buddy_picks}')
    print(f'Total youth: {len(youth_list)}')
    print(f'Youth with parents: {len([y for y in youth_list if y.parent_name])}')
    print(f'Youth with siblings: {len([y for y in youth_list if y.siblings_list])}')
    print(f'Centers: {[(c.name, len(c.crews)) for c in centers]}')
    if center_only_adults:
        print(f'Center-only leaders (algorithm assigns crew): {len(center_only_adults)}')
    if unassigned_adults:
        print(f'Unassigned leaders (algorithm assigns center & crew): {len(unassigned_adults)}')

    if no_reassignment:
        analyze_existing_assignments(year, youth_list, centers)
    else:
        run_optimization(
            year, youth_list, centers, center_only_adults, unassigned_adults,
            analyze_clusters_flag,
        )


if __name__ == '__main__':
    main()
