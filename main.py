"""Crew assignment optimizer entry point.

Two paths:

1. ``python main.py -y 2026`` — build the CP-SAT model and solve.
2. ``python main.py -y 2026 --no-reassignment --version v1`` — skip the solver
   and re-score ``data/results/<year>/vN/assignments_<year>.csv`` for dashboard /
   cluster output (PNG exports are regenerated in that same ``vN`` folder).

Fresh solver writes land under ``data/results/<year>/vN`` with the next unused
integer ``N`` (``v1``, ``v2``, …).

The re-analysis path uses a plain ``dict[(name, center, crew), int]`` rather
than a ``CpSolver``; downstream analysis / writer / clustering helpers accept
either via a thin :class:`AssignmentsLookup` wrapper.
"""

import argparse
import re
from collections.abc import Sequence
from pathlib import Path

import polars as pl
from ortools.sat.python import cp_model

from src.analysis import (
    PersonCrew,
    SolverLike,
    calculate_friend_match_buckets,
    calculate_friend_scores,
    print_crew_assignments,
    status_to_string,
    synthesize_centers_from_assignments,
)
from src.clustering import analyze_clusters
from src.config import CenterConfig, Config
from src.dashboard import render_center_dashboard
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


def normalize_run_version_label(version_arg: str) -> str:
    """Normalize CLI ``version`` into a directory basename ``v1``, ``v2``, …

    Accepts ``1`` or ``01`` → ``v1``, and ``v2`` / ``V2`` → ``v2``.
    """
    raw = version_arg.strip()
    if not raw:
        raise ValueError('Version label cannot be empty.')
    m_digit = re.fullmatch(r'(\d+)', raw)
    if m_digit:
        return f'v{int(m_digit.group(1), 10)}'
    m_v = re.fullmatch(r'[vV](\d+)', raw)
    if m_v:
        return f'v{int(m_v.group(1), 10)}'
    raise ValueError(
        f'Invalid version {version_arg!r}; expected e.g. v1 or 1.'
    )


def allocate_next_versioned_run_dir(results_base: Path, year: int) -> Path:
    """Create and return ``results_base / <year> / vN`` with the smallest unused ``N``, starting at 1 (``v1``, ``v2``, …).

    Existing sibling directories whose names match ``v`` + digits (e.g. ``v12``)
    bump the allocated version so repeated runs never overwrite prior outputs.
    """
    year_dir = results_base / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)

    highest = 0
    for entry in year_dir.iterdir():
        if not entry.is_dir():
            continue
        m = re.fullmatch(r'v(\d+)', entry.name, flags=re.ASCII)
        if not m:
            continue
        highest = max(highest, int(m.group(1)))

    run_dir = year_dir / f'v{highest + 1}'
    run_dir.mkdir(parents=False)
    return run_dir


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


def load_assignments_from_csv(
    assignments_csv: Path,
    youth_list: list[Youth],
    centers: list[Center],
) -> dict[tuple[str, str, str], int]:
    """Load a saved assignments workbook into a sparse ``(name, center, crew) -> 1`` dict.

    When the crews scaffold (``centers``) is non-empty, its ``(Center, Crew)``
    pairs are used as a typo guard against stray rows; an empty scaffold (e.g.
    all leaders are ``CENTER_ONLY`` so :func:`get_centers_from_adults_df`
    returns ``[]``) skips that gate so placements still load.
    """
    if not assignments_csv.is_file():
        raise FileNotFoundError(
            f'Assignments CSV not found: {assignments_csv}. '
            'Solve first to populate data/results/<year>/vN/assignments_<year>.csv, '
            'or pass matching --year and --no-reassignment --version vN.'
        )
    assignments_df = pl.read_csv(assignments_csv).with_columns(
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


def analyze_existing_assignments(
    year: int,
    youth_list: list[Youth],
    centers: list[Center],
    *,
    version_label: str,
    results_root: Path = RESULTS_DIR,
) -> None:
    """Re-score placements from ``results/<year>/<version>/assignments_<year>.csv`` without solving."""
    print('\n' + '=' * 50)
    print('ANALYZING EXISTING ASSIGNMENTS (--no-reassignment)')
    print('=' * 50)

    run_root = results_root / str(year) / version_label
    assignments_csv = run_root / f'assignments_{year}.csv'
    print(f'Source CSV: {assignments_csv}')

    person_crew = load_assignments_from_csv(assignments_csv, youth_list, centers)
    solver = AssignmentsLookup()

    effective_centers = centers if centers else synthesize_centers_from_assignments(person_crew)
    if not centers:
        synthesized = [(c.name, [crew.name for crew in c.crews]) for c in effective_centers]
        print(f'Centers (synthesized from assignments workbook): {synthesized}')

    print_crew_assignments(solver, person_crew, youth_list, effective_centers)

    center_friend_scores, _ = calculate_friend_scores(solver, person_crew, youth_list, effective_centers)
    buddy_match_counts = calculate_friend_match_buckets(solver, person_crew, youth_list, effective_centers)

    print(f'\nRegenerating dashboard and cluster visuals under {run_root}')

    render_center_dashboard(
        assignments_csv=assignments_csv,
        output_path=run_root / f'center_dashboard_{year}.png',
        year=year,
        friend_scores=center_friend_scores,
        buddy_match_counts=buddy_match_counts,
    )

    regular_youth = [y for y in youth_list if y.role == 'Youth']
    analyze_clusters(regular_youth, solver, person_crew, effective_centers, year=year, output_dir=str(run_root))

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
    run_dir = allocate_next_versioned_run_dir(RESULTS_DIR, year)
    print(f'\nWriting run artifacts under {run_dir}')

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
        assignments_csv_path=run_dir / f'assignments_{year}.csv',
    )

    center_friend_scores, _ = calculate_friend_scores(solver, person_crew, youth_list, centers)
    buddy_match_counts = calculate_friend_match_buckets(solver, person_crew, youth_list, centers)
    render_center_dashboard(
        assignments_csv=run_dir / f'assignments_{year}.csv',
        output_path=run_dir / f'center_dashboard_{year}.png',
        year=year,
        friend_scores=center_friend_scores,
        buddy_match_counts=buddy_match_counts,
    )

    if analyze_clusters_flag:
        regular_youth = [y for y in youth_list if y.role == 'Youth']
        analyze_clusters(regular_youth, solver, person_crew, centers, year=year, output_dir=str(run_dir))

    _print_friend_scores(solver, person_crew, youth_list, centers)


def _parse_cli_run_version(version_raw: str) -> str:
    try:
        return normalize_run_version_label(version_raw)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e


def main() -> None:
    parser = argparse.ArgumentParser(description='Run crew assignment optimization')
    parser.add_argument('-y', '--year', type=int, required=True, help='Year for the crew assignments')
    parser.add_argument(
        '--centers', nargs='*', help='Center specifications in format "CenterName:CrewCount" or "CenterName" (e.g., Fayette:11 Kanawha:12)'
    )
    parser.add_argument('--analyze-clusters', action='store_true', help='Run friend cluster analysis and generate visualization')
    parser.add_argument(
        '--no-reassignment',
        action='store_true',
        help=(
            'Skip optimization; read data/results/<year>/<version>/assignments_<year>.csv '
            'and regenerate dashboard + cluster visuals in that folder (requires --version).'
        ),
    )
    parser.add_argument(
        '--version',
        metavar='VN',
        default=None,
        type=_parse_cli_run_version,
        help=(
            'With --no-reassignment: subdirectory under data/results/<year>/, e.g. v1 or 2 (required there). '
            'Ignored during a normal solver run.'
        ),
    )

    args = parser.parse_args()

    year = args.year
    center_specs = args.centers
    analyze_clusters_flag = args.analyze_clusters
    no_reassignment = args.no_reassignment

    if args.version is not None and not no_reassignment:
        parser.error('--version is only used with --no-reassignment.')

    if no_reassignment and args.version is None:
        parser.error('--no-reassignment requires --version (e.g. --version v1).')

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
        retro_label = args.version
        assert retro_label is not None
        analyze_existing_assignments(year, youth_list, centers, version_label=retro_label)
    else:
        run_optimization(
            year, youth_list, centers, center_only_adults, unassigned_adults,
            analyze_clusters_flag,
        )


if __name__ == '__main__':
    main()
