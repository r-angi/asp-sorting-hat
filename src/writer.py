"""Write the solver's assignment output to a CSV (default: ``data/results/assignments_<year>.csv``)."""

from collections.abc import Sequence
from itertools import chain
from pathlib import Path

import polars as pl
from ortools.sat.python import cp_model

from src.models import Center, Leader, Youth


def _row(*, center_name: str, crew_name: str, name: str, role: str,
         gender: str | None, year: str, history: str | None) -> dict[str, str]:
    return {
        'Center': center_name,
        'Crew': crew_name,
        'Name': name,
        'Role': role,
        'Gender': gender or '',
        'Year': year,
        'History': history or '',
    }


def write_results_to_csv(
    solver: cp_model.CpSolver,
    person_crew: dict[tuple[str, str, str], cp_model.IntVar],
    youth_list: list[Youth],
    centers: list[Center],
    year: int,
    adult_crew: dict[tuple[str, str, str], cp_model.IntVar] | None = None,
    unassigned_adults: Sequence[Leader] | None = None,
    center_only_adults: Sequence[Leader] | None = None,
    *,
    assignments_csv_path: Path | None = None,
) -> Path:
    """Emit one row per (youth or leader, crew) placement.

    Pre-assigned leaders come from ``crew.adults`` (typed instances). Flexible
    leaders (Adult / Young Adult, center-only + unassigned) come from the matching
    ``adult_crew`` Boolean — collapsed into one loop using ``itertools.chain``.

    When ``assignments_csv_path`` is omitted, writes ``./data/results/assignments_<year>.csv``.
    Returns the path written.
    """
    output_file = assignments_csv_path or Path('./data/results') / f'assignments_{year}.csv'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    flex_pool: list[Leader] = list(chain(center_only_adults or [], unassigned_adults or []))
    rows: list[dict[str, str]] = []

    for center in centers:
        for crew in center.crews:
            for youth in youth_list:
                key = (youth.name, center.name, crew.name)
                if key not in person_crew or solver.Value(person_crew[key]) != 1:
                    continue
                rows.append(_row(
                    center_name=center.name, crew_name=crew.name, name=youth.name,
                    role='Youth', gender=youth.gender, year=youth.year, history=youth.history,
                ))

            for leader in crew.adults:
                rows.append(_row(
                    center_name=center.name, crew_name=crew.name, name=leader.name,
                    role=leader.role, gender=leader.gender, year='', history=leader.history,
                ))

            if adult_crew is not None:
                for leader in flex_pool:
                    key = (leader.name, center.name, crew.name)
                    if key not in adult_crew or solver.Value(adult_crew[key]) != 1:
                        continue
                    rows.append(_row(
                        center_name=center.name, crew_name=crew.name, name=leader.name,
                        role=leader.role, gender=leader.gender, year='', history=leader.history,
                    ))

    pl.DataFrame(rows).write_csv(output_file)
    print(f'\nResults written to {output_file}')
    return output_file
