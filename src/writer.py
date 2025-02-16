import polars as pl
import os
from ortools.sat.python import cp_model
from src.models import Youth, Center


def write_results_to_csv(
    solver: cp_model.CpSolver,
    person_crew: dict[tuple[str, str, str], int],
    youth_list: list[Youth],
    centers: list[Center],
    year: int,
) -> None:
    """Write all assignments and participant info to a CSV file."""
    output_path = f'./data/results/assignments_{year}.csv'
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = []

    # Add youth with their assignments and attributes
    for center in centers:
        for crew in center.crews:
            # Add youth in this crew
            crew_youth = [
                youth for youth in youth_list if solver.Value(person_crew[youth.name, center.name, crew.name]) == 1
            ]
            for youth in crew_youth:
                rows.append(
                    {
                        'Center': center.name,
                        'Crew': crew.name,
                        'Name': youth.name,
                        'Role': 'Youth',
                        'Gender': youth.gender,
                        'Year': youth.year,
                        'History': youth.history,
                    }
                )

            # Add adults in this crew
            for adult in crew.adults:
                rows.append(
                    {
                        'Center': center.name,
                        'Crew': crew.name,
                        'Name': adult,
                        'Role': 'Adult',
                        'Gender': '',
                        'Year': '',
                        'History': '',
                    }
                )

    # Convert to DataFrame and write to CSV
    results_df = pl.DataFrame(rows)
    results_df.write_csv(output_path)
    print(f'\nResults written to {output_path}')
