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
    adult_crew: dict[tuple[str, str, str], int] | None = None,
    unassigned_adults: list[str] | None = None,
    center_only_adults: list[tuple[str, str]] | None = None,
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

            # Add pre-assigned adults in this crew
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
            
            # Add center-only adults assigned to this crew
            if adult_crew and center_only_adults:
                for adult_name, assigned_center in center_only_adults:
                    if assigned_center == center.name:
                        if (adult_name, center.name, crew.name) in adult_crew:
                            if solver.Value(adult_crew[adult_name, center.name, crew.name]) == 1:
                                rows.append(
                                    {
                                        'Center': center.name,
                                        'Crew': crew.name,
                                        'Name': adult_name,
                                        'Role': 'Adult',
                                        'Gender': '',
                                        'Year': '',
                                        'History': '',
                                    }
                                )
            
            # Add unassigned adults assigned to this crew
            if adult_crew and unassigned_adults:
                for adult_name in unassigned_adults:
                    if (adult_name, center.name, crew.name) in adult_crew:
                        if solver.Value(adult_crew[adult_name, center.name, crew.name]) == 1:
                            rows.append(
                                {
                                    'Center': center.name,
                                    'Crew': crew.name,
                                    'Name': adult_name,
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
