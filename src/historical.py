"""Historical-crews dataset writers.

Two flows append to ``data/clean/historical_crews.csv``:

- :func:`convert_crews_to_historical` — for the in-flight ``crews_{year}.csv``
  (the one used to seed the solver). Used to backfill prior years.
- :func:`append_assignments_final_to_historical` — for the archived
  ``data/results/<year>/<vN>/assignments_<year>.csv`` snapshot (default slot
  is ``v1`` per year, with ``2025`` defaulting to ``v2`` where the finalized
  roster lives), after manual edits.

Both write the same three-column shape: ``name``, ``crew_year``, ``is_adult``.
The deduplication key is the row tuple itself, so re-runs are idempotent.
"""

import os
from pathlib import Path

import polars as pl


def convert_crews_to_historical(crews_path: str, year: int) -> None:
    """Convert crews data to historical_crews format and append to ``historical_crews.csv``.

    Args:
        crews_path: Path to the crews CSV file (e.g., ``./data/clean/crews_2024.csv``).
        year: Year to append to crew names.
    """
    if not os.path.exists(crews_path) or not crews_path.endswith('.csv'):
        raise ValueError(f'File {crews_path} does not exist or is not a csv')

    crews_df = pl.read_csv(crews_path)

    historical_df = crews_df.with_columns(
        [
            pl.concat_str([pl.col('Crew'), pl.lit(str(year))], separator=' ').alias('crew_year'),
            pl.when(pl.col('role') == 'Adult').then(pl.lit(True)).otherwise(pl.lit(False)).alias('is_adult'),
        ]
    ).select(['name', 'crew_year', 'is_adult'])

    historical_path = './data/clean/historical_crews.csv'
    if not os.path.exists(historical_path):
        historical_df.write_csv(historical_path)
        return
    existing_df = pl.read_csv(historical_path)
    pl.concat([existing_df, historical_df]).unique().write_csv(historical_path)


def default_versioned_assignments_path(year: int, *, results_root: Path | str = './data/results') -> Path:
    """Path to the default archival assignments workbook for ``year``.

    Uses ``results/<year>/v1/assignments_<year>.csv``, except ``2025`` uses
    ``v2`` (finalized roster) alongside solver output in ``v1``.
    """
    root = Path(results_root)
    version = 'v2' if year == 2025 else 'v1'
    return root / str(year) / version / f'assignments_{year}.csv'


def assignments_final_to_historical_df(assignments_df: pl.DataFrame, year: int) -> pl.DataFrame:
    """Map assignments workbook rows into ``historical_crews`` columns.

    Rows with blank ``Crew`` are skipped (e.g. unassigned placeholders).

    Adults (``Role == 'Adult'``) become ``is_adult`` true; Youth and Young Adult are false,
    consistent with :func:`convert_crews_to_historical`.
    """
    required = {'Name', 'Crew', 'Role'}
    missing = required - set(assignments_df.columns)
    if missing:
        raise ValueError(f'assignments CSV missing columns: {sorted(missing)}')
    if year <= 1900:
        raise ValueError(f'year must be plausible, got {year}')

    y = str(year)
    normalized = assignments_df.with_columns(
        crew_str=pl.col('Crew').cast(pl.Utf8).str.strip_chars(),
        name_clean=pl.col('Name').cast(pl.Utf8).str.replace(r'\s+', ' ').str.strip_chars(),
        role_clean=pl.col('Role').cast(pl.Utf8).str.strip_chars(),
    ).filter(pl.col('crew_str').is_not_null() & (pl.col('crew_str') != ''))

    return normalized.with_columns(
        pl.concat_str([pl.col('crew_str'), pl.lit(y)], separator=' ').alias('crew_year'),
        (pl.col('role_clean') == 'Adult').alias('is_adult'),
    ).select(pl.col('name_clean').alias('name'), pl.col('crew_year'), pl.col('is_adult'))


def append_assignments_final_to_historical(
    year: int,
    *,
    assignments_path: str | Path | None = None,
    historical_path: str | Path | None = None,
    dry_run: bool = False,
) -> pl.DataFrame:
    """Append rows from the versioned assignments workbook into ``historical_crews.csv``.

    De-duplicates on ``(name, crew_year, is_adult)`` after merge. When ``historical_path`` is
    missing, it is created with the new rows only (unless ``dry_run``).

    Args:
        year: Trip year appended to ``crew_year`` (e.g. crew ``F01`` → ``F01 2025``).
        assignments_path: Optional path; default :func:`default_versioned_assignments_path`.
        historical_path: Optional path; default ``./data/clean/historical_crews.csv``.
        dry_run: If true, compute the merged frame but do not write.

    Returns:
        The merged Polars dataframe (same as would be/was written).

    Raises:
        FileNotFoundError: Assignments CSV is missing.
        ValueError: Missing required columns or invalid year.
    """
    assign_path = Path(assignments_path) if assignments_path else default_versioned_assignments_path(year)
    hist_path = Path(historical_path) if historical_path else Path('./data/clean/historical_crews.csv')

    if not assign_path.is_file():
        raise FileNotFoundError(f'Assignments file not found: {assign_path}')

    assignments_df = pl.read_csv(assign_path)
    historical_df = assignments_final_to_historical_df(assignments_df, year)

    if hist_path.is_file():
        existing_df = pl.read_csv(hist_path)
        merged = pl.concat([existing_df, historical_df]).unique()
    else:
        merged = historical_df.unique()

    hist_path.parent.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        merged.write_csv(hist_path)

    return merged
