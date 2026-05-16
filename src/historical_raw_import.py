"""Convert vendor ``historical_crews_*_raw.csv`` exports into ``historical_crews.csv`` rows.

Shapes match :func:`src.historical.assignments_final_to_historical_df` /
:func:`src.historical.convert_crews_to_historical`: ``name``, ``crew_year``, ``is_leader``.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Final

import polars as pl

from src.crew_csv_normalize import proper_person_name
from src.historical import ensure_historical_leader_column

_VENDOR_ROLE_COL_FRAGMENT: Final[str] = 'i am registering for this asp trip'
_VENDOR_FIRST: Final[str] = "participant's name - first name"
_VENDOR_LAST: Final[str] = "participant's name - last name"


class HistoricalRawExportKind(StrEnum):
    AUTO = 'auto'
    """Infer from CSV headers."""

    CREWS_SPLIT = 'crews_split'
    """Columns like ``crews_*_raw`` (First Name, Last Name, Crew, Adult/YA)."""

    VENDOR_WIDE = 'vendor_wide'
    """Participant's Name + Crew + registration-role column."""

    VENDOR_SPLIT = 'vendor_split'
    """Separate participant first/last + Crew (+ optional Phone/Email)."""


def _normalized_headers(columns: tuple[str, ...]) -> dict[str, str]:
    """Map stripped lower-case header → original column label."""
    return {c.strip().lower(): c for c in columns}


def detect_historical_raw_kind(headers: tuple[str, ...]) -> HistoricalRawExportKind:
    lowered = _normalized_headers(headers)

    if _VENDOR_FIRST in lowered and _VENDOR_LAST in lowered and 'crew' in lowered:
        return HistoricalRawExportKind.VENDOR_SPLIT
    if (
        _role_column_from_wide_headers(headers) is not None
        and "participant's name" in lowered
        and 'crew' in lowered
    ):
        return HistoricalRawExportKind.VENDOR_WIDE
    if (
        'first name' in lowered
        and 'last name' in lowered
        and 'crew' in lowered
        and ('adult/ya' in lowered or 'adult_ya' in lowered)
    ):
        return HistoricalRawExportKind.CREWS_SPLIT

    kinds = ', '.join(
        repr(k.value)
        for k in HistoricalRawExportKind
        if k != HistoricalRawExportKind.AUTO
    )
    raise ValueError(
        f'Could not infer historical raw layout from columns {list(headers)!r}. '
        f'Pass explicit --format with one of: {kinds}'
    )


def _role_column_from_wide_headers(headers: tuple[str, ...]) -> str | None:
    for c in headers:
        lc = c.strip().lower()
        if _VENDOR_ROLE_COL_FRAGMENT in lc:
            return c
    return None


def _canonical_crew_code_from_first_last(raw_crew_expr: pl.Expr) -> pl.Expr:
    """Match :func:`scripts.clean_raw.clean_crews_2025_raw` crew normalization."""

    letter = raw_crew_expr.cast(pl.Utf8).str.strip_chars().str.slice(0, 1)
    rest = raw_crew_expr.cast(pl.Utf8).str.strip_chars().str.slice(1).str.zfill(2)
    return pl.concat_str([letter, rest])


def _standardize_name_column(col: pl.Expr) -> pl.Expr:
    return col.map_elements(
        lambda x: proper_person_name(str(x) if x is not None else None),
        return_dtype=pl.Utf8,
    )


def _expand_ya_role(col: pl.Expr) -> pl.Expr:
    s = col.cast(pl.Utf8).str.strip_chars()
    return pl.when(s == 'YA').then(pl.lit('Young Adult')).otherwise(s)


def _is_leader_from_role(role: pl.Expr) -> pl.Expr:
    r = _expand_ya_role(role).cast(pl.Utf8).str.strip_chars()
    return r.is_in(['Adult', 'Young Adult'])


def historical_rows_crews_split(df: pl.DataFrame, year: int) -> pl.DataFrame:
    lowered = _normalized_headers(tuple(df.columns))
    first_c = lowered['first name']
    last_c = lowered['last name']
    crew_c = lowered['crew']
    role_c = lowered.get('adult/ya') or lowered.get('adult_ya')
    if not role_c:
        raise ValueError(
            'Crews-split layout requires an Adult/YA column (adult/ya or adult_ya).'
        )

    y = str(year)
    return (
        df.select(
            pl.concat_str(
                [
                    pl.col(first_c).cast(pl.Utf8).str.strip_chars(),
                    pl.lit(' '),
                    pl.col(last_c).cast(pl.Utf8).str.strip_chars(),
                ]
            )
            .alias('_full'),
            pl.col(crew_c).cast(pl.Utf8).str.strip_chars().alias('_crew_raw'),
            pl.col(role_c).alias('_role'),
        )
        .with_columns(
            [
                _canonical_crew_code_from_first_last(pl.col('_crew_raw')).alias('_crew'),
                _is_leader_from_role(pl.col('_role')).alias('is_leader'),
            ]
        )
        .with_columns(
            [
                _standardize_name_column(pl.col('_full')).alias('name'),
                pl.concat_str([pl.col('_crew'), pl.lit(y)], separator=' ').alias('crew_year'),
            ]
        )
        .select('name', 'crew_year', 'is_leader')
    )


def historical_rows_vendor_wide(df: pl.DataFrame, year: int) -> pl.DataFrame:
    lowered = _normalized_headers(tuple(df.columns))
    name_c = lowered["participant's name"]
    crew_c = lowered['crew']
    role_col = _role_column_from_wide_headers(tuple(df.columns))
    if role_col is None:
        raise ValueError(
            'Vendor-wide layout requires a registration-role column (substring match on asp trip).'
        )

    y = str(year)
    return (
        df.select(
            pl.col(name_c).alias('_name'),
            pl.col(crew_c).alias('_crew_raw'),
            pl.col(role_col).alias('_role'),
        )
        .with_columns(
            [
                pl.col('_crew_raw').cast(pl.Utf8).str.strip_chars().alias('_crew'),
                _is_leader_from_role(pl.col('_role')).alias('is_leader'),
            ]
        )
        .with_columns(
            [
                _standardize_name_column(pl.col('_name')).alias('name'),
                pl.concat_str([pl.col('_crew'), pl.lit(y)], separator=' ').alias('crew_year'),
            ]
        )
        .select('name', 'crew_year', 'is_leader')
    )


def historical_rows_vendor_split(df: pl.DataFrame, year: int) -> pl.DataFrame:
    """Like :func:`scripts.clean_raw.clean_historical_crews_old`: split names plus crew only.

    If a ``Role`` column (exact, case-sensitive) exists, leader rows are roles
    ``Adult`` or ``Young Adult`` (after normalizing ``YA``).
    Otherwise every row whose *original* trimmed full name equals its upper-case spelling
    becomes ``is_leader`` true — same heuristic as ``clean_historical_crews_old``.
    """
    lowered = _normalized_headers(tuple(df.columns))
    first_src = lowered[_VENDOR_FIRST]
    last_src = lowered[_VENDOR_LAST]
    crew_src = lowered['crew']

    candidates = tuple(c for c in df.columns if c.strip().lower() == 'role')
    role_original = candidates[0] if candidates else None

    y = str(year)
    staged = (
        df.select(
            pl.concat_str(
                [
                    pl.col(first_src).cast(pl.Utf8).str.strip_chars(),
                    pl.lit(' '),
                    pl.col(last_src).cast(pl.Utf8).str.strip_chars(),
                ]
            ).alias('_raw_full'),
            pl.col(crew_src).cast(pl.Utf8).str.strip_chars().alias('_crew'),
        ).with_columns(
            pl.when(pl.col('_raw_full').cast(pl.Utf8).str.strip_chars() != '')
            .then(pl.col('_raw_full').cast(pl.Utf8).str.strip_chars())
            .otherwise(pl.lit(''))
            .alias('_name_before_case')
        )
    )

    if role_original is None:
        staged = staged.with_columns(
            (pl.col('_name_before_case') == pl.col('_name_before_case').str.to_uppercase())
            .fill_null(False)
            .alias('is_leader'),
        )
    else:
        staged = staged.with_columns(
            (_is_leader_from_role(pl.col(role_original))).alias('is_leader')
        )

    return (
        staged.with_columns(
            [
                _standardize_name_column(pl.col('_name_before_case')).alias('name'),
                pl.concat_str([pl.col('_crew'), pl.lit(y)], separator=' ').alias('crew_year'),
            ]
        )
        .select('name', 'crew_year', 'is_leader')
    )


def read_historical_raw_as_dataframe(kind: HistoricalRawExportKind, path: Path, year: int) -> pl.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pl.read_csv(path)
    inferred = detect_historical_raw_kind(tuple(df.columns)) if kind == HistoricalRawExportKind.AUTO else kind
    if inferred == HistoricalRawExportKind.CREWS_SPLIT:
        return historical_rows_crews_split(df, year)
    if inferred == HistoricalRawExportKind.VENDOR_WIDE:
        return historical_rows_vendor_wide(df, year)
    if inferred == HistoricalRawExportKind.VENDOR_SPLIT:
        return historical_rows_vendor_split(df, year)
    raise ValueError(f'Unsupported kind after inference: {inferred}')


def append_historical_raw_rows(
    new_rows: pl.DataFrame,
    historical_path: Path,
    *,
    dry_run: bool,
) -> pl.DataFrame:
    """Concatenate ``new_rows`` with existing history and dedupe on all columns."""

    expected = {'name', 'crew_year', 'is_leader'}
    if set(new_rows.columns) != expected:
        raise ValueError(f'new_rows must have exactly {sorted(expected)}, got {new_rows.columns!r}')
    merged: pl.DataFrame
    if historical_path.is_file():
        existing = ensure_historical_leader_column(pl.read_csv(historical_path))
        merged = pl.concat([existing, new_rows]).unique()
    else:
        merged = new_rows.unique()
    historical_path.parent.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        merged.write_csv(historical_path)
    return merged
