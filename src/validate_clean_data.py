"""Validate cleaned ``crews_{year}.csv`` and ``buddies_{year}.csv`` files.

Buddy choice columns may name another youth on the buddy roster or an Adult / Young Adult leader
on the crews CSV (the solver treats the latter as a soft same-center preference). Siblings and
anti-buddy values must still name youth on the buddy roster.
"""

import argparse
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import Final

import polars as pl

from src.schema import ALLOWED_GENDER, ALLOWED_HISTORY, ALLOWED_ROLES, ALLOWED_YEAR

CREWS_REQUIRED_COLUMNS: Final[tuple[str, ...]] = ('name', 'Center', 'Crew', 'role')
CREWS_OPTIONAL_COLUMNS: Final[tuple[str, ...]] = ('history', 'gender', 'parent')
# Tolerant gender vocabulary for crews CSV (long forms still appear in some exports).
CREW_GENDER_LONG_TO_SHORT: Final[dict[str, str]] = {
    'M': 'M',
    'F': 'F',
    'Male': 'M',
    'Female': 'F',
}
BUDDIES_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    'name',
    'history',
    'gender',
    'year',
    'first_choice',
    'second_choice',
    'third_choice',
    'siblings',
    'parent_name',
)
BUDDIES_OPTIONAL_DEFAULTS: Final[dict[str, str]] = {'supervision_group': '', 'anti_buddy': ''}

BUDDIES_COLUMNS: Final[tuple[str, ...]] = BUDDIES_REQUIRED_COLUMNS + tuple(BUDDIES_OPTIONAL_DEFAULTS.keys())

CHOICE_FIELDS: Final[tuple[str, ...]] = ('first_choice', 'second_choice', 'third_choice')

CREW_CODE_PATTERN: Final[str] = r'^[A-Za-z]\d{2}$'
NEAR_MATCH_CUTOFF: Final[float] = 0.82


@dataclass
class ValidationReport:
    """Aggregated validation results."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


def normalize_person_name(name: str | None) -> str:
    """Collapse internal whitespace and strip edges (matches typical cleaning behavior)."""
    if name is None:
        return ''
    return ' '.join(str(name).split())


def _split_pipe_field(value: str | None) -> list[str]:
    if not value or not str(value).strip():
        return []
    return [normalize_person_name(p) for p in str(value).split('|') if str(p).strip()]


def _near_miss_suggestions(value: str, candidates: list[str], *, cutoff: float = NEAR_MATCH_CUTOFF) -> list[str]:
    if not value or not candidates:
        return []
    return get_close_matches(value, candidates, n=3, cutoff=cutoff)


def _add_whitespace_issues(
    report: ValidationReport,
    label: str,
    df: pl.DataFrame,
    columns: tuple[str, ...],
) -> None:
    for col in columns:
        if col not in df.columns:
            continue
        for i, row in enumerate(df.iter_rows(named=True), start=2):
            raw = row.get(col)
            if raw is None:
                continue
            s = str(raw)
            normalized = normalize_person_name(s)
            if s != normalized:
                report.warnings.append(f'{label} row {i} column {col!r}: extra/leading/trailing spaces in {s!r}')


def _validate_crews_schema_and_values(crews: pl.DataFrame, report: ValidationReport) -> None:
    for col in CREWS_REQUIRED_COLUMNS:
        if col not in crews.columns:
            report.errors.append(f'Crews CSV missing required column {col!r}')

    if report.errors:
        return

    empty_name = crews.filter(pl.col('name').is_null() | (pl.col('name').cast(pl.Utf8).str.strip_chars() == ''))
    if len(empty_name) > 0:
        report.errors.append('Crews CSV has rows with empty name')

    center_blank = pl.col('Center').is_null() | (pl.col('Center').cast(pl.Utf8).str.strip_chars() == '')
    crew_blank = pl.col('Crew').is_null() | (pl.col('Crew').cast(pl.Utf8).str.strip_chars() == '')
    illegal_blank_center = crews.filter(center_blank & ~(pl.col('role').is_in(['Adult', 'Young Adult']) & crew_blank))
    if len(illegal_blank_center) > 0:
        samples = illegal_blank_center.select('name', 'role', 'Center', 'Crew').head(5).to_dicts()
        report.errors.append(
            'Crews CSV has rows with empty Center where it is not allowed '
            '(only fully unassigned Adult / Young Adult rows may leave both Center and Crew blank); '
            f'samples: {samples}'
        )

    bad_roles = crews.filter(~pl.col('role').is_in(list(ALLOWED_ROLES)))
    if len(bad_roles) > 0:
        bad_vals = bad_roles['role'].unique().to_list()
        report.errors.append(f'Crews CSV has invalid role values (expected one of {sorted(ALLOWED_ROLES)}): {bad_vals}')

    # Crew may be blank for center-only adults; if present, expect letter + two digits
    non_blank_crew = crews.filter(pl.col('Crew').is_not_null() & (pl.col('Crew').cast(pl.Utf8).str.strip_chars() != ''))
    bad_crew_fmt = non_blank_crew.filter(~pl.col('Crew').cast(pl.Utf8).str.contains(CREW_CODE_PATTERN))
    if len(bad_crew_fmt) > 0:
        samples = bad_crew_fmt.select('name', 'Crew').head(5).to_dicts()
        report.warnings.append(f'Crew codes not matching pattern letter + two digits (e.g. F01); samples: {samples}')

    dupes = crews.filter(pl.col('name').is_not_null()).group_by('name').len().filter(pl.col('len') > 1)
    if len(dupes) > 0:
        names = dupes['name'].to_list()
        report.errors.append(
            f'Crews CSV has duplicate leader names {names} — leader names must be unique so '
            'parent-center mapping and buddy-pick resolution stay deterministic'
        )

    if 'history' in crews.columns:
        adults_ya = crews.filter(pl.col('role').is_in(['Adult', 'Young Adult']))
        empty_hist = adults_ya.filter(pl.col('history').is_null() | (pl.col('history').cast(pl.Utf8).str.strip_chars() == ''))
        if len(empty_hist) > 0:
            samples = empty_hist.select('name').head(5).to_dicts()
            report.warnings.append(
                f'Crews CSV has Adult/Young Adult rows with empty history; adult vet/new balancing may be weaker on those crews; samples: {samples}'
            )
        bad_hist = adults_ya.filter(pl.col('history').cast(pl.Utf8).str.strip_chars() != '').filter(
            ~pl.col('history').cast(pl.Utf8).str.strip_chars().is_in(list(ALLOWED_HISTORY))
        )
        if len(bad_hist) > 0:
            vals = bad_hist['history'].unique().to_list()
            report.errors.append(f'Crews CSV invalid history values for Adult/Young Adult {vals} (allowed: {sorted(ALLOWED_HISTORY)})')

    if 'gender' in crews.columns:
        adults_ya_g = crews.filter(pl.col('role').is_in(['Adult', 'Young Adult']))
        bad_gender = adults_ya_g.filter(pl.col('gender').cast(pl.Utf8).str.strip_chars() != '').filter(
            ~pl.col('gender').cast(pl.Utf8).str.strip_chars().is_in(list(CREW_GENDER_LONG_TO_SHORT))
        )
        if len(bad_gender) > 0:
            vals = bad_gender['gender'].unique().to_list()
            report.errors.append(f'Crews CSV invalid gender values for Adult/Young Adult {vals} (allowed: {sorted(CREW_GENDER_LONG_TO_SHORT)})')

    for opt_col in CREWS_OPTIONAL_COLUMNS:
        if opt_col not in crews.columns:
            continue
        stray = crews.filter(pl.col('role') == 'Youth').filter(pl.col(opt_col).cast(pl.Utf8).str.strip_chars() != '')
        if len(stray) > 0:
            report.warnings.append(
                f'Crews CSV column {opt_col!r} has values on Youth rows (expected empty); '
                f'affected rows include {stray.select("name").head(5).to_dicts()}'
            )


def _validate_preassigned_adult_rules(crews: pl.DataFrame, report: ValidationReport) -> None:
    """Pre-flight: each fully pre-assigned (Center+Crew) crew must satisfy adult-side rules.

    Rules enforced (best-effort given available metadata):
      - At least one row with role == 'Adult' (the "driver"); 'Young Adult' does not count.
      - If 'history' is present and any leader on the crew has history == 'N',
        at least one leader (Adult OR Young Adult) on that crew must have history == 'V'.

    These rules apply only to crews where every assigned leader is fixed in the CSV.
    Center-only and unassigned adults are intentionally skipped: the solver places them.
    """
    pre_assigned = crews.filter(
        pl.col('role').is_in(['Adult', 'Young Adult'])
        & pl.col('Center').is_not_null()
        & (pl.col('Center').cast(pl.Utf8).str.strip_chars() != '')
        & pl.col('Crew').is_not_null()
        & (pl.col('Crew').cast(pl.Utf8).str.strip_chars() != '')
    )
    if len(pre_assigned) == 0:
        return

    has_history = 'history' in pre_assigned.columns

    grouped = pre_assigned.group_by(['Center', 'Crew']).agg(
        pl.col('role').alias('roles'),
        *([pl.col('history').alias('histories')] if has_history else []),
    )

    for row in grouped.iter_rows(named=True):
        center_name = row['Center']
        crew_name = row['Crew']
        roles = list(row['roles'])
        adult_count = sum(1 for r in roles if str(r) == 'Adult')
        if adult_count < 1:
            report.errors.append(f'Pre-assigned crew {center_name}/{crew_name} has no Adult (driver); roles present: {roles}')

        if has_history:
            histories = [str(h).strip() for h in row['histories']]
            new_count = sum(1 for h in histories if h == 'N')
            vet_count = sum(1 for h in histories if h == 'V')
            if new_count >= 1 and vet_count < 1:
                report.errors.append(
                    f'Pre-assigned crew {center_name}/{crew_name} has {new_count} New leader(s) '
                    f'and 0 Vet leaders; every New leader needs at least one Vet on the crew'
                )


def _validate_buddies_schema_and_values(buddies: pl.DataFrame, report: ValidationReport) -> None:
    for col in BUDDIES_COLUMNS:
        if col not in buddies.columns:
            report.errors.append(f'Buddies CSV missing required column {col!r}')

    if report.errors:
        return

    empty_name = buddies.filter(pl.col('name').is_null() | (pl.col('name').cast(pl.Utf8).str.strip_chars() == ''))
    if len(empty_name) > 0:
        report.errors.append('Buddies CSV has rows with empty name')

    for col_name, allowed in (
        ('history', ALLOWED_HISTORY),
        ('gender', ALLOWED_GENDER),
        ('year', ALLOWED_YEAR),
    ):
        empty_vals = buddies.filter(pl.col(col_name).is_null() | (pl.col(col_name).cast(pl.Utf8).str.strip_chars() == ''))
        if len(empty_vals) > 0:
            report.errors.append(f'Buddies CSV has rows with empty {col_name!r}')
        bad = buddies.filter(pl.col(col_name).cast(pl.Utf8).str.strip_chars() != '').filter(~pl.col(col_name).is_in(list(allowed)))
        if len(bad) > 0:
            vals = bad[col_name].unique().to_list()
            report.errors.append(f'Buddies CSV invalid {col_name} values {vals} (allowed: {sorted(allowed)})')

    dupes = buddies.filter(pl.col('name').is_not_null()).group_by('name').len().filter(pl.col('len') > 1)
    if len(dupes) > 0:
        names = dupes['name'].to_list()
        report.errors.append(f'Buddies CSV has duplicate name rows: {names}')

    for row in buddies.iter_rows(named=True):
        picks = [normalize_person_name(str(row.get(c) or '')) for c in CHOICE_FIELDS]
        non_empty = [p for p in picks if p]
        if len(non_empty) != len(set(non_empty)):
            duplicated = sorted({p for p in non_empty if non_empty.count(p) > 1})
            report.errors.append(
                f'{normalize_person_name(str(row["name"]))} has duplicate friend pick(s) across first/second/third choice: {duplicated}'
            )


def _validate_name_references(
    buddies: pl.DataFrame,
    crews: pl.DataFrame,
    report: ValidationReport,
) -> None:
    youth_names = sorted({normalize_person_name(n) for n in buddies['name'].drop_nulls().to_list() if n})
    youth_set = set(youth_names)
    adult_names_raw = crews.filter(pl.col('role').is_in(['Adult', 'Young Adult']))['name'].drop_nulls().to_list()
    adult_names = sorted({normalize_person_name(n) for n in adult_names_raw if n})
    adult_set = set(adult_names)
    buddy_target_names_sorted = sorted(youth_set | adult_set)

    # Friend choices → buddy roster youth or Adult/YA leader on crews (or empty)
    for row in buddies.iter_rows(named=True):
        youth = normalize_person_name(str(row['name']))
        for choice_key in CHOICE_FIELDS:
            choice_raw = row.get(choice_key)
            if choice_raw is None or str(choice_raw).strip() == '':
                continue
            choice = normalize_person_name(str(choice_raw))
            if choice not in youth_set and choice not in adult_set:
                sug = _near_miss_suggestions(choice, buddy_target_names_sorted)
                hint = f' (did you mean: {", ".join(sug)})?' if sug else ''
                report.errors.append(f"{youth}'s {choice_key} {choice!r} is not a youth in buddies nor an Adult/Young Adult on crews{hint}")

    # Siblings → subset of youth
    for row in buddies.iter_rows(named=True):
        youth = normalize_person_name(str(row['name']))
        for sib in _split_pipe_field(str(row.get('siblings', '') or '')):
            if sib not in youth_set:
                sug = _near_miss_suggestions(sib, youth_names)
                hint = f' (did you mean: {", ".join(sug)})?' if sug else ''
                report.errors.append(f"{youth}'s sibling {sib!r} is not in buddies roster{hint}")

    # Parents → subset of adult/YA crew list
    for row in buddies.iter_rows(named=True):
        youth = normalize_person_name(str(row['name']))
        for parent in _split_pipe_field(str(row.get('parent_name', '') or '')):
            if parent not in adult_set:
                sug = _near_miss_suggestions(parent, adult_names)
                hint = f' (did you mean: {", ".join(sug)})?' if sug else ''
                report.errors.append(f"{youth}'s parent {parent!r} not found in crews (Adult/Young Adult){hint}")

    # Anti-buddies → youth roster
    for row in buddies.iter_rows(named=True):
        youth = normalize_person_name(str(row['name']))
        for other in _split_pipe_field(str(row.get('anti_buddy', '') or '')):
            if other not in youth_set:
                sug = _near_miss_suggestions(other, youth_names)
                hint = f' (did you mean: {", ".join(sug)})?' if sug else ''
                report.errors.append(f"{youth}'s anti_buddy {other!r} is not in buddies roster{hint}")

    # Crews marked Youth should appear on buddy form
    youth_in_crews = crews.filter(pl.col('role') == 'Youth')
    for row in youth_in_crews.iter_rows(named=True):
        nm = normalize_person_name(str(row['name']))
        if nm and nm not in youth_set:
            sug = _near_miss_suggestions(nm, youth_names)
            hint = f' (did you mean: {", ".join(sug)})?' if sug else ''
            report.errors.append(f'Crews lists Youth {nm!r} who is not in buddies{hint}')


def validate_clean_data(
    year: int,
    *,
    data_dir: str | Path = './data/clean',
) -> ValidationReport:
    """Validate crews and buddies CSVs for ``year`` under ``data_dir``.

    Checks column presence, allowed enum fields, duplicates, whitespace quirks,
    and cross-file name references (buddy choices vs roster youth or crews leaders, siblings,
    parents, anti-buddy, Youth-in-crews vs buddy roster). Friend-choice typos get fuzzy
    suggestions from the union of those names.
    """
    report = ValidationReport()
    root = Path(data_dir)
    crews_path = root / f'crews_{year}.csv'
    buddies_path = root / f'buddies_{year}.csv'

    if not crews_path.is_file():
        report.errors.append(f'Missing crews file: {crews_path}')
    if not buddies_path.is_file():
        report.errors.append(f'Missing buddies file: {buddies_path}')
    if report.errors:
        return report

    crews_raw = pl.read_csv(crews_path)
    buddies_raw = pl.read_csv(buddies_path)

    if 'history' in crews_raw.columns and 'new/vet' in crews_raw.columns:
        report.errors.append("Crews CSV must not define both 'history' and 'new/vet' columns")
    elif 'new/vet' in crews_raw.columns:
        crews_raw = crews_raw.rename({'new/vet': 'history'})

    missing_c = [c for c in CREWS_REQUIRED_COLUMNS if c not in crews_raw.columns]
    missing_b = [c for c in BUDDIES_REQUIRED_COLUMNS if c not in buddies_raw.columns]
    if missing_c:
        report.errors.append(f'Crews CSV missing columns: {missing_c}')
    if missing_b:
        report.errors.append(f'Buddies CSV missing columns: {missing_b}')
    if report.errors:
        return report

    crews_cols = list(CREWS_REQUIRED_COLUMNS) + [c for c in CREWS_OPTIONAL_COLUMNS if c in crews_raw.columns]
    crews = crews_raw.select(crews_cols)
    buddies_work = buddies_raw
    for col, default in BUDDIES_OPTIONAL_DEFAULTS.items():
        if col not in buddies_work.columns:
            buddies_work = buddies_work.with_columns(pl.lit(default).alias(col))
    buddies = buddies_work.select(BUDDIES_COLUMNS)

    _validate_crews_schema_and_values(crews, report)
    _validate_buddies_schema_and_values(buddies, report)

    name_cols_crews: tuple[str, ...] = ('name',)
    name_cols_buddies: tuple[str, ...] = ('name', 'first_choice', 'second_choice', 'third_choice', 'siblings', 'parent_name', 'anti_buddy')

    _add_whitespace_issues(report, 'Crews', crews, name_cols_crews)
    _add_whitespace_issues(report, 'Buddies', buddies, name_cols_buddies)

    if not report.errors:
        _validate_name_references(buddies, crews, report)
        _validate_preassigned_adult_rules(crews, report)

    return report


def _print_report(report: ValidationReport) -> None:
    if report.warnings:
        print('Warnings:')
        for w in report.warnings:
            print(f'  - {w}')
    if report.errors:
        print('Errors:')
        for e in report.errors:
            print(f'  - {e}')
    else:
        print('No errors.')
    if report.ok:
        print('Validation passed.')
    else:
        print('Validation failed.')


def main() -> None:
    parser = argparse.ArgumentParser(description='Validate clean crews and buddies CSVs for a year.')
    parser.add_argument('-y', '--year', type=int, required=True, help='Year suffix on CSV files (e.g. 2026)')
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('./data/clean'),
        help='Directory containing crews_{year}.csv and buddies_{year}.csv',
    )
    args = parser.parse_args()
    rep = validate_clean_data(args.year, data_dir=args.data_dir)
    _print_report(rep)
    raise SystemExit(0 if rep.ok else 1)


if __name__ == '__main__':
    main()
