"""Normalize crews CSV column names and person names for the clean-data layout."""

import argparse
from pathlib import Path
from typing import Any, Final

import polars as pl

from src.schema import ADULT_ROLES

BASE_CREW_COLUMNS: Final[tuple[str, ...]] = ("name", "Center", "Crew", "role")
ADULT_DETAIL_COLUMNS: Final[tuple[str, ...]] = ("history", "gender", "parent")
OUTPUT_COLUMNS: Final[tuple[str, ...]] = BASE_CREW_COLUMNS + ADULT_DETAIL_COLUMNS


def proper_person_name(name: str | None) -> str:
    """Convert mixed or ALL-CAPS names to readable casing.

    - Collapses whitespace.
    - Two-letter alphabetic tokens (e.g. ``TJ``) stay uppercase.
    - ``Mc...`` surnames become ``McXxx`` (e.g. ``McMurray``).
    - Other tokens get ``str.capitalize()`` (first letter upper, rest lower).
    """
    if name is None or not str(name).strip():
        return ""
    words = str(name).split()
    fixed: list[str] = []
    for w in words:
        wl = w.lower()
        if len(wl) == 2 and wl.isalpha():
            fixed.append(wl.upper())
        elif len(wl) > 3 and wl.startswith("mc") and wl[2:].isalpha():
            fixed.append("Mc" + wl[2:].capitalize())
        else:
            fixed.append(wl.capitalize())
    return " ".join(fixed)


def _format_optional_field(val: Any) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    if not s:
        return ""
    return proper_person_name(s)


def _ensure_core_aliases(df: pl.DataFrame) -> pl.DataFrame:
    """Derive ``name`` / ``role`` from common export column names when missing."""
    by_lower = {c.strip().lower(): c for c in df.columns}
    out = df
    if "name" not in by_lower and "first name" in by_lower and "last name" in by_lower:
        out = out.with_columns(
            pl.concat_str(
                [pl.col(by_lower["first name"]), pl.col(by_lower["last name"])],
                separator=" ",
            ).alias("name")
        )
    if "role" not in by_lower and "adult/ya" in by_lower:
        src = by_lower["adult/ya"]
        out = (
            out.with_columns(
                pl.when(pl.col(src).cast(pl.Utf8).str.strip_chars() == "YA")
                .then(pl.lit("Young Adult"))
                .otherwise(pl.col(src))
                .alias("role")
            )
            .drop(src)
        )
    return out


def _canonical_header_map(columns: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for c in columns:
        key = c.strip().lower()
        if key == "name":
            mapping[c] = "name"
        elif key == "center":
            mapping[c] = "Center"
        elif key == "crew":
            mapping[c] = "Crew"
        elif key == "role":
            mapping[c] = "role"
        elif key in ("adult/ya", "adult_ya"):
            mapping[c] = "role"
        elif key == "history":
            mapping[c] = "history"
        elif key in ("new/vet", "new vet", "new_vet", "newvet"):
            mapping[c] = "history"
        elif key == "gender":
            mapping[c] = "gender"
        elif key == "parent":
            mapping[c] = "parent"
    return mapping


def normalize_crews_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Select core crew columns, optional adult metadata, and fix ``name`` casing.

    Columns ``history``, ``gender``, and ``parent`` are retained for Adult and Young Adult
    rows; they are cleared for Youth so downstream files keep a uniform schema.
    """
    work = _ensure_core_aliases(df)
    header_map = _canonical_header_map(work.columns)
    present = set(header_map.values())
    required = set(BASE_CREW_COLUMNS)
    if not required.issubset(present):
        raise ValueError(
            f"Could not resolve required columns {BASE_CREW_COLUMNS}; "
            f"mapped to {sorted(present)} from headers {work.columns!r}"
        )
    rename = {src: canon for src, canon in header_map.items()}
    renamed = work.rename(rename)

    meta_present = [c for c in ADULT_DETAIL_COLUMNS if c in renamed.columns]
    select_cols = list(BASE_CREW_COLUMNS) + meta_present
    out = renamed.select(select_cols)

    for col in ADULT_DETAIL_COLUMNS:
        if col not in out.columns:
            out = out.with_columns(pl.lit("").alias(col))

    out = out.with_columns(
        pl.when(pl.col("role").cast(pl.Utf8).str.strip_chars() == "YA")
        .then(pl.lit("Young Adult"))
        .otherwise(pl.col("role"))
        .alias("role"),
    )
    adult_mask = pl.col("role").cast(pl.Utf8).str.strip_chars().is_in(list(ADULT_ROLES))

    out = out.with_columns(
        pl.col("name").map_elements(
            lambda x: proper_person_name(str(x) if x is not None else None),
            return_dtype=pl.Utf8,
        )
    )
    for col in ADULT_DETAIL_COLUMNS:
        out = out.with_columns(
            pl.col(col)
            .map_elements(
                lambda x: _format_optional_field(x),
                return_dtype=pl.Utf8,
            )
            .alias(col)
        )

    out = out.with_columns(
        [
            pl.when(adult_mask).then(pl.col(col)).otherwise(pl.lit("")).alias(col)
            for col in ADULT_DETAIL_COLUMNS
        ]
    )

    out = out.with_columns(
        pl.col("Center").cast(pl.Utf8),
        pl.col("Crew").cast(pl.Utf8).fill_null(""),
    )
    return out.select(OUTPUT_COLUMNS)


def normalize_crews_csv_file(input_path: Path, output_path: Path | None = None) -> pl.DataFrame:
    df = pl.read_csv(input_path)
    result = normalize_crews_dataframe(df)
    result.write_csv(output_path if output_path is not None else input_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize crews CSV columns and name casing.")
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=None,
        help="CSV path for in-place normalize when --year is not used",
    )
    parser.add_argument(
        "-y",
        "--year",
        type=int,
        default=None,
        help="Read data/raw/crews_{year}_raw.csv and write data/clean/crews_{year}.csv",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory for crews_{year}_raw.csv when using --year",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/clean"),
        help="Output directory for crews_{year}.csv when using --year",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path when reading from positional path (default: overwrite input)",
    )
    args = parser.parse_args()

    if args.year is not None:
        source = args.raw_dir / f"crews_{args.year}_raw.csv"
        dest = args.data_dir / f"crews_{args.year}.csv"
        if not source.is_file():
            raise SystemExit(f"File not found: {source}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        normalize_crews_csv_file(source, dest)
        print(f"Wrote normalized crews CSV: {dest}")
        return

    if args.path is None:
        parser.error("Provide path or --year")

    target = args.path
    if not target.is_file():
        raise SystemExit(f"File not found: {target}")

    normalize_crews_csv_file(target, args.output)
    print(f"Wrote normalized crews CSV: {args.output or target}")


if __name__ == "__main__":
    main()
