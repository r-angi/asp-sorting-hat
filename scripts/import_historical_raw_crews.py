#!/usr/bin/env python3
"""Import a vendor historical roster CSV into ``data/clean/historical_crews.csv``.

Dry-run by default; pass ``--write`` only after inspecting the preview.

Supports the layouts described in :mod:`src.historical_raw_import` (auto-detected).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.historical_raw_import import (
    HistoricalRawExportKind,
    append_historical_raw_rows,
    read_historical_raw_as_dataframe,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Convert data/raw/historical_crews_<YEAR>_raw.csv into historical_crews rows.',
    )
    parser.add_argument(
        '--year',
        type=int,
        required=True,
        help='Trip year appended to crew_year (e.g. crew D01 becomes "D01 2025").',
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=None,
        help='Defaults to data/raw/historical_crews_<YEAR>_raw.csv',
    )
    parser.add_argument(
        '--historical',
        type=Path,
        default=None,
        help='Defaults to data/clean/historical_crews.csv',
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=[e.value for e in HistoricalRawExportKind],
        default=HistoricalRawExportKind.AUTO.value,
        help='Input layout (default auto from headers)',
    )
    parser.add_argument(
        '--write',
        action='store_true',
        help='Append merged rows into the historical CSV; otherwise preview only.',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=15,
        help='Rows to print in preview (default 15)',
    )
    args = parser.parse_args()

    year = args.year
    raw_path = args.input if args.input is not None else ROOT / 'data' / 'raw' / f'historical_crews_{year}_raw.csv'
    hist_path = args.historical if args.historical is not None else ROOT / 'data' / 'clean' / 'historical_crews.csv'

    kind = HistoricalRawExportKind(args.format)
    new_rows = read_historical_raw_as_dataframe(kind, raw_path, year)

    prior_unique: set[tuple[object, ...]] = set()
    if hist_path.is_file() and hist_path.stat().st_size > 0:
        prior_unique = set(pl.read_csv(hist_path).rows())

    merged = append_historical_raw_rows(new_rows, hist_path, dry_run=not args.write)

    print(f'Parsed new rows from {raw_path}: {new_rows.height}')
    preview_n = max(0, args.limit)
    if preview_n:
        print(f'Preview (first {preview_n}) new rows only:')
        print(new_rows.head(preview_n))
    dup_with_prior = prior_unique & set(new_rows.rows())
    if dup_with_prior:
        print(f'Note: {len(dup_with_prior)} new row(s) overlap existing historical tuples (skipped by .unique()).')
    print(f'Merged total row count {"(dry run)" if not args.write else "(written)"}: {merged.height}')
    if not args.write:
        print('Dry run: no file written; pass --write to persist.')


if __name__ == '__main__':
    main()
