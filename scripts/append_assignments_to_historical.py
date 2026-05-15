#!/usr/bin/env python3
"""Merge final assignments into historical_crews.csv for repeat-pair avoidance."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.historical import append_assignments_final_to_historical


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read the default data/results/<year>/vN/assignments_{year}.csv "
            "(v1 except 2025 → v2) and merge into data/clean/historical_crews.csv "
            "(same layout as crew CSV merges). Override with --assignments."
        )
    )
    parser.add_argument("--year", type=int, required=True, help="Trip year (e.g. 2025)")
    parser.add_argument(
        "--assignments",
        type=Path,
        default=None,
        help="Optional path to assignments CSV (default: data/results/<year>/v1/..., 2025 uses v2/)",
    )
    parser.add_argument(
        "--historical",
        type=Path,
        default=None,
        help="Optional path to historical CSV (default: data/clean/historical_crews.csv)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute merge and print row counts without writing.",
    )
    args = parser.parse_args()

    merged = append_assignments_final_to_historical(
        args.year,
        assignments_path=args.assignments,
        historical_path=args.historical,
        dry_run=args.dry_run,
    )
    print(f"Merged historical rows: {merged.height}")
    if args.dry_run:
        print("Dry run: historical file not modified.")


if __name__ == "__main__":
    main()
