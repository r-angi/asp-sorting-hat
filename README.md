# Crew Assignment System

This system optimizes the assignment of youth to crews within different centers using Google's OR-Tools CP-SAT solver.

> **Quick Start:** See [`QUICKSTART.md`](QUICKSTART.md) for installation and basic usage.

## Background

[Jesse Lee ASP](https://jesseleeasp.org/) is an organization based out of
[Jesse Lee Memorial UMC](https://jesseleechurch.com/) in Ridgefield, CT that runs a youth mission trip to Appalachia
every year through [Appalachia Service Project (ASP)](https://asphome.org/). Hundreds of youth volunteers join each year
and need to be assigned to crews across multiple centers with adult leaders. The crew assignment problem is solved
through this algorithm.

## Overview

The system takes youth preferences (buddy forms) and assigns each youth to a crew within a center while optimizing for:
- Friend preferences
- Gender diversity within crews (youth)
- Year diversity within crews
- Veteran/New youth balance within crews
- Adult-leader gender and veteran/new balance on each crew (when leader metadata is present)

## Constraints

1. **Basic Assignment**
   - Each youth must be assigned to exactly one crew within one center
   - Each crew must stay within minimum and maximum **total** headcount (configurable via `min_crew_size` and `max_crew_size`): everyone on the crew counts, including youth, pre-placed leaders, and any flexible leaders the solver assigns
   - Each crew must have 2–3 adults total (configurable via `min_adults_per_crew` and `max_adults_per_crew`)
   - Every crew must include **at least one** leader with role `Adult` (treated as the “driver” minimum). `Young Adult` leaders alone do not satisfy this rule, regardless of their placement mode
   - Adults and Young Adults share the same placement modes: fully pre-assigned (Center + Crew), center-only (Center, blank Crew), or fully unassigned (both blank). The solver picks any unspecified center/crew

2. **Family Constraints**
   - Youth must be in the same center as their parent (if they have one)
   - Youth cannot be in the same crew as their parent
   - Siblings must be assigned to the same center
   - Siblings cannot be in the same crew (to encourage independence)

3. **Friend Management**
   - Buddy choices cannot be on the same crew as the picker — applies uniformly whether the picked person is a roster youth or an Adult / Young Adult on the crews CSV (the youth still gets a leader they like at their worksite without being directly supervised by them)
   - At least one buddy choice must be at the youth's center — applies uniformly whether the picked person is a roster youth or an Adult / Young Adult on the crews CSV
   - Anti-buddies cannot be at the same center (enforces separation)

4. **Historical Constraints**
   - Youth cannot be assigned to crews led by any of their past leaders (from `historical_crews.csv`)

5. **Supervision Groups**
   - Maximum of 2 youth per center from each supervision group
   - Multiple groups can exist (A, B, C, etc.), each with independent limits

## Optimization Objectives

The system optimizes multiple objectives with configurable weights:

1. **Friend preferences (youth roster and leaders)** — `friend_weight` (default 2) / `adult_friend_weight` (defaults to `friend_weight`)
   - The "at least one pick at the youth's center" rule above is hard for both buddy-roster picks and leader picks; weights below are the soft tie-breaker that decides *which* of multiple eligible centers the solver prefers when freedom remains.
   - **Youth / Young Adult on the buddy roster**: first / second / third choice in the same center earns +3 / +2 / +1 points (each multiplied by `friend_weight`).
   - **Adult / Young Adult on `crews_YEAR.csv` only** (leader name used as a buddy pick): +3 / +2 / +1 points × `adult_friend_weight`. As above, both the same-center hard rule and the different-crew separation rule apply uniformly to leader picks.
   - Young Adult buddy-form rows still use the youth path when the pick is another roster youth; leader picks use the adult path when the pick is not on the buddy roster.
   - Printed friend scores summarize youth-to-youth matches only (leader picks are enforced in the objective but omitted from that summary).

2. **Youth Gender Diversity** (`gender_weight`, default=1)
   - Rewards crews that have a balanced male/female ratio among youth
   - Uses minimum of M/F count as the score

3. **Year Diversity** (`year_weight`, default=1)
   - Rewards crews that have representation from different years
   - +1 point for each year level (Fr/So/Jr/Sr) represented in the crew

4. **Youth Veteran/New Balance** (`history_weight`, default=1)
   - Rewards crews that have a mix of veteran and new youth
   - Uses minimum of vet/new count as the score

5. **Adult Leader Gender Balance** (`adult_gender_weight`, default=1)
   - Rewards balanced male/female leadership on each crew when the crews CSV supplies a `gender` value for the leaders placed on that crew
   - Missing gender metadata is skipped so legacy CSVs still solve

6. **Adult Leader Veteran/New Balance** (`adult_history_weight`, default=1)
   - Rewards mixing veteran and new adults on a crew when the crews CSV supplies a `V`/`N` `history` value
   - Missing history metadata is skipped

## Preparing clean data

Raw exports often need normalization and should be validated before running the solver.

1. **Normalize crews CSV** (column aliases, name casing, optional adult metadata):

   ```bash
   # Preferred: raw → clean for a given year
   python src/crew_csv_normalize.py --year 2026

   # Or normalize a specific file (default: overwrite in place; use -o for a copy)
   python src/crew_csv_normalize.py path/to/crews.csv -o path/to/out.csv
   ```

   Default paths: reads `data/raw/crews_{year}_raw.csv`, writes `data/clean/crews_{year}.csv`.

2. **Validate clean buddies + crews** (required columns, name hygiene, crew codes, cross-file checks):

   ```bash
   python src/validate_clean_data.py -y 2026
   ```

   Uses `data/clean/crews_{year}.csv` and `data/clean/buddies_{year}.csv` by default. Exits with status 1 if validation errors are found.

## Input Requirements

The solver expects cleaned CSVs under `data/clean/`:

### 1. Buddy Forms (`buddies_YEAR.csv`)

Contains youth information and friend preferences.

**Example:**
```csv
name,history,gender,year,first_choice,second_choice,third_choice,siblings,parent_name,supervision_group,anti_buddy
John Smith,V,M,Jr,Jane Doe,Bob Wilson,Mary Jane,Sarah Smith,Tom Smith,A,Mike Brown|Chris Lee
Jane Doe,N,F,So,John Smith,,,,,B,
Bob Wilson,V,M,Sr,John Smith,Jane Doe,,,,A,
Sarah Smith,N,F,Fr,,,,,Tom Smith,,
Mary Jane,V,F,Jr,Jane Doe,John Smith,Bob Wilson,,,,
```

**Column descriptions:**

| Column | Required | Description |
|--------|----------|-------------|
| name | Yes | Full name of youth |
| history | Yes | V (veteran) or N (new) |
| gender | Yes | M or F |
| year | Yes | Fr/So/Jr/Sr |
| first_choice | No | First preference: full name of another youth on the buddy roster, or full name of an Adult / Young Adult on `crews_YEAR.csv` |
| second_choice | No | Same as `first_choice` |
| third_choice | No | Same as `first_choice` |
| siblings | No | Pipe-separated sibling names |
| parent_name | No | Pipe-separated parent names (must exist in crews CSV) |
| supervision_group | No | Group identifier (A, B, C, etc.) for max 2 per center constraint |
| anti_buddy | No | Pipe-separated names of people who cannot be at the same center |

### 2. Crew Assignments (`crews_YEAR.csv`)

Contains adult and Young Adult assignments to centers and crews. Optional columns `history`, `gender`, and `parent` are **recommended for Adult and Young Adult rows** so the model can apply adult diversity objectives and the “at least one Adult per crew” rule uses accurate role data. Values may be left blank; unknown values are ignored rather than guessed.

**Example:**
```csv
name,Center,Crew,role,history,gender,parent
Glenn Smith,Fayette,F01,Adult,V,M,
Sandy Carpenter,Fayette,F02,Adult,N,F,
Matt Carpenter,Fayette,F01,Young Adult,N,M,
John Doe,Fayette,,Adult,V,M,
Jane Smith,,,Adult,N,F,
Pete Nichols,Kanawha,K01,Adult,V,M,
```

**Column descriptions:**

| Column | Required | Description |
|--------|----------|-------------|
| name | Yes | Full name of adult |
| Center | No | Center name (e.g., Fayette, Kanawha, Nicholas, Leslie). If blank, algorithm assigns center and crew |
| Crew | No | Crew identifier (e.g., F01, K02). If blank, algorithm assigns to a crew |
| role | Yes | Adult, Young Adult, or Youth |
| history | No | V or N (veteran/new), for leaders |
| gender | No | M or F (Male/Female accepted; normalized internally), for leaders |
| parent | No | Pipe-separated parent names when relevant for an adult row |

**Leader assignment modes (apply to both Adult and Young Adult rows):**

1. **Fully pre-assigned** (Center + Crew specified): leader is fixed to that specific crew
   - Example: `Glenn Smith,Fayette,F01,Adult`
   - Example: `Matt Carpenter,Fayette,F01,Young Adult`
2. **Center-only** (Center specified, Crew blank): algorithm assigns to any crew in that center
   - Example: `John Doe,Fayette,,Adult`
   - Example: `Casey Lopez,Fayette,,Young Adult`
3. **Unassigned** (Center and Crew both blank): algorithm assigns to any crew in any center
   - Example: `Jane Smith,,,Adult`
   - Example: `Sam Wright,,,Young Adult`

**Notes:**
- Young Adults are treated like Adults for crew placement and may also have friend preferences on the buddy form, but only `Adult` rows satisfy the per-crew driver minimum
- All leaders (Adult and Young Adult, regardless of placement mode) count toward the 2–3 adults per crew totals
- `main.py` only loads **`Adult` and `Young Adult`** rows from this file. Optional **`Youth`** rows are supported in validation/normalization for roster-style exports; they are not consumed by the solver entrypoint

### 3. Historical Crews (`historical_crews.csv`)

Contains past crew assignments for preventing repeat leader pairings.

**Example:**
```csv
name,crew_year,is_adult
John Smith,F01 2024,False
Glenn Smith,F01 2024,True
Jane Doe,K02 2023,False
```

**Column descriptions:**

| Column | Required | Description |
|--------|----------|-------------|
| name | Yes | Full name |
| crew_year | Yes | Crew and year (e.g., "F01 2024") |
| is_adult | Yes | True if person was an adult leader, False if youth |

## Output

The optimizer writes:

- **`data/results/assignments_{YEAR}.csv`** — solver output with columns: `Center`, `Crew`, `Name`, `Role`, `Gender`, `Year`, `History` (adult rows echo leader metadata when known)

For workflows that need a frozen roster (re-analysis, publishing, or updating history), maintain a finalized copy as **`data/results/assignments_{YEAR}_final.csv`**. That file is used by:

- **`--no-reassignment`** in `main.py` (print metrics and cluster analysis without re-solving)
- **`scripts/append_assignments_to_historical.py`** (merge the year into `historical_crews.csv`)

Console output still includes center/crew breakdowns, diversity and friend-fulfillment summaries, and friend scores.

### Refreshing history after a trip

After you finalize assignments for year Y:

```bash
python scripts/append_assignments_to_historical.py --year Y
# Optional: --assignments path/to/file.csv --historical path/to/historical_crews.csv --dry-run
```

This appends de-duplicated rows to `data/clean/historical_crews.csv` in the same shape as the existing file.

## Configuration

Adjustable parameters in the `Config` class (`src/config.py`):

- `min_crew_size` / `max_crew_size` — total people per crew, youth plus leaders (defaults 5–7)
- `min_adults_per_crew` / `max_adults_per_crew` — leader counts per crew (defaults 2–3)
- `friend_weight` — buddy-roster friend same-center preferences (default 2)
- `adult_friend_weight` — same-center preference toward a leader named on the crews CSV (default: same as `friend_weight`)
- `gender_weight` — youth gender diversity (default 1)
- `year_weight` — youth year diversity (default 1)
- `history_weight` — youth veteran/new mix (default 1)
- `adult_gender_weight` — adult leader gender balance (default 1)
- `adult_history_weight` — adult leader veteran/new balance (default 1)
- `solver_max_time_seconds` / `solver_num_workers` / `solver_relative_gap_limit` / `solver_log_progress` — CP-SAT runtime knobs

Factory helpers:

- `Config.with_high_friend_weight()` / `Config.with_high_diversity()` — bias the objective mix for experiments
- `Config.with_fast()` — short time limit + loose gap, for iteration / smoke runs
- `Config.with_optimal()` — long time limit + tight gap, for final solves

## Usage

### Basic usage

```bash
# Run with default settings (centers and crews extracted from adults CSV)
python main.py -y 2026

# Specify centers with crew counts
python main.py -y 2026 --centers Fayette:11 Kanawha:12 Nicholas:11 Leslie:11

# Specify centers only (crew counts from CSV)
python main.py -y 2026 --centers Fayette Kanawha Nicholas Leslie

# Mixed: some with counts, some from CSV
python main.py -y 2026 --centers Fayette:11 Kanawha Nicholas:11 Leslie

# Analyze an existing finalized roster (no solver): requires data/results/assignments_{year}_final.csv
python main.py -y 2026 --no-reassignment

# With cluster analysis (after a full solve, or included automatically with --no-reassignment)
python main.py -y 2026 --analyze-clusters

# Full example
python main.py -y 2026 --centers Fayette:11 Kanawha:12 --analyze-clusters
```

### Cluster analysis

The `--analyze-clusters` flag (on a full solve) enables friend cluster detection and visualization:

- Detects friend communities using the Louvain method
- Evaluates how clusters were split across centers given the assignments
- Writes `data/results/cluster_analysis_{YEAR}.png` (and related outputs in `data/results/`)

With **`--no-reassignment`**, cluster analysis always runs against the finalized CSV.

## Feature summary

- **Configurable centers** — CLI center list with optional per-center crew counts
- **Flexible leader placement** — Adults and Young Adults can be fully pre-assigned, center-only, or fully unassigned; the solver fills blanks
- **Driver / Adult minimum** — At least one `Adult` leader per crew (Young Adults never satisfy this rule)
- **Leader metadata** — Optional `history` / `gender` on crews CSV is loaded directly onto each `Adult` / `YoungAdult` model and powers adult-side objectives and the per-crew Adult-driver minimum
- **Supervision groups** — Cap youth per group per center
- **Anti-buddy lists** — Keep named youth at different centers
- **Validation & normalization** — `validate_clean_data` and `crew_csv_normalize` for repeatable data prep
- **History merge script** — Append finalized assignments into `historical_crews.csv`
- **Re-analysis mode** — `--no-reassignment` for reporting on a locked roster
