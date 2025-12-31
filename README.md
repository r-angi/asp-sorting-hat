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
- Gender diversity within crews
- Year diversity within crews
- Veteran/New youth balance within crews

## Constraints

1. **Basic Assignment**
   - Each youth must be assigned to exactly one crew within one center
   - Each crew must have a minimum and maximum size (configurable via min_crew_size and max_crew_size)
   - Each crew must have 2-3 adults (configurable via min_adults_per_crew and max_adults_per_crew)
   - Young Adults are pre-assigned to specific crews
   - Center-only adults are assigned to exactly one crew within their center

2. **Family Constraints**
   - Youth must be in the same center as their parent (if they have one)
   - Youth cannot be in the same crew as their parent
   - Siblings must be assigned to the same center
   - Siblings cannot be in the same crew (to encourage independence)

3. **Friend Management**
   - Friends listed on buddy forms cannot be in the same crew
   - At least one friend must be in the same center (if the person specified any friends)
   - Anti-buddies cannot be at the same center (enforces separation)

4. **Historical Constraints**
   - Youth cannot be assigned to crews led by any of their past leaders

5. **Supervision Groups**
   - Maximum of 2 youth per center from each supervision group
   - Multiple groups can exist (A, B, C, etc.), each with independent limits

## Optimization Objectives

The system optimizes multiple objectives with configurable weights:

1. **Friend Preferences** (weight=2)
   - First choice friend in same center: +3 points
   - Second choice friend in same center: +2 points
   - Third choice friend in same center: +1 point
   - Points are normalized per person in each center

2. **Gender Diversity** (weight=1)
   - Rewards crews that have a balanced male/female ratio
   - Uses minimum of M/F count as the score

3. **Year Diversity** (weight=1)
   - Rewards crews that have representation from different years
   - +1 point for each year level (Fr/So/Jr/Sr) represented in the crew

4. **Veteran/New Balance** (weight=1)
   - Rewards crews that have a mix of veteran and new youth
   - Uses minimum of vet/new count as the score

## Input Requirements

The system requires three CSV files with specific formats:

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
| first_choice | No | First choice friend (full name) |
| second_choice | No | Second choice friend (full name) |
| third_choice | No | Third choice friend (full name) |
| siblings | No | Pipe-separated sibling names |
| parent_name | No | Pipe-separated parent names (must exist in crews CSV) |
| supervision_group | No | Group identifier (A, B, C, etc.) for max 2 per center constraint |
| anti_buddy | No | Pipe-separated names of people who cannot be at the same center |

### 2. Crew Assignments (`crews_YEAR.csv`)

Contains adult assignments to centers and crews.

**Example:**
```csv
name,Center,Crew,role
Glenn Smith,Fayette,F01,Adult
Sandy Carpenter,Fayette,F02,Adult
Matt Carpenter,Fayette,F01,Young Adult
John Doe,Fayette,,Adult
Jane Smith,,,Adult
Pete Nichols,Kanawha,K01,Adult
```

**Column descriptions:**

| Column | Required | Description |
|--------|----------|-------------|
| name | Yes | Full name of adult |
| Center | No | Center name (e.g., Fayette, Kanawha, Nicholas, Leslie). If blank, algorithm assigns center and crew |
| Crew | No | Crew identifier (e.g., F01, K02). If blank, algorithm assigns to a crew |
| role | Yes | Adult or Young Adult |

**Adult Assignment Modes:**

1. **Fully Pre-assigned** (Center + Crew specified): Adult is fixed to that specific crew
   - Example: `Glenn Smith,Fayette,F01,Adult`
   
2. **Center-Only** (Center specified, Crew blank): Algorithm assigns to any crew in that center
   - Example: `John Doe,Fayette,,Adult`
   
3. **Unassigned** (Center and Crew both blank): Algorithm assigns to any crew in any center
   - Example: `Jane Smith,,,Adult`

**Notes:**
- Young Adults are treated like adults for crew leadership but may also have friend preferences
- All adults count toward the 2-3 adults per crew requirement

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

The system generates:
1. CSV file with final assignments
2. Detailed console output showing:
   - Center-by-center breakdown of assignments
   - Crew compositions and diversity metrics
   - Friend choice fulfillment statistics
   - Overall system performance metrics

## Configuration

Adjustable parameters in `Config` class:
- `min_crew_size`: Minimum people per crew (default=5)
- `max_crew_size`: Maximum people per crew (default=7)
- `min_adults_per_crew`: Minimum adults per crew (default=2)
- `max_adults_per_crew`: Maximum adults per crew (default=3)
- `friend_weight`: Weight for friend preferences (default=2)
- `gender_weight`: Weight for gender diversity (default=1)
- `year_weight`: Weight for year diversity (default=1)
- `history_weight`: Weight for vet/new balance (default=1)

## Usage

### Basic Usage

```bash
# Run with default settings (centers and crews extracted from adults CSV)
python main.py -y 2026

# Specify centers with crew counts
python main.py -y 2026 --centers Fayette:11 Kanawha:12 Nicholas:11 Leslie:11

# Specify centers only (crew counts from CSV)
python main.py -y 2026 --centers Fayette Kanawha Nicholas Leslie

# Mixed: some with counts, some from CSV
python main.py -y 2026 --centers Fayette:11 Kanawha Nicholas:11 Leslie

# With cluster analysis
python main.py -y 2026 --analyze-clusters

# Full example
python main.py -y 2026 --centers Fayette:11 Kanawha:12 --analyze-clusters
```

### Cluster Analysis

The `--analyze-clusters` flag enables friend cluster detection and visualization:
- Detects friend communities using Louvain community detection algorithm
- Analyzes how well clusters were kept together in assignments
- Generates a visualization showing cluster distribution across centers
- Computes cohesion scores (fraction of cluster in most common center)

The analysis is independent of assignments - clusters are detected purely from buddy form data, then evaluated against the assignment results.

## New Features for 2026

1. **Configurable Centers with Crew Counts** - Specify centers and optional crew counts via CLI
2. **Flexible Adult Pre-Assignment** - Adults can be assigned to a center without a specific crew
3. **Supervision Group Constraint** - Maximum of 2 per center per supervision group
4. **Anti-Buddy List** - Prevent specific youth from being at the same center
5. **Friend Cluster Analysis** - Detect friend groups and visualize distribution
