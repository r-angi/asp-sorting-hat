# Crew Assignment System

This system optimizes the assignment of youth to crews within different centers using Google's OR-Tools CP-SAT solver.

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
   - Young Adults are pre-assigned to specific crews

2. **Family Constraints**
   - Youth must be in the same center as their parent (if they have one)
   - Youth cannot be in the same crew as their parent
   - Siblings must be assigned to the same center
   - Siblings cannot be in the same crew (to encourage independence)

3. **Friend Management**
   - Friends listed on buddy forms cannot be in the same crew
   - At least one friend must be in the same center (if the person specified any friends)

4. **Historical Constraints**
   - Youth cannot be assigned to crews led by any of their past leaders

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

The system requires three CSV files:

1. **Buddy Forms** (`buddies_YEAR.csv`):
   - name: Youth's full name
   - gender: M/F
   - year: Fr/So/Jr/Sr
   - history: V (veteran) / N (new)
   - first_choice: First choice friend
   - second_choice: Second choice friend
   - third_choice: Third choice friend
   - parent_name: Name of parent (if applicable)
   - siblings: Pipe-separated list of sibling names
   - role: Youth/Young Adult

2. **Crew Assignments** (`crews_YEAR.csv`):
   - name: Person's name
   - Center: Center name
   - Crew: Crew name
   - role: Adult/Youth

3. **Historical Crews** (`historical_crews_YEAR.csv`):
   - name: Youth's name
   - crew_year: Crew and year assignment
   - is_adult: Boolean indicating if person was an adult

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
- `friend_weight`: Weight for friend preferences (default=2)
- `gender_weight`: Weight for gender diversity (default=1)
- `year_weight`: Weight for year diversity (default=1)
- `history_weight`: Weight for vet/new balance (default=1) 