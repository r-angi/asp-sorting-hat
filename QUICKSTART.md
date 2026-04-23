# Quick Start Guide

## Installation

```bash
pip install -r requirements.txt
```

## Basic Usage

```bash
# Run with default settings
python main.py -y 2026

# Specify centers with crew counts
python main.py -y 2026 --centers Fayette:11 Kanawha:12 Nicholas:11 Leslie:11

# With cluster analysis
python main.py -y 2026 --analyze-clusters

# Full featured
python main.py -y 2026 --centers Fayette:11 Kanawha:12 --analyze-clusters
```

## Required Files

Place these CSV files in `data/clean/`:

1. **`buddies_YEAR.csv`** - Youth information and friend preferences
2. **`crews_YEAR.csv`** - Adult assignments to centers/crews
3. **`historical_crews.csv`** - Past crew assignments

## New Features (2026)

- **`--centers`** - Specify centers and crew counts (e.g., `Fayette:11`)
- **`--analyze-clusters`** - Generate friend cluster visualization
- **Supervision Groups** - Add `supervision_group` column (max 2 per center per group)
- **Anti-Buddy Lists** - Add `anti_buddy` column (pipe-separated names)
- **Flexible Adults** - Leave `Crew` blank for algorithm assignment

## Output

- **Assignments:** `data/results/assignments_YEAR.csv`
- **Cluster Viz:** `data/results/cluster_analysis.png` (with `--analyze-clusters`)

## Troubleshooting

**Missing dependencies?**
```bash
pip install -r requirements.txt
```

**Need help?** See `README.md` for complete documentation.

