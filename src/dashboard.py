"""Center-level summary dashboard rendered as a single PNG.

Renders a multi-panel matplotlib figure summarizing the per-center diversity
and headcount stats already printed by :func:`print_crew_assignments`, so
mismatches across centers are visible at a glance instead of buried in stdout.

Buildable from any saved assignments workbook (e.g. solver output under
``data/results/<year>/vN/assignments_<year>.csv``); friend-preference metrics are passed in
as a precomputed mapping so the renderer never re-implements solver-side
weighting.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

YEAR_ORDER: Final[tuple[str, ...]] = ('Fr', 'So', 'Jr', 'Sr')
GENDER_ORDER: Final[tuple[str, ...]] = ('F', 'M')
HISTORY_ORDER: Final[tuple[str, ...]] = ('V', 'N')
BUDDY_MATCH_ORDER: Final[tuple[str, ...]] = ('0', '1', '2', '3')

ALL_LABEL: Final[str] = 'All'


YEAR_COLORS: Final[dict[str, str]] = {
    'Fr': '#56B4E9',  # sky blue
    'So': '#009E73',  # bluish green
    'Jr': '#E69F00',  # orange
    'Sr': '#CC79A7',  # reddish purple
}

GENDER_COLORS: Final[dict[str, str]] = {
    'F': '#C77B8A',  # muted coral
    'M': '#4C72B0',  # muted slate blue
}

HISTORY_COLORS: Final[dict[str, str]] = {
    'V': '#5B7E92',  # cool slate
    'N': '#E1A95F',  # warm sand
}

# Buddy match buckets are ordinal (0 worst to 3 best), so the palette is a
# light-to-dark sequential ramp that converges on ACCENT_PRIMARY at the top.
BUDDY_MATCH_COLORS: Final[dict[str, str]] = {
    '0': '#D6DCE3',
    '1': '#9FB1C7',
    '2': '#5E7B9F',
    '3': '#1F4E79',
}

# Headcount stack: youth vs crew adults vs young adults — teal / slate / coral,
# distinct from buddy-match navy ramp and percentage-stack palettes.
HEADCOUNT_ORDER: Final[tuple[str, ...]] = ('Youth', 'Adult', 'Young Adult')
HEADCOUNT_COLORS: Final[dict[str, str]] = {
    'Youth': '#1B7F72',
    'Adult': '#5D6D7E',
    'Young Adult': '#E07A5F',
}
HEADCOUNT_LEGEND_LABELS: Final[dict[str, str]] = {
    'Youth': 'Youth',
    'Adult': 'Adults',
    'Young Adult': 'Young adults',
}

ACCENT_PRIMARY: Final[str] = '#1F4E79'
ACCENT_SECONDARY: Final[str] = '#9AA0A6'
REFERENCE_LINE: Final[str] = '#7A7A7A'

FIG_FACECOLOR: Final[str] = '#FAFAFA'
AXES_FACECOLOR: Final[str] = '#FFFFFF'
TEXT_PRIMARY: Final[str] = '#1A1A1A'
TEXT_MUTED: Final[str] = '#5C5C5C'
DIVIDER_COLOR: Final[str] = '#D0D0D0'

FONT_STACK: Final[list[str]] = [
    'Inter', 'SF Pro Text', 'SF Pro Display', 'Helvetica Neue',
    'Segoe UI', 'system-ui', 'Arial', 'DejaVu Sans',
]


@dataclass(frozen=True)
class CenterSummary:
    """Per-center stats in a flat, plot-ready shape.

    Centers are sorted alphabetically so every panel's X axis ordering is
    identical and stable across years.

    ``adult_counts`` counts rows with role ``Adult``; ``young_adult_counts``
    counts ``Young Adult`` only (not combined).
    """

    centers: list[str]
    youth_counts: dict[str, int]
    adult_counts: dict[str, int]
    young_adult_counts: dict[str, int]
    year_counts: dict[str, dict[str, int]]
    gender_counts: dict[str, dict[str, int]]
    history_counts: dict[str, dict[str, int]]


def _category_counts_by_center(
    youth_df: pl.DataFrame,
    centers: list[str],
    column: str,
    valid: tuple[str, ...],
) -> dict[str, dict[str, int]]:
    """Group youth rows by ``(center, column)`` and inflate to a dense dict.

    Categories not in ``valid`` (typos, blanks) are dropped silently — the data
    is already validated upstream via :mod:`src.validate_clean_data`.
    """
    grouped = youth_df.group_by(['Center', column]).agg(pl.len().alias('n'))
    result: dict[str, dict[str, int]] = {c: dict.fromkeys(valid, 0) for c in centers}
    for row in grouped.iter_rows(named=True):
        center, category, count = row['Center'], row[column], row['n']
        if center in result and category in valid:
            result[center][category] = count
    return result


def compute_center_summary(assignments_df: pl.DataFrame) -> CenterSummary:
    """Aggregate per-center stats from a finalized assignments dataframe.

    Expects the writer schema: ``Center, Crew, Name, Role, Gender, Year, History``.
    Headcount splits ``Adult`` vs ``Young Adult`` roles; year / gender / history
    breakdowns are youth-only, matching :func:`print_crew_assignments`.
    """
    youth_df = assignments_df.filter(pl.col('Role') == 'Youth')

    youth_per_center = youth_df.group_by('Center').agg(pl.len().alias('youth'))
    adults_only = (
        assignments_df.filter(pl.col('Role') == 'Adult')
        .group_by('Center')
        .agg(pl.len().alias('adults'))
    )
    young_adults_only = (
        assignments_df.filter(pl.col('Role') == 'Young Adult')
        .group_by('Center')
        .agg(pl.len().alias('young_adults'))
    )
    centers_sorted = (
        youth_per_center.join(adults_only, on='Center', how='full', coalesce=True)
        .join(young_adults_only, on='Center', how='full', coalesce=True)
        .with_columns(
            pl.col('youth').fill_null(0),
            pl.col('adults').fill_null(0),
            pl.col('young_adults').fill_null(0),
        )
        .sort('Center')
    )

    centers: list[str] = centers_sorted.get_column('Center').to_list()
    youth_counts = dict(zip(centers, centers_sorted.get_column('youth').to_list(), strict=True))
    adult_counts = dict(zip(centers, centers_sorted.get_column('adults').to_list(), strict=True))
    young_adult_counts = dict(
        zip(centers, centers_sorted.get_column('young_adults').to_list(), strict=True),
    )

    year_counts = _category_counts_by_center(youth_df, centers, 'Year', YEAR_ORDER)
    gender_counts = _category_counts_by_center(youth_df, centers, 'Gender', GENDER_ORDER)
    history_counts = _category_counts_by_center(youth_df, centers, 'History', HISTORY_ORDER)

    return CenterSummary(
        centers=centers,
        youth_counts=youth_counts,
        adult_counts=adult_counts,
        young_adult_counts=young_adult_counts,
        year_counts=year_counts,
        gender_counts=gender_counts,
        history_counts=history_counts,
    )


def _dashboard_rc_params() -> dict[str, object]:
    """rcParams that override matplotlib defaults for a polished, report-ready look."""
    return {
        'figure.facecolor': FIG_FACECOLOR,
        'figure.dpi': 150,
        'savefig.dpi': 200,
        'savefig.facecolor': FIG_FACECOLOR,
        'savefig.bbox': 'tight',
        'font.family': 'sans-serif',
        'font.sans-serif': FONT_STACK,
        'font.size': 10.5,
        'axes.facecolor': AXES_FACECOLOR,
        'axes.edgecolor': '#BFBFBF',
        'axes.linewidth': 0.8,
        'axes.labelcolor': TEXT_PRIMARY,
        'axes.titlecolor': TEXT_PRIMARY,
        'axes.titlesize': 12.5,
        'axes.titleweight': '600',
        'axes.titlepad': 10,
        'axes.labelsize': 10.5,
        'axes.labelweight': '500',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.color': '#E5E5E5',
        'grid.linestyle': '-',
        'grid.linewidth': 0.6,
        'grid.alpha': 0.8,
        'xtick.color': TEXT_MUTED,
        'ytick.color': TEXT_MUTED,
        'xtick.labelsize': 9.5,
        'ytick.labelsize': 9.5,
        'xtick.major.size': 0,
        'ytick.major.size': 3,
        'legend.frameon': False,
        'legend.fontsize': 9.5,
        'legend.title_fontsize': 10,
    }


@contextmanager
def apply_dashboard_style() -> Iterator[None]:
    """Context manager that activates the polished dashboard rcParams."""
    with plt.rc_context(_dashboard_rc_params()):
        yield


def render_center_dashboard(
    assignments_csv: Path,
    output_path: Path,
    year: int,
    friend_scores: Mapping[str, float] | None = None,
    buddy_match_counts: Mapping[str, Mapping[int, int]] | None = None,
) -> None:
    """Render the per-center summary dashboard PNG from a finalized assignments CSV.

    ``friend_scores`` maps center name to per-youth normalized friend score (e.g.
    from :func:`calculate_friend_scores`); when omitted the friend-score panel
    shows a "buddy data not provided" placeholder so the surrounding panel grid
    stays balanced.

    ``buddy_match_counts`` maps center name to ``{0: n, 1: n, 2: n, 3: n}`` —
    the count of youth at that center with that many same-center buddy matches
    (e.g. from :func:`calculate_friend_match_buckets`). When omitted the
    matches panel is dropped entirely (keeping the figure compact).
    """
    if not assignments_csv.is_file():
        raise FileNotFoundError(f'Assignments CSV not found: {assignments_csv}')

    assignments_df = pl.read_csv(assignments_csv).with_columns(
        pl.col('Center').cast(pl.Utf8, strict=False),
        pl.col('Crew').cast(pl.Utf8, strict=False),
    )
    summary = compute_center_summary(assignments_df)

    if not summary.centers:
        print(f'Skipping {output_path}: no centers found in {assignments_csv}.')
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    buddy_str_counts = (
        _coerce_buddy_match_counts(buddy_match_counts, summary.centers)
        if buddy_match_counts is not None
        else None
    )
    with apply_dashboard_style():
        fig = _build_figure(summary, friend_scores or {}, buddy_str_counts, year, source=assignments_csv)
        fig.savefig(output_path)
        plt.close(fig)

    print(f'Center dashboard written to {output_path}')


def _coerce_buddy_match_counts(
    buddy_match_counts: Mapping[str, Mapping[int, int]],
    centers: list[str],
) -> dict[str, dict[str, int]]:
    """Convert int bucket keys to str so the shared stacked-bar plotter can consume them."""
    return {
        center: {
            str(bucket): int(buddy_match_counts.get(center, {}).get(bucket, 0))
            for bucket in (0, 1, 2, 3)
        }
        for center in centers
    }


def _build_figure(
    summary: CenterSummary,
    friend_scores: Mapping[str, float],
    buddy_match_counts: dict[str, dict[str, int]] | None,
    year: int,
    source: Path,
) -> Figure:
    has_buddy_panel = buddy_match_counts is not None
    if has_buddy_panel:
        mosaic = (
            'HF\n'
            'YB\n'
            'GI'
        )
    else:
        mosaic = (
            'HF\n'
            'YY\n'
            'GI'
        )

    fig, axes = plt.subplot_mosaic(
        mosaic,
        figsize=(13.5, 13.0),
        height_ratios=[1.0, 1.05, 1.0],
        gridspec_kw={'hspace': 0.55, 'wspace': 0.22},
    )

    _draw_headcount(axes['H'], summary)
    _draw_friend_score(axes['F'], summary, friend_scores)
    _draw_stacked_categories(
        axes['Y'], summary, summary.year_counts,
        order=YEAR_ORDER, colors=YEAR_COLORS,
        title='Class year mix  ·  % of youth (Fr to Sr)',
    )
    _draw_stacked_categories(
        axes['G'], summary, summary.gender_counts,
        order=GENDER_ORDER, colors=GENDER_COLORS,
        title='Gender mix  ·  % of youth',
    )
    _draw_stacked_categories(
        axes['I'], summary, summary.history_counts,
        order=HISTORY_ORDER, colors=HISTORY_COLORS,
        title='History mix  ·  % of youth (Vet vs New)',
    )
    if has_buddy_panel and buddy_match_counts is not None:
        _draw_stacked_categories(
            axes['B'], summary, buddy_match_counts,
            order=BUDDY_MATCH_ORDER, colors=BUDDY_MATCH_COLORS,
            title='Same-center buddy matches  ·  % of youth (0 to 3 friends together)',
        )

    fig.suptitle(
        f'Center summary  ·  Assignments {year}',
        fontsize=17, fontweight='600', color=TEXT_PRIMARY, y=0.998,
    )
    fig.text(
        0.5, 0.005,
        f'Source: {source.name}  ·  centers sorted alphabetically  ·  '
        '"All" reference shows cohort-wide ratio',
        ha='center', va='bottom', fontsize=8.5, color=TEXT_MUTED,
    )
    return fig


def _darker(hex_color: str, factor: float = 0.78) -> str:
    """Return a slightly darker variant of ``hex_color`` for crisp bar edges."""
    h = hex_color.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f'#{int(r * factor):02X}{int(g * factor):02X}{int(b * factor):02X}'


def _segment_label_color(hex_fill: str) -> str:
    """Pick white vs dark text for count annotations inside a filled segment."""
    h = hex_fill.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return '#FFFFFF' if luminance < 0.55 else TEXT_PRIMARY


def _draw_headcount(ax: Axes, summary: CenterSummary) -> None:
    """Stacked bars per center: youth, adults, young adults — counts inside segments."""
    centers = summary.centers
    x = np.arange(len(centers))
    width = 0.62

    counts_map: dict[str, dict[str, int]] = {
        c: {
            'Youth': summary.youth_counts[c],
            'Adult': summary.adult_counts[c],
            'Young Adult': summary.young_adult_counts[c],
        }
        for c in centers
    }

    bottoms = np.zeros(len(centers))
    annotate_frac_min = 0.06

    for role in HEADCOUNT_ORDER:
        heights = np.array([counts_map[c][role] for c in centers], dtype=float)
        color = HEADCOUNT_COLORS[role]
        ax.bar(
            x, heights, width=width, bottom=bottoms,
            label=HEADCOUNT_LEGEND_LABELS[role],
            color=color, edgecolor=_darker(color), linewidth=0.55,
        )
        label_clr = _segment_label_color(color)
        for xi_idx, (h, b) in enumerate(zip(heights, bottoms, strict=True)):
            cname = centers[xi_idx]
            total_bar = float(sum(counts_map[cname][r] for r in HEADCOUNT_ORDER))
            denom = total_bar if total_bar > 0 else 1.0
            if int(h) > 0 and float(h) / denom >= annotate_frac_min:
                ax.text(
                    float(x[xi_idx]), float(b) + float(h) / 2.0, str(int(h)),
                    ha='center', va='center', fontsize=8.5, color=label_clr,
                    fontweight='600',
                )
        bottoms += heights

    ymax = float(bottoms.max()) if len(bottoms) else 0.0
    pad = ymax * 0.055 if ymax > 0 else 0.5

    for xi_idx, total_top in enumerate(bottoms):
        total_people = int(round(total_top))
        if total_people <= 0:
            continue
        ax.text(
            float(x[xi_idx]), float(total_top) + pad * 0.08,
            str(total_people),
            ha='center', va='bottom', fontsize=9.5, color=TEXT_PRIMARY, fontweight='600',
        )

    ax.set_title('Headcount by center')
    ax.set_xticks(x)
    ax.set_xticklabels(centers)
    ax.set_ylabel('People')
    ax.set_ylim(0, ymax + pad if ymax > 0 else 1.0)
    ax.grid(axis='x', visible=False)
    ax.legend(
        loc='upper center', bbox_to_anchor=(0.5, -0.13),
        ncol=len(HEADCOUNT_ORDER), frameon=False,
    )
    ax.margins(x=0.06)


def _draw_friend_score(
    ax: Axes,
    summary: CenterSummary,
    friend_scores: Mapping[str, float],
) -> None:
    """Lollipop plot: head = per-center score, dotted line = simple mean."""
    centers = summary.centers
    if not friend_scores:
        ax.set_axis_off()
        ax.text(
            0.5, 0.6, 'Friend score',
            transform=ax.transAxes, ha='center', va='center',
            color=TEXT_PRIMARY, fontsize=12.5, fontweight='600',
        )
        ax.text(
            0.5, 0.42, 'Buddy data not provided',
            transform=ax.transAxes, ha='center', va='center',
            color=TEXT_MUTED, fontsize=10,
        )
        return

    x = np.arange(len(centers))
    scores = [float(friend_scores.get(c, 0.0)) for c in centers]
    mean = sum(scores) / len(scores) if scores else 0.0

    ax.vlines(x, ymin=0, ymax=scores, color=ACCENT_SECONDARY, linewidth=1.4)
    ax.scatter(
        x, scores, color=ACCENT_PRIMARY, s=85, zorder=3,
        edgecolors=_darker(ACCENT_PRIMARY), linewidths=0.8,
    )
    ax.axhline(mean, color=REFERENCE_LINE, linestyle=':', linewidth=1.2, zorder=1)
    ymax = max(scores + [mean]) if scores else 1.0
    ax.text(
        len(centers) - 0.55, mean, f'avg {mean:.2f}',
        color=TEXT_MUTED, fontsize=8.5, va='bottom', ha='right',
    )

    pad = ymax * 0.04 if ymax > 0 else 0.05
    for xi, s in zip(x, scores, strict=True):
        ax.text(float(xi), s + pad, f'{s:.2f}', ha='center', va='bottom',
                fontsize=9, color=TEXT_PRIMARY)

    ax.set_title('Friend score by center  ·  per-youth weighted')
    ax.set_xticks(x)
    ax.set_xticklabels(centers)
    ax.set_ylabel('Score')
    ax.set_ylim(0, ymax * 1.22 if ymax > 0 else 1.0)
    ax.grid(axis='x', visible=False)
    ax.margins(x=0.06)


def _stacked_columns_with_reference(
    counts: dict[str, dict[str, int]], centers: list[str], order: tuple[str, ...],
) -> tuple[list[str], np.ndarray]:
    """Build the 100% stacked matrix with an "All" reference column on the left.

    Returns ``(labels, share_matrix)`` where ``labels[0] == 'All'`` and
    ``share_matrix.shape == (len(order), len(labels))``.
    """
    labels: list[str] = [ALL_LABEL, *centers]
    cohort: dict[str, int] = dict.fromkeys(order, 0)
    for c in centers:
        for k in order:
            cohort[k] += counts[c].get(k, 0)

    share = np.zeros((len(order), len(labels)))
    for j, label in enumerate(labels):
        bucket = cohort if label == ALL_LABEL else counts[label]
        total = sum(bucket.values()) or 1
        for i, category in enumerate(order):
            share[i, j] = bucket.get(category, 0) / total * 100.0
    return labels, share


def _draw_stacked_categories(
    ax: Axes,
    summary: CenterSummary,
    counts: dict[str, dict[str, int]],
    order: tuple[str, ...],
    colors: dict[str, str],
    title: str,
) -> None:
    """100% stacked vertical bars per center, prefixed with an 'All' reference column."""
    labels, share = _stacked_columns_with_reference(counts, summary.centers, order)
    x = np.arange(len(labels))
    width = 0.62
    bottoms = np.zeros(len(labels))

    is_reference = np.array([label == ALL_LABEL for label in labels])

    for i, category in enumerate(order):
        seg = share[i]
        ax.bar(
            x[~is_reference], seg[~is_reference], width=width, bottom=bottoms[~is_reference],
            color=colors[category], edgecolor=_darker(colors[category]),
            linewidth=0.5, label=category,
        )
        ax.bar(
            x[is_reference], seg[is_reference], width=width, bottom=bottoms[is_reference],
            color=colors[category], edgecolor=_darker(colors[category]),
            linewidth=0.5, alpha=0.55,
        )
        for xi, s, b in zip(x, seg, bottoms, strict=True):
            if s >= 7.0:
                ax.text(float(xi), float(b) + float(s) / 2, f'{s:.0f}%',
                        ha='center', va='center',
                        fontsize=8.5, color='white', fontweight='600')
        bottoms += seg

    ax.axvline(0.5, color=DIVIDER_COLOR, linestyle='-', linewidth=0.8, zorder=0)

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.set_ylabel('% of youth')
    ax.grid(axis='x', visible=False)
    ax.legend(
        loc='upper center', bbox_to_anchor=(0.5, -0.13),
        ncol=len(order), frameon=False,
    )


