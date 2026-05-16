from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap, to_rgba
import networkx as nx  # type: ignore[import-untyped]
import numpy as np
import polars as pl
from community import community_louvain  # type: ignore

from src.analysis import (
    PersonCrew,
    SolverLike,
    build_name_to_center,
    calculate_youth_buddy_weights_by_name,
    is_person_at_center,
)
from src.models import Center, Youth

__all__ = (
    'is_person_at_center',
    'analyze_clusters',
    'detect_friend_clusters',
    'merge_friend_clusters_into_assignments_csv',
)


# Roster PNG + workbook: summed same-center roster buddy weights (4 + 2 + 1 max 7).
_MAX_PER_YOUTH_BUDDY_WEIGHT: float = 7.0
_SCORE_SURFACE_CMAP = LinearSegmentedColormap.from_list(
    'buddy_weight_seq',
    ['#f7fafc', '#d6e8f5', '#4a7598', '#1a4470'],
)


def _cmap_lut_rgba(cmap_key: str, n_colors: int) -> np.ndarray:
    """Evenly sample ``n_colors`` RGBA rows from matplotlib's named registry colormap."""
    if n_colors <= 0:
        return np.zeros((0, 4), dtype=float)
    cmap_obj = cast(Any, colormaps[cmap_key])
    return np.asarray(cmap_obj(np.linspace(0.0, 1.0, n_colors)), dtype=float)


# Qualitative hues (Paul Tol–style): strong separation on screen and in print; avoids Set3 pastels.
_CENTER_QUALITATIVE_HEX: tuple[str, ...] = (
    '#0077BB',
    '#EE7733',
    '#228833',
    '#CCBB44',
    '#AA3377',
    '#66CCEE',
    '#EE6677',
    '#009988',
    '#332288',
    '#CC6677',
    '#44AA99',
    '#882255',
    '#117733',
    '#DDCC77',
    '#6699CC',
    '#AA4499',
    '#997700',
    '#661100',
)


def _center_palette_rgba(n_colors: int) -> np.ndarray:
    """Distinct RGBA rows for center legends and roster cells (replaces ``Set3`` sampling)."""
    if n_colors <= 0:
        return np.zeros((0, 4), dtype=float)
    if n_colors <= len(_CENTER_QUALITATIVE_HEX):
        return np.asarray([to_rgba(h) for h in _CENTER_QUALITATIVE_HEX[:n_colors]], dtype=float)
    base = np.asarray([to_rgba(h) for h in _CENTER_QUALITATIVE_HEX], dtype=float)
    need_extra = n_colors - len(_CENTER_QUALITATIVE_HEX)
    tab20 = _cmap_lut_rgba('tab20', 20)
    order = list(range(0, 20, 2)) + list(range(1, 20, 2))
    extra = np.asarray([tab20[order[k % len(order)]] for k in range(need_extra)], dtype=float)
    return np.vstack([base, extra])


def _patch_rgba4(patch: Any) -> tuple[float, float, float, float]:
    """Primary facecolor of a patch as RGBA tuple (floats in [0, 1])."""
    flat = np.asarray(patch.get_facecolor(), dtype=float).reshape(-1)
    r = float(flat.flat[0])
    g = float(flat.flat[1])
    b = float(flat.flat[2])
    a = float(flat.flat[3]) if flat.size >= 4 else 1.0
    return (r, g, b, a)


def detect_friend_clusters(youth_list: list[Youth]) -> dict[str, int]:
    """Detect friend clusters using Louvain community detection.

    Uses only buddy form friend choices — completely independent of assignments.
    Edges are restricted to roster youth on both ends so leader picks (which
    appear in friend choice fields) do not introduce phantom nodes that would
    distort cluster sizes and cohesion scores.
    Returns mapping of youth name -> cluster_id.
    """
    roster: set[str] = {y.name for y in youth_list}
    G = nx.Graph()

    for youth in youth_list:
        G.add_node(youth.name)
        weights = {youth.first_choice: 4, youth.second_choice: 2, youth.third_choice: 1}
        for friend, weight in weights.items():
            if friend and friend in roster:
                G.add_edge(youth.name, friend, weight=weight)

    return community_louvain.best_partition(G, weight='weight')


def _friend_cluster_column_maps(
    clusters: dict[str, int],
    cohesion: dict[str, dict[str, Any]],
) -> tuple[dict[str, str], dict[str, str]]:
    """Map youth name → (viz-style label ``C{N}``, Louvain id string).

    ``N`` ranks clusters by descending size — same convention as :func:`render_cluster_roster_table`.
    """
    cluster_sizes = sorted(cohesion.items(), key=lambda x: x[1]['size'], reverse=True)
    cluster_keys_ordered = [cid for cid, _ in cluster_sizes]
    label_rank = {ck: i + 1 for i, ck in enumerate(cluster_keys_ordered)}
    display_by_name: dict[str, str] = {}
    raw_by_name: dict[str, str] = {}
    for name, nid in clusters.items():
        key = f'cluster_{nid}'
        rank = label_rank.get(key)
        if rank is None:
            continue
        display_by_name[name] = f'C{rank}'
        raw_by_name[name] = str(nid)
    return display_by_name, raw_by_name


def merge_friend_clusters_into_assignments_csv(
    path: Path | str,
    clusters: dict[str, int],
    cohesion: dict[str, dict[str, Any]],
    buddy_weights_by_name: Mapping[str, float] | None = None,
) -> None:
    """Add or replace ``FriendCluster`` / ``FriendClusterId`` (+ optional ``BuddyWeight``).

    Leader rows (anything other than role ``Youth``) get empty strings — friend
    clusters apply to roster youth only (:func:`detect_friend_clusters` input
    matches :func:`analyze_clusters`).

    ``BuddyWeight`` holds the summed same-center roster buddy preference weights
    (4+2+1) when ``buddy_weights_by_name`` is supplied; otherwise that column is
    omitted entirely.
    """
    if not cohesion:
        return
    csv_path = Path(path)
    if not csv_path.is_file():
        return

    df = pl.read_csv(csv_path)
    extra_cols = {'FriendCluster', 'FriendClusterId', 'BuddyWeight'}
    drop_existing = [c for c in df.columns if c in extra_cols]
    if drop_existing:
        df = df.drop(drop_existing)

    display_map, raw_map = _friend_cluster_column_maps(clusters, cohesion)

    friend_cluster: list[str] = []
    friend_cluster_id: list[str] = []
    buddy_weights_col: list[str] | None = [] if buddy_weights_by_name is not None else None
    for row in df.iter_rows(named=True):
        role = row.get('Role')
        role_str = str(role).strip() if role is not None else ''
        name = row.get('Name')
        name_str = str(name).strip() if name is not None else ''
        if role_str != 'Youth':
            friend_cluster.append('')
            friend_cluster_id.append('')
            if buddy_weights_col is not None:
                buddy_weights_col.append('')
            continue
        friend_cluster.append(display_map.get(name_str, ''))
        friend_cluster_id.append(raw_map.get(name_str, ''))
        if buddy_weights_col is not None:
            assert buddy_weights_by_name is not None
            wt = buddy_weights_by_name.get(name_str)
            buddy_weights_col.append('' if wt is None else str(int(round(wt))))

    out_series: list[pl.Series] = [
        pl.Series('FriendCluster', friend_cluster, dtype=pl.Utf8),
        pl.Series('FriendClusterId', friend_cluster_id, dtype=pl.Utf8),
    ]
    if buddy_weights_col is not None:
        out_series.append(pl.Series('BuddyWeight', buddy_weights_col, dtype=pl.Utf8))

    out = df.with_columns(out_series)
    out.write_csv(csv_path)
    print(f'Assignments workbook updated with friend cluster columns: {csv_path}')


def calculate_cluster_cohesion(
    clusters: dict[str, int],
    solver: SolverLike,
    person_crew: PersonCrew,
    centers: list[Center],
) -> dict[str, dict[str, Any]]:
    """Analyze how well clusters were kept together in center assignments.

    Returns per-cluster metrics including:
      - cluster_size: Total members in cluster
      - center_distribution: {center_name: count} showing where members landed
      - cohesion_score: Fraction of cluster in the most common center
    """
    cluster_members: dict[int, list[str]] = defaultdict(list)
    for name, cluster_id in clusters.items():
        cluster_members[cluster_id].append(name)

    results = {}
    for cluster_id, members in cluster_members.items():
        center_counts: dict[str, int] = defaultdict(int)
        for name in members:
            for center in centers:
                if is_person_at_center(solver, person_crew, name, center):
                    center_counts[center.name] += 1
                    break

        max_in_center = max(center_counts.values()) if center_counts else 0
        results[f'cluster_{cluster_id}'] = {
            'size': len(members),
            'center_distribution': dict(center_counts),
            'cohesion_score': max_in_center / len(members) if members else 0,
        }

    return results


def _build_friend_graph(youth_list: list[Youth]) -> nx.DiGraph:
    """Build directed graph of friend choices, restricted to roster youth on both ends."""
    roster: set[str] = {y.name for y in youth_list}
    G = nx.DiGraph()

    for youth in youth_list:
        G.add_node(youth.name)
        choices = [
            (youth.first_choice, 4),
            (youth.second_choice, 2),
            (youth.third_choice, 1),
        ]
        for friend, weight in choices:
            if friend and friend in roster:
                G.add_edge(youth.name, friend, weight=weight)

    return G


def visualize_cluster_distribution(
    clusters: dict[str, int],
    cohesion_data: dict[str, dict[str, Any]],
    centers: list[Center],
    solver: SolverLike,
    person_crew: PersonCrew,
    youth_list: list[Youth],
    output_path: str = 'cluster_analysis.png',
) -> None:
    """Create comprehensive 4-panel visualization of friend clusters and center assignments.

    Panel 1: Bar chart showing cluster distribution across centers with cohesion scores
    Panel 2: Network graph showing friend connections, colored by center assignment
    Panel 3: Grid of clusters (5 per row), each showing members colored by center
    Panel 4: Columns of centers (side-by-side), each showing members colored by cluster
    """
    if not centers:
        print(f'Skipping {output_path}: cluster visualization requires at least one center.')
        return

    # Get unique cluster IDs and sort by size
    cluster_sizes = [(cid, data['size']) for cid, data in cohesion_data.items()]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)
    cluster_ids = [cid for cid, _ in cluster_sizes]

    # Get center names and create color mapping
    center_names = [c.name for c in centers]
    center_colors = _center_palette_rgba(len(center_names))
    center_color_map = {name: center_colors[i] for i, name in enumerate(center_names)}

    # Build mapping of name -> center assignment
    name_to_center = {}
    for youth in youth_list:
        for center in centers:
            if is_person_at_center(solver, person_crew, youth.name, center):
                name_to_center[youth.name] = center.name
                break

    # Build cluster members dict for layout calculations
    cluster_members: dict[int, list[str]] = defaultdict(list)
    for name, louvain_cluster in clusters.items():
        cluster_members[louvain_cluster].append(name)

    # Calculate dynamic heights based on content
    clusters_per_row = 5
    num_cluster_rows = int(np.ceil(len(cluster_members) / clusters_per_row))
    max_cluster_size = max(len(members) for members in cluster_members.values())

    # For centers: find the tallest center column
    max_center_size = max(len([n for n, c in name_to_center.items() if c == cn]) for cn in center_names)

    # Calculate inches needed for each panel
    bar_chart_height = 6  # More space for bar chart
    network_height = 10  # More space for network graph
    # Clusters: each row needs space for title + members
    clusters_height = max(6, num_cluster_rows * (max_cluster_size * 0.15 + 0.6))
    # Centers: need to account for cluster sub-headers within each center
    # Estimate: max center size + extra for cluster grouping overhead
    num_unique_clusters = len(set(clusters.values()))
    # Assume clusters are somewhat evenly distributed, add overhead per estimated cluster per center
    estimated_clusters_per_center = max(3, num_unique_clusters // len(center_names))
    # Revert to reasonable height
    centers_height = max(14, max_center_size * 0.25 + estimated_clusters_per_center * 0.3 + 3)

    # Total figure height with spacing
    total_height = bar_chart_height + network_height + clusters_height + centers_height + 2

    # Use reasonable ratios that preserve proportions
    fig = plt.figure(figsize=(24, total_height))
    gs = fig.add_gridspec(4, 1, height_ratios=[bar_chart_height, network_height, clusters_height, centers_height], hspace=0.15)

    # ===== PANEL 1: Bar Chart =====
    ax_bar = fig.add_subplot(gs[0])

    # Create matrix for stacked bar chart
    data_matrix = np.zeros((len(cluster_ids), len(center_names)))
    for irow, cohesion_k in enumerate(cluster_ids):
        center_dist = cohesion_data[cohesion_k]['center_distribution']
        for j, center_nm in enumerate(center_names):
            data_matrix[irow, j] = center_dist.get(center_nm, 0)

    x = np.arange(len(cluster_ids))
    width = 0.8
    bottom = np.zeros(len(cluster_ids))

    for j_c, center_nm in enumerate(center_names):
        values = data_matrix[:, j_c]
        ax_bar.bar(x, values, width, label=center_nm, bottom=bottom, color=center_colors[j_c])
        bottom += values

    # Add cohesion scores as annotations
    for irow, cohesion_k in enumerate(cluster_ids):
        cohesion = cohesion_data[cohesion_k]['cohesion_score']
        size = cohesion_data[cohesion_k]['size']
        ax_bar.text(irow, bottom[irow] + 0.5, f'{cohesion:.1%}\n(n={size})', ha='center', va='bottom', fontsize=8)

    ax_bar.set_xlabel('Friend Cluster', fontsize=12)
    ax_bar.set_ylabel('Number of Youth', fontsize=12)
    ax_bar.set_title('Friend Cluster Distribution Across Centers\n(Percentage = Cohesion Score)', fontsize=14, pad=20)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f'C{i + 1}' for i in range(len(cluster_ids))], rotation=0)
    ax_bar.legend(title='Centers', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax_bar.grid(axis='y', alpha=0.3)

    # ===== PANEL 2: Network Graph =====
    ax_network = fig.add_subplot(gs[1])

    # Build directed graph
    G = _build_friend_graph(youth_list)

    # Position clusters in a grid with boundaries
    pos = {}
    num_clusters = len(cluster_members)
    cols = int(np.ceil(np.sqrt(num_clusters)))
    cluster_positions = {}  # Store cluster centers for labels

    for idx, (cluster_id, members) in enumerate(sorted(cluster_members.items(), key=lambda x: len(x[1]), reverse=True)):
        row = idx // cols
        col = idx % cols
        center_x = col * 10
        center_y = -row * 10
        cluster_positions[cluster_id] = (center_x, center_y)

        # Create subgraph for this cluster
        subgraph = G.subgraph(members)
        if len(members) > 1:
            sub_pos = nx.spring_layout(subgraph, k=1.2, iterations=50, seed=42)
            # Scale and translate
            for node in sub_pos:
                pos[node] = (center_x + sub_pos[node][0] * 4, center_y + sub_pos[node][1] * 4)
        else:
            # Single node
            pos[members[0]] = (center_x, center_y)

    # Draw cluster boundaries (subtle rectangles)
    for idx, (cluster_id, members) in enumerate(sorted(cluster_members.items(), key=lambda x: len(x[1]), reverse=True)):
        if len(members) == 0:
            continue
        # Get bounding box for cluster
        member_positions = [pos[m] for m in members if m in pos]
        if member_positions:
            xs = [p[0] for p in member_positions]
            ys = [p[1] for p in member_positions]
            min_x, max_x = min(xs) - 1, max(xs) + 1
            min_y, max_y = min(ys) - 1, max(ys) + 1
            width = max_x - min_x
            height = max_y - min_y

            # Draw boundary rectangle
            rect = plt.Rectangle((min_x, min_y), width, height, facecolor='lightgray', edgecolor='darkgray', linewidth=1.5, alpha=0.15, zorder=0)
            ax_network.add_patch(rect)

            # Add cluster label
            cluster_label = f'C{idx + 1}'
            ax_network.text(
                min_x + width / 2,
                max_y + 0.3,
                cluster_label,
                fontsize=9,
                ha='center',
                weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'),
            )

    # Draw nodes colored by center assignment
    for center_name, color in center_color_map.items():
        nodes_in_center = [name for name in G.nodes() if name_to_center.get(name) == center_name]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes_in_center, node_color=[color], node_size=300, ax=ax_network, alpha=0.8)

    # Draw edges with varying width based on friend choice weight
    for edge in G.edges():
        weight = G[edge[0]][edge[1]]['weight']
        nx.draw_networkx_edges(G, pos, edgelist=[edge], width=weight * 0.5, alpha=0.3, arrows=True, arrowsize=10, ax=ax_network, edge_color='gray')

    # Draw labels with smaller font and background
    nx.draw_networkx_labels(G, pos, font_size=5, font_weight='normal', ax=ax_network)

    ax_network.set_title('Social Network: Friend Choices (colored by center assignment)', fontsize=14, pad=20)
    ax_network.axis('off')

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=name) for name, color in center_color_map.items()
    ]
    ax_network.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # ===== PANEL 3: CLUSTERS Grid Layout =====
    ax_clusters = fig.add_subplot(gs[2])
    ax_clusters.axis('off')
    ax_clusters.set_xlim(0, 1)
    ax_clusters.set_ylim(0, 1)

    # Create cluster color map (generate distinct colors for clusters)
    cluster_color_map = {}
    unique_clusters = sorted(set(clusters.values()))

    # Use multiple colormaps to ensure enough distinct colors
    if len(unique_clusters) <= 20:
        cluster_colors_palette = _cmap_lut_rgba('tab20', len(unique_clusters))
    else:
        # Combine tab20 with tab20b and tab20c for more colors
        colors1 = _cmap_lut_rgba('tab20', 20)
        tail = len(unique_clusters) - 20
        colors2 = _cmap_lut_rgba('tab20b', min(20, tail))
        if len(unique_clusters) > 40:
            colors3 = _cmap_lut_rgba('tab20c', len(unique_clusters) - 40)
            cluster_colors_palette = np.vstack([colors1, colors2, colors3])
        else:
            cluster_colors_palette = np.vstack([colors1, colors2])

    for i, cluster_id in enumerate(unique_clusters):
        cluster_color_map[cluster_id] = cluster_colors_palette[i]

    # Title for clusters panel
    ax_clusters.text(
        0.5, 0.98, 'CLUSTERS (boxes colored by center assignment)', fontsize=11, weight='bold', transform=ax_clusters.transAxes, ha='center'
    )

    # Layout configuration for cluster grid (clusters_per_row already defined above)
    col_width = 1.0 / clusters_per_row
    box_height = 0.01
    box_width = col_width * 0.9  # Leave some margin

    y_start = 0.93
    row_start_y = y_start  # Track where each row starts
    row_min_y: dict[int, float] = {}  # Track the minimum y for each row (to know where next row should start)

    for idx_ck, cohesion_k in enumerate(cluster_ids):
        col = idx_ck % clusters_per_row
        row = idx_ck // clusters_per_row

        x_base = col * col_width + 0.01

        cluster_label = f'C{idx_ck + 1}'
        cluster_num = int(cohesion_k.split('_')[1])
        members = sorted(cluster_members[cluster_num])

        # Calculate y position for this cluster
        if col == 0:
            # Start a new row - position it below the previous row's longest cluster
            if row > 0:
                row_start_y = row_min_y.get(row - 1, row_start_y) - 0.03
            row_min_y[row] = row_start_y

        y_pos = row_start_y

        # Cluster header
        ax_clusters.text(
            x_base + box_width / 2,
            y_pos,
            f'{cluster_label} ({len(members)})',
            fontsize=7,
            weight='bold',
            transform=ax_clusters.transAxes,
            ha='center',
        )

        y_current = y_pos - 0.015

        # Draw each member as a colored box
        for name in members:
            assigned_cn = name_to_center.get(name)
            color_any: Any = center_color_map.get(assigned_cn, 'gray') if assigned_cn else 'gray'

            # Draw rectangle
            rect = plt.Rectangle(
                (x_base, y_current - box_height / 2),
                box_width,
                box_height,
                facecolor=color_any,
                edgecolor='black',
                linewidth=0.5,
                transform=ax_clusters.transAxes,
                alpha=0.9,
            )
            ax_clusters.add_patch(rect)

            # Draw name on top of rectangle
            ax_clusters.text(
                x_base + box_width / 2, y_current, name, fontsize=5, ha='center', va='center', transform=ax_clusters.transAxes, color='black'
            )
            y_current -= 0.012

        # Update the minimum y for this row
        row_min_y[row] = min(row_min_y.get(row, y_current), y_current)

    # ===== PANEL 4: CENTERS Layout =====
    ax_centers = fig.add_subplot(gs[3])
    ax_centers.axis('off')
    ax_centers.set_xlim(0, 1)
    ax_centers.set_ylim(0, 1)

    # Title for centers panel
    ax_centers.text(0.5, 0.98, 'CENTERS (boxes colored by cluster)', fontsize=11, weight='bold', transform=ax_centers.transAxes, ha='center')

    # Layout configuration for center columns
    num_centers = len(center_names)
    center_col_width = 1.0 / num_centers
    center_box_width = center_col_width * 0.9
    centers_y_start = 0.96  # Start higher to use more available space

    # Draw each center as a column, grouped by cluster
    for center_idx, center_name in enumerate(center_names):
        x_base = center_idx * center_col_width + 0.01
        names_in_center = [nm for nm, ctr_nm in name_to_center.items() if ctr_nm == center_name]

        # Group names by cluster
        center_clusters: dict[int, list[str]] = defaultdict(list)
        for name in names_in_center:
            cluster_id = clusters.get(name, -1)
            center_clusters[cluster_id].append(name)

        # Sort clusters by size (largest first) and then by cluster_id
        sorted_clusters = sorted(center_clusters.items(), key=lambda x: (-len(x[1]), x[0]))

        # Center header
        y_current = centers_y_start
        ax_centers.text(
            x_base + center_box_width / 2,
            y_current,
            f'{center_name} ({len(names_in_center)})',
            fontsize=7,
            weight='bold',
            transform=ax_centers.transAxes,
            ha='center',
        )

        y_current -= 0.025  # Reduced from 0.03

        # Draw each cluster group within this center
        for partition_id, members_of_partition in sorted_clusters:
            cluster_label_vis: str
            stripe_color: Any
            # Get cluster label and color
            if partition_id != -1 and partition_id in cluster_color_map:
                stripe_color = cluster_color_map[partition_id]
                cluster_label_candidate: str | None = None
                for ck_idx, cohesion_key_ck in enumerate(cluster_ids):
                    if int(cohesion_key_ck.split('_')[1]) == partition_id:
                        cluster_label_candidate = f'C{ck_idx + 1}'
                        break
                cluster_label_vis = cluster_label_candidate or 'C?'
            else:
                stripe_color = 'lightgray'
                cluster_label_vis = 'Other'

            # Cluster sub-header
            ax_centers.text(
                x_base + center_box_width / 2,
                y_current,
                cluster_label_vis,
                fontsize=6,
                weight='bold',
                style='italic',
                transform=ax_centers.transAxes,
                ha='center',
                color='darkgray',
            )
            y_current -= 0.015  # Reduced from 0.018

            # Draw each member in this cluster (alphabetically)
            for nm in sorted(members_of_partition):
                # Check if we have enough space (bottom of box should stay above 0.01)
                if y_current - box_height / 2 < 0.01:
                    print(f'Warning: Ran out of space in {center_name} column. Increase centers_height.')
                    break

                # Draw rectangle
                rect = plt.Rectangle(
                    (x_base, y_current - box_height / 2),
                    center_box_width,
                    box_height,
                    facecolor=stripe_color,
                    edgecolor='black',
                    linewidth=0.5,
                    transform=ax_centers.transAxes,
                    alpha=0.9,
                )
                ax_centers.add_patch(rect)

                # Draw name on top of rectangle
                ax_centers.text(
                    x_base + center_box_width / 2,
                    y_current,
                    nm,
                    fontsize=5,
                    ha='center',
                    va='center',
                    transform=ax_centers.transAxes,
                    color='black',
                )
                y_current -= 0.0095  # Reduced from 0.012 to fit more names

            # Add small spacing between cluster groups
            y_current -= 0.008

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'Cluster visualization saved to {output_path}')


def _cluster_id_to_palette_map(unique_clusters: list[int]) -> dict[int, Any]:
    """Map Louvain cluster ids to distinct RGBA colors (same scheme as visualize_cluster_distribution)."""
    n = len(unique_clusters)
    if n <= 20:
        palette = _cmap_lut_rgba('tab20', n)
    else:
        colors1 = _cmap_lut_rgba('tab20', 20)
        tail_len = min(20, n - 20)
        colors2 = _cmap_lut_rgba('tab20b', tail_len)
        if n > 40:
            colors3 = _cmap_lut_rgba('tab20c', n - 40)
            palette = np.vstack([colors1, colors2, colors3])
        else:
            palette = np.vstack([colors1, colors2])
    return dict(zip(unique_clusters, palette, strict=True))


def _cell_text_color(bg: tuple[float, float, float, float]) -> tuple[float, float, float]:
    """Pick black or white text for contrast against ``bg`` RGBA tuple."""
    r, g, b = bg[0], bg[1], bg[2]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return (0.0, 0.0, 0.0) if luminance > 0.55 else (1.0, 1.0, 1.0)


def render_cluster_roster_table(
    clusters: dict[str, int],
    cohesion_data: dict[str, dict[str, Any]],
    centers: list[Center],
    solver: SolverLike,
    person_crew: PersonCrew,
    youth_list: list[Youth],
    output_path: str = 'cluster_roster.png',
    *,
    buddy_weights: Mapping[str, float] | None = None,
) -> None:
    """One row per youth, grouped by friend cluster.

    Layout: Louvain cluster badge (far left); narrow summed buddy-weight cell
    (same-center roster 4/2/1); then roster columns ending with name-colored
    by assigned center.
    """
    if not centers or not youth_list:
        print(f'Skipping {output_path}: roster table requires at least one center and one youth.')
        return

    center_names = [c.name for c in centers]
    center_colors_arr = _center_palette_rgba(len(center_names))
    center_color_map = {name: center_colors_arr[i] for i, name in enumerate(center_names)}
    name_to_center_full = build_name_to_center(solver, person_crew, centers)

    cluster_sizes = [(cid, data['size']) for cid, data in cohesion_data.items()]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)
    cluster_ids_ordered = [cid for cid, _ in cluster_sizes]

    cluster_label_by_cohesion: dict[str, int] = {ck: i + 1 for i, ck in enumerate(cluster_ids_ordered)}

    cluster_numeric_by_key: dict[str, int] = {cid: int(cid.split('_')[1]) for cid in cluster_ids_ordered}

    youths_by_numeric: dict[int, list[str]] = defaultdict(list)
    for nm, nid in clusters.items():
        youths_by_numeric[nid].append(nm)

    ordered_rows: list[Youth] = []
    for cohesion_id in cluster_ids_ordered:
        nid = cluster_numeric_by_key[cohesion_id]
        for nm in sorted(youths_by_numeric[nid]):
            y = next((x for x in youth_list if x.name == nm), None)
            if y is not None:
                ordered_rows.append(y)

    n_data = len(ordered_rows)
    row_height = 0.32
    legend_y0 = (n_data + 1) * row_height
    total_h = legend_y0 + row_height
    # CSV-like layout: narrow figure so a row reads in a single horizontal scan,
    # constant row height in inches so names never overlap regardless of roster size.
    fig_width = 14.75
    per_row_in = 0.28
    fig_height_in = max(6.0, per_row_in * (n_data + 3))

    LABEL_COL_LEFT = -0.52
    LABEL_COL_RIGHT = 0.0
    SCORE_X0 = 0.03
    DATA_REGION_ORIGIN = 0.26
    DATA_X1 = DATA_REGION_ORIGIN + 6.0 + 0.04

    def cell_col_x0(col_idx: int) -> float:
        """Left inset for roster column ``col_idx`` (0=name, …, 5=siblings)."""
        return DATA_REGION_ORIGIN + float(col_idx) + 0.02

    fig = plt.figure(figsize=(fig_width, fig_height_in))
    ax = fig.add_axes((0.04, 0.03, 0.92, 0.93))
    ax.set_xlim(LABEL_COL_LEFT, DATA_X1)
    ax.set_ylim(0, total_h)
    ax.axis('off')

    UNKNOWN_BG = '#D3D3D3'
    headers = ('Name', 'First Buddy', 'Second Buddy', 'Third Buddy', 'Parent', 'Siblings')
    header_y0 = (n_data) * row_height
    fontsize_header = 11
    fontsize_cell = 8

    def cohesion_key(name: str) -> str | None:
        nid = clusters.get(name, -1)
        if nid < 0:
            return None
        return f'cluster_{nid}'

    cluster_row_spans: list[tuple[int, int, str | None]] = []
    if n_data:
        seg_start = 0
        prev_ck = cohesion_key(ordered_rows[0].name)
        for ri in range(1, n_data):
            ck = cohesion_key(ordered_rows[ri].name)
            if ck != prev_ck:
                cluster_row_spans.append((seg_start, ri - 1, prev_ck))
                seg_start = ri
                prev_ck = ck
        cluster_row_spans.append((seg_start, n_data - 1, prev_ck))

    # Center color legend (above column headers)
    leg_rect = plt.Rectangle(
        (LABEL_COL_LEFT + 0.02, legend_y0 + 0.02 * row_height),
        LABEL_COL_RIGHT - LABEL_COL_LEFT - 0.04,
        row_height * 0.96,
        facecolor='#F0F0F0',
        edgecolor='black',
        linewidth=0.8,
        clip_on=False,
    )
    ax.add_patch(leg_rect)
    ax.text(
        (LABEL_COL_LEFT + LABEL_COL_RIGHT) / 2,
        legend_y0 + row_height / 2,
        'Centers',
        ha='center',
        va='center',
        fontsize=max(8, fontsize_header - 2),
        fontweight='bold',
        clip_on=False,
    )
    n_cent = len(center_names)
    grid_left = DATA_REGION_ORIGIN
    grid_w = DATA_X1 - grid_left
    if n_cent:
        slot_w = grid_w / n_cent
        pad_x = 0.05
        swatch_w = min(0.12, max(0.06, slot_w * 0.2))
        swatch_h = row_height * 0.55
        swatch_y = legend_y0 + row_height * 0.22
        fontsize_legend = max(6, min(9, int(52 / max(n_cent, 1)) + 3))
        for i, cnm in enumerate(center_names):
            x_slot = grid_left + i * slot_w
            rgba = center_color_map[cnm]
            sw = plt.Rectangle(
                (x_slot + pad_x, swatch_y),
                swatch_w,
                swatch_h,
                facecolor=rgba,
                edgecolor='0.25',
                linewidth=0.55,
                clip_on=False,
                zorder=6,
            )
            ax.add_patch(sw)
            tx = x_slot + pad_x + swatch_w + 0.04
            ax.text(
                tx,
                legend_y0 + row_height / 2,
                cnm,
                ha='left',
                va='center',
                fontsize=fontsize_legend,
                color='0.1',
                clip_on=False,
                zorder=6,
            )
    leg_row_outline = plt.Rectangle(
        (grid_left, legend_y0),
        grid_w,
        row_height,
        facecolor='none',
        edgecolor='black',
        linewidth=0.8,
        clip_on=False,
        zorder=5,
    )
    ax.add_patch(leg_row_outline)

    # Cluster id column header
    h_rect = plt.Rectangle(
        (LABEL_COL_LEFT + 0.02, header_y0 + 0.02 * row_height),
        LABEL_COL_RIGHT - LABEL_COL_LEFT - 0.04,
        row_height * 0.96,
        facecolor='#E8E8E8',
        edgecolor='black',
        linewidth=0.8,
        clip_on=False,
    )
    ax.add_patch(h_rect)
    ax.text(
        (LABEL_COL_LEFT + LABEL_COL_RIGHT) / 2,
        header_y0 + row_height / 2,
        'Cluster',
        ha='center',
        va='center',
        fontsize=max(9, fontsize_header - 2),
        fontweight='bold',
        clip_on=False,
    )

    # Buddy-weight narrow header (summed roster same-center picks; max 7)
    sc_w_header = DATA_REGION_ORIGIN - SCORE_X0 - 0.01
    s_head = plt.Rectangle(
        (SCORE_X0, header_y0 + 0.02 * row_height),
        max(0.05, sc_w_header),
        row_height * 0.96,
        facecolor='#E8F0F6',
        edgecolor='black',
        linewidth=0.8,
        clip_on=False,
    )
    ax.add_patch(s_head)
    ax.text(
        SCORE_X0 + max(0.05, sc_w_header) / 2,
        header_y0 + row_height / 2,
        'Wt',
        ha='center',
        va='center',
        fontsize=max(8, fontsize_header - 3),
        fontweight='bold',
        color='0.2',
        clip_on=False,
    )

    for col, title in enumerate(headers):
        bx = cell_col_x0(col)
        bw = 0.96
        rect = plt.Rectangle(
            (bx, header_y0 + 0.02 * row_height),
            bw,
            row_height * 0.96,
            facecolor='#E8E8E8',
            edgecolor='black',
            linewidth=0.8,
            transform=None,
            clip_on=False,
        )
        ax.add_patch(rect)
        ax.text(
            bx + bw / 2,
            header_y0 + row_height / 2,
            title,
            ha='center',
            va='center',
            fontsize=fontsize_header,
            fontweight='bold',
            clip_on=False,
        )

    def draw_colored_stripes(
        col: int,
        y_bottom: float,
        stripe_height: float,
        names_with_centers: list[tuple[str, str | None]],
    ) -> None:
        """Draw vertically stacked stripes in one cell."""
        bx = cell_col_x0(col)
        bw = 0.96
        if not names_with_centers:
            return
        for si, (label, center_nm) in enumerate(names_with_centers):
            sty = y_bottom + si * stripe_height
            face: Any = UNKNOWN_BG
            if center_nm is not None and center_nm in center_color_map:
                face = center_color_map[center_nm]
            stripe_rect = plt.Rectangle(
                (bx, sty),
                bw,
                stripe_height * 0.98,
                facecolor=face,
                edgecolor='black',
                linewidth=0.4,
                clip_on=False,
            )
            ax.add_patch(stripe_rect)
            tc = _cell_text_color(_patch_rgba4(stripe_rect))
            ax.text(
                bx + bw / 2,
                sty + stripe_height / 2,
                label,
                ha='center',
                va='center',
                fontsize=fontsize_cell - 1,
                color=tc,
                clip_on=False,
            )

    for ri, youth in enumerate(ordered_rows):
        y_bottom = (n_data - 1 - ri) * row_height
        youth_center = name_to_center_full.get(youth.name)

        buddy_triples: list[tuple[str | None, str | None]] = []
        for pick in (youth.first_choice, youth.second_choice, youth.third_choice):
            trimmed = pick.strip() if pick else ''
            buddy_triples.append(
                ((trimmed, name_to_center_full.get(trimmed)) if trimmed else (None, None)),
            )

        parents: list[tuple[str, str | None]] = []
        for n in youth.parent_names_list:
            piece = n.strip()
            if not piece:
                continue
            p_center = name_to_center_full.get(piece)
            if p_center is None:
                p_center = youth_center
            parents.append((piece, p_center))
        sibling_entries: list[tuple[str, str | None]] = [
            (piece, name_to_center_full.get(piece))
            for n in youth.siblings_list
            if (piece := n.strip())
        ]

        wt_val = buddy_weights.get(youth.name) if buddy_weights is not None else None
        score_w_eff = DATA_REGION_ORIGIN - SCORE_X0 - 0.01
        pad_inner = y_bottom + 0.02 * row_height
        inner_score_h = row_height * 0.92
        score_label = '—'
        if buddy_weights is None:
            sw_fill: Any = '#EEEEEE'
        elif wt_val is None:
            sw_fill = UNKNOWN_BG
        else:
            t_norm = float(min(max(wt_val, 0.0), _MAX_PER_YOUTH_BUDDY_WEIGHT))
            rgba_tpl = tuple(_SCORE_SURFACE_CMAP(0.12 + (t_norm / _MAX_PER_YOUTH_BUDDY_WEIGHT) * 0.82))
            sw_fill = (float(rgba_tpl[0]), float(rgba_tpl[1]), float(rgba_tpl[2]), 1.0)
            score_label = str(int(round(float(wt_val))))

        sq = plt.Rectangle(
            (SCORE_X0, pad_inner),
            score_w_eff,
            inner_score_h,
            facecolor=sw_fill,
            edgecolor='black',
            linewidth=0.55,
            clip_on=False,
        )
        ax.add_patch(sq)
        sr, sg, sb, _sa = _patch_rgba4(sq)
        score_txt_clr = _cell_text_color((sr, sg, sb, _sa))
        ax.text(
            SCORE_X0 + score_w_eff / 2.0,
            y_bottom + row_height / 2.0,
            score_label,
            ha='center',
            va='center',
            fontsize=max(7, fontsize_cell - 2),
            fontweight='700',
            color=score_txt_clr,
            clip_on=False,
        )

        cell_bw = 0.96
        for col in range(6):
            bx = cell_col_x0(col)
            outer = plt.Rectangle(
                (bx, y_bottom), cell_bw, row_height, facecolor='none',
                edgecolor='black', linewidth=0.6, clip_on=False,
            )
            ax.add_patch(outer)

        # Name column — assigned center (not friend-cluster hue)
        ncol = 0
        cx_name = cell_col_x0(ncol)
        if youth_center is not None and youth_center in center_color_map:
            name_face: Any = center_color_map[youth_center]
        else:
            name_face = UNKNOWN_BG
        nrect = plt.Rectangle(
            (cx_name + 0.02, y_bottom + 0.01 * row_height),
            cell_bw - 0.035,
            row_height * 0.88,
            facecolor=name_face,
            edgecolor='none',
            clip_on=False,
        )
        ax.add_patch(nrect)
        nt = _cell_text_color(_patch_rgba4(nrect))
        ax.text(
            cx_name + cell_bw / 2,
            y_bottom + row_height / 2,
            youth.name,
            ha='center',
            va='center',
            fontsize=fontsize_cell,
            fontweight='600',
            color=nt,
            clip_on=False,
        )

        # Buddy columns — single stripe or empty cell (no interior fill when empty per plan)
        inner_h = row_height * 0.92
        y_pad = y_bottom + 0.04 * row_height
        for buddy_col, (buddy_name, buddy_center) in zip((1, 2, 3), buddy_triples, strict=True):
            if buddy_name:
                draw_colored_stripes(buddy_col, y_pad, inner_h, [(buddy_name, buddy_center)])

        # Parent column
        pcol = 4
        if parents:
            sh = (row_height * 0.92) / len(parents)
            y_in = y_bottom + 0.04 * row_height
            draw_colored_stripes(pcol, y_in, sh, [(lab, cen) for lab, cen in parents])

        # Siblings column
        scol = 5
        if sibling_entries:
            sh = (row_height * 0.92) / len(sibling_entries)
            y_in = y_bottom + 0.04 * row_height
            draw_colored_stripes(scol, y_in, sh, [(lab, cen) for lab, cen in sibling_entries])

    # Bold separators between friend clusters (full width including label column)
    for ri in range(n_data - 1):
        k0 = cohesion_key(ordered_rows[ri].name)
        k1 = cohesion_key(ordered_rows[ri + 1].name)
        if k0 != k1:
            y_line = (n_data - 1 - ri) * row_height
            ax.plot(
                [LABEL_COL_LEFT, DATA_X1],
                [y_line, y_line],
                color='black',
                linewidth=3.0,
                solid_capstyle='butt',
                clip_on=False,
                zorder=15,
            )

    # Cluster id labels (C1, C2, …) centered vertically on each block
    fontsize_cluster_badge = max(8, min(11, fontsize_cell + 1))
    for start_ri, end_ri, ck in cluster_row_spans:
        y_block_top = (n_data - 1 - start_ri) * row_height + row_height
        y_block_bot = (n_data - 1 - end_ri) * row_height
        y_mid = (y_block_top + y_block_bot) / 2.0
        if ck is not None and ck in cluster_label_by_cohesion:
            badge = f'C{cluster_label_by_cohesion[ck]}'
        else:
            badge = '—'
        ax.text(
            (LABEL_COL_LEFT + LABEL_COL_RIGHT) / 2,
            y_mid,
            badge,
            ha='center',
            va='center',
            fontsize=fontsize_cluster_badge,
            fontweight='bold',
            color='0.15',
            clip_on=False,
            zorder=5,
        )

    ax.plot(
        [LABEL_COL_RIGHT, LABEL_COL_RIGHT],
        [0.0, legend_y0 + row_height],
        color='0.35',
        linewidth=1.0,
        clip_on=False,
        zorder=4,
    )
    ax.plot(
        [DATA_REGION_ORIGIN, DATA_REGION_ORIGIN],
        [0.0, legend_y0 + row_height],
        color='0.35',
        linewidth=1.0,
        clip_on=False,
        zorder=4,
    )

    fig.suptitle(
        'Youth roster by friend cluster (Wt = summed same-center buddy weight 0–7; '
        "name column = assigned center color; buddies / parent / siblings = that person's center; "
        'thick rules separate clusters)',
        fontsize=12.5,
        y=0.985,
    )
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Cluster roster table saved to {output_path}')


def analyze_clusters(
    youth_list: list[Youth],
    solver: SolverLike,
    person_crew: PersonCrew,
    centers: list[Center],
    year: int | None = None,
    output_dir: str = '.',
    *,
    assignments_csv: Path | str | None = None,
) -> dict[str, Any]:
    """Full cluster analysis pipeline.

    When ``assignments_csv`` points to an existing workbook, friend cluster ids are written to
    that file as ``FriendCluster`` (``C1``, ``C2``, … ordered by roster size — same numbering as the
    cluster roster visualization), ``FriendClusterId`` (Louvain partition id strings), and
    ``BuddyWeight`` (summed same-center roster buddy preference weight 4+2+1, when placements exist).
    """
    print('\n' + '=' * 50)
    print('CLUSTER ANALYSIS')
    print('=' * 50)

    clusters = detect_friend_clusters(youth_list)
    cohesion = calculate_cluster_cohesion(clusters, solver, person_crew, centers)

    if not cohesion:
        print('No friend clusters detected (empty youth list).')
        return {'num_clusters': 0, 'avg_cohesion': 0.0, 'cluster_details': {}}

    buddy_weights: dict[str, float] | None = None
    if centers:
        buddy_weights = calculate_youth_buddy_weights_by_name(solver, person_crew, youth_list, centers)
        analysis_name = f'cluster_analysis_{year}.png' if year else 'cluster_analysis.png'
        roster_name = f'cluster_roster_{year}.png' if year else 'cluster_roster.png'
        out_base = Path(output_dir)
        visualize_cluster_distribution(clusters, cohesion, centers, solver, person_crew, youth_list, str(out_base / analysis_name))
        render_cluster_roster_table(
            clusters,
            cohesion,
            centers,
            solver,
            person_crew,
            youth_list,
            str(out_base / roster_name),
            buddy_weights=buddy_weights,
        )
    else:
        print('Skipping cluster visualization: no centers available to plot against.')

    avg_cohesion = sum(c['cohesion_score'] for c in cohesion.values()) / len(cohesion)
    num_clusters = len(set(clusters.values()))

    print(f'\nNumber of friend clusters detected: {num_clusters}')
    print(f'Average cluster cohesion: {avg_cohesion:.1%}')
    print('\nCluster Details:')
    for cluster_id, data in sorted(cohesion.items(), key=lambda x: x[1]['size'], reverse=True):
        print(f'  {cluster_id}: {data["size"]} members, cohesion={data["cohesion_score"]:.1%}, distribution={data["center_distribution"]}')

    if assignments_csv is not None:
        merge_friend_clusters_into_assignments_csv(
            assignments_csv, clusters, cohesion, buddy_weights_by_name=buddy_weights,
        )

    return {
        'num_clusters': num_clusters,
        'avg_cohesion': avg_cohesion,
        'cluster_details': cohesion,
    }
