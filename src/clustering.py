from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from community import community_louvain  # type: ignore
from ortools.sat.python import cp_model

from src.models import Center, Youth


def is_person_at_center(
    solver: cp_model.CpSolver,
    person_crew: dict,
    person_name: str,
    center: Center,
) -> bool:
    """Helper function to check if a person is at a center based on crew assignments."""
    for crew in center.crews:
        if solver.Value(person_crew[person_name, center.name, crew.name]) == 1:
            return True
    return False


def detect_friend_clusters(youth_list: list[Youth]) -> dict[str, int]:
    """Detect friend clusters using Louvain community detection.

    Uses only buddy form friend choices - completely independent of assignments.
    Returns mapping of youth name -> cluster_id.
    """
    G = nx.Graph()

    for youth in youth_list:
        G.add_node(youth.name)
        weights = {youth.first_choice: 3, youth.second_choice: 2, youth.third_choice: 1}
        for friend, weight in weights.items():
            if friend:
                G.add_edge(youth.name, friend, weight=weight)

    return community_louvain.best_partition(G, weight='weight')


def calculate_cluster_cohesion(
    clusters: dict[str, int],
    solver: cp_model.CpSolver,
    person_crew: dict,
    centers: list[Center],
) -> dict[str, dict]:
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
    """Build directed graph of friend choices with weights."""
    G = nx.DiGraph()

    for youth in youth_list:
        G.add_node(youth.name)
        choices = [
            (youth.first_choice, 3),
            (youth.second_choice, 2),
            (youth.third_choice, 1),
        ]
        for friend, weight in choices:
            if friend:
                G.add_edge(youth.name, friend, weight=weight)

    return G


def visualize_cluster_distribution(
    clusters: dict[str, int],
    cohesion_data: dict[str, dict],
    centers: list[Center],
    solver: cp_model.CpSolver,
    person_crew: dict,
    youth_list: list[Youth],
    output_path: str = 'cluster_analysis.png',
):
    """Create comprehensive visualization showing cluster distribution, network, and name lists."""
    # Get unique cluster IDs and sort by size
    cluster_sizes = [(cid, data['size']) for cid, data in cohesion_data.items()]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)
    cluster_ids = [cid for cid, _ in cluster_sizes]

    # Get center names and create color mapping
    center_names = [c.name for c in centers]
    center_colors = plt.cm.Set3(np.linspace(0, 1, len(center_names)))
    center_color_map = {name: center_colors[i] for i, name in enumerate(center_names)}

    # Build mapping of name -> center assignment
    name_to_center = {}
    for youth in youth_list:
        for center in centers:
            if is_person_at_center(solver, person_crew, youth.name, center):
                name_to_center[youth.name] = center.name
                break

    # Create multi-panel figure
    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 2, 1.5], hspace=0.3)

    # ===== PANEL 1: Bar Chart =====
    ax_bar = fig.add_subplot(gs[0])

    # Create matrix for stacked bar chart
    data_matrix = np.zeros((len(cluster_ids), len(center_names)))
    for i, cluster_id in enumerate(cluster_ids):
        center_dist = cohesion_data[cluster_id]['center_distribution']
        for j, center_name in enumerate(center_names):
            data_matrix[i, j] = center_dist.get(center_name, 0)

    x = np.arange(len(cluster_ids))
    width = 0.8
    bottom = np.zeros(len(cluster_ids))

    for j, center_name in enumerate(center_names):
        values = data_matrix[:, j]
        ax_bar.bar(x, values, width, label=center_name, bottom=bottom, color=center_colors[j])
        bottom += values

    # Add cohesion scores as annotations
    for i, cluster_id in enumerate(cluster_ids):
        cohesion = cohesion_data[cluster_id]['cohesion_score']
        size = cohesion_data[cluster_id]['size']
        ax_bar.text(i, bottom[i] + 0.5, f'{cohesion:.1%}\n(n={size})', ha='center', va='bottom', fontsize=8)

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

    # Create cluster-based layout with initial positions
    cluster_members: dict[int, list[str]] = defaultdict(list)
    for name, cluster_id in clusters.items():
        cluster_members[cluster_id].append(name)

    # Position clusters in a grid
    pos = {}
    num_clusters = len(cluster_members)
    cols = int(np.ceil(np.sqrt(num_clusters)))

    for idx, (cluster_id, members) in enumerate(sorted(cluster_members.items(), key=lambda x: len(x[1]), reverse=True)):
        row = idx // cols
        col = idx % cols
        center_x = col * 10
        center_y = -row * 10

        # Create subgraph for this cluster
        subgraph = G.subgraph(members)
        if len(members) > 1:
            sub_pos = nx.spring_layout(subgraph, k=0.5, iterations=50, seed=42)
            # Scale and translate
            for node in sub_pos:
                pos[node] = (center_x + sub_pos[node][0] * 3, center_y + sub_pos[node][1] * 3)
        else:
            # Single node
            pos[members[0]] = (center_x, center_y)

    # Draw nodes colored by center assignment
    for center_name, color in center_color_map.items():
        nodes_in_center = [name for name in G.nodes() if name_to_center.get(name) == center_name]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes_in_center, node_color=[color], node_size=300, ax=ax_network, alpha=0.8)

    # Draw edges with varying width based on friend choice weight
    for edge in G.edges():
        weight = G[edge[0]][edge[1]]['weight']
        nx.draw_networkx_edges(G, pos, edgelist=[edge], width=weight * 0.5, alpha=0.3, arrows=True, arrowsize=10, ax=ax_network, edge_color='gray')

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=7, font_weight='bold', ax=ax_network)

    ax_network.set_title('Social Network: Friend Choices (colored by center assignment)', fontsize=14, pad=20)
    ax_network.axis('off')

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=name) for name, color in center_color_map.items()
    ]
    ax_network.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # ===== PANEL 3: Name Lists =====
    ax_names = fig.add_subplot(gs[2])
    ax_names.axis('off')

    # Build text for left column (grouped by cluster)
    left_text = 'BY CLUSTER (colored by center):\n' + '=' * 40 + '\n'
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_label = f'C{idx + 1}'
        cluster_num = int(cluster_id.split('_')[1])
        members = sorted(cluster_members[cluster_num])
        left_text += f'\n{cluster_label} ({len(members)} members):\n'
        for name in members:
            left_text += f'  {name}\n'

    # Build text for right column (grouped by center)
    right_text = 'BY CENTER (colored by cluster):\n' + '=' * 40 + '\n'
    for center_name in center_names:
        names_in_center = sorted([name for name, center in name_to_center.items() if center == center_name])
        right_text += f'\n{center_name} ({len(names_in_center)} youth):\n'
        for name in names_in_center:
            cluster_id = clusters.get(name, -1)
            cluster_label = f'C{list(cluster_members.keys()).index(cluster_id) + 1}' if cluster_id in cluster_members else '?'
            right_text += f'  {name} [{cluster_label}]\n'

    # Display text in two columns
    ax_names.text(0.02, 0.98, left_text, transform=ax_names.transAxes, fontsize=8, verticalalignment='top', family='monospace')
    ax_names.text(0.52, 0.98, right_text, transform=ax_names.transAxes, fontsize=8, verticalalignment='top', family='monospace')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Cluster visualization saved to {output_path}')


def analyze_clusters(
    youth_list: list[Youth],
    solver: cp_model.CpSolver,
    person_crew: dict,
    centers: list[Center],
    year: int | None = None,
    output_dir: str = '.',
) -> dict:
    """Full cluster analysis pipeline."""
    print('\n' + '=' * 50)
    print('CLUSTER ANALYSIS')
    print('=' * 50)

    clusters = detect_friend_clusters(youth_list)
    cohesion = calculate_cluster_cohesion(clusters, solver, person_crew, centers)

    # Include year in filename if provided
    filename = f'cluster_analysis_{year}.png' if year else 'cluster_analysis.png'
    visualize_cluster_distribution(clusters, cohesion, centers, solver, person_crew, youth_list, f'{output_dir}/{filename}')

    avg_cohesion = sum(c['cohesion_score'] for c in cohesion.values()) / len(cohesion)
    num_clusters = len(set(clusters.values()))

    print(f'\nNumber of friend clusters detected: {num_clusters}')
    print(f'Average cluster cohesion: {avg_cohesion:.1%}')
    print('\nCluster Details:')
    for cluster_id, data in sorted(cohesion.items(), key=lambda x: x[1]['size'], reverse=True):
        print(f'  {cluster_id}: {data["size"]} members, cohesion={data["cohesion_score"]:.1%}, distribution={data["center_distribution"]}')

    return {
        'num_clusters': num_clusters,
        'avg_cohesion': avg_cohesion,
        'cluster_details': cohesion,
    }
