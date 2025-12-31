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


def visualize_cluster_distribution(
    clusters: dict[str, int],
    cohesion_data: dict[str, dict],
    centers: list[Center],
    solver: cp_model.CpSolver,
    person_crew: dict,
    output_path: str = 'cluster_analysis.png',
):
    """Create visualization showing cluster distribution across centers."""
    # Get unique cluster IDs and sort by size
    cluster_sizes = [(cid, data['size']) for cid, data in cohesion_data.items()]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)
    cluster_ids = [cid for cid, _ in cluster_sizes]

    # Get center names
    center_names = [c.name for c in centers]

    # Create matrix for stacked bar chart
    data_matrix = np.zeros((len(cluster_ids), len(center_names)))

    for i, cluster_id in enumerate(cluster_ids):
        center_dist = cohesion_data[cluster_id]['center_distribution']
        for j, center_name in enumerate(center_names):
            data_matrix[i, j] = center_dist.get(center_name, 0)

    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(cluster_ids))
    width = 0.8
    bottom = np.zeros(len(cluster_ids))

    colors = plt.cm.Set3(np.linspace(0, 1, len(center_names)))

    for j, center_name in enumerate(center_names):
        values = data_matrix[:, j]
        ax.bar(x, values, width, label=center_name, bottom=bottom, color=colors[j])
        bottom += values

    # Add cohesion scores as annotations
    for i, cluster_id in enumerate(cluster_ids):
        cohesion = cohesion_data[cluster_id]['cohesion_score']
        size = cohesion_data[cluster_id]['size']
        ax.text(i, bottom[i] + 0.5, f'{cohesion:.1%}\n(n={size})', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Friend Cluster', fontsize=12)
    ax.set_ylabel('Number of Youth', fontsize=12)
    ax.set_title('Friend Cluster Distribution Across Centers\n(Percentage = Cohesion Score)', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'C{i + 1}' for i in range(len(cluster_ids))], rotation=0)
    ax.legend(title='Centers', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
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
    visualize_cluster_distribution(clusters, cohesion, centers, solver, person_crew, f'{output_dir}/{filename}')

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
