import networkx as nx  # type: ignore
import matplotlib.pyplot as plt
from src.models import Center, Youth
from community import community_louvain  # type: ignore
import numpy as np
from ortools.sat.python import cp_model


def create_friend_network_visualization(
    youth_list: list[Youth],
    person_center: dict,
    solver: cp_model.CpSolver,
    centers: list[Center],
    output_path: str = 'friend_network.png',
):
    """
    Creates a visualization showing community groupings within each center.
    Points of the same color represent youth in the same friend community.
    Communities are arranged vertically with controlled horizontal spread.

    Args:
        youth_list: List of Youth objects with friend choices
        person_center: Dictionary mapping (person, center) to solver variables
        solver: Solved CP model
        centers: List of Center objects
        output_path: Where to save the visualization
    """
    # Create network graph for community detection
    G = nx.Graph()
    center_names = [center.name for center in centers]

    for youth in youth_list:
        # Find which center this youth is assigned to
        for center_name in center_names:
            if solver.Value(person_center[youth.name, center_name]) == 1:
                G.add_node(youth.name, center=center_name)
                break

        friend_weights = {youth.first_choice: 3, youth.second_choice: 2, youth.third_choice: 1}
        for friend, weight in friend_weights.items():
            if friend:
                G.add_edge(youth.name, friend, weight=weight)

    # Detect communities
    communities = community_louvain.best_partition(G)
    num_communities = max(communities.values()) + 1

    # Set up colors
    community_colors = plt.get_cmap('Set3')(np.linspace(0, 1, num_communities))

    # Create figure with subplots (one for each center)
    fig, axes = plt.subplots(1, len(centers), figsize=(5 * len(centers), 5))
    if len(centers) == 1:
        axes = [axes]  # Make it iterable if there's only one center

    center_axes = dict(zip(center_names, axes))

    # Calculate vertical spacing
    vertical_padding = 0.1
    available_height = 1 - 2 * vertical_padding

    # Group youth by center and community
    for center_name, ax in center_axes.items():
        # Get youth in this center using solver results
        center_youth = [
            (youth.name, communities[youth.name])
            for youth in youth_list
            if solver.Value(person_center[youth.name, center_name]) == 1
        ]

        # Group by community
        community_groups: dict[int, list[str]] = {}
        for name, comm_id in center_youth:
            if comm_id not in community_groups:
                community_groups[comm_id] = []
            community_groups[comm_id].append(name)

        # Sort communities by size (optional)
        sorted_communities = sorted(community_groups.items(), key=lambda x: len(x[1]), reverse=True)

        # Calculate positions for each community
        spacing = available_height / max(len(sorted_communities), 1)

        # Plot each community group
        for i, (comm_id, members) in enumerate(sorted_communities):
            num_members = len(members)
            if num_members > 0:
                # Calculate vertical position for this community
                y_pos = 1 - vertical_padding - (i * spacing)

                # Calculate horizontal positions
                max_spread = 0.4  # Maximum horizontal spread from center
                x_positions = np.linspace(-max_spread, max_spread, num_members) if num_members > 1 else np.array([0.0])

                # Add small random jitter to prevent perfect alignment
                x_jitter = np.random.normal(0, 0.02, num_members)
                y_jitter = np.random.normal(0, 0.01, num_members)

                # Create position arrays
                x = np.full(num_members, 0.5) + x_positions + x_jitter
                y = np.full(num_members, y_pos) + y_jitter

                # Plot points
                ax.scatter(x, y, color=community_colors[comm_id], s=100, alpha=0.6, label=f'Community {comm_id + 1}')

                # Add labels
                for j, name in enumerate(members):
                    ax.annotate(
                        name.split()[0],  # First name only
                        (x[j], y[j]),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8,
                        alpha=0.7,
                    )

        # Customize subplot
        ax.set_title(f'{center_name}\n({len(center_youth)} youth)')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    # Add legend to the right of the subplots
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, title='Friend Communities', loc='center right', bbox_to_anchor=(1.15, 0.5))

    plt.suptitle('Friend Communities by Center Assignment', y=1.02, fontsize=14)
    plt.tight_layout()

    # Save and close
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
