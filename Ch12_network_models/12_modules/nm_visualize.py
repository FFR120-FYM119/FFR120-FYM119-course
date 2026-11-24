"""Draw network
"""

import networkx as nx
from matplotlib.patches import FancyArrowPatch

def draw_network_labels(G, pos=None, ax=None, node_size=500):
    """Draw a NetworkX graph using Matplotlib."""

    if ax is None:
        fig, ax = plt.subplots()

    if pos is None:
        pos = nx.spring_layout(G, seed=42)

    for u, v in G.edges():
        ax.add_patch(FancyArrowPatch(
            pos[u], pos[v],
            arrowstyle="-",
            color="black",
            linewidth=1.5,
            shrinkA=15,
            shrinkB=15
        ))

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color="lightskyblue",
        node_size=node_size
    )
    nx.draw_networkx_labels(G, pos, ax=ax)

    ax.autoscale_view()
    ax.margins(0.15)
    ax.set_aspect("equal")
    ax.axis("off")

    return ax


def draw_network_dots(G, pos=None, ax=None, node_size=50):
    """Draw a network with dots and no labels."""

    if ax is None:
        fig, ax = plt.subplots()

    if pos is None:
        pos = nx.circular_layout(G)

    for u, v in G.edges():
        ax.add_patch(FancyArrowPatch(
            pos[u], pos[v],
            arrowstyle="-",
            color="black",
            linewidth=0.6,
            shrinkA=5,
            shrinkB=5
        ))

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color="C0",
        node_size=node_size
    )

    ax.autoscale_view()
    ax.margins(0.2)
    ax.set_aspect("equal")
    ax.axis("off")

    return ax
