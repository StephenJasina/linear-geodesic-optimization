import argparse
import itertools
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch

from linear_geodesic_optimization.data.utility import mercator


def get_edge_volumes(blob):
    """
    Aggregate traffic volume onto each directed link.

    Each entry of `blob['traffic']` gives a route (a sequence of node
    IDs) and the volume of traffic that flows along it. The volume of
    a link is the sum, over all routes using that link, of the route's
    volume.
    """
    volumes = {(link['source_id'], link['target_id']): 0. for link in blob['links']}
    for route_info in blob.get('traffic', []):
        for u, v in itertools.pairwise(route_info['route']):
            volumes[(u, v)] += route_info['volume']
    return volumes


def get_traffic_matrix(blob):
    """
    Build a source/destination traffic matrix.

    Returns a sorted list of node IDs together with a 2D array
    `matrix` where `matrix[i, j]` is the total volume of traffic whose
    route starts at the `i`th node and ends at the `j`th node.
    """
    node_ids = sorted(node['id'] for node in blob['nodes'])
    index = {node_id: i for i, node_id in enumerate(node_ids)}

    matrix = np.zeros((len(node_ids), len(node_ids)))
    for route_info in blob.get('traffic', []):
        route = route_info['route']
        matrix[index[route[0]], index[route[-1]]] += route_info['volume']

    return node_ids, matrix


def plot_traffic_graph(
    blob, ax=None,
    min_linewidth=0.5, max_linewidth=8., cmap='Blues',
    node_color='black'
):
    """
    Draw the graph, with nodes placed via a Mercator projection of
    their latitude/longitude, and directed links drawn as curved
    arrows both colored and thickened according to the volume of
    traffic flowing through them.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1)
    fig = ax.get_figure()
    ax.set_aspect('equal')
    ax.axis('off')

    positions = {
        node['id']: mercator(node['longitude'], node['latitude'])
        for node in blob['nodes']
    }

    volumes = get_edge_volumes(blob)
    volume_max = max(volumes.values(), default=0.)
    norm = Normalize(vmin=0., vmax=volume_max if volume_max > 0. else 1.)
    colormap = plt.get_cmap(cmap)

    # Plot the links as arrows, curved so that the two directions of a
    # reciprocal pair of links don't overlap. Thickness and color are
    # both scaled with the volume of traffic flowing through the link.
    for (u, v), volume in volumes.items():
        if volume <= 0.:
            continue

        linewidth = min_linewidth \
            + (max_linewidth - min_linewidth) * volume / volume_max

        ax.add_patch(FancyArrowPatch(
            positions[u], positions[v],
            connectionstyle='arc3,rad=0.08',
            arrowstyle='-|>',
            mutation_scale=10 + linewidth,
            shrinkA=8, shrinkB=8,
            linewidth=linewidth,
            color=colormap(norm(volume)),
            zorder=1,
        ))

    # Plot the nodes
    xs, ys = zip(*positions.values())
    ax.scatter(xs, ys, s=20, color=node_color, zorder=2)
    for node_id, (x, y) in positions.items():
        ax.annotate(
            node_id, (x, y),
            textcoords='offset points', xytext=(5, 5),
            fontsize=8, zorder=3
        )

    margin = 0.05
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)

    fig.colorbar(
        ScalarMappable(norm=norm, cmap=colormap), ax=ax,
        label='Link volume', fraction=0.046, pad=0.04
    )

    return fig


def plot_traffic_matrix(blob, ax=None, cmap='YlOrRd'):
    """
    Draw a heatmap of the source/destination traffic matrix, with rows
    as source nodes and columns as destination nodes.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1)
    fig = ax.get_figure()

    node_ids, matrix = get_traffic_matrix(blob)
    n = len(node_ids)

    im = ax.imshow(matrix, cmap=cmap)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(node_ids)
    ax.set_yticklabels(node_ids)
    ax.set_xlabel('Destination')
    ax.set_ylabel('Source')

    # Annotate each cell with its value
    value_max = matrix.max()
    for i in range(n):
        for j in range(n):
            value = matrix[i, j]
            if value <= 0.:
                continue
            text_color = 'white' if value > 0.6 * value_max else 'black'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                    fontsize=7, color=text_color)

    fig.colorbar(im, ax=ax, label='Traffic volume', fraction=0.046, pad=0.04)

    return fig


def plot_traffic_overview(blob):
    """Show the network diagram alongside the traffic matrix heatmap."""
    fig, (ax_graph, ax_matrix) = plt.subplots(1, 2, figsize=(13, 5.5))

    plot_traffic_graph(blob, ax=ax_graph)
    ax_graph.set_title('Link volumes')

    plot_traffic_matrix(blob, ax=ax_matrix)
    ax_matrix.set_title('Source/destination traffic matrix')

    fig.tight_layout()
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize a routing graph with traffic volumes.'
    )
    parser.add_argument('json_filename', metavar='json-file')
    parser.add_argument('--output', '-o', dest='output_filename', metavar='filename')
    args = parser.parse_args()

    with open(args.json_filename) as file:
        blob = json.load(file)

    fig = plot_traffic_overview(blob)

    if args.output_filename is None:
        plt.show()
    else:
        fig.savefig(args.output_filename, dpi=200, bbox_inches='tight')
