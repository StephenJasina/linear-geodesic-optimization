import itertools
import json
import os
import pickle
import sys

import numpy as np

from linear_geodesic_optimization.plot import get_line_plot, get_scatter_fig, \
    combine_scatter_figs, get_heat_map, get_network_map, Animation3D

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python3 <directory name>')
        sys.exit(0)

    directory = sys.argv[1]

    if not os.path.exists(os.path.join(directory, '0')):
        print('Error: supplied directory must contain file named "0"')
        sys.exit(0)

    lses = []
    L_smooths = []
    L_curvatures = []
    Ls = []

    scatter_fig_before = None
    scatter_fig_after = None

    animation_3D = Animation3D()

    for i in itertools.count():
        path = os.path.join(directory, str(i))
        with open(path, 'rb') as f:
            hierarchy = pickle.load(f)
            hierarchy.cores = 1
            break

            animation_3D.add_frame(hierarchy.mesh)

            lse, L_smooth, L_curvature = hierarchy.get_forwards()
            lses.append(lse)
            L_smooths.append(L_smooth)
            L_curvatures.append(L_curvature)
            Ls.append(hierarchy.lambda_geodesic * lse
                      + hierarchy.lambda_smooth * L_smooth
                      + hierarchy.lambda_curvature * L_curvature)

            if i == 0:
                scatter_fig_before = get_scatter_fig(hierarchy, True)

            path_next = os.path.join(directory, str(i + 1))
            if not os.path.exists(path_next):
                scatter_fig_after = get_scatter_fig(hierarchy, False)
                break

    # TODO: Make these plots a lot nicer (maybe aggregate them into one
    # visualization?)
    # get_line_plot(lses, 'Least Squares Loss').show()
    # get_line_plot(L_smooths, 'Smoothness Loss').show()
    # get_line_plot(L_curvatures, 'Curvature Loss').show()
    # get_line_plot(Ls, 'Total Loss').show()

    mesh = hierarchy.mesh
    vertices = mesh.get_vertices()
    coordinates = None
    label_to_index = {}
    with open(os.path.join('..', 'data', 'toy', 'position.json')) as f:
        position_json = json.load(f)

        label_to_index = {label: index for index, label in enumerate(position_json)}

        coordinates = [None for _ in range(len(position_json))]
        for vertex, position in position_json.items():
            coordinates[label_to_index[vertex]] = position

    network_vertices = [vertices[i,:2]
                        for i in mesh.coordinates_to_indices(coordinates)]

    network_edges = []
    ts = {i: [] for i in range(len(network_vertices))}
    with open(os.path.join('..', 'data', 'toy', 'latency.json')) as f:
        latency_json = json.load(f)

        for edge, latency in latency_json.items():
            u = label_to_index[edge[0]]
            v = label_to_index[edge[1]]

            network_edges.append((u, v))

            ts[u].append((v, latency))
            ts[v].append((u, latency))

    x = list(sorted(set(vertices[:,0])))
    y = list(sorted(set(vertices[:,1])))
    z = vertices[:,2].reshape(len(x), len(y))
    get_heat_map(x, y, z, network_vertices, network_edges).show()
    # get_network_map(coordinates, network_edges).show()

    # combine_scatter_figs(scatter_fig_before, scatter_fig_after).show()
    # animation_3D.get_fig(duration=50).show()
