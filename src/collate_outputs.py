import itertools
import json
import os
import pathlib
import sys
import typing

import matplotlib as mpl
import numpy as np
import potpourri3d as pp3d

from linear_geodesic_optimization.data import utility
from linear_geodesic_optimization.graph import boundary
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh


tableau_colors = list(mpl.colors.TABLEAU_COLORS.values())
def get_tableau_color(index):
    color = mpl.colors.to_rgb(tableau_colors[index % len(tableau_colors)])
    return [int(channel * 255) for channel in color]

def compute_geodesics_from_graph(mesh: RectangleMesh, network_vertices, network_edges, geodesic_index_pairs):
    mesh_scale = mesh.get_scale()

    network_indices = set()
    bad_indices = set()
    for (index_source, index_target) in geodesic_index_pairs:
        network_indices.add(index_source)
        network_indices.add(index_target)
    for network_index in network_indices:
        network_vertex = network_vertices[network_index]
        if network_vertex.tolist() not in (mesh.get_coordinates()[:, :2].tolist()):
            try:
                mesh.add_vertex_at_coordinates(network_vertex, 0.0001)
            except ValueError:
                bad_indices.add(network_index)

    # fig, ax = plt.subplots(1, 1)
    # topology = mesh.get_topology()
    # coordinates = mesh.get_coordinates()
    # for edge in topology.edges():
    #     u, v = edge.vertices()
    #     ax.plot([coordinates[u.index][0], coordinates[v.index][0]], [coordinates[u.index][1], coordinates[v.index][1]], 'k-')
    # ax.set_aspect('equal')
    # plt.show()

    path_solver = pp3d.EdgeFlipGeodesicSolver(
        mesh.get_coordinates(),
        np.array([
            [vertex.index for vertex in face.vertices()]
            for face in mesh.get_topology().faces()
        ])
    )

    geodesics = []
    for (index_source, index_target) in geodesic_index_pairs:
        if index_source in bad_indices or index_target in bad_indices:
            continue

        source = mesh.nearest_vertex(network_vertices[index_source]).index
        target = mesh.nearest_vertex(network_vertices[index_target]).index

        if source == target:
            continue
        else:
            geodesic = path_solver.find_geodesic_path(source, target)
            geodesics.append((geodesic[:, :2] / mesh_scale).tolist())

    return geodesics

def collate_outputs(
    directories_outputs,
    path_output_collated,
    *,
    times=None,
    geodesic_label_color_pairs=[],
    height_scale=0.15,
    use_convex_hull=False,
    bubble_size = np.inf,
):
    """
    Join many optimization outputs into a single animation.

    `directories_outputs` is a collection of directories, each of which
    contains an optimization output file called 'output.json'.
    Optionally, these are each associated with timestamps in `times`

    The data are then combined and written to `path_output_collated`.

    Geodesic paths can be computed using `geodesic_label_color_pairs`,
    which is a list with elements of the form
      `((source, target), [colorR, colorG, colorB])`,
    where the colors are encoded with numbers between 0 and 255.
    """

    # Grab the data from the files
    outputs = []
    for directory_output in directories_outputs:
        with open(directory_output / 'output.json', 'r') as file_output:
            outputs.append(json.load(file_output))
    if times is None:
        times = list(range(len(outputs)))

    # Clustering should always return the same set of clusters, but we might
    # end up omiting nodes that have no associated measurements at a
    # snapshot. We might also choose a different cluster representative.
    # So, we need to ensure our network vertices are uniform across time.

    # Determine a consistent set of cluster representatives, their
    # coordinates, and their associated data
    node_labels_to_representatives = {}
    node_representatives_to_labels = {}
    node_labels_to_coordinates = {}
    node_labels_to_data = {}
    node_data_keys = set()
    for output in outputs:
        network = output['network']
        graph_data, vertex_data, edge_data = network

        # Cases depending on whether clustering was used
        if 'elements' in vertex_data:
            for index, (node_label, elements, coordinates) in enumerate(zip(graph_data['labels'], vertex_data['elements'], graph_data['coordinates'])):
                node_representative = min(elements)
                for element in elements:
                    node_labels_to_representatives[element] = node_representative
                node_labels_to_coordinates[node_representative] = coordinates
                node_labels_to_data[node_representative] = {
                    key: value[index]
                    for key, value in vertex_data.items()
                }
        else:
            for index, (node_label, coordinates) in enumerate(zip(graph_data['labels'], graph_data['coordinates'])):
                node_labels_to_representatives[node_label] = node_label
                node_labels_to_coordinates[node_label] = coordinates
                node_labels_to_data[node_label] = {
                    key: value[index]
                    for key, value in vertex_data.items()
                }

        for key in vertex_data.keys():
            node_data_keys.add(key)

    node_indices_to_labels = list(sorted(node_labels_to_coordinates))
    node_labels_to_indices = {label: index for index, label in enumerate(node_indices_to_labels)}
    for node_label, node_representative in node_labels_to_representatives.items():
        node_labels_to_indices[node_label] = node_labels_to_indices[node_representative]
    for node_label, node_representative in node_labels_to_representatives.items():
        if node_representative not in node_representatives_to_labels:
            node_representatives_to_labels[node_representative] = [node_label]
        else:
            node_representatives_to_labels[node_representative].append(node_label)

    # Combine the computed data into an appropriate format
    node_labels = node_indices_to_labels
    node_coordinates = [node_labels_to_coordinates[label] for label in node_labels]
    node_data = {
        key: [
            node_labels_to_data[label][key] if key in node_labels_to_data[label] else None
            for label in node_labels
        ]
        for key in node_data_keys
    }

    # Relabel the networks
    for output in outputs:
        network = output['network']
        graph_data, vertex_data, edge_data = network

        graph_data_new = {
            'coordinates': node_coordinates,
            'edges': [
                (node_labels_to_indices[source_label], node_labels_to_indices[target_label])
                for source_index, target_index in graph_data['edges']
                for source_label in (graph_data['labels'][source_index],)
                for target_label in (graph_data['labels'][target_index],)
            ],
            'labels': node_indices_to_labels,
            'bounding_box': graph_data['bounding_box'] if 'bounding_box' in graph_data else None
        }
        vertex_data_new = node_data
        edge_data_new = edge_data

        output['network'] = (graph_data_new, vertex_data_new, edge_data_new)

    # Normalize the z-coordinates across time
    zs = []
    for output in outputs:
        z = np.array(output['final']) - np.array(output['initial'])
        zs.append(z - np.mean(z))

    # Assume most parameters don't change across snapshots
    parameters = outputs[0]['parameters']
    width = parameters['width']
    height = parameters['height']
    mesh_scale = parameters['mesh_scale']
    coordinates_scale = parameters['coordinates_scale']
    mesh = RectangleMesh(width, height, mesh_scale)

    # Assume bounding box information is constant
    graph_data, vertex_data, edge_data = outputs[0]['network']
    bounding_box = graph_data['bounding_box']
    network_vertices = mesh.map_coordinates_to_support(np.array(node_coordinates), coordinates_scale, bounding_box)

    # Figure out vertex coordinates for the frontend
    animation_vertices = [
        {
            'label': '/'.join(sorted(node_representatives_to_labels[label])),
            'coordinates': [network_vertex[0] / mesh_scale, network_vertex[1] / mesh_scale],
        }
        for network_vertex, label in zip(network_vertices, node_indices_to_labels)
    ]

    # Figure out the edges for the frontend
    animation_edges = [
        [
            {
                'source': edge[0],
                'target': edge[1],
                'curvature': curvature,
                'throughput': throughput,
            }
            for edge, curvature, throughput in zip(
                graph_data['edges'],
                edge_data['ricciCurvature'] if 'ricciCurvature' in edge_data else itertools.repeat(0.),
                edge_data['throughput'] if 'throughput' in edge_data else itertools.repeat(1.)
            )
        ]
        for output in outputs
        for (graph_data, vertex_data, edge_data) in (output['network'],)
    ]

    # Compute the network borders
    network_borders = []
    hulls = []
    distances_to_networks = []
    for output in outputs:
        network = output['network']
        graph_data, vertex_data, edge_data = network
        network_edges = graph_data['edges']

        if use_convex_hull:
            convex_hulls_indices = boundary.compute_connected_convex_hulls(network_vertices, network_edges)
            network_border = [
                [
                    list(network_vertices[index])
                    for index in border_part
                ]
                for border_part in convex_hulls_indices
            ]
        else:
            network_border = boundary.compute_border(network_vertices, network_edges)

        network_borders.append(network_border)
        distances_to_networks.append(np.array([
            boundary.distance_to_network(
                np.array(vertex_coordinate),
                network_border
            )
            for vertex_coordinate in mesh.get_coordinates()[:, :2]
        ]))
        hulls.append(np.where(distances_to_networks[-1] / mesh_scale <= bubble_size)[0])

    # Determine values for vertical scaling
    z_max = -np.inf
    z_min = np.inf
    for z, hull in zip(zs, hulls):
        z_max = max(z_max, np.max(z[hull]))
        z_min = min(z_min, np.min(z[hull]))

    geodesic_labels = [geodesic_label for geodesic_label, _ in geodesic_label_color_pairs]
    edge_colors = [list(color) for _, color in geodesic_label_color_pairs]

    animation_data = []
    for t, z, distance_to_network, hull, edges, network_border, output in zip(times, zs, distances_to_networks, hulls, animation_edges, network_borders, outputs):
        z_original = np.copy(z)

        distance_to_bubble = np.zeros(distance_to_network.shape) if np.isposinf(bubble_size) else np.maximum(distance_to_network / mesh_scale - bubble_size, 0.)
        z = z - z_min
        if z_max != z_min and height_scale is not None:
            z = z / (z_max - z_min) * height_scale
        z = (z + 0.05) * np.exp(-1000 * distance_to_bubble**2) - 0.05

        mesh.set_parameters(z)
        # TODO: Make this safer
        mesh.trim_to_set(hull)
        geodesics = compute_geodesics_from_graph(
            mesh, network_vertices,
            network_edges,
            [
                (
                    node_labels_to_indices[node_source],
                    node_labels_to_indices[node_target]
                )
                for (node_source, node_target) in geodesic_labels
            ]
        )
        mesh.remove_added_vertices()
        mesh.restore_removed_vertices()

        traffic = output['traffic'] if 'traffic' in output else None
        if 'traffic' in output:
            paths = [[[] for _ in range(len(network_vertices))] for _ in range(len(network_vertices))]
            for (node_source, node_target), geodesic in zip(geodesic_labels, geodesics):
                paths[node_labels_to_indices[node_source]][node_labels_to_indices[node_target]] = geodesic
        else:
            paths = None

        animation_data.append({
            'time': t,
            'height': z.reshape((width, height)).tolist(),
            'edges': edges,
            'geodesics': geodesics,
            'edgeColors': edge_colors,
            'border': network_border,
            'traffic': traffic,
            'trafficPaths': paths
        })

    # Set the map data
    # TODO: Make this more robust
    if node_coordinates:
        coordinates = np.array(node_coordinates)
        center_xy = (np.amin(coordinates, axis=0) + np.amax(coordinates, axis=0)) / 2.
        center = utility.inverse_mercator(*center_xy)
        left, _ = utility.inverse_mercator(np.amin(coordinates[:,0]), 0)
        right, _ = utility.inverse_mercator(np.amax(coordinates[:,0]), 0)
        zoom_factor = coordinates_scale * 360. / (right - left)
        map_data = {
            'center': center,
            'zoomFactor': zoom_factor,
        }
    else:
        map_data = {
            'center': (0., 0.),
            'zoomFactor': 1.,
        }

    os.makedirs(path_output_collated.parent, exist_ok=True)
    with open(path_output_collated, 'w') as file_output:
        json.dump(
            {
                'nodes': animation_vertices,
                'animation': animation_data,
                'map': map_data,
            },
            file_output, ensure_ascii=False
        )

def main_routing_with_volumes():
    superdirectory = pathlib.PurePath('..', 'outputs', 'toy', 'routing_with_volumes')
    directories_outputs = [
        superdirectory / directory / '0.002_50_50'
        for directory in sorted(os.listdir(superdirectory))
    ]

    geodesic_label_color_pairs = [
        ((u, v), [0, 0, 0])
        for u in 'ABCDEF'
        for v in 'ABCDEF'
        if u != v
    ]

    for directory in directories_outputs:
        collate_outputs(
            [directory],
            pathlib.PurePath('..', 'outputs', 'animations', 'routing_with_volumes', f'{directory.parent.stem}.json'),
            geodesic_label_color_pairs=geodesic_label_color_pairs,
            bubble_size=0.05,
            # height_scale=None,
        )

def main_routing_with_volumes_animated():
    superdirectory = pathlib.PurePath('..', 'outputs', 'toy', 'routing_with_volumes', 'graphs_single_route_change')
    directories_outputs = [
        superdirectory / directory / '0.002_50_50'
        for directory in sorted(os.listdir(superdirectory))
    ]

    geodesic_label_color_pairs = [
        ((u, v), get_tableau_color(index))
        for index, (u, v) in enumerate([
            (u, v)
            for u in 'ABCDEF'
            for v in 'ABCDEF'
            if u != v
        ])
    ]

    collate_outputs(
        directories_outputs,
        pathlib.PurePath('..', 'outputs', 'animations', 'routing_with_volumes', f'test.json'),
        geodesic_label_color_pairs=geodesic_label_color_pairs,
        bubble_size=0.05,
    )

if __name__ == '__main__':
    pass
    # main_routing_with_volumes()
    main_routing_with_volumes_animated()
