import argparse
import itertools
import json
import os
import pathlib
import sys
import typing

import matplotlib as mpl
import networkx as nx
import numpy as np
import potpourri3d as pp3d

import linear_geodesic_optimization.batch as batch
from linear_geodesic_optimization.data import utility
from linear_geodesic_optimization.graph import boundary
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh


# List of parameters that correspond to extant files. This is used to
# help generate output directory names (by removing extensions when they
# would be unneeded) and determining actual locations of input files (by
# prepending ../data)
parameter_name_filenames = [
    'filename_probes',
    'filename_links',
    'filename_graphml',
    'filename_json',
]

def argument_to_string(
    arguments,
    parameter_name
):
    if parameter_name in parameter_name_filenames:
        return pathlib.PurePath(arguments[parameter_name]).stem
    else:
        return str(arguments[parameter_name])

tableau_colors = list(mpl.colors.TABLEAU_COLORS.values())
def get_tableau_color(index: int):
    """
    Return a color selected from the Tableau palette.
    """
    color = mpl.colors.to_rgb(tableau_colors[index % len(tableau_colors)])
    return [int(channel * 255) for channel in color]

def remove_loops(path: typing.List[int]):
    """
    Remove loops in a path.

    As input, take a list of vertex indices. This traces out a path that
    might intersect itself. This function simply removes all loops, so
    the resulting path is a sequence of unique vertex indices starting
    and ending at the same location, and each edge in the new path is
    one of the edges in the original path.
    """
    indices = {}
    path_new = []

    for v in path:
        if v in indices:
            for _ in range(len(path_new) - indices[v] - 1):
                indices.pop(path_new.pop())
        else:
            indices[v] = len(path_new)
            path_new.append(v)

    return path_new

def compute_geodesics_from_graph(mesh: RectangleMesh, network_vertices, network_edges, geodesic_routes):
    mesh_scale = mesh.get_scale()

    network_indices = set()
    bad_indices = set()
    for geodesic_route in geodesic_routes:
        for network_index in geodesic_route:
            network_indices.add(network_index)
    for network_index in network_indices:
        network_vertex = network_vertices[network_index]
        try:
            mesh.add_vertex_at_coordinates(network_vertex, 0.0001)
        except ValueError:
            bad_indices.add(network_index)

    coordinates = mesh.get_coordinates()
    topology = mesh.get_topology()
    mesh_graph = nx.Graph()
    for edge in topology.edges():
        source, target = edge.vertices()
        source = source.index
        target = target.index
        weight = np.linalg.norm(coordinates[source] - coordinates[target])
        mesh_graph.add_edge(source, target, length=weight)

    network_vertex_indices_to_mesh_vertex_indices = [
        mesh.nearest_vertex(network_vertex).index
        for network_vertex in network_vertices
    ]
    # TODO: Would using something like this be more efficient?
    # approximate_network_edge_traces = [
    #     nx.dijkstra_path(
    #         mesh_graph,
    #         network_vertex_indices_to_mesh_vertex_indices[u_index],
    #         network_vertex_indices_to_mesh_vertex_indices[v_index],
    #         'length'
    #     )
    #     for (u_index, v_index) in network_edges
    # ]

    path_solver = pp3d.EdgeFlipGeodesicSolver(
        coordinates,
        np.array([
            [vertex.index for vertex in face.vertices()]
            for face in mesh.get_topology().faces()
        ])
    )

    geodesics = []
    for geodesic_route in geodesic_routes:
        is_bad = False
        for index_node in geodesic_route:
            if index_node in bad_indices:
                is_bad = True
        if is_bad:
            continue

        approximate_trace = []
        for index_source, index_target in itertools.pairwise(geodesic_route):
            approximate_trace.extend(
                nx.dijkstra_path(
                    mesh_graph,
                    network_vertex_indices_to_mesh_vertex_indices[index_source],
                    network_vertex_indices_to_mesh_vertex_indices[index_target],
                    'length'
                )
            )
        approximate_trace = remove_loops(approximate_trace)

        if len(approximate_trace) < 2:
            geodesics.append([])
        else:
            geodesic = path_solver.find_geodesic_path_poly(approximate_trace)
            geodesics.append((geodesic[:, :2] / mesh_scale).tolist())

    return geodesics

def collate_outputs(
    directories_outputs,
    path_output_collated,
    *,
    times=None,
    geodesic_label_color_pairs=None,
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

    node_indices_to_labels = list(sorted(node_labels_to_coordinates.keys()))
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

    # Read some parameters. Assume most don't change across snapshots
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

    # Compute the network borders and normalize the z-coordinates across time
    network_borders = []
    hulls = []
    distances_to_networks = []
    zs = []
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
        hull = np.where(distances_to_networks[-1] / mesh_scale <= bubble_size)[0]
        hulls.append(hull)

        # TODO: Is this the right strategy? Should we make this a
        # parameter somewhere?
        if parameters['initial_radius'] is None:
            z_0 = np.array(output['initial'])
        else:
            z_0 = np.array([
                (parameters['initial_radius']**2
                    - (i / (width - 1) - 0.5)**2
                    - (j / (height - 1) - 0.5)**2)**0.5
                for i in range(width)
                for j in range(height)
            ]).reshape((width * height,))
        z = np.array(output['final']) - z_0
        zs.append(z - np.mean(z[hull]))

    # Determine values for vertical scaling
    z_max = -np.inf
    z_min = np.inf
    for z, hull in zip(zs, hulls):
        z_max = max(z_max, np.max(z[hull]))
        z_min = min(z_min, np.min(z[hull]))

    if geodesic_label_color_pairs is None:
        # For now, just use the routes in the first snapshot
        # TODO: Is this exactly what we want?
        high_traffic_route_pairs = list(sorted([
            (traffic, route)
            for traffic, route in zip(outputs[0]['traffic'], outputs[0]['routes'])
            if route[0] != route[-1]
        ], reverse=True))
        geodesic_label_color_pairs = [
            (route, get_tableau_color(index))
            for index, (_, route) in enumerate(high_traffic_route_pairs)
        ]
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
        # mesh.set_parameters(z_original)
        # TODO: Make this safer
        mesh.trim_to_set(hull)
        geodesics = compute_geodesics_from_graph(
            mesh, network_vertices,
            network_edges,
            [
                [
                    node_labels_to_indices[node]
                    for node in node_list
                ]
                for node_list in geodesic_labels
            ]
        )
        mesh.remove_added_vertices()
        mesh.restore_removed_vertices()

        traffic = output['traffic'] if 'traffic' in output else None
        traffic_matrix = [[0. for _ in range(len(network_vertices))] for _ in range(len(network_vertices))]
        if traffic is not None:
            for route, volume in zip(output['routes'], output['traffic']):
                traffic_matrix[node_labels_to_indices[route[0]]][node_labels_to_indices[route[-1]]] += volume

        if traffic is not None:
            paths = [[[] for _ in range(len(network_vertices))] for _ in range(len(network_vertices))]
            for geodesic_label, geodesic in zip(geodesic_labels, geodesics):
                paths[node_labels_to_indices[geodesic_label[0]]][node_labels_to_indices[geodesic_label[-1]]] = geodesic
        else:
            paths = None

        animation_data.append({
            'time': t,
            'height': z.reshape((width, height)).tolist(),
            'edges': edges,
            'geodesics': geodesics,
            'edgeColors': edge_colors,
            'border': network_border,
            'traffic': traffic_matrix,
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    args = parser.parse_args()
    config_file = pathlib.PurePath(args.config_file)

    # Read JSON config file and generate optimization parameters from it
    defaults = {
        'filename_probes': None,
        'filename_links': None,
        'filename_graphml': None,
        'filename_json': None,
        'latency_threshold': None,
        'clustering_distance': None,
        'ricci_curvature_alpha': 0.,
        'lambda_curvature': 1.,
        'lambda_smooth': 0.,
        'initial_radius': 20.,
        'sides': 50,
        'mesh_scale': 1.,
        'coordinates_scale': 0.8,
        'network_trim_radius': None,
        'maxiter': None,
        'initialization_file_path': None,
        'index': None # Additional unique ID
    }
    with open(config_file, 'r') as f:
        arguments, settings = batch.parse_json(f, defaults)

    # Determine where the outputs are going
    if 'output_format' in settings:
        output_format = settings['output_format']
    else:
        output_format = ['filename_json']
    # Generate the output directories from the given format. Check if
    # they exist
    for index, argument_dict in enumerate(arguments):
        argument_dict['index'] = index
        # Also prepend ../outputs (which is the path of the outputs
        # directory relative to the script)
        directory_output = pathlib.PurePath('..', 'outputs') / pathlib.PurePath(*(settings['directory_output'] + [
            argument_to_string(argument_dict, output_format_part) if isinstance(output_format_part, str) else
            '_'.join([argument_to_string(argument_dict, output_format_part_part) for output_format_part_part in output_format_part])
            for output_format_part in output_format
        ]))
        if not os.path.exists(directory_output):
            raise ValueError(f'{str(directory_output)} does not exist')
        argument_dict['directory_output'] = directory_output

    if 'initialization' in settings:
        initialization = settings['initialization']
    else:
        initialization = 'sphere'
    collate_outputs(
        [argument_dict['directory_output'] for argument_dict in arguments[(1 if initialization == 'first' else 0):]],
        pathlib.PurePath('..', 'outputs') / pathlib.PurePath(*settings['directory_output']) / 'animation.json',
        geodesic_label_color_pairs=None,  # TODO: Add custom functionality
        bubble_size=0.05,
    )

if __name__ == '__main__':
    main()
