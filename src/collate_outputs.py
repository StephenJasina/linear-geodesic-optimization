import itertools
import json
import os
import pathlib
import typing

import numpy as np
import potpourri3d as pp3d

from linear_geodesic_optimization.data import input_mesh, utility
from linear_geodesic_optimization.graph import convex_hull
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh

# Outputs are stored in `directory_outputs` / <output number> / `subdirectory_output`
# directory_outputs = pathlib.PurePath('..', 'outputs', 'throughputs', 'elbow', 'removed_AG', '0.001_20.0_30_30_1.0')
# directory_outputs = pathlib.PurePath('..', 'outputs', 'throughputs', 'elbow', 'removed_FL', '0.0004_20.0_30_30_1.0')
# subdirectory_output = pathlib.PurePath()
# directories_outputs = list(sorted([
#     (float(directory_output), directory_outputs / directory_output / subdirectory_output)
#     for directory_output in os.listdir(directory_outputs)
#     if os.path.isdir(directory_outputs / directory_output)
# ]))
# directories_outputs = [(0., directory_outputs)]

# For ESnet data
epsilon = 10
directories_outputs = [
    (0., pathlib.PurePath('..', 'outputs', 'esnet', '1742868000000', f'{epsilon}_0.0002_50_50')),
    (1., pathlib.PurePath('..', 'outputs', 'esnet', '1742871600000', f'{epsilon}_0.0002_50_50')),
    (2., pathlib.PurePath('..', 'outputs', 'esnet', '1742875200000', f'{epsilon}_0.0002_50_50')),
    (3., pathlib.PurePath('..', 'outputs', 'esnet', '1742878800000', f'{epsilon}_0.0002_50_50')),
    (4., pathlib.PurePath('..', 'outputs', 'esnet', '1742882400000', f'{epsilon}_0.0002_50_50')),
    (5., pathlib.PurePath('..', 'outputs', 'esnet', '1742886000000', f'{epsilon}_0.0002_50_50')),
    (6., pathlib.PurePath('..', 'outputs', 'esnet', '1742889600000', f'{epsilon}_0.0002_50_50')),
    (7., pathlib.PurePath('..', 'outputs', 'esnet', '1742893200000', f'{epsilon}_0.0002_50_50')),
    (8., pathlib.PurePath('..', 'outputs', 'esnet', '1742896800000', f'{epsilon}_0.0002_50_50')),
    (9., pathlib.PurePath('..', 'outputs', 'esnet', '1742900400000', f'{epsilon}_0.0002_50_50')),
    (10., pathlib.PurePath('..', 'outputs', 'esnet', '1742904000000', f'{epsilon}_0.0002_50_50')),
    (11., pathlib.PurePath('..', 'outputs', 'esnet', '1742907600000', f'{epsilon}_0.0002_50_50')),
    (12., pathlib.PurePath('..', 'outputs', 'esnet', '1742911200000', f'{epsilon}_0.0002_50_50')),
    (13., pathlib.PurePath('..', 'outputs', 'esnet', '1742914800000', f'{epsilon}_0.0002_50_50')),
]

# For ESnet toy data
# directories_outputs = [
#     (0., pathlib.PurePath('..', 'outputs', 'toy', 'esnet', '0', '0.001_30_30')),
#     (1., pathlib.PurePath('..', 'outputs', 'toy', 'esnet', '1', '0.001_30_30')),
#     (2., pathlib.PurePath('..', 'outputs', 'toy', 'esnet', '2', '0.001_30_30')),
# ]

path_output_collated = pathlib.PurePath('..', 'outputs', 'esnet', f'output_geodesics_{epsilon}.json')

def get_nearest_vertex(mesh: RectangleMesh, vertex):
    """
    Find the xy-coordinates of the nearest mesh point to some coordinates.
    """
    nearest_vertex = mesh.get_coordinates()[mesh.nearest_vertex(vertex).index]
    return [nearest_vertex[0], nearest_vertex[1]]

def compute_geodesics_from_graph(mesh: RectangleMesh, network_vertices, network_edges):
    mesh_scale = mesh.get_scale()
    n_vertices = len(network_vertices)

    path_solver = pp3d.EdgeFlipGeodesicSolver(
        mesh.get_coordinates(),
        np.array([
            [vertex.index for vertex in face.vertices()]
            for face in mesh.get_topology().faces()
        ])
    )

    geodesics = []
    for (index_source, index_target) in network_edges:
        source = mesh.nearest_vertex(network_vertices[index_source]).index
        target = mesh.nearest_vertex(network_vertices[index_target]).index
        if source == target:
            geodesics.append([(mesh.get_coordinates()[source] / mesh_scale)[:2].tolist()])
        else:
            geodesic = path_solver.find_geodesic_path(source, target)
            geodesics.append((geodesic[:, :2] / mesh_scale).tolist())

    return geodesics

# Grab the data from the files
outputs = []
times = []
for t, directory_output in directories_outputs:
    with open(directory_output / 'output.json', 'r') as file_output:
        outputs.append(json.load(file_output))
        times.append(t)

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
        'coordinates': [vertex[0] / mesh_scale, vertex[1] / mesh_scale],
    }
    # [network_vertex[0] / mesh_scale, network_vertex[1] / mesh_scale]
    for network_vertex, label in zip(network_vertices, node_indices_to_labels)
    for vertex in (get_nearest_vertex(mesh, network_vertex),)
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
        for edge, curvature, throughput in zip(graph_data['edges'], edge_data['ricciCurvature'], edge_data['throughput'] if 'throughput' in edge_data else itertools.repeat(1.))
    ]
    for output in outputs
    for (graph_data, vertex_data, edge_data) in (output['network'],)
]

# Compute the network convex hulls
convex_hulls = []
distances_to_convex_hulls = []
for output in outputs:
    network = output['network']
    graph_data, vertex_data, edge_data = network

    network_edges = graph_data['edges']
    network_convex_hulls = convex_hull.compute_connected_convex_hulls(network_vertices, network_edges)
    distances_to_convex_hulls.append(np.array([
        convex_hull.distance_to_convex_hulls(
            np.array(vertex_coordinate),
            network_vertices,
            network_convex_hulls
        )
        for vertex_coordinate in mesh.get_coordinates()[:, :2]
    ]))
    convex_hulls.append(np.where(distances_to_convex_hulls[-1] == 0.)[0])

# Determine values for vertical scaling
z_max = -np.inf
z_min = np.inf
for z, hull in zip(zs, convex_hulls):
    z_max = max(z_max, np.max(z[hull]))
    z_min = min(z_min, np.min(z[hull]))

animation_data = []
for t, z, distance_to_convex_hull, edges in zip(times, zs, distances_to_convex_hulls, animation_edges):
    distance_to_convex_hull = np.maximum(distance_to_convex_hull - 0.05, 0.)
    z = z - z_min
    z = z / (z_max - z_min) * 0.25  # TODO: Finalize height scaling
    z = (z + 0.05) * np.exp(-1000 * distance_to_convex_hull**2) - 0.05

    mesh.set_parameters(z)
    geodesics = compute_geodesics_from_graph(
        mesh, network_vertices,
        [
            (
                node_labels_to_indices[node_source],
                node_labels_to_indices[node_target]
            )
            # for (node_source, node_target) in [
            #     ('SALT', 'CHIC'),
            #     ('NEWY32AOA', 'LASV'),
            #     ('SAND', 'ATLA'),
            #     ('SAND', 'WASH'),
            # ]
            for (node_source, node_target) in itertools.product(
                ['SALT', 'SAND', 'SACR', 'DENV', 'SEAT'],
                ['WASH', 'NEWY32AOA', 'CHIC', 'ATLA', 'HOUS']
            )
        ]
    )

    animation_data.append({
        'time': t,
        'height': z.reshape((width, height)).tolist(),
        'edges': edges,
        'geodesics': geodesics
    })

# Set the map data
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

with open(path_output_collated, 'w') as file_output:
    json.dump(
        {
            'nodes': animation_vertices,
            'animation': animation_data,
            'map': map_data,
        },
        file_output, ensure_ascii=False
    )
