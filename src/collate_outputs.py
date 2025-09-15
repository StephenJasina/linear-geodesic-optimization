import itertools
import json
import os
import pathlib
import sys
import typing

import numpy as np
import potpourri3d as pp3d

from linear_geodesic_optimization.data import utility
from linear_geodesic_optimization.graph import boundary
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh

# Outputs are stored in `directory_outputs` / <output number> / `subdirectory_output`
epsilon = 10
# directory_outputs = pathlib.PurePath('..', 'outputs', 'ripe', 'ASN_20473')
# directory_outputs = pathlib.PurePath('..', 'outputs', 'ripe', 'ASN_20473_longer')
# directory_outputs = pathlib.PurePath('..', 'outputs', 'ripe', 'ASN_20473')
# directory_outputs = pathlib.PurePath('..', 'outputs', 'ripe', 'high_variation_US')
directory_outputs = pathlib.PurePath('..', 'outputs', 'graph_US')
# directory_outputs = pathlib.PurePath('..', 'outputs', 'graph_US')
subdirectory_output = pathlib.PurePath(f'{epsilon}_0.0002_50_50')
directories_outputs = list(sorted([
    (
        float(directory_output),
        directory_outputs / directory_output / subdirectory_output,
    )
    for i, directory_output in enumerate(sorted(os.listdir(directory_outputs)))
    if os.path.isdir(directory_outputs / directory_output)
]))
# path_output_collated = pathlib.PurePath('..', 'outputs', 'animations', f'ASN_20473_{epsilon}.json')
# path_output_collated = pathlib.PurePath('..', 'outputs', 'animations', f'ASN_20473_{epsilon}_longer.json')
# path_output_collated = pathlib.PurePath('..', 'outputs', 'animations', 'ASN_20473_16_0.002.json')
# path_output_collated = pathlib.PurePath('..', 'outputs', 'animations', f'high_variation_US_{epsilon}.json')
path_output_collated = pathlib.PurePath('..', 'outputs', 'animations', f'graph_US_{epsilon}_convex_hull.json')

height_scale = 0.10
use_convex_hull = True

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
                mesh.add_vertex_at_coordinates(network_vertex)
            except ValueError:
                bad_indices.add(network_index)

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
        for edge, curvature, throughput in zip(graph_data['edges'], edge_data['ricciCurvature'], edge_data['throughput'] if 'throughput' in edge_data else itertools.repeat(1.))
    ]
    for output in outputs
    for (graph_data, vertex_data, edge_data) in (output['network'],)
]

# Compute the network borders
network_borders = []
hulls = []
distances_to_borders = []
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
    distances_to_borders.append(np.array([
        boundary.distance_to_border(
            np.array(vertex_coordinate),
            network_border
        )
        for vertex_coordinate in mesh.get_coordinates()[:, :2]
    ]))
    hulls.append(np.where(distances_to_borders[-1] == 0.)[0])

# Determine values for vertical scaling
z_max = -np.inf
z_min = np.inf
for z, hull in zip(zs, hulls):
    z_max = max(z_max, np.max(z[hull]))
    z_min = min(z_min, np.min(z[hull]))

# geodesic_label_color_pairs = [
#     (('ALBQ', 'SALT'), [31, 119, 180]),
#     (('ALBQ', 'SAND'), [31, 119, 180]),
#     (('ALBQ', 'FNALGCC'), [31, 119, 180]),
#     (('EQXDC4', 'WASH'), [31, 119, 180]),
#     (('EQXDC4', 'SAND'), [31, 119, 180]),
#     (('EQXDC4', 'FNALGCC'), [174, 199, 232]),
#     (('ATLA', 'SAND'), [174, 199, 232]),
#     (('LBNL50', 'LBNL59'), [174, 199, 232]),
#     (('LBNL50', 'STAR'), [174, 199, 232]),
#     (('LBNL59', 'SAND'), [255, 127, 14]),
#     (('BOIS', 'EQXCH2'), [255, 127, 14]),
#     (('BOIS', 'DENV'), [255, 127, 14]),
#     (('BOIS', 'KANS'), [255, 127, 14]),
#     (('BOIS', 'GA'), [255, 187, 120]),
#     (('BOST', 'SAND'), [255, 187, 120]),
#     (('CHAT', 'CHIC'), [255, 187, 120]),
#     (('CHAT', 'SAND'), [255, 187, 120]),
#     (('EQXCH2', 'LBNL50'), [44, 160, 44]),
#     (('EQXCH2', 'STAR'), [44, 160, 44]),
#     (('EQXCH2', 'INLEIL'), [44, 160, 44]),
#     (('CHIC', 'NASH'), [44, 160, 44]),
#     (('EQXCH2', 'ORNL1064'), [152, 223, 138]),
#     (('EQXCH2', 'SACR'), [152, 223, 138]),
#     (('CHIC', 'SALT'), [152, 223, 138]),
#     (('EQXCH2', 'SAND'), [152, 223, 138]),
#     (('CHIC', 'SEAT'), [214, 39, 40]),
#     (('EQXCH2', 'FNALGCC'), [214, 39, 40]),
#     (('DENV', 'LBNL50'), [214, 39, 40]),
#     (('DENV', 'INLEIL'), [214, 39, 40]),
#     (('DENV', 'LASV'), [255, 152, 150]),
#     (('DENV', 'SACR'), [255, 152, 150]),
#     (('DENV', 'SALT'), [255, 152, 150]),
#     (('DENV', 'SAND'), [255, 152, 150]),
#     (('DENV', 'SEAT'), [148, 103, 189]),
#     (('DENV', 'FNALGCC'), [148, 103, 189]),
#     (('FRIB', 'ORNL5600'), [148, 103, 189]),
#     (('ELPA', 'GA'), [148, 103, 189]),
#     (('ELPA', 'FNALGCC'), [197, 176, 213]),
#     (('HOUS', 'SAND'), [197, 176, 213]),
#     (('KANS', 'LBNL50'), [197, 176, 213]),
#     (('KANS', 'SACR'), [197, 176, 213]),
#     (('KANS', 'SALT'), [140, 86, 75]),
#     (('KANS', 'SAND'), [140, 86, 75]),
#     (('KANS', 'SEAT'), [140, 86, 75]),
#     (('LASV', 'SAND'), [140, 86, 75]),
#     (('ANL541B', 'LBNL50'), [196, 156, 148]),
#     (('ANL541B', 'INLEIL'), [196, 156, 148]),
#     (('ANL221', 'ANL541B'), [196, 156, 148]),
#     (('ANL221', 'ORAU'), [196, 156, 148]),
#     (('ANL221', 'FNALFCC'), [227, 119, 194]),
#     (('LLNL', 'SAND'), [227, 119, 194]),
#     (('LANLTA50', 'LBNL50'), [227, 119, 194]),
#     (('LOSA', 'SAND'), [227, 119, 194]),
#     (('SLAC50N', 'SLAC50S'), [247, 182, 210]),
#     (('NASH', 'STAR'), [247, 182, 210]),
#     (('NASH', 'SAND'), [247, 182, 210]),
#     (('NEWY1118TH', 'NEWY32AOA'), [247, 182, 210]),
#     (('NEWY32AOA', 'SAND'), [127, 127, 127]),
#     (('ORAU', 'STAR'), [127, 127, 127]),
#     (('ORNL1064', 'ORNL5600'), [127, 127, 127]),
#     (('ORAU', 'SAND'), [127, 127, 127]),
#     (('PANTEX', 'SAND'), [199, 199, 199]),
#     (('SACR', 'STAR'), [199, 199, 199]),
#     (('SACR', 'SAND'), [199, 199, 199]),
#     (('SALT', 'STAR'), [199, 199, 199]),
#     (('SALT', 'SAND'), [188, 189, 34]),
#     (('GA', 'LBNL59'), [188, 189, 34]),
#     (('SAND', 'STAR'), [188, 189, 34]),
#     (('GA', 'LASV'), [188, 189, 34]),
#     (('GA', 'LLNL'), [219, 219, 141]),
#     (('SAND', 'WASH'), [219, 219, 141]),
#     (('GA', 'SALT'), [219, 219, 141]),
#     (('GA', 'SAND'), [219, 219, 141]),
#     (('SAND', 'SEAT'), [23, 190, 207]),
#     (('EQXSV5', 'SAND'), [23, 190, 207]),
#     (('EQXSV5', 'FNALGCC'), [23, 190, 207]),
#     (('BNL515', 'SAND'), [23, 190, 207]),
#     (('BNL515', 'BNL725'), [158, 218, 229]),
#     (('FNALGCC', 'LBNL59'), [158, 218, 229]),
#     (('FNALGCC', 'SLAC50S'), [158, 218, 229]),
#     (('FNALFCC', 'ORNL5600'), [158, 218, 229]),
#     (('FORRESTAL', 'GERMANTOWN'), [158, 218, 229]),
# ]
# geodesic_label_color_pairs = [
#     (('BOST', 'SEAT'), [0, 0, 0]),
# ]
# geodesic_label_color_pairs = [
#     (('6989', '6987'), [0, 0, 0]),
#     (('6832', '6946'), [0, 0, 0]),
#     (('6989', '6946'), [0, 0, 0]),
#     (('6832', '6987'), [0, 0, 0]),
#     (('6875', '6987'), [0, 0, 0]),
#     (('6517', '6987'), [0, 0, 0]),
#     (('6574', '6987'), [0, 0, 0]),
#     (('7380', '6987'), [0, 0, 0]),
#     (('7193', '6436'), [0, 0, 0]),
#     (('6725', '6987'), [0, 0, 0]),
#     (('6649', '6957'), [0, 0, 0]),
#     (('6875', '6946'), [0, 0, 0]),
#     (('7193', '6927'), [0, 0, 0]),
#     (('6517', '6946'), [0, 0, 0]),
#     (('7193', '6972'), [0, 0, 0]),
#     (('7193', '6783'), [0, 0, 0]),
#     (('7253', '7121'), [0, 0, 0]),
#     (('6379', '6092'), [0, 0, 0]),
#     (('6574', '6946'), [0, 0, 0]),
#     (('7380', '6946'), [0, 0, 0]),
#     (('7193', '7380'), [0, 0, 0]),
#     (('6725', '6946'), [0, 0, 0]),
#     (('6436', '6957'), [0, 0, 0]),
#     (('6973', '7363'), [0, 0, 0]),
#     (('7227', '6525'), [0, 0, 0]),
#     (('6408', '6989'), [0, 0, 0]),
#     (('6989', '6408'), [0, 0, 0]),
#     (('7348', '7065'), [0, 0, 0]),
#     (('7193', '7065'), [0, 0, 0]),
#     (('7253', '6947'), [0, 0, 0]),
#     (('6411', '7011'), [0, 0, 0]),
#     (('6411', '7043'), [0, 0, 0]),
#     (('6588', '7065'), [0, 0, 0]),
#     (('6411', '6794'), [0, 0, 0]),
#     (('6411', '6434'), [0, 0, 0]),
#     (('7193', '6588'), [0, 0, 0]),
#     (('7193', '6524'), [0, 0, 0]),
#     (('6898', '6411'), [0, 0, 0]),
#     (('7193', '6231'), [0, 0, 0]),
#     (('7193', '6899'), [0, 0, 0]),
#     (('7193', '6517'), [0, 0, 0]),
#     (('6783', '7065'), [0, 0, 0]),
#     (('6992', '6947'), [0, 0, 0]),
#     (('6992', '7011'), [0, 0, 0]),
#     (('6973', '7011'), [0, 0, 0]),
#     (('7014', '6454'), [0, 0, 0]),
#     (('6898', '6407'), [0, 0, 0]),
#     (('6898', '7011'), [0, 0, 0]),
#     (('6898', '7043'), [0, 0, 0]),
#     (('7193', '6957'), [0, 0, 0]),
#     (('6898', '6434'), [0, 0, 0]),
#     (('6411', '6898'), [0, 0, 0]),
#     (('7187', '6428'), [0, 0, 0]),
#     (('6898', '6794'), [0, 0, 0]),
#     (('7227', '6947'), [0, 0, 0]),
#     (('7253', '6407'), [0, 0, 0]),
#     (('6899', '7065'), [0, 0, 0]),
#     (('6783', '7380'), [0, 0, 0]),
#     (('6408', '7360'), [0, 0, 0]),
#     (('6832', '7043'), [0, 0, 0]),
#     (('6408', '6992'), [0, 0, 0]),
#     (('6992', '6408'), [0, 0, 0]),
#     (('7360', '6408'), [0, 0, 0]),
#     (('6973', '6434'), [0, 0, 0]),
#     (('6989', '6783'), [0, 0, 0]),
#     (('6649', '6947'), [0, 0, 0]),
#     (('6989', '6434'), [0, 0, 0]),
#     (('6769', '7348'), [0, 0, 0]),
#     (('7380', '6783'), [0, 0, 0]),
#     (('6411', '6947'), [0, 0, 0]),
#     (('6989', '6794'), [0, 0, 0]),
#     (('6898', '6649'), [0, 0, 0]),
#     (('6832', '6585'), [0, 0, 0]),
#     (('6832', '6407'), [0, 0, 0]),
#     (('7227', '6492'), [0, 0, 0]),
#     (('6973', '6407'), [0, 0, 0]),
#     (('6524', '6903'), [0, 0, 0]),
#     (('6898', '7363'), [0, 0, 0]),
#     (('6524', '6643'), [0, 0, 0]),
#     (('6408', '6492'), [0, 0, 0]),
#     (('7227', '6411'), [0, 0, 0]),
#     (('6574', '6875'), [0, 0, 0]),
#     (('6875', '6574'), [0, 0, 0]),
#     (('6972', '6797'), [0, 0, 0]),
#     (('6408', '6599'), [0, 0, 0]),
#     (('7187', '6794'), [0, 0, 0]),
#     (('6524', '7061'), [0, 0, 0]),
#     (('6524', '7346'), [0, 0, 0]),
#     (('7014', '7334'), [0, 0, 0]),
#     (('6832', '7011'), [0, 0, 0]),
#     (('6524', '6875'), [0, 0, 0]),
#     (('6411', '6492'), [0, 0, 0]),
#     (('6973', '6947'), [0, 0, 0]),
#     (('6898', '6525'), [0, 0, 0]),
#     (('6797', '6972'), [0, 0, 0]),
#     (('6992', '6434'), [0, 0, 0]),
#     (('6898', '7395'), [0, 0, 0]),
#     (('6379', '6987'), [0, 0, 0]),
#     (('7348', '6454'), [0, 0, 0]),
#     (('7157', '6492'), [0, 0, 0]),
#     (('6988', '6988'), [0, 0, 0]),
#     (('7193', '7381'), [0, 0, 0]),
#     (('6898', '7361'), [0, 0, 0]),
#     (('6649', '6898'), [0, 0, 0]),
#     (('7193', '6629'), [0, 0, 0]),
#     (('6898', '6898'), [0, 0, 0]),
#     (('7158', '6875'), [0, 0, 0]),
#     (('6649', '6649'), [0, 0, 0]),
#     (('6992', '7043'), [0, 0, 0]),
#     (('6411', '7363'), [0, 0, 0]),
#     (('6517', '6794'), [0, 0, 0]),
#     (('6989', '7043'), [0, 0, 0]),
#     (('6989', '6928'), [0, 0, 0]),
#     (('6898', '7121'), [0, 0, 0]),
#     (('6379', '7014'), [0, 0, 0]),
#     (('6988', '6454'), [0, 0, 0]),
#     (('6649', '7363'), [0, 0, 0]),
#     (('6989', '7363'), [0, 0, 0]),
#     (('7014', '6379'), [0, 0, 0]),
#     (('6992', '6407'), [0, 0, 0]),
# ]
# geodesic_label_color_pairs = [
#     ((source, target), [0, 0, 0])
#     for source in graph_data['labels']
#     for target in graph_data['labels']
#     if source < target
# ]
geodesic_label_color_pairs = [
    (('6144', '6066'), [0, 0, 0]),
    (('6097', '6288'), [0, 0, 0]),
]
# geodesic_label_color_pairs = []
geodesic_labels = [geodesic_label for geodesic_label, _ in geodesic_label_color_pairs]
edge_colors = [list(color) for _, color in geodesic_label_color_pairs]

animation_data = []
for t, z, distance_to_network, edges, network_border in zip(times, zs, distances_to_borders, animation_edges, network_borders):
    z_original = np.copy(z)

    hull = [index for index in range(mesh.get_topology().n_vertices) if distance_to_network[index] < 0.025]
    distance_to_network = np.maximum(distance_to_network - 0.025, 0.)
    z = z - z_min
    z = z / (z_max - z_min) * height_scale
    z = (z + 0.05) * np.exp(-1000 * distance_to_network**2) - 0.05

    mesh.set_parameters(z)
    # TODO: Make this safer
    # mesh.trim_to_set(hull)
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

    animation_data.append({
        'time': t,
        'height': z.reshape((width, height)).tolist(),
        'edges': edges,
        'geodesics': geodesics,
        'edgeColors': edge_colors,
        'border': network_border
    })

# Set the map data
# TODO: Make this more robust
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
