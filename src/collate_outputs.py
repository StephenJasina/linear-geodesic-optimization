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
# directories_outputs = [
#     (0., pathlib.PurePath('..', 'outputs', 'esnet', '1742868000000', '0.0002_50_50')),
#     (1., pathlib.PurePath('..', 'outputs', 'esnet', '1742871600000', '0.0002_50_50')),
#     (2., pathlib.PurePath('..', 'outputs', 'esnet', '1742875200000', '0.0002_50_50')),
#     (3., pathlib.PurePath('..', 'outputs', 'esnet', '1742878800000', '0.0002_50_50')),
#     (4., pathlib.PurePath('..', 'outputs', 'esnet', '1742882400000', '0.0002_50_50')),
#     (5., pathlib.PurePath('..', 'outputs', 'esnet', '1742886000000', '0.0002_50_50')),
#     (6., pathlib.PurePath('..', 'outputs', 'esnet', '1742889600000', '0.0002_50_50')),
#     (7., pathlib.PurePath('..', 'outputs', 'esnet', '1742893200000', '0.0002_50_50')),
#     (8., pathlib.PurePath('..', 'outputs', 'esnet', '1742896800000', '0.0002_50_50')),
#     (9., pathlib.PurePath('..', 'outputs', 'esnet', '1742900400000', '0.0002_50_50')),
#     (10., pathlib.PurePath('..', 'outputs', 'esnet', '1742904000000', '0.0002_50_50')),
#     (11., pathlib.PurePath('..', 'outputs', 'esnet', '1742907600000', '0.0002_50_50')),
#     (12., pathlib.PurePath('..', 'outputs', 'esnet', '1742911200000', '0.0002_50_50')),
#     (13., pathlib.PurePath('..', 'outputs', 'esnet', '1742914800000', '0.0002_50_50')),
# ]

# For ESnet toy data
directories_outputs = [
    (0., pathlib.PurePath('..', 'outputs', 'toy', 'esnet', '0', '0.001_30_30')),
    (1., pathlib.PurePath('..', 'outputs', 'toy', 'esnet', '1', '0.001_30_30')),
    (2., pathlib.PurePath('..', 'outputs', 'toy', 'esnet', '2', '0.001_30_30')),
]

path_output_collated = pathlib.PurePath('..', 'outputs', 'toy', 'esnet', 'output_0.001.json')

def get_nearest_vertex(mesh: RectangleMesh, vertex):
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
zs = []
convex_hulls = []
convex_hulls_boundaries = []
network_data = []
meshes = []
networks_for_animation = []
for t, directory_output in directories_outputs:
    print(t)
    with open(directory_output / 'output.json', 'r') as file_output:
        output = json.load(file_output)

        width = output['parameters']['width']
        height = output['parameters']['height']
        mesh_scale = output['parameters']['mesh_scale']
        coordinates_scale = output['parameters']['coordinates_scale']


        z_final = np.array(output['final'])
        z_initial = np.array(output['initial'])
        if not zs:
            zs.append(z_final - z_initial)
        else:
            zs.append(z_final - z_initial + np.mean(zs[-1] - (z_final - z_initial)))

        # TODO: Ensure that the vertices are all the same
        graph_data, vertex_data, edge_data = output['network']
        network_data.append((graph_data, vertex_data, edge_data))

        mesh = RectangleMesh(width, height, mesh_scale)
        meshes.append(mesh)

        coordinates = np.array(graph_data['coordinates'])
        bounding_box = graph_data['bounding_box']
        network_vertices = mesh.map_coordinates_to_support(coordinates, coordinates_scale, bounding_box)
        network_edges = graph_data['edges']
        network_convex_hulls = convex_hull.compute_connected_convex_hulls(network_vertices, network_edges)

        convex_hulls_boundaries.append(network_convex_hulls)

        mesh_coordinates = mesh.get_coordinates()[:, :2]
        distances_to_convex_hulls = np.array([
            convex_hull.distance_to_convex_hulls(
                np.array(vertex_coordinate),
                network_vertices,
                network_convex_hulls
            )
            for vertex_coordinate in mesh_coordinates
        ])
        convex_hulls.append(np.where(distances_to_convex_hulls == 0.)[0])

        vertices = [
            [vertex[0] / mesh_scale, vertex[1] / mesh_scale]
            # [network_vertex[0] / mesh_scale, network_vertex[1] / mesh_scale]
            for network_vertex in network_vertices
            for vertex in (get_nearest_vertex(mesh, network_vertex),)
        ]
        if 'throughput' not in edge_data:
            edge_data['throughput'] = itertools.cycle([1])
        edges = [
            {
                'source': edge[0],
                'target': edge[1],
                'curvature': curvature,
                'throughput': throughput,
            }
            for edge, curvature, throughput in zip(network_edges, edge_data['ricciCurvature'], edge_data['throughput'])
        ]
        networks_for_animation.append({
            'vertices': vertices,
            'edges': edges,
        })

# Determine values for vertical scaling
max_height = -np.inf
min_height = np.inf
for i in range(len(directories_outputs)):
    max_height = max(max_height, np.max(zs[i][convex_hulls[i]]))
    min_height = min(min_height, np.min(zs[i][convex_hulls[i]]))

animation_data = []
for i, (t, _) in enumerate(directories_outputs):
    # Rescale heights
    mesh = meshes[i]
    vertices = mesh.get_coordinates()[:, :2]
    z = zs[i]
    convex_hulls_boundary = convex_hulls_boundaries[i]
    distances_to_convex_hulls = np.array([
        convex_hull.distance_to_convex_hulls(
            np.array(vertex_coordinate),
            network_vertices,
            network_convex_hulls
        )
        for vertex_coordinate in vertices
    ])
    distances_to_convex_hulls = np.maximum(distances_to_convex_hulls - 0.05, 0.)
    z = z - min_height
    z = z / (max_height - min_height) * 0.25  # TODO: Finalize height scaling
    z = (z + 0.05) * np.exp(-1000 * distances_to_convex_hulls**2) - 0.05

    animation_data.append({
        'time': t,
        'height': z.reshape((width, height)).tolist(),
        'network': networks_for_animation[i],
        'geodesics': compute_geodesics_from_graph(
            mesh, network_vertices,
            [
                # (source_index, target_index)
                # for connected_component in get_connected_components(len(network_vertices), network_edges)
                # for source_index in connected_component
                # for target_index in connected_component
                # if source_index < target_index
            ]
        ),
    })

# # Set the map data
graph_data, _, _ = network_data[0]
coordinates = np.array(graph_data['coordinates'])
center_xy = (np.amin(coordinates, axis=0) + np.amax(coordinates, axis=0)) / 2.
center = utility.inverse_mercator(*center_xy)
left, _ = utility.inverse_mercator(np.amin(coordinates[:,0]), 0)
right, _ = utility.inverse_mercator(np.amax(coordinates[:,0]), 0)
zoom_factor = coordinates_scale * 360. / (right - left)
map_data = {
    'center': center,
    'zoomFactor': zoom_factor,
}

# with open(directory_outputs / '..' / 'output.json', 'w') as file_output:
with open(path_output_collated, 'w') as file_output:
    json.dump(
        {
            'animation': animation_data,
            'map': map_data,
        },
        file_output, ensure_ascii=False
    )
