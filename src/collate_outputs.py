import itertools
import json
import os
import pathlib
import typing

import numpy as np
import potpourri3d as pp3d

from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.data import input_mesh, utility

directory_outputs = pathlib.PurePath('..', 'outputs', 'json_test', 'toy', 'three_clusters')
subdirectory_output = '1.0_0.005_20.0_50_50_0.7'
directories_outputs = list(sorted([
    (float(directory_output), directory_outputs / directory_output / subdirectory_output)
    for directory_output in os.listdir(directory_outputs)
    if os.path.isdir(directory_outputs / directory_output)
]))

def get_nearest_vertex(mesh: RectangleMesh, vertex):
    nearest_vertex = mesh.get_coordinates()[mesh.nearest_vertex(vertex).index]
    return [nearest_vertex[0], nearest_vertex[1]]

def compute_geodesics(mesh: RectangleMesh, network_vertices, network_edges):
    mesh_scale = mesh.get_scale()

    path_solver = pp3d.EdgeFlipGeodesicSolver(
        mesh.get_coordinates(),
        np.array([
            [vertex.index for vertex in face.vertices()]
            for face in mesh.get_topology().faces()
        ])
    )

    geodesics = []
    # for (index_source, index_target) in network_edges:
    for (index_source, index_target) in itertools.product(range(len(network_vertices)), range(len(network_vertices))):
        source = mesh.nearest_vertex(network_vertices[index_source]).index
        target = mesh.nearest_vertex(network_vertices[index_target]).index
        if source == target:
            geodesics.append([(mesh.get_coordinates()[source] / mesh_scale)[:2].tolist()])
        else:
            geodesic = path_solver.find_geodesic_path(source, target)
            geodesics.append((geodesic[:, :2] / mesh_scale).tolist())

    return geodesics

coordinates_scale = None
network_data = None

# Set the animation data
animation_data = []
for t, directory_output in directories_outputs:
    print(t)
    with open(directory_output / 'output.json', 'r') as file_output:
        output = json.load(file_output)

        width = output['parameters']['width']
        height = output['parameters']['height']

        coordinates_scale = output['parameters']['coordinates_scale']
        mesh_scale = output['parameters']['mesh_scale']

        # Make a mesh, pretty much just so that we compute the trim mapping
        mesh = input_mesh.get_mesh(
            output['final'],
            width,
            height,
            output['network'],
            coordinates_scale,
            mesh_scale,
            network_trim_radius=output['parameters']['network_trim_radius']
        )

        graph_data, vertex_data, edge_data = output['network']
        coordinates = np.array(graph_data['coordinates'])
        bounding_box = graph_data['bounding_box']
        network_edges = graph_data['edges']
        network_vertices = mesh.map_coordinates_to_support(coordinates, coordinates_scale, bounding_box)
        vertices = [
            [vertex[0] / mesh_scale, vertex[1] / mesh_scale]
            for network_vertex in network_vertices
            for vertex in (get_nearest_vertex(mesh, network_vertex),)
        ]
        edges = [
            {
                'source': edge[0],
                'target': edge[1],
                'weight': weight,
            }
            for edge, weight in zip(network_edges, edge_data['throughput'])
        ]

        z = np.full(width * height, -0.5)
        z_trim_mapping = np.array(output['final']) - np.array(output['initial'])
        z_trim_mapping = z_trim_mapping - np.amin(z_trim_mapping)
        z_trim_mapping = z_trim_mapping / np.amax(z_trim_mapping)
        # z_trim_mapping = np.array(output['final'])
        z[mesh.get_trim_mapping()] = z_trim_mapping
        # mesh.set_parameters(z_trim_mapping)

        animation_data.append({
            'time': t,
            'height': z.reshape((width, height)).tolist(),
            'network': {
                'vertices': vertices,
                'edges': edges,
            },
            'geodesics': compute_geodesics(mesh, network_vertices, network_edges),
        })

        if coordinates_scale is None:
            coordinates_scale = output['parameters']['coordinates_scale']
        if network_data is None:
            network_data = output['network']

# Set the map data
graph_data, _, _ = network_data
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

with open(directory_outputs / 'output.json', 'w') as file_output:
    json.dump(
        {
            'animation': animation_data,
            'map': map_data,
        },
        file_output, ensure_ascii=False
    )
