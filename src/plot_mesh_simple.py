import itertools
import pathlib
import pickle
import sys
import typing

import dcelmesh
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from linear_geodesic_optimization import geometry
from linear_geodesic_optimization.data import input_mesh, input_network
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization.curvature import Computer as Curvature
from linear_geodesic_optimization.optimization.laplacian import Computer as Laplacian
from linear_geodesic_optimization.plot import get_rectangular_mesh_plot, \
    get_color_from_float, \
    get_face_colors_curvature_true, \
    get_face_colors_curvature_desired
from linear_geodesic_optimization.connected_components import compute_holes_vertices


# mesh_data_directory = pathlib.PurePath('..', 'outputs', 'test_mesh_scale', 'three_clusters', 'trimmed_0.2', '1.0_0.01_20.0_50_50_0.7')
mesh_data_directory = pathlib.PurePath('..', 'outputs', 'toy', 'three_clusters_animation', '24', '1.0_0.005_20.0_50_50_0.7')
with open(mesh_data_directory / 'output', 'rb') as f:
    mesh_data = pickle.load(f)

parameters = mesh_data['parameters']
network = mesh_data['network']
width = parameters['width']
height = parameters['height']

# First, just figure out which vertices were used in the optimization
mesh_true = input_mesh.get_mesh(
    # np.array(mesh_data['final']) - np.array(mesh_data['initial']),
    mesh_data['final'],
    width, height,
    network,
    parameters['coordinates_scale'],
    parameters['mesh_scale'],
    network_trim_radius = parameters['network_trim_radius'],
)
vertex_index_list = mesh_true.get_trim_mapping()

mesh = RectangleMesh(width, height)
topology = mesh.get_topology()
holes_vertices, _ = compute_holes_vertices(topology, vertex_index_list)

z = np.zeros(mesh.get_topology().n_vertices)
z[vertex_index_list] = np.array(mesh_data['final']) - np.array(mesh_data['initial'])
z = z - np.amin(z[vertex_index_list])
z = z / np.amax(z[vertex_index_list])

for hole in holes_vertices:
    z[list(hole)] = -0.5

# TODO (extremely low priority): Add smoothing around the boundaries of holes
# mesh_coordinates = mesh.get_coordinates()[:,:2]
# for hole, boundary in zip(holes_vertices, boundaries_vertices):
#     boundary_coordinates = np.array([
#         mesh_coordinates[vertex]
#         for vertex in boundary
#     ])[:2]
#     for vertex in hole:
#         distance = np.amin(np.linalg.norm(boundary_coordinates - mesh_coordinates[vertex], axis=1))
#         z[vertex] = np.exp(-1000 * distance**2)

mesh = input_mesh.get_mesh(
    z,
    width, height,
    network,
    parameters['coordinates_scale']
)

face_colors = get_face_colors_curvature_true(mesh_true)
# face_colors = get_face_colors_curvature_desired(mesh_true, network, parameters['coordinates_scale'])

graph_data, vertex_data, edge_data = network
vertex_coordinates = mesh.map_coordinates_to_support(
    np.array(graph_data['coordinates']),
    parameters['coordinates_scale'],
    graph_data['bounding_box']
)
edges = graph_data['edges']

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
get_rectangular_mesh_plot(
    z.reshape((width, height)),
    face_colors,
    '',
    network=(vertex_coordinates, edges, edge_data['ricciCurvature'], graph_data['labels']),
    ax=ax
)

plt.show()
