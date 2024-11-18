import itertools
import pathlib
import pickle
import typing

import dcelmesh
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from linear_geodesic_optimization.data import input_mesh, input_network
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization.curvature import Computer as Curvature
from linear_geodesic_optimization.optimization.laplacian import Computer as Laplacian
from linear_geodesic_optimization.plot import get_mesh_plot, get_rectangular_mesh_plot, get_heat_map


def compute_connected_components_faces(topology: dcelmesh.Mesh, face_indices) -> typing.List[int]:
    """
    Given a topology and faces in the topology, compute the connected
    components of the faces.
    """
    connected_components = []
    unused_face_indices = set(face_indices)
    while unused_face_indices:
        stack = [next(iter(unused_face_indices))]
        unused_face_indices.remove(stack[0])
        connected_component = set(stack)
        while stack:
            face = topology.get_face(stack.pop())
            for neighbor in face.faces():
                if neighbor.index in unused_face_indices:
                    stack.append(neighbor.index)
                    unused_face_indices.remove(neighbor.index)
                    connected_component.add(neighbor.index)
        connected_components.append(connected_component)

    return connected_components

def compute_holes_faces(topology: dcelmesh.Mesh, vertex_indices) -> typing.List[typing.List[int]]:
    """
    Given a topology and vertices in the topology, compute the
    connected components of the faces not entirely in the set of
    vertices.
    """
    vertex_indices = set(vertex_indices)
    face_indices = []
    for face in topology.faces():
        for vertex in face.vertices():
            if vertex.index not in vertex_indices:
                face_indices.append(face.index)
                break
    return compute_connected_components_faces(
        topology, face_indices
    )

def compute_holes_vertices(topology: dcelmesh.Mesh, vertex_indices) -> typing.Tuple[typing.List[typing.Set[int]], typing.List[typing.Set[int]]]:
    holes_faces = compute_holes_faces(topology, vertex_indices)
    holes = []
    boundaries = []
    for hole_faces in holes_faces:
        halfedge_indices = set(
            halfedge.index
            for face_index in hole_faces
            for halfedge in topology.get_face(face_index).halfedges()
        )
        boundary_halfedge_indices = [
            halfedge_index
            for halfedge_index in halfedge_indices
            if topology.get_halfedge(halfedge_index).is_on_boundary()
        ]
        boundary_vertex_indices = set(
            topology.get_halfedge(boundary_halfedge_index).origin.index
            for boundary_halfedge_index in boundary_halfedge_indices
        )
        hole_vertex_indices = set(
            vertex.index
            for face_index in hole_faces
            for vertex in topology.get_face(face_index).vertices()
        )
        holes.append(hole_vertex_indices)
        boundaries.append(boundary_vertex_indices)
    return holes, boundaries

def get_color(value: float, value_min: float, value_max: float) -> typing.Tuple[float, float, float]:
    return list(mpl.colormaps['RdBu']((value - value_min) / (value_max - value_min)))[:3]

# mesh_data_directory = pathlib.PurePath('..', 'outputs', 'toy', 'three_clusters', 'untrimmed', '1.0_0.005_20.0_50_50_1.0')
mesh_data_directory = pathlib.PurePath('..', 'outputs', 'toy', 'three_clusters', 'trimmed_0.2', '1.0_0.005_20.0_50_50_1.0')
with open(mesh_data_directory / 'output', 'rb') as f:
    mesh_data = pickle.load(f)

parameters = mesh_data['parameters']
network = mesh_data['network']
width = parameters['width']
height = parameters['height']

# First, just figure out which vertices were used in the optimization
mesh = input_mesh.get_mesh(
    # np.array(mesh_data['final']) - np.array(mesh_data['initial']),
    mesh_data['final'],
    width, height,
    network,
    parameters['coordinates_scale'],
    network_trim_radius = parameters['network_trim_radius'],
)
vertex_index_list = mesh.get_trim_mapping()

curvature = Curvature(mesh, Laplacian(mesh))
curvature.forward()
kappa_G = curvature.kappa_G
kappa_G_min = min(kappa_G)
kappa_G_max = max(kappa_G)
kappa_G_bound = max(kappa_G_max, -kappa_G_min)
print(kappa_G_min, kappa_G_max)

mesh = RectangleMesh(width, height)
topology = mesh.get_topology()
holes_vertices, boundaries_vertices = compute_holes_vertices(topology, vertex_index_list)
holes_faces = compute_holes_faces(topology, vertex_index_list)

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

vertex_index_lookup = {vertex_index: index for index, vertex_index in enumerate(vertex_index_list)}
# face_colors = np.array([
#     [0.6, 0.6, 0.6] if face.index in itertools.chain(holes_faces) else [0.3, 0.3, 0.3]
#     for face in mesh.get_topology().faces()
# ])
face_colors = np.array([
    get_color(kappa_G[vertex_index_lookup[vertex.index]], -kappa_G_bound, kappa_G_bound) if vertex.index in vertex_index_lookup else [0.3, 0.3, 0.3]
    for vertex in mesh.get_topology().vertices()
]).reshape((width, height, 3))

graph_data, vertex_data, edge_data = network
vertex_coordinates = mesh.map_coordinates_to_support(
    np.array(graph_data['coordinates']),
    parameters['coordinates_scale'],
    graph_data['bounding_box']
)
edges = graph_data['edges']

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# get_mesh_plot(
#     mesh,
#     face_colors = face_colors,
#     network = (vertex_coordinates, edges, edge_data['ricciCurvature'], graph_data['labels']),
#     ax = ax
# )
get_rectangular_mesh_plot(
    z.reshape((width, height)),
    face_colors,
    '',
    network=(vertex_coordinates, edges, edge_data['ricciCurvature'], graph_data['labels']),
    ax=ax
)

# curvature = Curvature(mesh, Laplacian(mesh))
# curvature.forward()
# kappa_G = curvature.kappa_G
# kappa_G_min = min(kappa_G)
# kappa_G_max = max(kappa_G)
# kappa_G_bound = max(kappa_G_max, -kappa_G_min)
# print(kappa_G_min, kappa_G_max, kappa_G_bound)
# get_heat_map(list(range(width)), list(range(height)), np.array(curvature.kappa_G).reshape((width, height)), v_range=(-kappa_G_bound, kappa_G_bound))

plt.show()
