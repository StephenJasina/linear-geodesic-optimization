import csv
import json
import os
import pickle
import typing

import networkx as nx
import numpy as np
from utils import *

from linear_geodesic_optimization import convex_hull
from linear_geodesic_optimization.mesh.mesh import Mesh
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh

def mercator(longitude: np.float64, latitude: np.float64) \
        -> typing.Tuple[np.float64, np.float64]:
    '''
    Given a longitude in [-180, 180] and a latitude in [-90, 90], return an
    (x, y) pair representing the location on a Mercator projection. Assuming
    the latitude is no larger/smaller than +/- 85 (approximately), the pair
    will lie in [-0.5, 0.5]^2.
    '''

    x = longitude / 360.
    y = np.log(np.tan(np.pi / 4. + latitude * np.pi / 360.)) / (2. * np.pi)
    return (x, y)

def inverse_mercator(x: np.float64, y: np.float64) \
        -> typing.Tuple[np.float64, np.float64]:
    longitude = x * 360.
    latitude = np.arctan(np.exp(y * 2. * np.pi)) * 360. / np.pi - 90.
    return (longitude, latitude)

def read_graphml(data_file_path: str,
                 latencies_file_path: typing.Optional[str] = None,
                 with_labels: bool = False) \
        -> typing.Union[
            typing.Tuple[
                typing.List[typing.Tuple[np.float64, np.float64]],
                typing.Optional[typing.Tuple[np.float64, np.float64,
                                             np.float64, np.float64]],
                typing.List[typing.Tuple[int, int]],
                typing.List[np.float64],
                typing.List[typing.Tuple[typing.Tuple[int, int], np.float64]]
            ],
            typing.Tuple[
                typing.List[typing.Tuple[np.float64, np.float64]],
                typing.Optional[typing.Tuple[np.float64, np.float64,
                                             np.float64, np.float64]],
                typing.List[typing.Tuple[int, int]],
                typing.List[np.float64],
                typing.List[typing.Tuple[typing.Tuple[int, int], np.float64]],
                typing.List[str]
            ]
        ]:
    """
    Parse a graphml file and return a tuple of relevant data.

    Given the path of a graphml file, the optional path of a csv file,
    and an optional boolean, return:
    * A list of (x, y) coordinate pairs representing vertices
    * An optional bounding box aruond the coordinates
    * A list of pairs of indices into the vertex list representing edges
    * A list of (Ollivier-Ricci) curvatures of each of the edges
    * A list of list of pairs representing measured latencies. This is
      stored in the format ((i_index, j_index), latency)
    * Optionally, a list of labels for each of the vertices
    """
    network = nx.read_graphml(data_file_path)
    coordinates: typing.List[typing.Tuple[np.float64, np.float64]] \
        = [mercator(node['long'], node['lat'])
           for node in network.nodes.values()]
    bounding_box = None
    if 'long_min' in network.graph:
        coordiate_min = mercator(network.graph['long_min'],
                                 network.graph['lat_min'])
        coordiate_max = mercator(network.graph['long_max'],
                                 network.graph['lat_max'])
        bounding_box = (
            coordiate_min[0], coordiate_max[0],
            coordiate_min[1], coordiate_max[1]
        )
    label_to_index = {label: index
                      for index, label in enumerate(network.nodes)}
    network_edges: typing.List[typing.Tuple[int, int]] \
        = [(label_to_index[u], label_to_index[v])
           for u, v in network.edges]
    network_curvatures: typing.List[np.float64] \
        = [edge['ricciCurvature']
           for edge in network.edges.values()]
    network_latencies: typing.List[typing.Tuple[typing.Tuple[int, int],
                                                np.float64]] = []

    if latencies_file_path is not None:
        with open(latencies_file_path) as latencies_file:
            latencies_reader = csv.DictReader(latencies_file)
            for row in latencies_reader:
                source_id = row['source_id']
                target_id = row['target_id']
                if source_id in label_to_index and target_id in label_to_index:
                    network_latencies.append((
                        (label_to_index[source_id],
                         label_to_index[target_id]),
                        row['rtt']
                    ))

    if with_labels:
        network_city : typing.List[str] = [node['city'] for node in network.nodes.values()]

        return coordinates, bounding_box, network_edges, network_curvatures, \
            network_latencies, list(network.nodes), network_city
    else:
        return coordinates, bounding_box, network_edges, network_curvatures, \
            network_latencies

def map_latencies_to_mesh(
        mesh: Mesh,
        network_vertices: typing.List[typing.Tuple[np.float64, np.float64]],
        network_latencies: typing.List[typing.Tuple[typing.Tuple[int, int],
                                                    np.float64]]
    ) -> typing.List[typing.Tuple[typing.Tuple[int, int], np.float64]]:
    """
    Convert latencies from a network to a mesh.

    As input, take a mesh, a list of coordinates, and latencies (as
    returned by `read_graphml`). Return a list of latencies stored along
    with the pair of corresponding mesh vertex indices, in the form
    `((i, j), latency)`.

    This format is used for space efficiency (not many points on the
    mesh are expected to have measured latencies).
    """
    latencies: typing.List[typing.Tuple[typing.Tuple[int, int], np.float64]] \
        = []

    # Can't do a dict comprehension since multiple vertices could map to the
    # same mesh point
    for (i, j), latency in network_latencies:
        i_key = mesh.nearest_vertex(network_vertices[i]).index()
        j_key = mesh.nearest_vertex(network_vertices[j]).index()
        latencies.append(((i_key, j_key), latency))

    return latencies

def get_mesh_output(
        directory: str,
        max_iterations: typing.Optional[int] = None,
        postprocessed: bool = False,
        initialization_path: typing.Optional[str] = None
) -> RectangleMesh:
    """
    Given a directory of output, return a mesh of the final output.

    This function can be used to get the height map from precomputed
    output that has been pickled. Some extra postprocessing is
    optionally done to make the output a bit more aesthetically
    pleasing.
    """

    with open(os.path.join(directory, 'parameters'), 'rb') as f:
        parameters = pickle.load(f)

        data_file_name = os.path.join('..', 'data' ,parameters['data_file_name'])
        width = parameters['width']
        height = parameters['height']

    if initialization_path is None:
        initialization_path = os.path.join(directory, '0')
    with open(initialization_path, 'rb') as f:
        iteration_data = pickle.load(f)
        z_0 = np.array(iteration_data['mesh_parameters'])

    iteration = max(int(name)
                    for name in os.listdir(directory)
                    if name.isdigit())
    if max_iterations is not None:
        iteration = min(iteration, max_iterations)

    with open(os.path.join(directory, str(iteration)), 'rb') as f:
        iteration_data = pickle.load(f)
        z = np.array(iteration_data['mesh_parameters'])

    mesh = RectangleMesh(width, height)

    coordinates, bounding_box, network_edges, _, _, labels, name \
        = read_graphml(data_file_name, with_labels=True)
    coordinates = np.array(coordinates)
    network_vertices = mesh.map_coordinates_to_support(coordinates, np.float64(0.8), bounding_box)
    nearest_vertex_indices = [mesh.nearest_vertex(network_vertex).index()
                              for network_vertex in network_vertices]
    network_convex_hulls = convex_hull.compute_connected_convex_hulls(
        network_vertices, network_edges)

    if postprocessed:
        vertices = mesh.get_coordinates()
        x = list(sorted(set(vertices[:,0])))
        y = list(sorted(set(vertices[:,1])))
        distances = np.array([
            convex_hull.distance_to_convex_hulls(
                np.array([px, py]),
                network_vertices,
                network_convex_hulls
            )
            for px in x
            for py in y
        ])
        # Add a small amount of space around the convex hull
        # TODO: Maybe make this "smarter"
        distances = np.maximum(distances - 0.05, 0.)
        z = z - np.array(z_0)
        z = (z - np.amin(z[distances == 0.], initial=np.amin(z))) \
            * np.exp(-1000 * distances**2)
        z = z - np.amin(z)

    mesh.set_parameters(z)
    return mesh
