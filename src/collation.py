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
import pygeodesic.geodesic as pgeo

import linear_geodesic_optimization.batch as batch
import linear_geodesic_optimization.driver as driver
from linear_geodesic_optimization.data import utility
from linear_geodesic_optimization.graph import boundary
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization.laplacian import Computer as Laplacian
from linear_geodesic_optimization.optimization.curvature import Computer as Curvature


tableau_colors = list(mpl.colors.TABLEAU_COLORS.values())
def get_tableau_color(index: int):
    """
    Return a color selected from the Tableau palette.
    """
    color = mpl.colors.to_rgb(tableau_colors[index % len(tableau_colors)])
    return [int(channel * 255) for channel in color]

def compute_geodesics_from_graph(mesh: RectangleMesh, network_vertices, network_edges, geodesic_routes, exact_geodesics=False):
    """
    If `exact_geodesics` is false (the default), geodesic paths are
    computed with `potpourri3d`'s edge-flip solver: an approximate
    trace through the mesh graph (via Dijkstra's algorithm) is
    tightened into a taut geodesic. If `exact_geodesics` is true,
    `pygeodesic`'s exact algorithm is used instead to compute the true
    shortest path between each pair of consecutive route waypoints
    directly on the mesh surface.
    """
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
    faces = np.array([
        [vertex.index for vertex in face.vertices()]
        for face in topology.faces()
    ])

    network_vertex_indices_to_mesh_vertex_indices = [
        mesh.nearest_vertex(network_vertex).index
        for network_vertex in network_vertices
    ]

    if exact_geodesics:
        path_solver = pgeo.PyGeodesicAlgorithmExact(coordinates, faces)
    else:
        mesh_graph = nx.Graph()
        for edge in topology.edges():
            source, target = edge.vertices()
            source = source.index
            target = target.index
            weight = np.linalg.norm(coordinates[source] - coordinates[target])
            mesh_graph.add_edge(source, target, length=weight)
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

        path_solver = pp3d.EdgeFlipGeodesicSolver(coordinates, faces)

    geodesics = []
    for geodesic_route in geodesic_routes:
        is_bad = False
        for index_node in geodesic_route:
            if index_node in bad_indices:
                is_bad = True
        if is_bad:
            continue

        if exact_geodesics:
            trace_points = []
            for index_source, index_target in itertools.pairwise(geodesic_route):
                _, segment_points = path_solver.geodesicDistance(
                    network_vertex_indices_to_mesh_vertex_indices[index_source],
                    network_vertex_indices_to_mesh_vertex_indices[index_target]
                )
                # `geodesicDistance` returns points ordered from target
                # to source, so reverse to match the route's direction
                segment_points = segment_points[::-1]
                if trace_points:
                    # Avoid duplicating the shared waypoint between
                    # consecutive segments
                    segment_points = segment_points[1:]
                trace_points.extend(segment_points)

            if len(trace_points) < 2:
                geodesics.append([])
            else:
                geodesic = np.array(trace_points)
                geodesics.append((geodesic[:, :2] / mesh_scale).tolist())
        else:
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
            approximate_trace = utility.remove_loops(approximate_trace)

            if len(approximate_trace) < 2:
                geodesics.append([])
            else:
                geodesic = path_solver.find_geodesic_path_poly(approximate_trace)
                geodesics.append((geodesic[:, :2] / mesh_scale).tolist())

    return geodesics

def load_batch(
    directories_outputs,
    *,
    times=None,
    geodesic_label_color_pairs=None,
):
    """
    Read a batch of optimization outputs and do the collation work that
    doesn't depend on postprocessing, so that `render_animation` can
    reuse it for any number of postprocessing variants.

    `directories_outputs` is a collection of directories, each of which
    contains an optimization output file called 'output.json'.
    Optionally, these are each associated with timestamps in `times`

    Geodesic paths can be computed using `geodesic_label_color_pairs`,
    which is a list with elements of the form
      `((source, target), [colorR, colorG, colorB])`,
    where the colors are encoded with numbers between 0 and 255.

    Return a dictionary of the shared data.
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

    # Create the mesh that will be used for curvature and geodesic
    # computations
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

    if geodesic_label_color_pairs is None:
        # For now, just use the routes in the first snapshot
        # TODO: Is this exactly what we want?
        if 'traffic' in outputs[0] and outputs[0]['traffic'] is not None:
            high_traffic_route_pairs = list(sorted([
                (traffic, route)
                for traffic, route in zip(outputs[0]['traffic'], outputs[0]['routes'])
                if route[0] != route[-1]
            ], reverse=True))
        else:
            high_traffic_route_pairs = list(sorted([
                (0., [source, destination])
                for source in node_indices_to_labels
                for destination in node_indices_to_labels
                if source != destination
            ]))
        geodesic_label_color_pairs = [
            (route, get_tableau_color(index))
            for index, (_, route) in enumerate(high_traffic_route_pairs)
        ][:100]
        # TODO: revert
        # geodesic_label_color_pairs = []
    geodesic_labels = [geodesic_label for geodesic_label, _ in geodesic_label_color_pairs]
    edge_colors = [list(color) for _, color in geodesic_label_color_pairs]

    # The curvatures of the unprocessed surfaces and the traffic data
    curvatures = []
    traffic_matrices = []
    traffic_paths_needed = []
    for output in outputs:
        mesh.set_parameters(output['final'])
        # TODO: Avoid having to recreate these
        laplacian = Laplacian(mesh)
        curvature = Curvature(mesh, laplacian)
        curvature.forward()
        curvatures.append(np.array(curvature.kappa_G).reshape((width, height)).tolist())

        traffic = output['traffic'] if 'traffic' in output else None
        traffic_matrix = [[0. for _ in range(len(network_vertices))] for _ in range(len(network_vertices))]
        if traffic is not None:
            for route, volume in zip(output['routes'], output['traffic']):
                traffic_matrix[node_labels_to_indices[route[0]]][node_labels_to_indices[route[-1]]] += volume
        traffic_matrices.append(traffic_matrix)
        traffic_paths_needed.append(traffic is not None)

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

    return {
        'outputs': outputs,
        'times': times,
        'parameters': parameters,
        'mesh': mesh,
        'network_vertices': network_vertices,
        'node_labels_to_indices': node_labels_to_indices,
        'animation_vertices': animation_vertices,
        'animation_edges': animation_edges,
        'geodesic_labels': geodesic_labels,
        'edge_colors': edge_colors,
        'curvatures': curvatures,
        'traffic_matrices': traffic_matrices,
        'traffic_paths_needed': traffic_paths_needed,
        'map_data': map_data,
        # Caches keyed by `use_convex_hull`, filled by `render_animation`
        'borders': {},
        'distances_to_networks': {},
        # Cache keyed by `exact_geodesics`, filled by `get_original_frames`
        'original_frames': {},
    }

def get_network_borders(batch_data, use_convex_hull):
    """
    Return the network border of each snapshot, along with the edges of
    the last snapshot, caching the borders in `batch_data`.
    """
    if use_convex_hull not in batch_data['borders']:
        network_vertices = batch_data['network_vertices']
        network_borders = []
        for output in batch_data['outputs']:
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
        batch_data['borders'][use_convex_hull] = (network_borders, network_edges)
    return batch_data['borders'][use_convex_hull]

def get_distances_to_networks(batch_data, use_convex_hull):
    """
    Return, for each snapshot, the distance from each mesh vertex to
    the network border, caching the distances in `batch_data`.
    """
    if use_convex_hull not in batch_data['distances_to_networks']:
        network_borders, _ = get_network_borders(batch_data, use_convex_hull)
        mesh = batch_data['mesh']
        batch_data['distances_to_networks'][use_convex_hull] = [
            np.array([
                boundary.distance_to_network(
                    np.array(vertex_coordinate),
                    network_border
                )
                for vertex_coordinate in mesh.get_coordinates()[:, :2]
            ])
            for network_border in network_borders
        ]
    return batch_data['distances_to_networks'][use_convex_hull]

def compute_frame_geodesics(batch_data, z, hull, exact_geodesics, traffic_paths_needed):
    """
    Set the mesh heights to `z` and compute the geodesics of one
    snapshot. If `hull` is not `None`, the mesh is first trimmed to it.

    Return the geodesics and, if `traffic_paths_needed` is true, the
    matrix of traffic paths (otherwise `None`).
    """
    mesh = batch_data['mesh']
    network_vertices = batch_data['network_vertices']
    node_labels_to_indices = batch_data['node_labels_to_indices']
    geodesic_labels = batch_data['geodesic_labels']
    # Use the edges of the last snapshot, as `get_network_borders` does
    graph_data, _, _ = batch_data['outputs'][-1]['network']
    network_edges = graph_data['edges']

    mesh.set_parameters(z)
    if hull is not None:
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
        ],
        exact_geodesics=exact_geodesics
    )
    mesh.remove_added_vertices()
    mesh.restore_removed_vertices()

    if traffic_paths_needed:
        paths = [[[] for _ in range(len(network_vertices))] for _ in range(len(network_vertices))]
        for geodesic_label, geodesic in zip(geodesic_labels, geodesics):
            paths[node_labels_to_indices[geodesic_label[0]]][node_labels_to_indices[geodesic_label[-1]]] = geodesic
    else:
        paths = None

    return geodesics, paths

def get_original_frames(batch_data, exact_geodesics):
    """
    Return, for each snapshot, the unprocessed surface: its heights
    (`output['final']` centered by its own mean and divided by the mesh
    scale), and the geodesics and traffic paths on the untrimmed mesh.
    These are cached in `batch_data`, since they don't depend on any
    postprocessing option.
    """
    if exact_geodesics not in batch_data['original_frames']:
        parameters = batch_data['parameters']
        width = parameters['width']
        height = parameters['height']
        mesh_scale = parameters['mesh_scale']
        frames = []
        for output, traffic_paths_needed in zip(batch_data['outputs'], batch_data['traffic_paths_needed']):
            z = np.array(output['final'])
            z = z - np.mean(z)
            geodesics, paths = compute_frame_geodesics(
                batch_data, z, None, exact_geodesics, traffic_paths_needed
            )
            frames.append({
                'height': (z / mesh_scale).reshape((width, height)).tolist(),
                'geodesics': geodesics,
                'trafficPaths': paths,
            })
        batch_data['original_frames'][exact_geodesics] = frames
    return batch_data['original_frames'][exact_geodesics]

def render_animation(
    batch_data,
    *,
    height_scale=0.15,
    use_convex_hull=False,
    bubble_size=np.inf,
    postprocess=True,
    exact_geodesics=False,
    postprocess_subtract_initialization=True
):
    """
    Turn a batch loaded by `load_batch` into a single animation, using
    the given postprocessing options. Return the animation data.

    All coordinates sent to the viewer (node and border x and y,
    geodesic x and y, and the heights z) are divided by the mesh scale,
    so the viewer shows the surface with its true proportions.

    If `postprocess` is true, the z-coordinates (heights) are put
    through a pipeline that subtracts a reference height, centers and
    normalizes them across snapshots (to a range of `height_scale`
    times the mesh width), applies a smoothing decay near
    the trim boundary, and trims the mesh to a bubble around the
    network before computing geodesics. If `postprocess` is false,
    none of that happens: each snapshot's z-coordinates are just
    `output['final']` (converted to a Numpy array and centered by
    their own mean, so the animation doesn't jump between frames), and
    geodesics are computed on the untrimmed mesh.

    If `postprocess` is true, each frame also has an `'original'` field
    holding the unprocessed surface's `'height'`, `'geodesics'`, and
    `'trafficPaths'` (the same values `postprocess=False` would give),
    so the viewer can show both. See `get_original_frames`.

    If `exact_geodesics` is true, geodesic paths are computed with
    `pygeodesic`'s exact algorithm instead of the default
    `potpourri3d`-based approximation. See
    `compute_geodesics_from_graph` for details.
    """
    outputs = batch_data['outputs']
    parameters = batch_data['parameters']
    width = parameters['width']
    height = parameters['height']
    mesh_scale = parameters['mesh_scale']

    network_borders, _ = get_network_borders(batch_data, use_convex_hull)

    # Normalize the z-coordinates across time
    distances_to_networks = []
    hulls = []
    zs = []
    if postprocess:
        distances_to_networks = get_distances_to_networks(batch_data, use_convex_hull)
        for index, output in enumerate(outputs):
            # TODO: Make this more intelligent. In particular, ensure that
            # the bubble around solitary edges is wide enough so that the
            # hull does not become disconnected
            hull = np.where(distances_to_networks[index] / mesh_scale <= bubble_size)[0]
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
            if postprocess_subtract_initialization:
                z = np.array(output['final']) - z_0
            else:
                z = np.array(output['final'])
            zs.append(z - np.mean(z[hull]))

    if postprocess:
        # Determine values for vertical scaling
        z_max = -np.inf
        z_min = np.inf
        for z, hull in zip(zs, hulls):
            z_max = max(z_max, np.max(z[hull]))
            z_min = min(z_min, np.min(z[hull]))

    originals = get_original_frames(batch_data, exact_geodesics)

    animation_data = []
    zs_iter = zs if postprocess else itertools.repeat(None)
    distances_to_networks_iter = distances_to_networks if postprocess else itertools.repeat(None)
    hulls_iter = hulls if postprocess else itertools.repeat(None)
    for t, z, distance_to_network, hull, edges, network_border, curvature, traffic_matrix, traffic_paths_needed, original in zip(
        batch_data['times'], zs_iter, distances_to_networks_iter, hulls_iter,
        batch_data['animation_edges'], network_borders, batch_data['curvatures'],
        batch_data['traffic_matrices'], batch_data['traffic_paths_needed'],
        originals,
    ):
        if postprocess:
            distance_to_bubble = np.zeros(distance_to_network.shape) if np.isposinf(bubble_size) else np.maximum(distance_to_network / mesh_scale - bubble_size, 0.)
            z = z - z_min
            if z_max != z_min and height_scale is not None:
                z = z / (z_max - z_min) * height_scale * mesh_scale
            offset = 0.05 * mesh_scale
            z = (z + offset) * np.exp(-1000 * distance_to_bubble**2) - offset

            geodesics, paths = compute_frame_geodesics(
                batch_data, z, hull, exact_geodesics, traffic_paths_needed
            )
            z_height = (z / mesh_scale).reshape((width, height)).tolist()
        else:
            # The unprocessed surface is exactly the original one
            z_height = original['height']
            geodesics = original['geodesics']
            paths = original['trafficPaths']

        frame = {
            'time': t,
            'height': z_height,
            'curvature': curvature,
            'edges': edges,
            'geodesics': geodesics,
            'edgeColors': batch_data['edge_colors'],
            'border': [
                [[coordinate / mesh_scale for coordinate in point] for point in border_part]
                for border_part in network_border
            ],
            'traffic': traffic_matrix,
            'trafficPaths': paths
        }
        if postprocess:
            frame['original'] = original
        animation_data.append(frame)

    return {
        'nodes': batch_data['animation_vertices'],
        'animation': animation_data,
        'map': batch_data['map_data'],
    }

def collate_outputs(
    directories_outputs,
    variants,
    *,
    times=None,
    geodesic_label_color_pairs=None,
):
    """
    Join many optimization outputs into animations, one per collation
    variant.

    `directories_outputs` is a collection of directories, each of which
    contains an optimization output file called 'output.json'. The
    outputs are read and processed once (see `load_batch`). Then, for
    each `(collation_arguments, path_output_collated)` pair in
    `variants`, an animation is made with `render_animation`, using
    `collation_arguments` as its keyword arguments, and written to
    `path_output_collated`.
    """
    batch_data = load_batch(
        directories_outputs,
        times=times,
        geodesic_label_color_pairs=geodesic_label_color_pairs,
    )
    for collation_arguments, path_output_collated in variants:
        animation = render_animation(batch_data, **collation_arguments)
        os.makedirs(path_output_collated.parent, exist_ok=True)
        with open(path_output_collated, 'w') as file_output:
            json.dump(animation, file_output, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    args = parser.parse_args()
    config_file = pathlib.PurePath(args.config_file)

    # Read JSON config file and generate optimization and collation
    # parameters from it
    arguments, arguments_collation, settings, defaults = driver.load_config(config_file)

    # Determine where the outputs are. Check that they exist, since
    # we're reading already-produced optimization output.
    output_format = driver.get_output_format(settings, defaults)
    driver.assign_output_directories(arguments, settings, output_format, existence_check='must_exist')

    # Each batch's outputs are read once and reused for every collation
    # variant
    initialization = driver.get_initialization(settings)
    jobs = driver.assign_collation_paths(
        arguments, arguments_collation, settings, defaults, initialization
    )
    batch.run_multiprocessed(collate_outputs, jobs)

if __name__ == '__main__':
    main()
