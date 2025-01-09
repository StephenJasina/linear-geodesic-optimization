import itertools
import os
import pathlib
import pickle
import sys
import time

import networkx as nx
import pandas as pd
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
from matplotlib import animation as animation
import numpy as np

sys.path.append('.')
from linear_geodesic_optimization.data import input_network, input_mesh, utility
from linear_geodesic_optimization.plot \
    import get_rectangular_mesh_plot, get_image_data, get_face_colors_curvature_true
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh

lambda_curvature = 1.
lambda_smooth = 0.005
initial_radius = 20.
width = 50
height = 50
scale = 0.7
ip_type = 'ipv4'
threshold = 1

directory = pathlib.Path('..', 'outputs', 'toy', 'three_clusters_animation')
subdirectory_name = f'{lambda_curvature}_{lambda_smooth}_{initial_radius}_{width}_{height}_{scale}'

output_filepaths = list(sorted(
    output_directory / subdirectory_name / 'output'
    for output_directory in directory.iterdir()
))

fps = 24
animation_length = 12.  # in seconds

include_line_graph = False
def investigating_graph(k=3):
    path_to_graphml = f'../data/{ip_type}/graph_Europe_hourly/{threshold}/'
    path_to_probes = f'../data/{ip_type}/graph_Europe_hourly/probes.csv'
    path_to_latency = f'../data/{ip_type}/graph_Europe_hourly/'
    # Get all files with .graphml extension
    files = [f for f in os.listdir(path_to_graphml) if f.endswith('.graphml')]
    files_latency = [f for f in os.listdir(path_to_latency) if f.endswith('.csv') and not(f.startswith('probes'))]
    curvature_data = {}
    latency_data = {}

    # Read Probes.csv file
    probes = pd.read_csv(path_to_probes)

    for file in files:
        graph = nx.read_graphml(os.path.join(path_to_graphml, file))
        # Populate the curvature data for each edge across all graphs
        for edge in graph.edges(data=True):
            edge_key = tuple(
                sorted([edge[0], edge[1]]))  # Ensure edge key is consistent (node order doesn't matter)
            curvature_value = edge[2].get('ricciCurvature', -3)  # Default to -3 if no curvature value found

            if edge_key not in curvature_data:
                curvature_data[edge_key] = []

            curvature_data[edge_key].append(curvature_value)

    # Investigate which edges have appeared and disappeared in the graph
    appeared_edges = {}
    disappeared_edges = {}

    prev_edges = set()
    for i, file in enumerate(sorted(files)):
        graph = nx.read_graphml(os.path.join(path_to_graphml, file))
        current_edges = set(graph.edges())

        if i != 0:  # Skip the comparison for the first file
            appeared = current_edges - prev_edges
            disappeared = prev_edges - current_edges

            if appeared:
                for edge in list(appeared):
                    if edge not in appeared_edges:
                        appeared_edges[edge] = []
                    appeared_edges[edge].append(i)
            if disappeared:
                for edge in list(disappeared):
                    if edge not in disappeared_edges:
                        disappeared_edges[edge] = []
                    disappeared_edges[edge].append(i)

        prev_edges = current_edges

    # print("Edges that appeared:")
    # for file, edges in appeared_edges.items():
    #     print(f"In {file}:")
    #     for edge in edges:
    #         print(edge)

    # print("\nEdges that disappeared:")
    # for file, edges in disappeared_edges.items():
    #     print(f"In {file}:")
    #     print(edges)

    # select top k edges with the most disappearance and appearance
    # sorted_edges = sorted(disappeared_edges, key=lambda x: len(disappeared_edges[x]), reverse=True)[:k]
    # top_appeared = sorted(appeared_edges, key=lambda x: len(appeared_edges[x]), reverse=True)[:k]

    # Calculate the change in curvature for each edge
    curvature_changes = {edge: abs(max(curvatures) - min(curvatures)) for edge, curvatures in
                         curvature_data.items()}

    # Sort the edges based on curvature change
    sorted_edges = sorted(curvature_changes.keys(), key=lambda x: curvature_changes[x], reverse=True)[:k]

    time_series = []

    for file in files_latency:
        latency = pd.read_csv(os.path.join(path_to_latency, file))
        ### change dtype of latency
        latency['source_id'] = latency['source_id'].astype('int32')
        latency['target_id'] = latency['target_id'].astype('int32')
        ### change src and dst to str
        latency['source_id'] = latency['source_id'].apply(lambda x: str(x))
        latency['target_id'] = latency['target_id'].apply(lambda x: str(x))
        for src,dst,lat in latency[['source_id','target_id','rtt']].values:
            if (src,dst) not in curvature_data:
                continue
            if (src,dst) not in latency_data:
                latency_data[(src,dst)] = []
            latency_data[(src,dst)].append(lat)
    latency_changes = {edge: abs(max(latency) - min(latency)) for edge, latency in
                         latency_data.items()}

    # Sort the edges based on curvature change
    # sorted_edges = sorted(latency_changes.keys(), key=lambda x: latency_changes[x], reverse=True)[:k]

    # Sort and get the top N entries
    top_latency_changes = sorted(latency_changes.items(), key=lambda x: x[1], reverse=True)[:k]

    city_level_top_latency_changes = []

    for key in top_latency_changes:
        city_level_top_latency_changes.append((probes[probes['id'] == int(key[0][0])]['city'].values[0],probes[probes['id'] == int(key[0][1])]['city'].values[0],key[1]))
    print(city_level_top_latency_changes)
    print(top_latency_changes)
    ### sorted list(latency_changes.values())
    name_series = []
    for edge in sorted_edges:
        if (edge[1], edge[0]) in latency_data:
            latency_data[edge] = latency_data[(edge[1], edge[0])]
        time_series.append(latency_data[edge])
        name_series.append(probes[probes['id'] == int(edge[0])]['city'].values[0] + '-' + probes[probes['id'] == int(edge[1])]['city'].values[0])
    return time_series, name_series

if __name__ == '__main__':
    # Assume the first word of each path is a number indicating time
    paths = list(sorted(
        (
            float(output_filepath.parent.parent.name.split(' ')[0]),
            output_filepath
        )
        for output_filepath in output_filepaths
    ))
    manifold_count = len(paths)

    zs = []
    with open(output_filepaths[0], 'rb') as f:
        parameters = pickle.load(f)['parameters']
        width = parameters['width']
        height = parameters['height']
        if 'graphml_filename' in parameters:
            graph = nx.read_graphml(parameters['graphml_filename'])
            latencies = []
        else:
            probes_filename = parameters['probes_filename']
            probes_file_path = os.path.join('..', 'data', probes_filename)
            latencies_filename = parameters['latencies_filenames']
            if not isinstance(latencies_filename, str):
                latencies_filename = latencies_filename[0]
            latencies_file_path = os.path.join('..', 'data', latencies_filename)
            epsilon = parameters['epsilon']
            clustering_distance = parameters['clustering_distance']
            should_remove_tivs = parameters['should_remove_TIVs']
            ricci_curvature_alpha = parameters['ricci_curvature_alpha']
            graph, latencies = input_network.get_graph_from_paths(
                probes_file_path, latencies_file_path,
                epsilon=epsilon,
                clustering_distance=clustering_distance,
                should_remove_tivs=should_remove_tivs,
                should_include_latencies=True,
                ricci_curvature_alpha=ricci_curvature_alpha
            )
        coordinates_scale = parameters['coordinates_scale']
        mesh_scale = parameters['mesh_scale']
        network_trim_radius = parameters['network_trim_radius']
        network_data = input_network.get_network_data(graph)
        coordinates = network_data[0]['coordinates']

    # Use curvature values for coloring
    face_colors = []

    for output_filepath in output_filepaths:
        with output_filepath.open('rb') as f:
            data = pickle.load(f)
            mesh = input_mesh.get_mesh(
                data['final'], width, height,
                network_data, coordinates_scale, mesh_scale,
                network_trim_radius=network_trim_radius,
                # postprocessed=True, z_0=data['initial']
            )

            # Use curvature values for coloring
            face_colors.append(get_face_colors_curvature_true(mesh))

            # Fill in the missing z values (and postprocess)
            # TODO: Make this less ad hoc
            z = np.full(width * height, -0.5)
            z_trim_mapping = np.array(data['final']) - np.array(data['initial'])
            z_trim_mapping = z_trim_mapping - np.amin(z_trim_mapping)
            z_trim_mapping = z_trim_mapping / np.amax(z_trim_mapping)
            z[mesh.get_trim_mapping()] = z_trim_mapping
            zs.append(z)

    z_max = np.amax(zs)

    mesh = RectangleMesh(width, height, scale)

    # Use map colors for coloring
    # resolution = 500
    # face_colors = [np.flipud(get_image_data(
    #     coordinates, resolution, coordinates_scale,
    #     'coral', 'aqua'
    # )[0]).swapaxes(0, 1) / 256.] * len(zs)

    # Use solid colors for coloring
    # face_colors = [np.full((width, height, 3), 0.4)] * len(zs)

    fig = plt.figure()

    if include_line_graph:
        time_series, name_series= investigating_graph(k=3)

        # Define the grid shape (total rows, total cols)
        gs = gridspec.GridSpec(4, 1, height_ratios=[16, 1, 1, 1], hspace=0)

        ax = fig.add_subplot(gs[0], projection='3d', facecolor='#808080')
        ax_ts1 = fig.add_subplot(gs[1])
        ax_ts2 = fig.add_subplot(gs[2])
        ax_ts3 = fig.add_subplot(gs[3])

        # Remove x ticks and labels from ax
        # ax.set_xticks([])
        # ax.set_xlabel('')

        ts_data1 = time_series[0]
        ts_data2 = time_series[1]
        ts_data3 = time_series[2]

        ax_ts3.set_xlim(0, 24)
        # ax_ts1.plot(ts_data1)
        # ax_ts2.plot(ts_data2)
        # ax_ts3.plot(ts_data3)


        # ax_ts1.set_ylim(0, 100)
        # ax_ts2.set_ylim(0, 100)
        # ax_ts3.set_ylim(0, 100)
        # ax_ts1.clear()
        # ax_ts1.plot(ts_data1)
        #
        # ax_ts2.clear()
        # ax_ts2.plot(ts_data2)
    else:
        ax = fig.add_subplot(projection='3d')
        ax.set_facecolor((0.5, 0.5, 0.5))

    def get_frame(t):
        print(f'Computing frame at t={t}')

        for left_index in range(len(paths) + 1):
            if left_index == len(paths) or paths[left_index][0] > t:
                break
        left_index -= 1
        for right_index in range(len(paths) - 1, -2, -1):
            if right_index == -1 or paths[right_index][0] < t:
                break
        right_index += 1

        left = paths[left_index][0]
        right = paths[right_index][0]

        lam = 0. if right == left else (t - left) / (right - left)
        z = (1 - lam) * zs[left_index] + lam * zs[right_index]
        z = np.reshape(z, (width, height))

        entry = paths[left_index if lam < 0.5 else right_index][1]
        with (entry.parent / 'parameters').open('rb') as f:
            parameters = pickle.load(f)
            if 'graphml_filename' in parameters:
                network = nx.read_graphml(parameters['graphml_filename'])
                latencies = []
            else:
                probes_filename = parameters['probes_filename']
                probes_file_path = os.path.join('..', 'data', probes_filename)
                latencies_filename = parameters['latencies_filenames']
                if not isinstance(latencies_filename, str):
                    latencies_filename = latencies_filename[0]
                latencies_file_path = os.path.join('..', 'data', latencies_filename)
                epsilon = parameters['epsilon']
                clustering_distance = parameters['clustering_distance']
                should_remove_tivs = parameters['should_remove_TIVs']
                ricci_curvature_alpha = parameters['ricci_curvature_alpha']
                network, latencies = input_network.get_graph_from_paths(
                    probes_file_path, latencies_file_path,
                    epsilon=epsilon,
                    clustering_distance=clustering_distance,
                    should_remove_tivs=should_remove_tivs,
                    should_include_latencies=True,
                    ricci_curvature_alpha=ricci_curvature_alpha
                )
            coordinates_scale = parameters['coordinates_scale']
            graph_data, vertex_data, edge_data = input_network.get_network_data(network)
            coordinates = graph_data['coordinates']
            bounding_box = graph_data['bounding_box']
            network_edges = graph_data['edges']
            network_city = vertex_data['city']
            network_curvatures = edge_data['ricciCurvature']
        coordinates = np.array(coordinates)
        network_vertices = mesh.map_coordinates_to_support(coordinates, coordinates_scale, bounding_box)

        if include_line_graph:
            # Update timeseries plot
            line1, = ax_ts1.plot(ts_data1[:int(t) + 1], '-o', markersize=3, color='blue', label = name_series[0])
            line2, = ax_ts2.plot(ts_data2[:int(t) + 1], '-o', markersize=3, color='red', label= name_series[1])
            line3, = ax_ts3.plot(ts_data3[:int(t) + 1], '-o', markersize=3, color='green', label = name_series[2])
            legend1 = ax_ts1.legend([line1], [name_series[0]], loc='upper right', fontsize=4)
            legend2 = ax_ts2.legend([line2], [name_series[1]], loc='upper right', fontsize=4)
            legend3 = ax_ts3.legend([line3], [name_series[2]], loc='upper right', fontsize= 4)
            ax.set_xlim(0,24)
            ax_ts1.tick_params(axis='both', which='major', labelsize=6)
            ax_ts2.tick_params(axis='both', which='major', labelsize=6)
            ax_ts3.tick_params(axis='both', which='major', labelsize=6)

            # Remove x ticks and labels from ax_ts1 and ax_ts2 since they're stacked
            ax_ts1.set_xticks([])
            ax_ts1.set_xlabel('')
            # ax_ts1.set_yticks(np.arange(0, max(time_series[0]), 5))
            ax_ts2.set_xticks([])
            # ax_ts2.set_yticks(np.arange(0, max(time_series[1]), 5))
            ax_ts2.set_xlabel('')
            # ax_ts3.set_yticks(np.arange(0, max(time_series[2]), 5))
            ax_ts3.set_xlabel('Time (hours)')
            # ax_ts3.set_xticks([])
            ax_ts1.set_xlim(0, 24)
            ax_ts2.set_xlim(0, 24)
            ax_ts3.set_xlim(0, 24)
            ax_ts3.set_xticks(np.arange(0, 24, 1))
            # ax_ts2.set_xticks(np.arange(0, 24, 1))
            # ax_ts1.set_xticks(np.arange(0, 24, 1))

        ax.clear()

        return [
            get_rectangular_mesh_plot(z, face_colors[left_index], None,
                np.amax(z) / z_max * 0.25,
                [network_vertices, network_edges, network_curvatures, network_city],
                ax
            ),
            # ax.text2D(0.05, 0.95, f'{left:02}:{round(lam*60):02}',
            #           transform=ax.transAxes)
            ax.text2D(0.05, 0.95, f'{t:02}',
                      transform=ax.transAxes)
        ] + (
            [line1, line2, line3, legend1, legend2, legend3]
            if include_line_graph
            else []
        )

    ani = animation.FuncAnimation(fig, get_frame,
                                  np.linspace(paths[0][0], paths[-1][0],
                                              int(round(fps * animation_length)) + 1),
                                  interval=1000/fps,
                                  blit=True)
    ani.save(os.path.join('..', 'animation.mp4'), dpi=300)
