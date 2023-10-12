import os
import pickle

import networkx as nx
import pandas as pd
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
from matplotlib import animation as animation
from mpl_toolkits.basemap import Basemap
import numpy as np
import time
from utils import *

from linear_geodesic_optimization import data
from linear_geodesic_optimization.plot \
    import get_mesh_plot, get_rectangular_mesh_plot
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh

directory = os.path.join('..', 'out_US_hourly')

lambda_curvature = 1.
lambda_smooth = 0.004
lambda_geodesic = 0.
initial_radius = 20.
width = 50
height = 50
scale = 1.
subdirectory_name = f'{lambda_curvature}_{lambda_smooth}_{lambda_geodesic}_{initial_radius}_{width}_{height}_{scale}'

manifold_count = 24
fps = 6

def get_image_data(data_file_path, resolution=100):
    coordinates, _, _, _ = data.read_graphml(data_file_path)
    coordinates = np.array(coordinates)
    center = np.mean(coordinates, axis=0)
    scale_factor = 0.8
    coordinates = center + (coordinates - center) / scale_factor

    coordinates_left = np.amin(coordinates[:,0])
    coordinates_right = np.amax(coordinates[:,0])
    coordinates_bottom = np.amin(coordinates[:,1])
    coordinates_top = np.amax(coordinates[:,1])

    if coordinates_right - coordinates_left > coordinates_top - coordinates_bottom:
        center = (coordinates_bottom + coordinates_top) / 2.
        scale_factor = (coordinates_top - coordinates_bottom) / (coordinates_right - coordinates_left)
        coordinates_bottom = center + (coordinates_bottom - center) / scale_factor
        coordinates_top = center + (coordinates_top - center) / scale_factor
    else:
        center = (coordinates_left + coordinates_right) / 2.
        scale_factor = (coordinates_right - coordinates_left) / (coordinates_top - coordinates_bottom)
        coordinates_left = center + (coordinates_left - center) / scale_factor
        coordinates_right = center + (coordinates_right - center) / scale_factor

    left, _ = data.inverse_mercator(coordinates_left, 0.)
    right, _ = data.inverse_mercator(coordinates_right, 0.)
    _, bottom = data.inverse_mercator(0., coordinates_bottom)
    _, top = data.inverse_mercator(0., coordinates_top)

    map = Basemap(llcrnrlon=left, urcrnrlon=right,
                  llcrnrlat=bottom, urcrnrlat=top, epsg=3857)
    fig, ax = plt.subplots()
    image_data = map.arcgisimage(service='USA_Topo_Maps', ax=ax,
                                 xpixels=resolution, y_pixels=resolution).get_array()
    image_data = np.flipud(image_data).swapaxes(0, 1) / 255
    plt.close(fig)
    return image_data


def investigating_graph(k=3):
    path_to_graphml = f'../data/{ip_type}/graph_Europe_hourly/{threshold}/'
    path_to_probes = f'../data/{ip_type}/'
    path_to_latency = f'../data/{ip_type}/graph_Europe_hourly/'
    # Get all files with .graphml extension
    files = [f for f in os.listdir(path_to_graphml) if f.endswith('.graphml')]
    files_latency = [f for f in os.listdir(path_to_latency) if f.endswith('.csv') and not(f.startswith('probes'))]
    curvature_data = {}
    latency_data = {}

    # Read Probes.csv file
    probes = pd.read_csv(os.path.join(path_to_probes, f'probes_{ip_type}.csv'))
    print(probes)

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
    initialization_path = os.path.join(directory, 'graph_0', subdirectory_name, '0')

    time_series, name_series= investigating_graph(k=3)
    zs = []
    for i in range(manifold_count):
        print(f'Reading data from manifold {i}')

        current_directory = os.path.join(
            directory, f'graph_{i}_{threshold}', subdirectory_name
        )

        iteration = max(
            int(name)
            for name in os.listdir(current_directory)
            if name.isdigit()
        )
        path = os.path.join(current_directory, str(iteration))
        mesh = data.get_mesh_output(
            current_directory, postprocessed=True,
            intialization_path=initialization_path
        )

        zs.append(mesh.get_parameters())

    mesh = RectangleMesh(width, height, scale)

    with open(os.path.join(directory, 'graph_0', subdirectory_name, 'parameters'), 'rb') as f:
        parameters = pickle.load(f)
        data_file_name = parameters['data_file_name']
        data_file_path = os.path.join('..', 'data', data_file_name)
    resolution = 500
    face_colors = get_image_data(data_file_path, resolution)

    fig = plt.figure()

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
    def get_frame(i):
        start_time = time.time()
        ### print time taken for each loop
        print(f'Computing frame at t={i}')
        left = int(i)
        right = min(left + 1, manifold_count - 1)
        lam = i - left
        z = (1 - lam) * zs[left] + lam * zs[right]
        # mesh.set_parameters(z)
        z = np.reshape(z, (width, height))

        with open(os.path.join(directory, f'graph_{right}_{threshold}', subdirectory_name, 'parameters'), 'rb') as f:
            parameters = pickle.load(f)
            data_file_name = parameters['data_file_name']
            data_file_path = os.path.join('..', 'data', data_file_name)
            coordinates, network_edges, network_curvatures, \
            network_latencies, network_nodes, network_city = data.read_graphml(data_file_path, with_labels=True)
        coordinates = np.array(coordinates)
        network_vertices = mesh.map_coordinates_to_support(coordinates, np.float64(0.8))


        # Update timeseries plot
        line1, = ax_ts1.plot(ts_data1[:int(i) + 1], '-o', markersize=3, color='blue', label = name_series[0])
        line2, = ax_ts2.plot(ts_data2[:int(i) + 1], '-o', markersize=3, color='red', label= name_series[1])
        line3, = ax_ts3.plot(ts_data3[:int(i) + 1], '-o', markersize=3, color='green', label = name_series[2])
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
        elapsed_time = time.time() - start_time
        print(f"Time taken for iteration {i}: {elapsed_time:.4f} seconds")

        return [
            # get_rectangular_mesh_plot(z, face_colors, None,
            #               [network_vertices, network_edges, network_curvatures],
            #               ax),
            get_mesh_plot(mesh, None, None,
                          [network_vertices, network_edges, network_curvatures, network_city],
                          ax),
            ax.text2D(0.05, 0.95, f'{left:02}:{round(lam*60):02}',
                      transform=ax.transAxes),
            line1, line2, line3, legend1, legend2, legend3
        ]


    ani = animation.FuncAnimation(fig, get_frame,
                                  np.linspace(0, manifold_count - 1,
                                              (manifold_count - 1) * fps + 1),
                                  interval=1000/fps,
                                  blit=True)
    ani.save(os.path.join('..', 'animation_test.mp4'), dpi=300)
