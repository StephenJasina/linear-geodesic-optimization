import os
import pickle

import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import animation as animation
# from mpl_toolkits.basemap import Basemap
import numpy as np
import time


from linear_geodesic_optimization import data
from linear_geodesic_optimization.plot import get_mesh_plot
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh

directory = os.path.join('..', 'out_Europe_hourly')

lambda_curvature = 1.
lambda_smooth = 0.004
lambda_geodesic = 0.
initial_radius = 20.
width = 50
height = 50
scale = 1.
subdirectory_name = f'{lambda_curvature}_{lambda_smooth}_{lambda_geodesic}_{initial_radius}_{width}_{height}_{scale}'

manifold_count = 24
fps = 24

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
    image_data = np.flipud(image_data) / 255
    plt.close(fig)
    return image_data


def investigating_graph(k=3):
    path_to_graphml = '../data/graph_Europe_hourly/'

    # Get all files with .graphml extension
    files = [f for f in os.listdir(path_to_graphml) if f.endswith('.graphml')]
    files_latency = [f for f in os.listdir(path_to_graphml) if f.endswith('.csv') and not(f.startswith('probes'))]
    curvature_data = {}
    latency_data = {}

    # Read Probes.csv file
    probes = pd.read_csv(os.path.join(path_to_graphml, 'probes.csv'))
    print(probes)

    for file in files:
        graph = nx.read_graphml(os.path.join(path_to_graphml, file))
        # Populate the curvature data for each edge across all graphs
        for edge in graph.edges(data=True):
            edge_key = tuple(
                sorted([edge[0], edge[1]]))  # Ensure edge key is consistent (node order doesn't matter)
            curvature_value = edge[2].get('ricciCurvature', 0)  # Default to 0 if no curvature value found

            if edge_key not in curvature_data:
                curvature_data[edge_key] = []

            curvature_data[edge_key].append(curvature_value)

    # Calculate the change in curvature for each edge
    curvature_changes = {edge: abs(max(curvatures) - min(curvatures)) for edge, curvatures in
                         curvature_data.items()}

    # Sort the edges based on curvature change
    sorted_edges = sorted(curvature_changes.keys(), key=lambda x: curvature_changes[x], reverse=True)[:k]

    time_series = []

    for file in files_latency:
        latency = pd.read_csv(os.path.join(path_to_graphml, file))
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
    sorted_edges = sorted(latency_changes.keys(), key=lambda x: latency_changes[x], reverse=True)[:k]

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
            directory, f'graph_{i}', subdirectory_name
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
    resolution = 1000
    # image_data = get_image_data(data_file_path, resolution)
    mesh_coordinates = mesh.get_coordinates()
    # face_colors = [
    #     image_data[int(resolution * (face_center[1] / scale + 0.5)),
    #                int(resolution * (face_center[0] / scale + 0.5))]
    #     for face in mesh.get_topology().faces()
    #     for face_center in (sum(mesh_coordinates[v.index()]
    #                             for v in face.vertices()) / 3,)
    # ]

    import matplotlib.gridspec as gridspec

    fig = plt.figure()
    gs = gridspec.GridSpec(4, 1, height_ratios=[16, 1, 1,1])  # Adjust as necessary for more timeseries

    ax = fig.add_subplot(gs[0], projection='3d', facecolor='#808080')
    ax_ts1 = fig.add_subplot(gs[1])
    ax_ts2 = fig.add_subplot(gs[2])
    ax_ts3 = fig.add_subplot(gs[3])
    # Dummy timeseries data, replace with your actual data
    ts_data1 = time_series[0]
    ts_data2 = time_series[1]
    ts_data3 = time_series[2]

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
        mesh.set_parameters(z)

        with open(os.path.join(directory, f'graph_{right}', subdirectory_name, 'parameters'), 'rb') as f:
            parameters = pickle.load(f)
            data_file_name = parameters['data_file_name']
            data_file_path = os.path.join('..', 'data', data_file_name)
            coordinates, network_edges, network_curvatures, \
            network_latencies, network_nodes, network_city = data.read_graphml(data_file_path, with_labels=True)
        coordinates = np.array(coordinates)
        network_vertices = mesh.map_coordinates_to_support(coordinates, np.float64(0.8))

        ax.clear()

        # Update timeseries plot
        line1, = ax_ts1.plot(ts_data1[:int(i) + 1], '-o', markersize=3, color='blue', label = name_series[0])
        line2, = ax_ts2.plot(ts_data2[:int(i) + 1], '-o', markersize=3, color='red', label= name_series[1])
        line3, = ax_ts3.plot(ts_data3[:int(i) + 1], '-o', markersize=3, color='green', label = name_series[2])
        ax_ts1.legend(loc='upper left')
        ax_ts2.legend(loc='upper left')
        ax_ts3.legend(loc='upper left')
        elapsed_time = time.time() - start_time
        # print(f"Time taken for iteration {i}: {elapsed_time:.4f} seconds")

        return [
            get_mesh_plot(mesh, None, None,
                          [network_vertices, network_edges, network_curvatures, network_city],
                          ax),
            ax.text2D(0.05, 0.95, f'{left:02}:{round(lam*60):02}',
                      transform=ax.transAxes),
            line1, line2, line3
        ]


    ani = animation.FuncAnimation(fig, get_frame,
                                  np.linspace(0, manifold_count - 1,
                                  (manifold_count - 1) * fps + 1),
                                  interval=1000/ fps,
                                  blit=True)
    # ani = animation.FuncAnimation(fig, get_frame,
    #                               np.linspace(0, manifold_count - 1,
    #                                           (manifold_count - 1) * fps + 1),
    #                               interval=1000/fps,
    #                               blit=True)
    ani.save(os.path.join('..', 'animation_Europe.mp4'), dpi=300)
