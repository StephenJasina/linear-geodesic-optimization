import os
import pickle

from matplotlib import pyplot as plt
from matplotlib import animation as animation
from mpl_toolkits.basemap import Basemap
import numpy as np

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

if __name__ == '__main__':
    initialization_path = os.path.join(directory, 'graph_0', subdirectory_name, '0')

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
    image_data = get_image_data(data_file_path, resolution)
    mesh_coordinates = mesh.get_coordinates()
    face_colors = [
        image_data[int(resolution * (face_center[1] / scale + 0.5)),
                   int(resolution * (face_center[0] / scale + 0.5))]
        for face in mesh.get_topology().faces()
        for face_center in (sum(mesh_coordinates[v.index()]
                                for v in face.vertices()) / 3,)
    ]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d', facecolor='#808080')
    def get_frame(i):
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
        coordinates, network_edges, network_curvatures, _ = data.read_graphml(data_file_path)
        coordinates = np.array(coordinates)
        network_vertices = mesh.map_coordinates_to_support(coordinates, np.float64(0.8))

        ax.clear()
        return [
            get_mesh_plot(mesh, None, face_colors,
                          [network_vertices, network_edges, network_curvatures],
                          ax),
            ax.text2D(0.05, 0.95, f'{left:02}:{round(lam*60):02}',
                      transform=ax.transAxes),
        ]
    ani = animation.FuncAnimation(fig, get_frame,
                                  np.linspace(0, manifold_count - 1,
                                              (manifold_count - 1) * fps + 1),
                                  interval=1000/fps,
                                  blit=True)
    ani.save(os.path.join('..', 'animation_Europe.mp4'), dpi=300)
