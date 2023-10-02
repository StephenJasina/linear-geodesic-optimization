import os
import pickle
import sys

from matplotlib import pyplot as plt
from matplotlib import animation as animation
import numpy as np

from linear_geodesic_optimization import data
from linear_geodesic_optimization.plot import get_mesh_plot
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
fps = 10

if __name__ == '__main__':
    initialization_path = os.path.join(directory, 'graph_0', subdirectory_name, '0')

    zs = []
    for i in range(manifold_count):
        print(f'Reading data from manifold {i}')
        with open(os.path.join(directory, 'graph_0', subdirectory_name, 'parameters'), 'rb') as f:
            parameters = pickle.load(f)
            data_file_name = parameters['data_file_name']

        data_file_path = os.path.join('..', 'data', data_file_name)
        data_name, _ = os.path.splitext(os.path.basename(data_file_name))

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

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    def get_frame(i):
        print(f'Computing frame at t={i}')
        left = int(i)
        right = min(left + 1, manifold_count - 1)
        lam = i - left
        z = (1 - lam) * zs[left] + lam * zs[right]

        mesh.set_parameters(z)
        ax.clear()
        return [get_mesh_plot(mesh, f'Mesh', False, ax)]
    ani = animation.FuncAnimation(fig, get_frame,
                                  np.linspace(0, manifold_count - 1,
                                              (manifold_count - 1) * fps + 1),
                                  interval=1000/fps,
                                  blit=True)
    ani.save(os.path.join('..', 'animation_test_2.mp4'), dpi=300)
