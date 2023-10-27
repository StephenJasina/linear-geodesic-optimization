import itertools
import os

import numpy as np

import optimization

if __name__ == '__main__':
    lambda_curvature = 1.
    lambda_smooth = 0.0004
    lambda_geodesic = 0.
    initial_radius = 20.
    sides = 50
    width = height = sides
    scale = 1.
    leaveout_proportion = 1.
    ip_type = 'ipv4'

    input_dir = os.path.join(ip_type, 'graph_Europe_clustered_fine_threshold')
    output_dir = os.path.join('..', f'out_{os.path.basename(input_dir)[6:]}')

    max_iterations = 2
    i_values = [
        '{:.2f}'.format(i)
        for i in np.arange(1., 15.25, 0.25)
    ]
    for i, i_previous in zip(i_values, itertools.chain([None], i_values)):
        initialiazation_file_path = None
        if i_previous is not None:
            directory = os.path.join(
                output_dir, f'graph_{i_previous}',
                f'{lambda_curvature}_{lambda_smooth}_{lambda_geodesic}_{initial_radius}_{width}_{height}_{scale}'
            )
            iteration = max(
                int(name)
                for name in os.listdir(directory)
                if name.isdigit()
            )
            initialiazation_file_path = os.path.join(directory, str(iteration))

        optimization.main(
            os.path.join(input_dir, f'graph_{i}.graphml'),
            os.path.join(input_dir, '..', 'graph_Europe', 'latencies.csv'), # maybe change this in the future
            lambda_curvature, lambda_smooth, lambda_geodesic,
            initial_radius, sides, scale, leaveout_proportion,
            max_iterations,
            output_dir, initialiazation_file_path
        )
