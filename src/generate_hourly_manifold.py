import os

import optimization

if __name__ == '__main__':
    lambda_curvature = 1.
    lambda_smooth = 0.004
    lambda_geodesic = 0.
    initial_radius = 20.

    sides = 50
    width = height = sides
    scale = 1.

    leaveout_proportion = 1.

    max_iterations = 200

    for i in range(24):
        initialiazation_file_path = None
        if i != 0:
            directory = os.path.join(
                '..', 'out_Europe_hourly', f'graph_{i - 1}',
                f'{lambda_curvature}_{lambda_smooth}_{lambda_geodesic}_{initial_radius}_{width}_{height}_{scale}'
            )
            iteration = max(
                int(name)
                for name in os.listdir(directory)
                if name.isdigit()
            )
            initialiazation_file_path = os.path.join(directory, str(iteration))

        optimization.main(
            os.path.join('graph_Europe_hourly', f'graph_{i}.graphml'),
            os.path.join('graph_Europe_hourly', f'latencies_{i}.csv'),
            lambda_curvature, lambda_smooth, lambda_geodesic,
            initial_radius, sides, scale, leaveout_proportion,
            max_iterations,
            os.path.join('..', 'out_test'),
            initialiazation_file_path
        )
