### contains all the meta information needed to run the script
import os

directory = os.path.join('..', 'out_Europe_hourly')
ip_type = 'ipv4'
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
