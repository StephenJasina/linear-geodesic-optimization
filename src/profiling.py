import cProfile
import io
import pstats

import numpy as np

from linear_geodesic_optimization.mesh.sphere import Mesh as SphereMesh
from linear_geodesic_optimization.optimization import laplacian, geodesic, linear_regression, smooth

# Construct the mesh
frequency = 5
M = SphereMesh(frequency)
directions = M.get_directions()
V = directions.shape[0]
dif_v = {l: directions[l] for l in range(V)}

lat_long_pairs = [
    (0, 0),
    (0, 180),
]
directions = [SphereMesh.latitude_longitude_to_direction(lat, long) for (lat, long) in lat_long_pairs]
s_indices = [M.nearest_direction_index(direction) for direction in directions]
ts = {si: [(sj, np.arccos(dsi @ dsj))
           for sj, dsj in zip(s_indices, directions)]
      for si, dsi in zip(s_indices, directions)}

# Construct the differentiation heirarchy
laplacian_forward = laplacian.Forward(M)
geodesic_forward = geodesic.Forward(M, laplacian_forward)
linear_regression_forward = linear_regression.Forward()
smooth_forward = smooth.Forward(M, laplacian_forward)
laplacian_reverse = laplacian.Reverse(M, laplacian_forward)
geodesic_reverse = geodesic.Reverse(M, geodesic_forward, laplacian_reverse)
linear_regression_reverse = linear_regression.Reverse(linear_regression_forward)
smooth_reverse = smooth.Reverse(M, laplacian_forward, laplacian_reverse)

lam = 0.1
eta = 1

pr = cProfile.Profile()
pr.enable()

gamma = [s_indices[0]]
s_connected, t = zip(*ts[s_indices[0]])
s_connected = list(s_connected)
t = np.array(t)

dif_L = np.zeros(V)

# Compute the geodesic loss gradient
geodesic_forward.calc(gamma)
phi = geodesic_forward.phi
for l in range(V):
    geodesic_reverse.calc(gamma, dif_v[l], l)
    dif_phi = geodesic_reverse.dif_phi[s_connected]
    linear_regression_reverse.calc(phi[s_connected], t, dif_phi, l)
    dif_lse = linear_regression_reverse.dif_lse
    dif_L[l] += dif_lse

# Compute the smooth loss gradient
for l in range(V):
    smooth_reverse.calc(dif_v[l], l)
    dif_L_smooth = smooth_reverse.dif_L_smooth
    dif_L[l] += lam * dif_L_smooth

pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s)
ps.strip_dirs()
ps.sort_stats(pstats.SortKey.CUMULATIVE)
ps.print_stats()
s = s.getvalue().split('\n')
for line in s:
    if '(_calc_' in line or 'filename:lineno(function)' in line:
        print(line)
