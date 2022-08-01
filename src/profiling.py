import itertools

import numpy as np
from scipy import linalg

from linear_geodesic_optimization.mesh.mesh import Animation3D
from linear_geodesic_optimization.mesh.sphere import Mesh as SphereMesh
from linear_geodesic_optimization.optimization import laplacian, geodesic, linear_regression, smooth

def approximate_geodesics_fpi(mesh, phi, initial_vertices):
    e = mesh.get_edges()
    c = mesh.get_c()
    vertices = set()
    to_process = []
    processed = set()
    for b in initial_vertices:
        for i in e[b]:
            cbi = c[b,i]
            if phi[b] > phi[i] or phi[b] > phi[cbi]:
                to_process.append((b, i, cbi))

    while to_process:
        (i, j, k) = to_process[-1]
        del to_process[-1]
        if j < i and j < k:
            j, k, i = i, j, k
        elif k < i and k < j:
            k, i, j = i, j, k
        if (i, j, k) in processed:
            continue

        vertices.add(i)
        vertices.add(j)
        vertices.add(k)

        cji = c[j,i]
        if phi[cji] < phi[i] or phi[cji] < phi[j]:
            to_process.append((j, i, cji))

        ckj = c[k,j]
        if phi[ckj] < phi[j] or phi[ckj] < phi[k]:
            to_process.append((k, j, ckj))

        cik = c[i,k]
        if phi[cik] < phi[k] or phi[cik] < phi[i]:
            to_process.append((i, k, cik))

        processed.add((i, j, k))

    return vertices

# Construct the mesh
frequency = 1
M = SphereMesh(frequency)
rng = np.random.default_rng()
directions = M.get_directions()
V = directions.shape[0]
dif_v = {l: directions[l] for l in range(V)}
rho = rng.random(V) + 0.5
rho /= sum(linalg.norm(rho[l]) for l in range(V)) / V
M.set_rho(rho)

# Get some (phony) latency measurements
lat_long_pairs = [
    (0, 0),
    (0, 90),
    (90, 0),
    (0, 180),
    (0, -90),
    (-90, 0),
]
directions = [SphereMesh.latitude_longitude_to_direction(lat, long) for (lat, long) in lat_long_pairs]
s_indices = [M.nearest_direction_index(direction) for direction in directions]
ts = {si: [(sj, np.arccos(dsi @ dsj))
           for j, (sj, dsj) in enumerate(zip(s_indices, directions)) if (i - j) % 6 != 0]
      for i, (si, dsi) in enumerate(zip(s_indices, directions))}

# Construct the differentiation heirarchy
laplacian_forward = laplacian.Forward(M)
geodesic_forward = geodesic.Forward(M, laplacian_forward)
linear_regression_forward = linear_regression.Forward()
smooth_forward = smooth.Forward(M)
laplacian_reverse = laplacian.Reverse(M, laplacian_forward)
geodesic_reverse = geodesic.Reverse(M, geodesic_forward, laplacian_reverse)
linear_regression_reverse = linear_regression.Reverse(linear_regression_forward)
smooth_reverse = smooth.Reverse(M)

# Run gradient descent

lam = 0.1
max_iterations = 1

def get_losses(s_indices, ts):
    lse = 0
    for s_index in np.random.permutation(s_indices):
        gamma = [s_index]
        s_connected, t = zip(*ts[s_index])
        s_connected = list(s_connected)
        t = np.array(t)
        phi = geodesic_forward.calc_phi(gamma)
        lse += linear_regression_forward.calc_lse(phi[s_connected], t)
    L_smooth = smooth_forward.calc_L_smooth()
    return lse, L_smooth

rng = np.random.default_rng()

animation_3D = Animation3D()

import cProfile
import io
import pstats
from pstats import SortKey
pr = cProfile.Profile()
pr.enable()
for i in itertools.count(1):
    if i > max_iterations:
        break

    eta = 1 / i

    lse, L_smooth = get_losses(s_indices, ts)
    print(f'iteration {i}:\n\tlse: {lse:.6f}\n\tL_smooth: {L_smooth:.6f}\n\tLoss: {(lse + lam * L_smooth):.6f}')

    for s_index in np.random.permutation(s_indices):
        animation_3D.add_frame(M)
        gamma = [s_index]
        s_connected, t = zip(*ts[s_index])
        s_connected = list(s_connected)
        t = np.array(t)

        dif_L = np.zeros(V)
        dif_L_smooth = smooth_reverse.calc_dif_L_smooth(dif_v)
        for l in range(V):
            dif_L[l] +=  lam * dif_L_smooth[l]

        phi = geodesic_forward.calc_phi(gamma)
        ls = approximate_geodesics_fpi(M, phi, s_connected)
        dif_phi = geodesic_reverse.calc_dif_phi(gamma, dif_v, ls)
        dif_phi = {l: dif[s_connected] for l, dif in dif_phi.items()}
        dif_lse = linear_regression_reverse.calc_dif_lse(phi[s_connected], t, dif_phi, ls)

        for l in dif_lse:
            dif_L[l] += dif_lse[l]

        rho = np.maximum(rho - eta * dif_L, 0.01)
        rho /= sum(linalg.norm(rho[l]) for l in range(V)) / V
        M.set_rho(rho)

lse, L_smooth = get_losses(s_indices, ts)
print(f'\nFinal lse: {lse:.6f}\nFinal L_smooth: {L_smooth:.6f}\nFinal Loss: {(lse + lam * L_smooth):.6f}')
animation_3D.add_frame(M)

pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())
