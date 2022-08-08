import itertools

import numpy as np
from scipy import linalg

from linear_geodesic_optimization.plot import plot_scatter, Animation3D
from linear_geodesic_optimization.mesh.sphere import Mesh as SphereMesh
from linear_geodesic_optimization.optimization \
    import laplacian, geodesic, linear_regression, smooth
from linear_geodesic_optimization.optimization.partial_selection \
    import approximate_geodesics_fpi
from linear_geodesic_optimization.data import phony

# Construct the mesh
frequency = 3
M = SphereMesh(frequency)
directions = M.get_directions()
V = directions.shape[0]
rho = np.random.random(V) + 0.5
rho /= sum(linalg.norm(rho[l]) for l in range(V)) / V
M.set_rho(rho)

dif_v = {l: directions[l] for l in range(V)}

# Get some (phony) latency measurements
s_indices, ts = phony.sphere_true(M)

# Construct the differentiation heirarchy
laplacian_forward = laplacian.Forward(M)
geodesic_forward = geodesic.Forward(M, laplacian_forward)
linear_regression_forward = linear_regression.Forward()
smooth_forward = smooth.Forward(M, laplacian_forward)
laplacian_reverse = laplacian.Reverse(M, laplacian_forward)
geodesic_reverse = geodesic.Reverse(M, geodesic_forward, laplacian_reverse)
linear_regression_reverse = linear_regression.Reverse(linear_regression_forward)
smooth_reverse = smooth.Reverse(M, laplacian_forward, laplacian_reverse)

def get_losses(s_indices, ts):
    lse = 0
    for s_index in np.random.permutation(s_indices):
        gamma = [s_index]
        s_connected, t = zip(*ts[s_index])
        s_connected = list(s_connected)
        t = np.array(t)
        geodesic_forward.calc(gamma)
        phi = geodesic_forward.phi
        linear_regression_forward.calc(phi[s_connected], t)
        lse += linear_regression_forward.lse
    smooth_forward.calc()
    L_smooth = smooth_forward.L_smooth
    return lse, L_smooth

plot_scatter(geodesic_forward, ts)

# Run gradient descent

lam = 0.1
max_iterations = 5

animation_3D = Animation3D()

for i in itertools.count(1):
    if i > max_iterations:
        break

    eta = 1 / i

    lse, L_smooth = get_losses(s_indices, ts)
    print(f'iteration {i}: \n\tlse: {lse:.6f}\n\tL_smooth: {L_smooth:.6f}\n\tLoss: {(lse + lam * L_smooth):.6f}')

    for s_index in np.random.permutation(s_indices):
        animation_3D.add_frame(M)
        gamma = [s_index]
        s_connected, t = zip(*ts[s_index])
        s_connected = list(s_connected)
        t = np.array(t)

        dif_L = np.zeros(V)

        geodesic_forward.calc(gamma)
        phi = geodesic_forward.phi
        ls = approximate_geodesics_fpi(M, phi, s_connected)
        for l in range(V):
            # Compute the geodesic loss gradient
            if l in ls:
                geodesic_reverse.calc(gamma, dif_v[l], l)
                dif_phi = geodesic_reverse.dif_phi[s_connected]
                linear_regression_reverse.calc(phi[s_connected], t, dif_phi, l)
                dif_lse = linear_regression_reverse.dif_lse
                dif_L[l] += dif_lse

            # Compute the smooth loss gradient
            smooth_reverse.calc(dif_v[l], l)
            dif_L_smooth = smooth_reverse.dif_L_smooth
            dif_L[l] += lam * dif_L_smooth

        # Apply the gradient step, and then normalize rho
        rho = np.maximum(rho - eta * dif_L, 0.01)
        rho /= sum(linalg.norm(rho[l]) for l in range(V)) / V

        M.set_rho(rho)

lse, L_smooth = get_losses(s_indices, ts)
print(f'\nFinal lse: {lse:.6f}\nFinal L_smooth: {L_smooth:.6f}\nFinal Loss: {(lse + lam * L_smooth):.6f}')
animation_3D.add_frame(M)

animation_3D.get_fig(duration=50).show()

# The following values should hopefully both be near 1.0
print(np.min(rho), np.max(rho))

plot_scatter(geodesic_forward, ts)
