import itertools

import numpy as np
from scipy import linalg

from linear_geodesic_optimization.plot import plot_scatter, Animation3D
from linear_geodesic_optimization.mesh.sphere import Mesh as SphereMesh
from linear_geodesic_optimization.optimization \
    import laplacian, geodesic, linear_regression, smooth, standard
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
geodesic_forwards = {s_index: geodesic.Forward(M, laplacian_forward)
                     for s_index in s_indices}
linear_regression_forward = linear_regression.Forward()
smooth_forward = smooth.Forward(M, laplacian_forward)
laplacian_reverse = laplacian.Reverse(M, laplacian_forward)
geodesic_reverses = {s_index: geodesic.Reverse(M, geodesic_forwards[s_index],
                                               laplacian_reverse)
                     for s_index in s_indices}
linear_regression_reverse =  linear_regression.Reverse(linear_regression_forward)
smooth_reverse = smooth.Reverse(M, laplacian_forward, laplacian_reverse)

def get_forwards(s_indices=s_indices):
    phis = []
    lse = 0
    for s_index in s_indices:
        gamma = [s_index]
        s_connected, t = zip(*ts[s_index])
        s_connected = list(s_connected)
        t = np.array(t)
        geodesic_forwards[s_index].calc(gamma)
        phi = geodesic_forwards[s_index].phi
        phis.append(phi)
        linear_regression_forward.calc(phi[s_connected], t)
        lse += linear_regression_forward.lse
    smooth_forward.calc()
    L_smooth = smooth_forward.L_smooth
    return phis, lse, L_smooth

def get_reverses(s_indices=s_indices):
    dif_lse = np.zeros(V)
    dif_L_smooth = np.zeros(V)

    # Avoid recomputing these values too many times
    phis = {}
    ls = {}
    s_connecteds = {}
    Ts = {} # Capitalized due to unfortunate naming clash
    for s_index in s_indices:
        s_connected, t = zip(*ts[s_index])
        s_connected = list(s_connected)
        s_connecteds[s_index] = s_connected
        Ts[s_index] = np.array(t)
        geodesic_forwards[s_index].calc([s_index])
        phi = geodesic_forwards[s_index].phi

        phis[s_index] = phi[s_connected]
        ls[s_index] = approximate_geodesics_fpi(M, phi, s_connected)

    for l in range(V):
        for s_index in s_indices:
            if l in ls[s_index]:
                geodesic_reverses[s_index].calc([s_index], dif_v[l], l)
                dif_phi = geodesic_reverses[s_index].dif_phi[s_connecteds[s_index]]
                linear_regression_reverse.calc(phis[s_index], Ts[s_index], dif_phi, l)
                dif_lse[l] += linear_regression_reverse.dif_lse

        smooth_reverse.calc(dif_v[l], l)
        dif_L_smooth[l] += smooth_reverse.dif_L_smooth

    return dif_lse, dif_L_smooth

lam = 0.1

def get_loss_callback(lam=lam, s_indices=s_indices):
    def loss(rho):
        M.set_rho(rho)
        _, lse, L_smooth = get_forwards(s_indices)
        return lse + lam * L_smooth
    return loss

def get_dif_loss_callback(lam=lam, s_indices=s_indices):
    def dif_loss(rho):
        M.set_rho(rho)
        dif_lse, dif_L_smooth = get_reverses(s_indices)
        return dif_lse + lam * dif_L_smooth
    return dif_loss

def descent_step(rho, lam=lam, s_indices=s_indices):
    f = get_loss_callback(lam, s_indices)
    g = get_dif_loss_callback(lam, s_indices)
    d = -g(rho)
    alpha = standard.wolfe(rho, d, f, g)
    if alpha is None:
        print('Could not find good step size. Using alpha=0.5 instead')
        alpha = 0.5
    rho = M.set_rho(rho + alpha * d)
    return rho

# plot_scatter(geodesic_forwards, ts)

max_iterations = 10

animation_3D = Animation3D()

stochastic = False
for i in itertools.count(1):
    if i > max_iterations:
        break

    # Diagnostic information
    _, lse, L_smooth = get_forwards()
    print(f'iteration {i}: \n'
          + f'\tlse: {lse:.6f}\n'
          + f'\tL_smooth: {L_smooth:.6f}\n'
          + f'\tLoss: {(lse + lam * L_smooth):.6f}')
    animation_3D.add_frame(M)

    if stochastic:
        for s_index in np.random.permutation(s_indices):
            rho = descent_step(M, rho, lam / len(s_indices), [s_index])
    else:
        rho = descent_step(rho, lam, s_indices)

# Diagnostic information
_, lse, L_smooth = get_forwards()
print(f'iteration {i}: \n'
      + f'\tlse: {lse:.6f}\n'
      + f'\tL_smooth: {L_smooth:.6f}\n'
      + f'\tLoss: {(lse + lam * L_smooth):.6f}')
animation_3D.add_frame(M)

animation_3D.get_fig(duration=50).show()

# The following values should hopefully both be near 1.0
print(np.min(rho), np.max(rho))

# plot_scatter(geodesic_forwards, ts)
