import numpy as np
import scipy.optimize

from linear_geodesic_optimization.data import measured
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization import optimization
from linear_geodesic_optimization.plot import get_scatter_fig, \
    combine_scatter_figs, Animation3D

if __name__ == '__main__':
    # Construct the mesh
    width = 14
    height = 14
    mesh = RectangleMesh(width, height)
    partials = mesh.get_partials()
    V = partials.shape[0]
    z = mesh.set_parameters(np.random.normal(0., 0.1, width * height))

    dif_v = {l: partials[l] for l in range(V)}

    # Get some (maybe phony) latency measurements
    s_indices, ts = measured.rectangle_north_america(mesh)

    lam = 0.01
    hierarchy = optimization.DifferentiationHierarchy(mesh, ts, lam)

    animation_3D = Animation3D()
    iteration = 0
    def diagnostics(_):
        global iteration
        lse, L_smooth = hierarchy.get_forwards()
        print(f'iteration {iteration}:')
        print(f'\tlse: {lse:.6f}')
        print(f'\tL_smooth: {L_smooth:.6f}\n')
        print(f'\tLoss: {(lse + lam * L_smooth):.6f}')

        animation_3D.add_frame(mesh)

        iteration = iteration + 1

    f = hierarchy.get_loss_callback(s_indices)
    g = hierarchy.get_dif_loss_callback(s_indices)

    before = get_scatter_fig(hierarchy, True)

    diagnostics(None)
    scipy.optimize.minimize(f, z, method='L-BFGS-B', jac=g, callback=diagnostics)

    after = get_scatter_fig(hierarchy, False)
    animation_3D.get_fig(duration=50).show()

    combine_scatter_figs(before, after).show()
