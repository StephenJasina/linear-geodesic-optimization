import datetime
import os

import scipy.optimize

from linear_geodesic_optimization.data import phony
from linear_geodesic_optimization.mesh.sphere import Mesh as SphereMesh
from linear_geodesic_optimization.optimization import optimization

if __name__ == '__main__':
    directory = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    os.mkdir(directory)

    # Construct the mesh
    frequency = 2
    mesh = SphereMesh(frequency)
    partials = mesh.get_partials()
    V = partials.shape[0]
    rho = mesh.get_parameters()

    dif_v = {l: partials[l] for l in range(V)}

    # Get some (phony) latency measurements
    ts = phony.sphere_random(mesh)

    lam = 0.01
    hierarchy = optimization.DifferentiationHierarchy(mesh, ts, lam, directory)

    f = hierarchy.get_loss_callback()
    g = hierarchy.get_dif_loss_callback()

    hierarchy.diagnostics(None)
    scipy.optimize.minimize(f, rho, method='L-BFGS-B', jac=g,
                            callback=hierarchy.diagnostics)
