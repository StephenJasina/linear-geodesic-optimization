import datetime
import os

import numpy as np
import scipy.optimize

from linear_geodesic_optimization.data import measured
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization import optimization

if __name__ == '__main__':
    directory = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    os.mkdir(directory)

    # Construct the mesh
    width = 20
    height = 20
    mesh = RectangleMesh(width, height)
    partials = mesh.get_partials()
    V = partials.shape[0]
    z = mesh.set_parameters(np.random.normal(0., 0.1, width * height))

    dif_v = {l: partials[l] for l in range(V)}

    # Get some (maybe phony) latency measurements
    ts = measured.rectangle_north_america(mesh)
    s_indices = ts.keys()

    lam = 0.01
    hierarchy = optimization.DifferentiationHierarchy(mesh, ts, lam)

    f = hierarchy.get_loss_callback(s_indices)
    g = hierarchy.get_dif_loss_callback(s_indices)

    hierarchy.diagnostics(None)
    scipy.optimize.minimize(f, z, method='L-BFGS-B', jac=g,
                            callback=hierarchy.diagnostics)
