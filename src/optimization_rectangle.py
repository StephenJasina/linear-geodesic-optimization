import numpy as np

from linear_geodesic_optimization.data import phony, measured
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization import optimization, standard
from linear_geodesic_optimization.plot import get_scatter_fig, Animation3D

# Construct the mesh
width = 8
height = 8
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
def diagnostics(iteration=None):
    _, lse, L_smooth = hierarchy.get_forwards()
    if iteration is None:
        print('final iteration:')
    else:
        print(f'iteration {iteration}:')
    print(f'\tlse: {lse:.6f}')
    print(f'\tL_smooth: {L_smooth:.6f}\n')
    print(f'\tLoss: {(lse + lam * L_smooth):.6f}')

    animation_3D.add_frame(mesh)

f = hierarchy.get_loss_callback(s_indices)
g = hierarchy.get_dif_loss_callback(s_indices)
max_iterations = 10

get_scatter_fig(hierarchy).show()

standard.steepest_descent(z, mesh.set_parameters, f, g, max_iterations, diagnostics)

get_scatter_fig(hierarchy).show()
animation_3D.get_fig(duration=50).show()
