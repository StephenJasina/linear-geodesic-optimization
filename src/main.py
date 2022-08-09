from linear_geodesic_optimization.data import phony
from linear_geodesic_optimization.mesh.sphere import Mesh as SphereMesh
from linear_geodesic_optimization.optimization import optimization, standard
from linear_geodesic_optimization.plot import plot_scatter, Animation3D

# Construct the mesh
frequency = 2
mesh = SphereMesh(frequency)
directions = mesh.get_directions()
V = directions.shape[0]
rho = mesh.get_rho()

dif_v = {l: directions[l] for l in range(V)}

# Get some (phony) latency measurements
s_indices, ts = phony.sphere_random(mesh)

lam = 0.1
hierarchy = optimization.DifferentiationHierarchy(mesh, ts, lam)

def print_diagnostics(iteration=None):
    _, lse, L_smooth = hierarchy.get_forwards()
    if iteration is None:
        print('final iteration:\n')
    else:
        print(f'iteration {iteration}: \n')
    print(f'\tlse: {lse:.6f}\n'
          + f'\tL_smooth: {L_smooth:.6f}\n'
          + f'\tLoss: {(lse + lam * L_smooth):.6f}')

def descent_step(rho, s_indices=s_indices):
    f = hierarchy.get_loss_callback(s_indices)
    g = hierarchy.get_dif_loss_callback(s_indices)
    d = -g(rho)
    alpha = standard.wolfe(rho, d, f, g)
    if alpha is None:
        print('Could not find good step size. Skipping iteration.')
        return rho
    rho = mesh.set_rho(rho + alpha * d)
    return rho

plot_scatter(hierarchy)

max_iterations = 3

animation_3D = Animation3D()

f = hierarchy.get_loss_callback(s_indices)
g = hierarchy.get_dif_loss_callback(s_indices)
for i in range(max_iterations):
    print_diagnostics(i)
    animation_3D.add_frame(mesh)

    d = -g(rho)
    alpha = standard.wolfe(rho, d, f, g)
    rho = mesh.set_rho(rho + alpha * d)

print_diagnostics()
animation_3D.add_frame(mesh)

animation_3D.get_fig(duration=50).show()

plot_scatter(hierarchy)
