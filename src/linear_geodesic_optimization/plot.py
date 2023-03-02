from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np

from linear_geodesic_optimization.optimization import linear_regression

# Allow TeX to be used in titles, axes, etc.
plt.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
})

def get_line_plot(data, title, x_max=None, y_max=None):
    fig, ax = plt.subplots(1, 1)

    ax.plot(range(len(data)), data)
    ax.set_xlim(xmax=x_max)
    ax.set_ylim(ymin=0, ymax=y_max)
    ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')

    return fig

def get_scatter_plot(before_data, after_data, title):
    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal')
    linear_regression_forward = linear_regression.Forward()

    lim_min = min(
        min(before_data[0]),
        min(before_data[1]),
        min(after_data[0]),
        min(after_data[1])
    )
    lim_max = max(
        max(before_data[0]),
        max(before_data[1]),
        max(after_data[0]),
        max(after_data[1])
    )

    # Plot the before
    ax.plot(before_data[0], before_data[1], 'b.')
    beta0, beta1 = linear_regression_forward.get_beta(before_data[0],
                                                      before_data[1])
    ax.plot([lim_min, lim_max],
            [beta0 + beta1 * lim_min, beta0 + beta1 * lim_max],
            'b-')

    # Plot the after
    ax.plot(after_data[0], after_data[1], 'r.')
    beta0, beta1 = linear_regression_forward.get_beta(after_data[0],
                                                      after_data[1])
    ax.plot([lim_min, lim_max],
            [beta0 + beta1 * lim_min, beta0 + beta1 * lim_max],
            'r-')

    ax.set_title(title)
    ax.set_xlabel('True Latency')
    ax.set_ylabel('Predicted Latency')

    # We want the scale to be square since the relationship between the
    # true and predicted latencies should be the identity function.
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)

    return fig

def get_heat_map(x=None, y=None, z=None, title='',
                 network_vertices=[], network_curvatures=[],
                 v_range=(None, None)):
    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal')

    # Plot the heat map
    if x is not None and y is not None and z is not None:
        im = ax.imshow(z, origin='lower',
                       extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)),
                       vmin=v_range[0], vmax=v_range[1],
                       cmap=mpl.colormaps['gray'])
        fig.colorbar(im)

    # Plot the edges
    for (u, v), curvature in network_curvatures:
        ax.plot([network_vertices[u][0], network_vertices[v][0]],
                [network_vertices[u][1], network_vertices[v][1]],
                color=mpl.colormaps['RdBu']((curvature + 2) / 3))

    # Plot the vertices
    for vertex in network_vertices:
        ax.plot(vertex[0], vertex[1], '.', color='purple')

    ax.set_title(title)
    ax.set_xlim(np.amin(x), np.amax(x))
    ax.set_ylim(np.amin(y), np.amax(y))

    return fig

def get_mesh_plot(mesh, title, remove_boundary=True):
    vertices = mesh.get_vertices()
    x, y, z = vertices[:,0], vertices[:,1], vertices[:,2]

    boundary_vertices = mesh.get_boundary_vertices()
    faces = [
        face for face in mesh.get_faces()
        if not remove_boundary or
           (face[0] not in boundary_vertices and
            face[1] not in boundary_vertices and
            face[2] not in boundary_vertices)
    ]

    interior_vertices = [v for v in range(len(z)) if v not in boundary_vertices]
    z_interior = z[interior_vertices]
    if len(z_interior) != 0:
        z_min = np.amin(z_interior)
        z_max = np.amax(z_interior)
        if z_min != z_max:
            z = (z - z_min) / (z_max - z_min) / 4.


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(x, y, z, triangles=faces)
    ax.set_aspect('equal')
    return fig
