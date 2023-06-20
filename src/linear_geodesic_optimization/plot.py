from collections.abc import Iterable

from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np

# Allow TeX to be used in titles, axes, etc.
plt.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
})

def get_line_plot(data, title, x_max=None, y_max=None):
    fig, ax = plt.subplots(1, 1)

    ax.plot(range(len(data)), data)
    ax.set_xlim(xmax=x_max)
    if y_max is None:
        y_max = np.amax(data) * 1.2
    if y_max == np.float64(0.):
        y_max = np.float64(1.)
    ax.set_ylim(ymin=0, ymax=y_max)
    ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')

    return fig

def get_scatter_plot(before_data, after_data, title):
    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal')

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

    # Plot the after
    ax.plot(after_data[0], after_data[1], 'r.')

    # Plot the "expected" line
    ax.plot([lim_min, lim_max],
            [lim_min, lim_max],
            'k-')

    ax.set_title(title)
    ax.set_xlabel('True Latency')
    ax.set_ylabel('Predicted Latency')

    # We want the scale to be square since the relationship between the
    # true and predicted latencies should be the identity function.
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)

    return fig

def get_heat_map(x=None, y=None, z=None, title='',
                 network_vertices=[], network_edges=[], network_curvatures=[], extra_points=[],
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
    for edge, curvature in zip(network_edges, network_curvatures):
        if edge == []:
            continue

        if isinstance(edge[0], Iterable):
            ax.plot(edge[:,0], edge[:,1],
                    color=mpl.colormaps['RdBu']((curvature + 2) / 4))
        else:
            u, v = edge
            ax.plot([network_vertices[u][0], network_vertices[v][0]],
                    [network_vertices[u][1], network_vertices[v][1]],
                    color=mpl.colormaps['RdBu']((curvature + 2) / 4))

    # Plot the vertices
    for vertex in extra_points:
        ax.plot(vertex[0], vertex[1], '.', color='purple')

    ax.set_title(title)
    ax.set_xlim(np.amin(x), np.amax(x))
    ax.set_ylim(np.amin(y), np.amax(y))

    return fig

def get_mesh_plot(mesh, title, remove_boundary=True):
    vertices = mesh.get_coordinates()
    x, y, z = vertices[:,0], vertices[:,1], vertices[:,2]

    faces = []
    for face in mesh.get_topology().faces():
        v0, v1, v2 = face.vertices()
        if not remove_boundary or not (v0.is_on_boundary()
                                       or v1.is_on_boundary()
                                       or v2.is_on_boundary()):
            faces.append([v0.index(), v1.index(), v2.index()])

    interior_vertices = [v.index() for v in mesh.get_topology().vertices()
                         if not remove_boundary or not v.is_on_boundary()]
    z_interior = z[interior_vertices]
    if len(z_interior) != 0:
        z_min = np.amin(z_interior)
        z_max = np.amax(z_interior)
        if z_min != z_max:
            z = (z - z_min) / (z_max - z_min) / 4.

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(x, y, z, triangles=faces)
    ax.set_title(title)
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_aspect('equal')
    return fig
