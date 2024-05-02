from collections.abc import Iterable

from adjustText import adjust_text
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LightSource
import numpy as np

from linear_geodesic_optimization.data import utility


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

def get_network_plot(graph, trim_vertices=False):
    fig, ax = plt.subplots(1, 1, facecolor='#808080')
    ax.set_aspect('equal')
    ax.axis('off')

    # Plot the edges
    for u, v, d in graph.edges(data=True):
        color = mpl.colormaps['RdBu']((d['ricciCurvature'] + 2) / 4)

        x_u, y_u = utility.mercator(graph.nodes[u]['long'],
                                    graph.nodes[u]['lat'])
        x_v, y_v = utility.mercator(graph.nodes[v]['long'],
                                    graph.nodes[v]['lat'])
        ax.plot([x_u, x_v], [y_u, y_v], color=color)

    # Plot the vertices
    for node, d in graph.nodes(data=True):
        # If trim_vertices is set, then only plot the vertices with
        # incident edges
        if not trim_vertices or graph[node]:
            x, y = utility.mercator(graph.nodes[node]['long'],
                                    graph.nodes[node]['lat'])
            ax.plot(x, y, '.', ms=4, color='green')

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
                       cmap=mpl.colormaps['PiYG'])
        fig.colorbar(im)

    # Plot the edges
    edges_curvatures = list(zip(network_edges, network_curvatures))
    rng = np.random.default_rng()
    rng.shuffle(edges_curvatures)
    for edge, curvature in edges_curvatures:
        if edge == []:
            continue

        if curvature is None:
            color = 'k'
        else:
            color = mpl.colormaps['RdBu']((curvature + 2) / 4)

        if isinstance(edge[0], Iterable):
            ax.plot(edge[:,0], edge[:,1],
                    color=color)
        else:
            u, v = edge
            ax.plot([network_vertices[u][0], network_vertices[v][0]],
                    [network_vertices[u][1], network_vertices[v][1]],
                    color=color)

    # Plot the vertices
    for vertex in extra_points:
        ax.plot(vertex[0], vertex[1], '.', ms=4, color='green')

    ax.set_title(title)
    ax.set_xlim(np.amin(x), np.amax(x))
    ax.set_ylim(np.amin(y), np.amax(y))

    return fig

def get_mesh_plot(mesh, title = '', face_colors=None, network=None, ax=None):
    vertices = mesh.get_coordinates()
    x, y, z = vertices[:,0], vertices[:,1], vertices[:,2]

    faces = []
    for face in mesh.get_topology().faces():
        faces.append([v.index() for v in face.vertices()])

    z_min = np.amin(z)
    z_max = np.amax(z)

    to_return = None
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        p3dc = ax.plot_trisurf(x, y, z, triangles=faces, color='tab:blue')
        if face_colors is not None:
            p3dc.set_fc(face_colors)
        to_return = fig
    else:
        to_return = ax.plot_trisurf(x, y, z, triangles=faces, color='tab:blue',
                                    animated=True)
        if face_colors is not None:
            to_return.set_fc(face_colors)

    if network is not None:
        network_vertices, network_edges, network_curvatures, network_name = network
        # Plot the edges
        for (u, v), curvature in zip(network_edges, network_curvatures):
            color = mpl.colormaps['RdBu']((curvature + 2) / 4)

            ax.plot([network_vertices[u][0], network_vertices[v][0]],
                    [network_vertices[u][1], network_vertices[v][1]],
                    [0.7, 0.7], color=color)

        # # Plot the vertices
        for vertex in network_vertices:
            ax.plot(vertex[0], vertex[1], 0.7, '.', ms=4, color='green')
        # Plot the name of a few cities above the vertices
        # for vertex, name in zip(network_vertices, network_name):
        # texts = [ax.text(vertex[0], vertex[1], 0.7, name, fontsize=4) for (vertex,name) in zip(network_vertices, network_name)]
        ### only show the text if it is not overlapping with another text
        # adjust_text(texts)

        def get_min_distance_based_on_label(label, base_distance=1.0, scaling_factor=0.1):
            """Determine spacing based on label length."""
            return base_distance + scaling_factor * len(label)

        def is_too_close(new_vertex, labeled_vertices, label, base_distance=0.05, scaling_factor=0.01):
            """Check if the new vertex is too close to any of the already labeled vertices."""
            min_distance = get_min_distance_based_on_label(label, base_distance, scaling_factor)
            if label in ['Tallinn', 'Tuusula','Battipaglia']:
                return False
            if len(label) > 8:
                return True
            for v, l in labeled_vertices:
                dist = ((new_vertex[0] - v[0]) ** 2 + (new_vertex[1] - v[1]) ** 2) ** 0.5
                if dist < min_distance:
                    return True
            return False

        labeled_vertices = []
        texts = []

        for vertex, name in zip(network_vertices, network_name):
            if not is_too_close(vertex, labeled_vertices, name):
                texts.append(ax.text(vertex[0]-0.01, vertex[1]+0.01, 0.7, name, fontsize=4))
                labeled_vertices.append((vertex, name))


    ax.set_title(title)
    ax.set_xlim([-mesh._scale / 2, mesh._scale / 2])
    ax.set_ylim([-mesh._scale / 2, mesh._scale / 2])
    ax.set_aspect('equal')
    ax.set_axis_off()

    return to_return

def get_rectangular_mesh_plot(z, face_colors, title, vertical_scale=0.25,
                              network=None, ax=None):
    width, height = face_colors.shape[:2]
    x, y = np.meshgrid(
        np.linspace(-0.5, 0.5, width),
        np.linspace(-0.5, 0.5, height),
        indexing='ij'
    )

    # Resize z to match face_colors, interpolating if needed
    z = np.array([
        [
            (1 - lambda_j) * ((1 - lambda_i) * z[i_left,j_bottom] + lambda_i * z[i_right,j_bottom])
                + lambda_j * ((1 - lambda_i) * z[i_left,j_top] + lambda_i * z[i_right,j_top])
            for j in np.linspace(0, z.shape[1] - 1, height)
            for lambda_j in (j % 1,)
            for j_bottom in (np.floor(j).astype(int),)
            for j_top in (np.ceil(j).astype(int),)
        ]
        for i in np.linspace(0, z.shape[0] - 1, width)
        for lambda_i in (i % 1,)
        for i_left in (np.floor(i).astype(int),)
        for i_right in (np.ceil(i).astype(int),)
    ])

    z_min = np.amin(z)
    z_max = np.amax(z)
    if z_min != z_max:
        z = (z - z_min) / (z_max - z_min) * vertical_scale

    to_return = None
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(x, y, z, facecolors=face_colors,
                        rcount=height, ccount=width)
        to_return = fig
    else:
        to_return = ax.plot_surface(x, y, z, facecolors=face_colors,
                                    rcount=height, ccount=width)

    if network is not None:
        network_vertices, network_edges, network_curvatures, network_name = network
        # Plot the edges
        for (u, v), curvature in zip(network_edges, network_curvatures):
            color = mpl.colormaps['RdBu']((curvature + 2) / 4)

            ax.plot([network_vertices[u][0], network_vertices[v][0]],
                    [network_vertices[u][1], network_vertices[v][1]],
                    [0.7, 0.7], color=color)

        # Plot the vertices
        for vertex in network_vertices:
            ax.plot(vertex[0], vertex[1], 0.7, '.', ms=4, color='green')
        # Plot the name of a few cities above the vertices
        # for vertex, name in zip(network_vertices, network_name):
        # texts = [ax.text(vertex[0], vertex[1], 0.7, name, fontsize=4) for (vertex,name) in zip(network_vertices, network_name)]
        ### only show the text if it is not overlapping with another text
        # adjust_text(texts)

        def get_min_distance_based_on_label(label, base_distance=1.0, scaling_factor=0.1):
            """Determine spacing based on label length."""
            return base_distance + scaling_factor * len(label)

        def is_too_close(new_vertex, labeled_vertices, label, base_distance=0.05, scaling_factor=0.01):
            """Check if the new vertex is too close to any of the already labeled vertices."""
            min_distance = get_min_distance_based_on_label(label, base_distance, scaling_factor)
            if label in ['Tallinn', 'Tuusula','Battipaglia']:
                return False
            if len(label) > 8:
                return True
            for v, l in labeled_vertices:
                dist = ((new_vertex[0] - v[0]) ** 2 + (new_vertex[1] - v[1]) ** 2) ** 0.5
                if dist < min_distance:
                    return True
            return False

        labeled_vertices = []
        texts = []

        for vertex, name in zip(network_vertices, network_name):
            if not is_too_close(vertex, labeled_vertices, name):
                texts.append(ax.text(vertex[0]-0.01, vertex[1]+0.01, 0.7, name, fontsize=4))
                labeled_vertices.append((vertex, name))

    ax.set_title(title)
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_aspect('equal')
    ax.set_axis_off()

    return to_return
