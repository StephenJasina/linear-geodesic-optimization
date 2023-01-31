from matplotlib import pyplot as plt
import numpy as np

from linear_geodesic_optimization.optimization import linear_regression

# Allow TeX to be used in titles, axes, etc.
plt.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
})

def get_line_plot(data, title):
    fig, ax = plt.subplots(1, 1)

    ax.plot(range(len(data)), data)
    ax.set_title(title)
    ax.set_xlabel('iteration')
    ax.set_ylabel('Loss')

    return fig

def get_scatter_plot(before_data, after_data, title):
    fig, ax = plt.subplots(1, 1)
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

def get_heat_map(x, y, z, title, network_vertices=[], network_edges=[]):
    fig, ax = plt.subplots(1, 1)

    # Plot the heat map
    im = ax.imshow(z, origin='lower', extent=(-0.5, 0.5, -0.5, 0.5))

    # Plot the edges
    for u, v in network_edges:
        ax.plot([network_vertices[u][0], network_vertices[v][0]],
                [network_vertices[u][1], network_vertices[v][1]],
                'k-')

    # Plot the vertices
    for vertex in network_vertices:
        ax.plot(vertex[0], vertex[1], 'w.')

    ax.set_title(title)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    fig.colorbar(im)

    return fig
