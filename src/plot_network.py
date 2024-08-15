import argparse
import itertools
import sys

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.basemap import Basemap
import numpy as np
import networkx as nx

from linear_geodesic_optimization.data import input_network, utility
from linear_geodesic_optimization import plot


def get_image_data(coordinates, resolution, scale = 1.):
    """
    Get the image data to plot a square map.

    `coordinates` is a list of pairs of longitudes and latitudes.

    `resolution` is the number of pixels on one side of the image.

    `scale` is the size of the space taken up in the map by the
    coordinates (smaller number = zoom out).
    """
    coordinates = np.array(coordinates)
    center = np.mean(coordinates)
    coordinates = center + (coordinates - center) / scale

    # Find the least disruptive break in the horizontal coordinates
    # (as they are cyclic)
    coordinates_x = list(sorted(coordinates[:, 0] % 1.))
    coordinates_x.append(1 + coordinates_x[0])
    difference_max = -1.
    difference_index = -1
    for index, (x_left, x_right) in enumerate(itertools.pairwise(coordinates_x)):
        difference = x_right - x_left
        if difference > difference_max:
            difference_max = difference
            difference_index = index
    left = coordinates_x[difference_index + 1] - 1.
    right = coordinates_x[difference_index]
    bottom = np.amin(coordinates[:, 1])
    top = np.amax(coordinates[:, 1])

    # Expand the coordinates out so they are a square
    height = top - bottom
    width = right - left
    if height > width:
        center = (left + right) / 2.
        left = center - height / 2.
        right = center + height / 2.
    else:
        center = (bottom + top) / 2.
        bottom = center - width / 2.
        top = center + width / 2.

    # Make a fake image just to grab the map data

    # Create the figure and axes. If errors appear due to invalid fontsize,
    # try increasing the dpi
    dpi = 1.
    fig = plt.figure(figsize = (resolution / dpi, resolution / dpi), dpi = dpi)
    ax = fig.add_subplot()

    # We need the axes to be hidden, but we can't just turn them off
    # (otherwise, the water on the exterior portion of the map will not
    # appear). As a result, we manually hide the axes and tick marks
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.)
    ax.margins(0.)
    fig.tight_layout(pad = 0.)

    m = Basemap(
        projection = 'merc',
        llcrnrlon = utility.inverse_mercator(x = left),
        urcrnrlon = utility.inverse_mercator(x = right),
        llcrnrlat = utility.inverse_mercator(y = bottom),
        urcrnrlat = utility.inverse_mercator(y = top),
        resolution = 'i'
    )
    m.drawcoastlines(ax = ax, linewidth = 0.)
    m.fillcontinents(ax = ax, color = 'coral',lake_color = 'aqua')
    m.drawmapboundary(ax = ax, linewidth = 0., fill_color = 'aqua')

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    canvas_data = np.asarray(canvas.buffer_rgba())

    plt.close(fig)

    return canvas_data, (left, right, bottom, top)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graphml', '-g', dest='graphml_filename', metavar='graphml-file')
    parser.add_argument('--probes', '-p', dest='probes_filename', metavar='probes-file')
    parser.add_argument('--latencies', '-l', dest='latencies_filename', metavar='latencies-file')
    parser.add_argument('--epsilon', '-e', dest='epsilon', metavar='epsilon', type=float)
    parser.add_argument('--show-map', '-m', dest='show_map', action='store_true')
    parser.add_argument('--output', '-o', dest='output_filename', metavar='filename')
    args = parser.parse_args()
    graphml_filename = args.graphml_filename
    probes_filename = args.probes_filename
    latencies_filename = args.latencies_filename
    epsilon = args.epsilon
    show_map = args.show_map
    output_filename = args.output_filename

    if graphml_filename is not None:
        graph = nx.graphml.read_graphml(graphml_filename)
    else:
        if probes_filename is None or latencies_filename is None:
            print('Need to supply input files', file=sys.stderr)
            sys.exit(-1)

        graph = input_network.get_graph_from_paths(
            probes_filename, latencies_filename,
            epsilon, ricci_curvature_alpha=0.9999, clustering_distance=500000
        )

    coordinates = np.array([
        utility.mercator(data['long'], data['lat'])
        for _, data in graph.nodes(data = True)
    ])

    fig, ax = plt.subplots()

    if show_map:
        scale = 0.8
        image_data, extent = get_image_data(coordinates, 1000, scale)
        ax.imshow(image_data, extent = extent)

    plot.get_network_plot(
        graph, ax = ax
    )

    if output_filename is None:
        plt.show()
    else:
        fig.savefig(output_filename, dpi=1000, bbox_inches='tight')
