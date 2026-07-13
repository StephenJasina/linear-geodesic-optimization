import argparse
import glob
import os
import re

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import networkx as nx
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap


def _parse_n(path):
    m = re.search(r'graph(\d+)\.graphml$', path)
    return int(m.group(1))


def main(output_path, residual_max=None):
    graph_dir = os.path.join(os.path.dirname(__file__), '..')
    paths = sorted(glob.glob(os.path.join(graph_dir, 'graph*.graphml')), key=_parse_n)

    node_pos = {}
    edge_residual = {}

    for path in paths:
        n = _parse_n(path)
        g = nx.read_graphml(path)

        for node, data in g.nodes(data=True):
            if node not in node_pos:
                node_pos[node] = (data['long'], data['lat'])

        for u, v in g.edges():
            key = (min(u, v), max(u, v))
            if key not in edge_residual:
                edge_residual[key] = n

    residual_values = list(edge_residual.values())
    residual_min = min(residual_values)
    if residual_max is None:
        residual_max = max(residual_values)
    norm = mcolors.Normalize(vmin=residual_min, vmax=residual_max)
    cmap = cm.viridis

    xs = [pos[0] for pos in node_pos.values()]
    ys = [pos[1] for pos in node_pos.values()]
    pad_lon = (max(xs) - min(xs)) * 0.05
    pad_lat = (max(ys) - min(ys)) * 0.05

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('white')

    m = Basemap(
        projection='merc',
        llcrnrlon=min(xs) - pad_lon,
        urcrnrlon=max(xs) + pad_lon,
        llcrnrlat=min(ys) - pad_lat,
        urcrnrlat=max(ys) + pad_lat,
        resolution='i',
        ax=ax,
    )
    m.drawmapboundary(fill_color='white')
    m.fillcontinents(color='black', lake_color='white')
    m.drawcoastlines(linewidth=0.3, color='#444444')

    segments = []
    colors = []
    for (u, v), residual in edge_residual.items():
        if residual > residual_max:
            continue
        if u in node_pos and v in node_pos:
            x_u, y_u = m(*node_pos[u])
            x_v, y_v = m(*node_pos[v])
            segments.append([(x_u, y_u), (x_v, y_v)])
            colors.append(cmap(norm(residual)))

    lc = LineCollection(segments, colors=colors, linewidths=0.8, zorder=3)
    ax.add_collection(lc)

    proj_xs, proj_ys = m(xs, ys)
    ax.scatter(proj_xs, proj_ys, s=4, color='white', zorder=4)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.1)
    fig.colorbar(sm, cax=cax, label='Residual Latency')

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', dest='output', metavar='file', default=None)
    args = parser.parse_args()
    main(args.output, 22)
