import itertools
import os
import pickle
import sys

import numpy as np

from linear_geodesic_optimization.plot import get_line_plot, get_scatter_fig, \
    combine_scatter_figs, Animation3D

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python3 <directory name>')
        sys.exit(0)

    directory = sys.argv[1]

    if not os.path.exists(os.path.join(directory, '0')):
        print('Error: supplied directory must contain file named "0"')
        sys.exit(0)

    lses = []
    L_smooths = []
    Ls = []

    scatter_fig_before = None
    scatter_fig_after = None

    animation_3D = Animation3D()

    for i in itertools.count():
        path = os.path.join(directory, str(i))
        with open(path, 'rb') as f:
            hierarchy = pickle.load(f)
            hierarchy.cores = 2

            lse, L_smooth = hierarchy.get_forwards()
            lses.append(lse)
            L_smooths.append(L_smooth)
            Ls.append(lse + hierarchy.lam * L_smooth)

            if i == 0:
                scatter_fig_before = get_scatter_fig(hierarchy, True)

            path_next = os.path.join(directory, str(i + 1))
            if not os.path.exists(path_next):
                scatter_fig_after = get_scatter_fig(hierarchy, False)
                break

            animation_3D.add_frame(hierarchy.mesh)

    get_line_plot(lses, 'Least Squares Loss').show()
    get_line_plot(L_smooths, 'Smoothness Loss').show()
    get_line_plot(Ls, 'Total Loss').show()

    combine_scatter_figs(scatter_fig_before, scatter_fig_after).show()
    animation_3D.get_fig(duration=50).show()
