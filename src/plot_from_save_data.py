import itertools
import os
import pickle
import sys

from linear_geodesic_optimization.plot import get_scatter_fig, \
    combine_scatter_figs, Animation3D

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python3 <directory name>')
        sys.exit(0)

    directory = sys.argv[1]
    scatter_fig_before = None
    scatter_fig_after = None

    animation_3D = Animation3D()

    for i in itertools.count():
        path = os.path.join(directory, str(i))
        if not os.path.exists(path):
            break

        with open(path, 'rb') as f:
            hierarchy = pickle.load(f)

            if i == 0:
                scatter_fig_before = get_scatter_fig(hierarchy, True)

            path_next = os.path.join(directory, str(i + 1))
            if not os.path.exists(path_next):
                scatter_fig_after = get_scatter_fig(hierarchy, False)

            animation_3D.add_frame(hierarchy.mesh)

    combine_scatter_figs(scatter_fig_before, scatter_fig_after).show()
    animation_3D.get_fig(duration=50).show()
