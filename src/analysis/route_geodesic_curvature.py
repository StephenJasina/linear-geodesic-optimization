"""
Report how far network links and routes are from geodesics on optimized
surfaces.

Each link and each route (as a polyline through its nodes, and as a
natural cubic spline through them) is lifted onto a run's heights, and
its geodesic curvature kappa_g is measured. See
`linear_geodesic_optimization.geodesic_curvature`. The headline number
is the total absolute geodesic turning in radians (0 for a geodesic),
compared with the turning the same curve has in the flat plane.

Run from `src/`:

    python -m analysis.route_geodesic_curvature \\
        ../outputs/<experiment>/<run or batch directory> [--output out.json]

The directory may be a single run (containing `output.json`) or a
directory of runs.
"""

import argparse
import json
import pathlib
import re

import numpy as np

from linear_geodesic_optimization import geodesic_curvature as gc
from linear_geodesic_optimization.mesh.rectangle import Mesh


def find_runs(directory):
    if (directory / 'output.json').exists():
        return [directory]
    runs = [path.parent for path in directory.glob('*/output.json')]

    # Order by the leading index of names like `9_graph_0`
    def key(path):
        match = re.match(r'(\d+)', path.name)
        return (int(match.group(1)) if match else np.inf, path.name)

    return sorted(runs, key=key)


def summarize(result):
    return {
        key: result[key] for key in (
            'length', 'turning', 'smooth_turning', 'corner_turning',
            'planar_turning', 'bending', 'max_kappa_g',
        )
    }


def analyze_run(path, heights, spacing_fraction):
    with open(path / 'output.json') as file:
        output = json.load(file)
    parameters = output['parameters']
    graph_data, vertex_data, edge_data = output['network']
    mesh = Mesh(
        parameters['width'], parameters['height'], parameters['mesh_scale']
    )
    fields = gc.derivative_fields(mesh, np.array(output[heights]))
    spacing = spacing_fraction * mesh.get_scale() / (mesh.get_width() - 1)
    vertices = mesh.map_coordinates_to_support(
        np.array(graph_data['coordinates']),
        parameters['coordinates_scale'], graph_data['bounding_box'],
    )

    labels = graph_data['labels']
    label_to_index = {str(label): index for index, label in enumerate(labels)}
    for index, elements in enumerate(vertex_data.get('elements', [])):
        for element in elements or []:
            label_to_index.setdefault(str(element), index)

    links = []
    for (i, j), throughput in zip(
        graph_data['edges'], edge_data.get('throughput', [None] * len(labels))
    ):
        result = gc.route_curvature(
            mesh, fields, vertices[[i, j]], spacing
        )
        links.append({
            'link': [labels[i], labels[j]],
            'throughput': throughput,
            **summarize(result),
        })

    routes = []
    for route, traffic in zip(output['routes'], output['traffic']):
        if len(route) < 2:
            continue
        waypoints = vertices[[label_to_index[str(label)] for label in route]]
        routes.append({
            'route': route,
            'traffic': traffic,
            'polyline': summarize(
                gc.route_curvature(mesh, fields, waypoints, spacing)
            ),
            'spline': summarize(gc.route_curvature(
                mesh, fields, waypoints, spacing, curve='spline'
            )),
        })
    return {'run': path.name, 'links': links, 'routes': routes}


def weighted_mean(routes, curve, key):
    weights = np.array([route['traffic'] for route in routes])
    values = np.array([route[curve][key] for route in routes])
    return float((weights * values).sum() / weights.sum())


def print_report(runs):
    print('Routes (traffic-weighted means, radians of geodesic turning; '
          'a geodesic has 0)')
    print(f'{"run":<12} {"routes":>6}  {"polyline":>8} {"corners":>8} '
          f'{"along":>8} {"planar":>8}  {"spline":>8} {"planar":>8}')
    for run in runs:
        routes = run['routes']
        print(
            f'{run["run"]:<12} {len(routes):>6}  '
            f'{weighted_mean(routes, "polyline", "turning"):>8.3f} '
            f'{weighted_mean(routes, "polyline", "corner_turning"):>8.3f} '
            f'{weighted_mean(routes, "polyline", "smooth_turning"):>8.3f} '
            f'{weighted_mean(routes, "polyline", "planar_turning"):>8.3f}  '
            f'{weighted_mean(routes, "spline", "turning"):>8.3f} '
            f'{weighted_mean(routes, "spline", "planar_turning"):>8.3f}'
        )

    print('\nLinks: integral of |kappa_g| ds along each straight link '
          '(radians)')
    names = ['-'.join(link['link']) for link in runs[0]['links']]
    print(f'{"link":<6}' + ''.join(f'{run["run"]:>11}' for run in runs))
    for index, name in enumerate(names):
        print(f'{name:<6}' + ''.join(
            f'{run["links"][index]["turning"]:>11.3f}' for run in runs
        ))

    print('\nLeast geodesic routes in each run (polyline turning vs. '
          'planar turning)')
    for run in runs:
        worst = sorted(
            run['routes'], key=lambda route: -route['polyline']['turning']
        )[:3]
        print(f'{run["run"]:<12} ' + '; '.join(
            f'{"-".join(route["route"])} '
            f'{route["polyline"]["turning"]:.2f} vs '
            f'{route["polyline"]["planar_turning"]:.2f}'
            for route in worst
        ))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('directory', type=pathlib.Path)
    parser.add_argument('--heights', choices=('final', 'initial'),
                        default='final')
    parser.add_argument('--spacing', type=float, default=0.25,
                        help='sample spacing, as a fraction of the grid '
                             'spacing')
    parser.add_argument('--output', type=pathlib.Path,
                        help='write per-link and per-route results here')
    args = parser.parse_args()

    runs = [
        analyze_run(path, args.heights, args.spacing)
        for path in find_runs(args.directory)
    ]
    if not runs:
        raise SystemExit(f'No output.json files under {args.directory}')
    print_report(runs)
    if args.output is not None:
        with open(args.output, 'w') as file:
            json.dump({'heights': args.heights, 'runs': runs}, file, indent=4)


if __name__ == '__main__':
    main()
