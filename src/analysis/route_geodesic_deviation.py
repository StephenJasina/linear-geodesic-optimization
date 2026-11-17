"""
Report how far the geodesics drawn by the viewer are from the routes
they stand for.

collation.py draws, for each route in `geodesicRoutes`, the shortest
path between the route's endpoints near the route (a Dijkstra trace
through its waypoints, tightened into a geodesic). The geodesic does not
have to pass through the route's intermediate waypoints. For each frame
this prints, in units of the mesh width:
* `max dev`: the mean over routes of the largest distance from a point
  of the geodesic to the route's polyline (and the worst route),
* `miss`: the mean over routes with intermediate waypoints of the
  largest distance from a waypoint to the geodesic (and the worst),
* `hit`: the fraction of those routes whose geodesic passes within
  `--tolerance` of every intermediate waypoint.

Run from `src/`:

    python -m analysis.route_geodesic_deviation <animation.json> [...]
"""

import argparse
import json
import pathlib

import numpy as np


def segment_distances(points, a, b):
    direction = b - a
    t = np.clip(((points - a) @ direction) / (direction @ direction), 0, 1)
    return np.linalg.norm(points - (a + t[:, None] * direction), axis=1)


def polyline_distances(points, polyline):
    return np.min([
        segment_distances(points, a, b)
        for a, b in zip(polyline[:-1], polyline[1:])
        if np.any(a != b)
    ], axis=0)


def frame_deviation(positions, routes, geodesics, tolerance):
    deviations, misses = [], []
    for route, geodesic in zip(routes, geodesics):
        if len(geodesic) < 2:
            continue
        geodesic = np.array(geodesic)
        polyline = positions[route]
        deviations.append(polyline_distances(geodesic, polyline).max())
        if len(route) > 2:
            misses.append(polyline_distances(polyline[1:-1], geodesic).max())
    deviations, misses = np.array(deviations), np.array(misses)
    return {
        'max_deviation_mean': float(deviations.mean()),
        'max_deviation_worst': float(deviations.max()),
        'miss_mean': float(misses.mean()) if len(misses) else 0.,
        'miss_worst': float(misses.max()) if len(misses) else 0.,
        'hit_fraction': float((misses < tolerance).mean()) if len(misses) else 1.,
    }


def animation_deviation(path, tolerance):
    with open(path) as file:
        animation = json.load(file)
    positions = np.array([node['coordinates'] for node in animation['nodes']])
    return [
        frame_deviation(
            positions, animation['geodesicRoutes'], frame['geodesics'],
            tolerance,
        )
        for frame in animation['animation']
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('animations', type=pathlib.Path, nargs='+')
    parser.add_argument('--tolerance', type=float, default=0.01,
                        help='waypoint hit distance, in units of the mesh width')
    parser.add_argument('--frames', action='store_true',
                        help='print every frame, not just the mean')
    args = parser.parse_args()
    keys = ('max_deviation_mean', 'max_deviation_worst', 'miss_mean',
            'miss_worst', 'hit_fraction')
    print(f'{"max dev":>8} {"worst":>7} {"miss":>7} {"worst":>7} {"hit":>5}  animation')
    for path in args.animations:
        frames = animation_deviation(path, args.tolerance)
        rows = [(f'  frame {index}', frame) for index, frame in enumerate(frames)] \
            if args.frames else []
        rows.append((str(path), {
            key: float(np.mean([frame[key] for frame in frames])) for key in keys
        }))
        for name, values in rows:
            print(' '.join(
                f'{values[key]:>{8 if index == 0 else 7 if index < 4 else 5}.{2 if key == "hit_fraction" else 4}f}'
                for index, key in enumerate(keys)
            ) + f'  {name}')


if __name__ == '__main__':
    main()
