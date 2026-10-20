"""Export multiple route-conditioned manifold outputs for the browser viewer.

From src, use a manifest with two to four variants::

    python site/scripts/build_route_comparison.py --manifest route-variants.json \\
        --output site/public/route-comparison.json

The legacy two-output arguments remain supported.
"""

import argparse
import json
from pathlib import Path
import re

import numpy as np
from pygeodesic import geodesic

from linear_geodesic_optimization.mesh.rectangle import Mesh


def corridor_fraction(path: np.ndarray, route: np.ndarray, width: float) -> float:
    """Approximate the share of surface length in a projected route corridor."""
    segments = path[1:] - path[:-1]
    lengths = np.linalg.norm(segments, axis=1)
    subdivisions = 16
    fractions = (np.arange(subdivisions) + 0.5) / subdivisions
    samples = (
        path[:-1, None, :] + fractions[None, :, None] * segments[:, None, :]
    ).reshape(-1, 3)[:, :2]
    weights = np.repeat(lengths / subdivisions, subdivisions)
    distances = np.full(len(samples), np.inf)
    for start, end in zip(route[:-1], route[1:]):
        direction = end - start
        denominator = float(direction @ direction)
        if denominator == 0:
            continue
        projection = np.clip(((samples - start) @ direction) / denominator, 0, 1)
        closest = start + projection[:, None] * direction
        distances = np.minimum(distances, np.linalg.norm(samples - closest, axis=1))
    if not np.isfinite(distances).all():
        raise ValueError('The route must contain distinct consecutive waypoints')
    return float(weights[distances <= width].sum() / weights.sum())


def mean_polyline_distance(source: np.ndarray, target: np.ndarray) -> float:
    """Length-weighted mean projected distance from one polyline to another."""
    segments = source[1:] - source[:-1]
    lengths = np.linalg.norm(segments, axis=1)
    samples_per_segment = 32
    fractions = (np.arange(samples_per_segment) + 0.5) / samples_per_segment
    samples = (source[:-1, None, :] + fractions[None, :, None]
               * segments[:, None, :]).reshape(-1, 2)
    weights = np.repeat(lengths / samples_per_segment, samples_per_segment)
    distances = np.full(len(samples), np.inf)
    for start, end in zip(target[:-1], target[1:]):
        direction = end - start
        squared = float(direction @ direction)
        if squared == 0:
            continue
        fraction = np.clip(((samples - start) @ direction) / squared, 0, 1)
        distances = np.minimum(
            distances,
            np.linalg.norm(samples - (start + fraction[:, None] * direction), axis=1),
        )
    return float(np.dot(distances, weights) / weights.sum())


def export_gallery(variants: list[dict], width: float) -> dict:
    """Compute exact geodesics for manifolds sharing one original surface."""
    if not 0 < width < 1:
        raise ValueError('corridor width must be between 0 and 1')
    if not 2 <= len(variants) <= 4:
        raise ValueError('The comparison viewer supports two to four variants')
    ids = [item['id'] for item in variants]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r'[a-z0-9_-]+', value) for value in ids):
        raise ValueError('Variant ids must be unique lowercase identifiers')
    baseline = variants[0]['output']
    params = baseline['parameters']
    if params['width'] != params['height']:
        raise ValueError('The comparison viewer requires a square mesh')
    graph = baseline['network'][0]
    first_route = variants[0]['route_labels']
    if len(first_route) < 2 or first_route[0] == first_route[-1]:
        raise ValueError('Each route needs distinct endpoints')
    for variant in variants:
        output = variant['output']
        labels = variant['route_labels']
        if (len(labels) < 2 or labels[0] != first_route[0]
                or labels[-1] != first_route[-1]):
            raise ValueError('All routes must share their endpoints')
        for key in ('width', 'height', 'mesh_scale', 'coordinates_scale'):
            if output['parameters'][key] != params[key]:
                raise ValueError(f'Variants differ in {key}')
        for key in ('labels', 'coordinates', 'edges', 'bounding_box'):
            if output['network'][0][key] != graph[key]:
                raise ValueError(f'Variants differ in network {key}')
        if output.get('routes') is not None and labels not in output['routes']:
            raise ValueError(f'The route for {variant["id"]} is absent from its traffic input')
    routes = [tuple(item['route_labels']) for item in variants]
    if len(set(routes)) != len(routes):
        raise ValueError('Each variant needs a different traffic route')
    original_heights = np.asarray(baseline['final'], dtype=np.float64)
    for variant in variants[1:]:
        if not np.array_equal(np.asarray(variant['output']['initial']), original_heights):
            raise ValueError(f'{variant["id"]} did not start from the original manifold')

    mesh = Mesh(params['width'], params['height'], params['mesh_scale'])
    scale = mesh.get_scale()
    xy = np.asarray(mesh.get_coordinates())[:, :2]
    network = mesh.map_coordinates_to_support(
        np.asarray(graph['coordinates']), params['coordinates_scale'], graph['bounding_box'],
    )
    def route_points(labels):
        try:
            indices = [graph['labels'].index(label) for label in labels]
        except ValueError as error:
            raise ValueError(f'Unknown route waypoint: {error}') from error
        return np.asarray(network[indices, :2] / scale)
    endpoints = [
        int(np.argmin(np.sum((xy - network[graph['labels'].index(label), :2]) ** 2, axis=1)))
        for label in (first_route[0], first_route[-1])
    ]
    faces = np.asarray([
        [vertex.index for vertex in face.vertices()]
        for face in mesh.get_topology().faces()
    ], dtype=np.int32)

    surfaces = []
    max_change = 0.0
    for variant in variants:
        output = variant['output']
        heights = np.asarray(output['final'], dtype=np.float64)
        if heights.size != xy.shape[0]:
            raise ValueError(f'{variant["id"]} has the wrong number of mesh heights')
        points = np.column_stack((xy, heights))
        distance, path = geodesic.PyGeodesicAlgorithmExact(points, faces).geodesicDistance(
            *endpoints
        )
        path = path[::-1].copy() / scale
        route = route_points(variant['route_labels'])
        change = float(np.max(np.abs(heights - original_heights)) / scale)
        max_change = max(max_change, change)
        surfaces.append({
            'id': variant['id'],
            'label': variant['label'],
            'routeLabels': variant['route_labels'],
            'route': np.round(route, 8).tolist(),
            'heights': np.round(heights / scale, 8).tolist(),
            'geodesic': np.round(path, 8).tolist(),
            'geodesicLength': round(float(distance), 8),
            'corridorFraction': corridor_fraction(path, route, width),
            'routeDeviation': mean_polyline_distance(route, path[:, :2]),
            'optimization': output.get('optimization'),
            'maxHeightChange': change,
        })
    original_path = np.asarray(surfaces[0]['geodesic'])
    for surface in surfaces:
        route = np.asarray(surface['route'])
        surface['baselineCorridorFraction'] = corridor_fraction(original_path, route, width)
        surface['baselineRouteDeviation'] = mean_polyline_distance(route, original_path[:, :2])

    incomplete = [
        item for item in surfaces if item['optimization'] is not None
        and not item['optimization']['converged']
    ]
    status = (
        'All runs stopped at their iteration limits. '
        if len(incomplete) == len(surfaces) and all(
            'LIMIT' in item['optimization']['message'].upper() for item in incomplete
        ) else 'Some runs did not converge. ' if incomplete else ''
    )
    return {
        'title': f'{first_route[0]} → {first_route[-1]} route alternatives',
        'side': params['width'],
        'corridorWidth': width,
        'network': {
            'labels': graph['labels'],
            'points': np.round(network[:, :2] / scale, 8).tolist(),
            'edges': graph['edges'],
        },
        'surfaces': surfaces,
        'heightChange': {'maximumAbsolute': max_change},
        'note': (
            'Each refit starts from the original optimized manifold. Its traffic route '
            'and exact-geodesic route preference change; all other traffic and graph '
            'topology remain fixed. ' + status +
            'Compare each route against its own original-surface baseline and check '
            'curvature and smoothness losses alongside the fit.'
        ),
    }


def export_comparison(before: dict, after: dict, before_labels: list[str],
                      after_labels: list[str], width: float) -> dict:
    """Compatibility wrapper for a two-output comparison."""
    return export_gallery([
        {'id': 'before', 'label': 'Original routing', 'output': before,
         'route_labels': before_labels},
        {'id': 'after', 'label': 'Changed routing', 'output': after,
         'route_labels': after_labels},
    ], width)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path, nargs='?')
    parser.add_argument('after', type=Path, nargs='?')
    parser.add_argument('--before-route', nargs='+')
    parser.add_argument('--after-route', nargs='+')
    parser.add_argument('--manifest', type=Path,
                        help='JSON with variants: [{id, label, output, route}]')
    parser.add_argument('--corridor-width', type=float, default=0.055)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        variants = []
        for item in manifest['variants']:
            output_path = args.manifest.parent / item['output']
            variants.append({
                'id': item['id'], 'label': item['label'],
                'route_labels': item['route'],
                'output': json.loads(output_path.read_text()),
            })
        result = export_gallery(variants, args.corridor_width)
    else:
        if not all((args.before, args.after, args.before_route, args.after_route)):
            parser.error('Provide --manifest or both outputs and route arguments')
        result = export_comparison(
            json.loads(args.before.read_text()), json.loads(args.after.read_text()),
            args.before_route, args.after_route, args.corridor_width,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(',', ':')))
    print(args.output)


if __name__ == '__main__':
    main()
