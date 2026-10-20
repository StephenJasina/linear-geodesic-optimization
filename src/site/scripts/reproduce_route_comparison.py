"""Refit one manifold per traffic route, all from the same original surface.

Run from src::

    python site/scripts/reproduce_route_comparison.py --output-dir site/examples/route-change

Pass --changed-route repeatedly to choose one to three alternatives.
"""

import argparse
import json
from pathlib import Path

from optimization import optimize
from build_route_comparison import export_gallery


DEFAULT_ROUTES = [
    ['C', 'D', 'E', 'H', 'I', 'J'],
    ['C', 'F', 'E', 'H', 'G', 'J'],
    ['C', 'D', 'E', 'H', 'G', 'J'],
]
VARIANT_IDS = ['after', 'third', 'fourth']


def reroute_traffic(source: dict, original_route: list[str],
                    changed_route: list[str]) -> dict:
    """Replace one traffic route and its reverse without changing graph topology."""
    if (len(original_route) < 2 or len(changed_route) < 2
            or original_route[0] != changed_route[0]
            or original_route[-1] != changed_route[-1]):
        raise ValueError('The old and new routes need the same distinct endpoints')
    if original_route == changed_route:
        raise ValueError('The changed route must differ from the original route')
    edges = {(link['source_id'], link['target_id']) for link in source['links']}
    if any((a, b) not in edges for a, b in zip(changed_route[:-1], changed_route[1:])):
        raise ValueError('The changed route uses a missing directed network link')
    document = json.loads(json.dumps(source))
    replacements = [
        (original_route, changed_route),
        (original_route[::-1], changed_route[::-1]),
    ]
    for old, new in replacements:
        matching = [flow for flow in document['traffic'] if flow['route'] == old]
        if len(matching) != 1:
            raise ValueError(f'Expected one traffic flow with route {old}')
        matching[0]['route'] = new
    return document


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-graph', type=Path, default=repo / 'data/toy/routing_with_volumes/traffic/two_clusters/graph_4.json')
    parser.add_argument('--original-route', nargs='+', default=['C', 'F', 'G', 'J'])
    parser.add_argument('--changed-route', nargs='+', action='append',
                        help='Repeat for up to three alternative routes')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--comparison-json', type=Path,
                        default=repo / 'src/site/public/route-comparison.json')
    parser.add_argument('--route-weight', type=float, default=100_000)
    parser.add_argument('--baseline-iterations', type=int, default=1500)
    parser.add_argument('--refit-iterations', type=int, default=200)
    args = parser.parse_args()
    routes = args.changed_route or DEFAULT_ROUTES
    if not 1 <= len(routes) <= 3:
        parser.error('Specify one to three changed routes')
    if len({tuple(route) for route in routes + [args.original_route]}) != len(routes) + 1:
        parser.error('All routes must differ')

    source = json.loads(args.source_graph.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    before_dir = args.output_dir / 'before'
    common = dict(
        latency_threshold=4, clustering_distance=500000,
        ricci_curvature_alpha=0, ricci_curvature_reweight='distribution',
        lambda_curvature=1.0, lambda_smooth=0.005,
        initial_radius=80.0, sides=50, mesh_scale=0.25,
        coordinates_scale=0.8, method='lbfgs', backend='torch',
    )
    optimize(filename_json=str(args.source_graph), directory_output=before_dir,
             maxiter=args.baseline_iterations, **common)
    before = json.loads((before_dir / 'output.json').read_text())
    variants = [{
        'id': 'before', 'label': 'Original upper route',
        'output': before, 'route_labels': args.original_route,
    }]
    manifest = [{
        'id': 'before', 'label': 'Original upper route',
        'output': 'before/output.json', 'route': args.original_route,
    }]
    for index, route in enumerate(routes):
        variant_id = VARIANT_IDS[index]
        changed = reroute_traffic(source, args.original_route, route)
        graph_path = (args.output_dir / 'changed_graph.json' if index == 0
                      else args.output_dir / f'changed_graph_{variant_id}.json')
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        graph_path.write_text(json.dumps(changed, indent=2))
        fit_dir = args.output_dir / variant_id
        optimize(filename_json=str(graph_path), directory_output=fit_dir,
                 maxiter=args.refit_iterations, lambda_route=args.route_weight,
                 route_metric='continuous',
                 route_constraints=[{'route': route, 'weight': 1.0}],
                 route_corridor_width=0.055, route_margin=0.0,
                 initialization_file_path=str(before_dir / 'output.json'), **common)
        default_labels = ['Lower route via I', 'Upper route via E and H',
                          'Lower route via G']
        label = default_labels[index] if routes == DEFAULT_ROUTES else f'Alternative {index + 1}'
        variants.append({
            'id': variant_id, 'label': label,
            'output': json.loads((fit_dir / 'output.json').read_text()),
            'route_labels': route,
        })
        manifest.append({
            'id': variant_id, 'label': label,
            'output': f'{variant_id}/output.json', 'route': route,
        })
    (args.output_dir / 'variants.json').write_text(json.dumps({'variants': manifest}, indent=2))
    comparison = export_gallery(variants, 0.055)
    args.comparison_json.parent.mkdir(parents=True, exist_ok=True)
    args.comparison_json.write_text(json.dumps(comparison, separators=(',', ':')))
    print(args.comparison_json)


if __name__ == '__main__':
    main()
