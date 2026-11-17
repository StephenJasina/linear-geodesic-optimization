"""
Compare the surface's Gaussian curvature with the Ollivier-Ricci
curvature (ORC) of each network edge, in more detail than the single
per-edge mean in output.json.

For each run and each edge with a target, the vertices in the edge's
fat tube (the tube the curvature loss uses, of radius
`pipeline.fat_edge_width`, interior vertices only) give:
* `mean`, `median`, `std` of kappa_G, and `sign_agree`, the fraction of
  tube vertices whose kappa_G has the target's sign,
* `bias2 = (mean - target)^2` and `var = std^2`. Their sum is the mean
  squared error over the tube, so `weight * (bias2 + var)` is the edge's
  contribution to L_curvature (weight = the edge's share of total
  network length),
* `profile`, the mean kappa_G in `--bins` bins of the position along the
  edge (0 at the first endpoint, 1 at the second),
* `mean_wide`, the mean over tubes `--wide` times as wide, to show how
  much the fit depends on the tube,
* `mean_band`, the mean over bands of fixed half-widths `--bands`
  (fractions of mesh_scale), which don't shrink as the mesh is refined,
* `cross`, kappa_G / target over the middle 60% of the edge, binned by
  distance from the edge line (`--cross` bin edges, fractions of
  mesh_scale).

The summary's `band width` is the distance out to which the median of
`cross` over all edges stays at least 0.5 (the matching band's
half-width, in mesh_scale units).

Run from `src/`:

    python -m analysis.edge_curvature_fit <batch directory> [...] --output fit.json
"""

import argparse
import json
import pathlib
import re

import numpy as np

from linear_geodesic_optimization import pipeline
from linear_geodesic_optimization.mesh.rectangle import Mesh
from linear_geodesic_optimization.optimization.torch_optimization import Computer


def tube(points, a, b, radius):
    """Return the indices within `radius` of segment ab and their positions along it."""
    direction = b - a
    fraction = np.clip(((points - a) @ direction) / (direction @ direction), 0., 1.)
    distance = np.linalg.norm(points - (a + fraction[:, None] * direction), axis=1)
    members = np.flatnonzero(distance < radius)
    return members, fraction[members]


def analyze_run(path, bins, wide, bands=(), cross=()):
    with open(path / 'output.json') as file:
        output = json.load(file)
    parameters = output['parameters']
    sides, scale = parameters['width'], parameters['mesh_scale']
    mesh = Mesh(sides, sides, scale)
    graph_data, _, edge_data = output['network']
    vertices = mesh.map_coordinates_to_support(
        np.array(graph_data['coordinates']), parameters['coordinates_scale'],
        graph_data['bounding_box'],
    )[:, :2]
    edges, targets = graph_data['edges'], edge_data['ricciCurvature']
    epsilon = pipeline.fat_edge_width(sides, scale)
    computer = Computer(
        mesh, vertices, edges, targets, epsilon,
        parameters['lambda_curvature'], parameters['lambda_smooth'],
    )
    computer.forward(np.array(output['final']))
    kappa = computer.kappa_G
    points = mesh.get_coordinates()[:, :2]
    interior = np.array([not v.is_on_boundary() for v in mesh.get_topology().vertices()])

    lengths = np.array([np.linalg.norm(vertices[i] - vertices[j]) for i, j in edges])
    weights = lengths / lengths.sum()
    labels = graph_data['labels']
    throughputs = edge_data.get('throughput') or [None] * len(edges)
    results = []
    for (i, j), target, weight, throughput in zip(edges, targets, weights, throughputs):
        if target is None:
            continue
        members, fraction = tube(points, vertices[i], vertices[j], epsilon)
        keep = interior[members]
        members, fraction = members[keep], fraction[keep]
        values = kappa[members]
        profile = [
            float(values[(fraction >= lo) & (fraction < hi)].mean())
            if np.any((fraction >= lo) & (fraction < hi)) else None
            for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1],
                              np.linspace(0, 1, bins + 1)[1:] + np.r_[np.zeros(bins - 1), 1e-9])
        ]
        wide_members, _ = tube(points, vertices[i], vertices[j], wide * epsilon)
        wide_members = wide_members[interior[wide_members]]
        mean = float(values.mean())
        direction = vertices[j] - vertices[i]
        along = np.clip(((points - vertices[i]) @ direction) / (direction @ direction), 0., 1.)
        distance = np.linalg.norm(points - (vertices[i] + along[:, None] * direction), axis=1)
        middle = (along > 0.2) & (along < 0.8) & interior
        cross_section = []
        for lo, hi in zip(cross[:-1], cross[1:]):
            ring = middle & (distance >= lo * scale) & (distance < hi * scale)
            cross_section.append(float(kappa[ring].mean() / target) if np.any(ring) and target else None)
        in_band = [(distance < band * scale) & interior for band in bands]
        mean_band = [float(kappa[m].mean()) for m in in_band]
        # Spread within each band: RMS error relative to the target's
        # size, and the fraction of band vertices with the target's sign
        rmse_band = [float(np.sqrt(np.mean((kappa[m] - target)**2))) for m in in_band]
        sign_band = [float(np.mean(np.sign(kappa[m]) == np.sign(target))) for m in in_band]
        results.append({
            'edge': f'{labels[i]}-{labels[j]}',
            'target': float(target),
            'throughput': None if throughput is None else float(throughput),
            'weight': float(weight),
            'n': int(len(members)),
            'mean': mean,
            'median': float(np.median(values)),
            'std': float(values.std()),
            'sign_agree': float(np.mean(np.sign(values) == np.sign(target))),
            'bias2': (mean - target)**2,
            'var': float(values.var()),
            'loss': float(weight * np.mean((values - target)**2)),
            'profile': profile,
            'mean_wide': float(kappa[wide_members].mean()),
            'mean_band': mean_band,
            'rmse_band': rmse_band,
            'sign_band': sign_band,
            'cross': cross_section,
        })
    return {
        'bands': list(bands),
        'cross_edges': list(cross),
        'run': path.name,
        'L_curvature': float(computer.curvature_loss.loss),
        'edges': results,
    }


def find_runs(directory):
    runs = [path.parent for path in directory.glob('*/output.json')]
    return sorted(runs, key=lambda path: int(re.search(r'graph_(\d+)', path.name).group(1))
                  if re.search(r'graph_(\d+)', path.name) else 0)


def band_width(run):
    """The distance out to which the median kappa_G / target stays >= 0.5."""
    cross = np.array([
        [np.nan if value is None else value for value in edge['cross']]
        for edge in run['edges']
    ])
    if cross.size == 0:
        return None
    median = np.nanmedian(cross, axis=0)
    width = 0.
    for value, hi in zip(median, run['cross_edges'][1:]):
        if not value >= 0.5:
            break
        width = hi
    return float(width)


def summarize(runs):
    rows = []
    for run in runs:
        edges = run['edges']
        target = np.array([e['target'] for e in edges])
        mean = np.array([e['mean'] for e in edges])
        slope, intercept = np.polyfit(target, mean, 1)
        loss = sum(e['loss'] for e in edges)
        rows.append({
            'run': run['run'],
            'L_curvature': run['L_curvature'],
            'corr': float(np.corrcoef(target, mean)[0, 1]),
            'slope': float(slope),
            'intercept': float(intercept),
            'mae': float(np.mean(np.abs(mean - target))),
            'sign_mean': float(np.mean(np.sign(mean) == np.sign(target))),
            'sign_vertex': float(np.mean([e['sign_agree'] for e in edges])),
            'bias_share': float(sum(e['weight'] * e['bias2'] for e in edges) / loss),
            'corr_wide': float(np.corrcoef(target, [e['mean_wide'] for e in edges])[0, 1]),
            'corr_band': [
                float(np.corrcoef(target, [e['mean_band'][k] for e in edges])[0, 1])
                for k in range(len(run['bands']))
            ],
            'slope_band': [
                float(np.polyfit(target, [e['mean_band'][k] for e in edges], 1)[0])
                for k in range(len(run['bands']))
            ],
            'band_width': band_width(run),
            'rmse_band': [
                float(np.average([e['rmse_band'][k] for e in edges], weights=[e['weight'] for e in edges]))
                for k in range(len(run['bands']))
            ],
            'sign_band': [
                float(np.mean([e['sign_band'][k] for e in edges]))
                for k in range(len(run['bands']))
            ],
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('directories', type=pathlib.Path, nargs='+')
    parser.add_argument('--bins', type=int, default=10)
    parser.add_argument('--wide', type=float, default=4.)
    parser.add_argument('--bands', type=float, nargs='*', default=[0.025, 0.0375],
                        help='band half-widths, as fractions of mesh_scale')
    parser.add_argument('--cross', type=float, nargs='*',
                        default=[0, 0.00625, 0.0125, 0.01875, 0.025, 0.03125, 0.0375, 0.05, 0.075, 0.1],
                        help='cross-section bin edges, as fractions of mesh_scale')
    parser.add_argument('--output', type=pathlib.Path)
    args = parser.parse_args()
    report = {}
    for directory in args.directories:
        runs = [analyze_run(path, args.bins, args.wide, args.bands, args.cross) for path in find_runs(directory)]
        report[str(directory)] = {'runs': runs, 'summary': summarize(runs)}
        print(directory)
        print(f'  {"run":<12} {"L_curv":>7} {"corr":>6} {"slope":>6} {"icpt":>6} {"MAE":>6} '
              f'{"sign":>5} {"sgn/v":>5} {"bias%":>5} {"corr4x":>6}'
              + ''.join(f' {"corr@" + str(b):>11} {"slope":>6} {"rmse":>6} {"sign":>5}' for b in args.bands) + f' {"band":>7}')
        for row in report[str(directory)]['summary']:
            print(f'  {row["run"]:<12} {row["L_curvature"]:>7.3f} {row["corr"]:>6.3f} '
                  f'{row["slope"]:>6.3f} {row["intercept"]:>6.3f} {row["mae"]:>6.3f} '
                  f'{row["sign_mean"]:>5.2f} {row["sign_vertex"]:>5.2f} '
                  f'{100 * row["bias_share"]:>5.1f} {row["corr_wide"]:>6.3f}'
                  + ''.join(f' {c:>11.3f} {sl:>6.3f} {r:>6.3f} {sg:>5.2f}' for c, sl, r, sg in zip(row['corr_band'], row['slope_band'], row['rmse_band'], row['sign_band']))
                  + f' {row["band_width"]:>7.4f}')
    if args.output is not None:
        with open(args.output, 'w') as file:
            json.dump(report, file, indent=1)


if __name__ == '__main__':
    main()
