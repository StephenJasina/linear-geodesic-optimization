"""
Compare each run's losses with the best L_curvature any surface can reach.

The curvature loss is a weighted sum over (mesh vertex, target) pairs,

    L_curvature = sum_i w_i (K[v_i] - kappa_i)^2,

where the pairs come from the tubes around the network edges (with
their tapered falloff, if any). A vertex in several tubes gets several
targets but has only one Gaussian curvature K, so no surface can do
better than putting every vertex at the weighted mean of its targets.
With S0 = sum w, S1 = sum w kappa, S2 = sum w kappa^2 over a vertex's
pairs, the vertex's unavoidable cost is

* S2 - S1^2 / S0 at an interior vertex (K* = S1 / S0),
* S2 at a boundary vertex, where K is fixed at 0.

Their sum, L_LB, is a lower bound on L_curvature, and lambda_curvature *
L_LB is a lower bound on the total loss (L_smooth >= 0). On the toy
ridges runs it is also reachable: L-BFGS on L_curvature alone converges
to within 1e-7 of it, because K can be set nearly independently at each
vertex. So L_curvature - L_LB is the part of the misfit that a better
surface could remove.

The bound is computed from the loss computer that produced the run
(`pipeline.computer_from_output`, the last stage's tube), so it follows
`curvature_tube_radius` and `curvature_tube_falloff`. Route losses
(lambda_geodesic, lambda_stretch) are left out of `total`.

Run from `src/`:

    python -m analysis.loss_bounds <batch directory> [...] [--output bounds.json]
"""

import argparse
import copy
import json
import pathlib
import warnings

import numpy as np

from analysis.edge_curvature_fit import find_runs
from linear_geodesic_optimization import pipeline
from linear_geodesic_optimization.mesh.rectangle import Mesh


def vertex_floor(computer):
    """
    Return `(k_star, floor)`, per-vertex arrays for a torch loss
    computer: the Gaussian curvature that minimizes the vertex's part of
    L_curvature (NaN where the vertex has no target; 0 on the boundary)
    and that part's minimum.
    """
    n = computer._n_vertices
    indices = computer._fat_indices.numpy()
    targets = computer._targets.numpy()
    weights = computer._weights.numpy()
    interior = computer._interior.numpy()

    s0 = np.bincount(indices, weights, n)
    s1 = np.bincount(indices, weights * targets, n)
    s2 = np.bincount(indices, weights * targets**2, n)
    has_target = s0 > 0
    safe_s0 = np.where(has_target, s0, 1.)

    k_star = np.where(has_target, np.where(interior, s1 / safe_s0, 0.), np.nan)
    floor = np.where(interior, s2 - np.where(has_target, s1**2 / safe_s0, 0.), s2)
    # Rounding can leave tiny negative values where the targets agree
    return k_star, np.maximum(floor, 0.)


def analyze_run(path, arrays=False):
    """
    Return the losses of the run in directory `path` (its output.json's
    final surface) with the lower bound L_LB. With `arrays`, also return
    the per-vertex `k_star`, `floor` and `kappa_G` and the mesh `sides`.
    """
    path = pathlib.Path(path)
    with open(path / 'output.json') as file:
        output = json.load(file)
    parameters = output['parameters']
    if parameters.get('network_trim_radius') is not None:
        raise ValueError(f'{path}: trimmed meshes are not supported')

    # The bound reads the torch computer's (vertex, target, weight)
    # arrays; the legacy backend computes the same loss
    if parameters.get('backend', 'torch') != 'torch':
        output = copy.copy(output)
        output['parameters'] = {**parameters, 'backend': 'torch'}
    mesh = Mesh(parameters['width'], parameters['height'], parameters['mesh_scale'])
    computer = pipeline.computer_from_output(output, mesh)
    total = computer.forward(np.array(output['final']))
    l_curvature = computer.curvature_loss.loss
    l_smooth = computer.smooth_loss.loss

    k_star, floor = vertex_floor(computer)
    interior = computer._interior.numpy()
    counts = np.bincount(computer._fat_indices.numpy(), minlength=computer._n_vertices)
    lambda_curvature = parameters['lambda_curvature']
    lambda_smooth = parameters['lambda_smooth']
    l_lb = float(floor.sum())

    result = {
        'run': path.name,
        'lambda_curvature': lambda_curvature,
        'lambda_smooth': lambda_smooth,
        'total': total,
        'L_curvature': l_curvature,
        'L_smooth': l_smooth,
        'lambda_smooth_L_smooth': lambda_smooth * l_smooth,
        'L_LB': l_lb,
        'L_LB_boundary': float(floor[~interior].sum()),
        'n_shared': int(np.sum((counts > 1) & interior)),
        'excess': l_curvature - l_lb,
        'L_curvature_over_LB': l_curvature / l_lb if l_lb > 0 else None,
        'total_over_LB': total / (lambda_curvature * l_lb) if l_lb > 0 else None,
    }

    recorded = output.get('construction', {}).get('polish', {}).get('losses')
    if recorded is not None:
        result['recorded_loss'] = recorded['loss']
        if abs(recorded['loss'] - total) > 1e-8:
            warnings.warn(
                f'{path}: recomputed loss {total} differs from the recorded {recorded["loss"]}'
            )
    omitted = [
        name for name in ('lambda_geodesic', 'lambda_stretch')
        if parameters.get(name)
    ]
    if omitted:
        result['omitted_losses'] = omitted

    if arrays:
        result.update(
            k_star=k_star, floor=floor, kappa_G=computer.kappa_G.copy(),
            sides=parameters['width'],
        )
    return result


def print_table(results):
    print(f'{"run":<14} {"total":>8} {"L_curv":>8} {"λ·L_smooth":>10} '
          f'{"L_LB":>8} {"total/LB":>9} {"L_curv/LB":>9}')
    for r in results:
        print(f'{r["run"]:<14} {r["total"]:8.4f} {r["L_curvature"]:8.4f} '
              f'{r["lambda_smooth_L_smooth"]:10.4f} {r["L_LB"]:8.4f} '
              f'{r["total_over_LB"] or float("nan"):9.3f} '
              f'{r["L_curvature_over_LB"] or float("nan"):9.3f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[1].strip())
    parser.add_argument('directories', type=pathlib.Path, nargs='+')
    parser.add_argument('--output', type=pathlib.Path)
    args = parser.parse_args()

    batches = {}
    for directory in args.directories:
        runs = find_runs(directory)
        if not runs:
            parser.error(f'no runs (*/output.json) in {directory}')
        results = [analyze_run(run) for run in runs]
        print(directory)
        print_table(results)
        print()
        batches[str(directory)] = results
    if args.output is not None:
        with open(args.output, 'w') as file:
            json.dump(batches, file, indent=1)


if __name__ == '__main__':
    main()
