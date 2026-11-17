"""
Postprocess finished surfaces by greedily nudging their heights so that
their geodesics follow the network routes (see
linear_geodesic_optimization/geodesic_nudge.py).

The input is a batch directory of optimization runs (each a directory
holding an output.json). For each value of `--lambda-fit`, every run is
nudged independently and written, with the same directory names, to

    <output root>/fit_<lambda_fit>/<batch name>/<run name>/output.json

where the output root defaults to the batch's parent directory with
`_nudge` appended. Each output.json keeps the source's `initial`, has
the nudged heights as `final`, and records the settings and the
per-iteration history in a `geodesic_nudge` block.

With `--collate`, each nudged batch, and the source batch (under
`<output root>/source/<batch name>`), is collated into
`animation.json` (postprocessed, as in the source configs) and
`animation_true.json` (`postprocess: false`, the true surface), both
with the default route-following geodesics that
analysis/route_geodesic_deviation.py scores.

Run from `src/`:

    python geodesic_nudge.py ../outputs/.../<batch> --lambda-fit 0.1 0.01 --collate
"""

import argparse
import functools
import json
import multiprocessing
import os
import pathlib
import re
import sys
import time


def run_key(path: pathlib.Path):
    match = re.match(r'(\d+)', path.name)
    return (int(match.group(1)) if match else float('inf'), path.name)


def run_directories(batch: pathlib.Path):
    return sorted(
        (path for path in batch.iterdir() if (path / 'output.json').is_file()),
        key=run_key,
    )


def nudge_run(job, settings):
    import numpy as np

    from linear_geodesic_optimization import geodesic_nudge as gn
    from linear_geodesic_optimization import pipeline
    from linear_geodesic_optimization.mesh.rectangle import Mesh

    source, destination, lambda_fit = job
    with open(source / 'output.json') as file:
        output = json.load(file)
    parameters = output['parameters']
    if parameters.get('network_trim_radius') is not None:
        raise ValueError(f'{source}: trimmed meshes are not supported')
    mesh = Mesh(parameters['width'], parameters['height'], parameters['mesh_scale'])
    computer = pipeline.computer_from_output(output, mesh)
    network_vertices, _, _ = pipeline.network_from_output(output, mesh)
    routes, weights = gn.routes_from_output(output, network_vertices)

    mesh_scale = parameters['mesh_scale']
    nudge = gn.GeodesicNudge(
        mesh, routes, weights, computer,
        lambda_fit=lambda_fit,
        step_smoothing=settings['step_smoothing'] * mesh_scale,
        waypoint_weight=settings['waypoint_weight'],
        trust=settings['trust'],
        inner_maxiter=settings['inner_maxiter'],
    )
    # A destination without output.json is an incomplete earlier run
    pipeline.reset_output_directory(destination)
    time_start = time.time()
    with open(destination / 'nudge.log', 'w') as log_file:
        def log(line):
            print(line, file=log_file, flush=True)
        log(f'source: {source}\nlambda_fit: {lambda_fit}\nsettings: {settings}')
        z, history = nudge.run(np.array(output['final']), settings['iterations'], log)

    extra = {
        key: value for key, value in output.items()
        if key not in ('parameters', 'initial', 'final', 'network', 'routes', 'traffic')
    }
    extra['geodesic_nudge'] = {
        'source': str(source),
        'lambda_fit': lambda_fit,
        **settings,
        'time': time.time() - time_start,
        'history': history,
    }
    pipeline.write_output(
        destination,
        parameters=parameters,
        initial=np.array(output['initial']),
        final=z,
        network=output['network'],
        routes=output['routes'],
        traffic=output['traffic'],
        **extra,
    )
    first, last = history[0], history[-1]
    accepted = [entry for entry in history if entry.get('accepted')]
    best = accepted[-1]
    return (
        f'{destination}: max dev {first["max_deviation_mean"]:.4f} -> {best["max_deviation_mean"]:.4f},'
        f' miss {first["miss_mean"]:.4f} -> {best["miss_mean"]:.4f},'
        f' L_curv {first["L_curvature"]:.4f} -> {best["L_curvature"]:.4f}'
        f' ({len(accepted) - 1} of {last["iteration"]} steps accepted, {time.time() - time_start:.0f} s)'
    )


COLLATION_VARIANTS = {
    'animation.json': {
        'postprocess': True,
        'postprocess_subtract_initialization': False,
        'vertical_centering': 'min',
    },
    'animation_true.json': {
        'postprocess': False,
        'vertical_centering': 'min',
    },
    # Endpoint-only geodesics, which can switch paths between frames
    'animation_endpoints.json': {
        'postprocess': False,
        'vertical_centering': 'min',
        'geodesic_endpoints': True,
    },
}


def collate(batch: pathlib.Path):
    import collation
    directories = run_directories(batch)
    collation.collate_outputs(
        directories,
        [(arguments, batch / name) for name, arguments in COLLATION_VARIANTS.items()],
    )
    return str(batch)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('batch', type=pathlib.Path,
                        help='directory of runs, each with an output.json')
    parser.add_argument('--output-root', type=pathlib.Path, default=None)
    parser.add_argument('--lambda-fit', type=float, nargs='+', default=[0.1],
                        help='weight of the (normalized) optimizer loss against the (normalized) route offset')
    parser.add_argument('--iterations', type=int, default=15)
    parser.add_argument('--step-smoothing', type=float, default=0.035,
                        help='width of the Gaussian blur of each step, as a fraction of mesh_scale')
    parser.add_argument('--waypoint-weight', type=float, default=1.)
    parser.add_argument('--trust', type=float, default=1.,
                        help='initial trust-region weight mu')
    parser.add_argument('--inner-maxiter', type=int, default=100)
    parser.add_argument('--processes', type=int, default=1,
                        help='parallel runs; each holds a full torch loss for its mesh '
                             '(measure one run\'s memory before raising this)')
    parser.add_argument('--collate', action='store_true')
    parser.add_argument('--collate-only', action='store_true',
                        help='skip the nudge and collate existing outputs')
    args = parser.parse_args()

    batch = args.batch.resolve()
    runs = run_directories(batch)
    if not runs:
        sys.exit(f'No runs with an output.json in {batch}')
    root = args.output_root or batch.parent.with_name(batch.parent.name + '_nudge')
    settings = {
        'iterations': args.iterations,
        'step_smoothing': args.step_smoothing,
        'waypoint_weight': args.waypoint_weight,
        'trust': args.trust,
        'inner_maxiter': args.inner_maxiter,
    }
    batches = {
        lambda_fit: root / f'fit_{lambda_fit:g}' / batch.name
        for lambda_fit in args.lambda_fit
    }

    with multiprocessing.Pool(args.processes) as pool:
        if not args.collate_only:
            jobs = [
                (run, destination / run.name, lambda_fit)
                for lambda_fit, destination in batches.items()
                for run in runs
            ]
            # Finished runs are skipped, so an interrupted batch can resume
            done = [job for job in jobs if (job[1] / 'output.json').is_file()]
            if done:
                print(f'Skipping {len(done)} finished runs', flush=True)
            jobs = [job for job in jobs if job not in done]
            for line in pool.imap_unordered(functools.partial(nudge_run, settings=settings), jobs):
                print(line, flush=True)

        if args.collate or args.collate_only:
            # Link the source runs so the source batch is collated with
            # the same variants, next to the nudged ones
            source_batch = root / 'source' / batch.name
            os.makedirs(source_batch, exist_ok=True)
            for run in runs:
                link = source_batch / run.name
                if not link.exists():
                    os.makedirs(link)
                    os.symlink(run / 'output.json', link / 'output.json')
            for line in pool.imap_unordered(collate, [source_batch, *batches.values()]):
                print(f'collated {line}', flush=True)


if __name__ == '__main__':
    # One thread per process; the runs are parallelized across processes
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(variable, '1')
    main()
