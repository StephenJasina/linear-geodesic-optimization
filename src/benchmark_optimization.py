"""Compare end-to-end legacy and Torch optimization on the same network.

Run from src with ``python benchmark_optimization.py``. The built-in GraphML
fixture has fixed curvature targets, so network preprocessing cannot change
between backends. Both runs use the same mesh, initialization, and iteration
cap. Timings include optimizer setup, minimization, and output serialization.
"""

import argparse
import contextlib
import io
import json
import platform
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

from optimization import optimize


DEFAULT_GRAPHML = Path(__file__).parent / "test" / "fixtures" / "optimization_parity.graphml"


def run_once(graphml, directory, backend, sides, maxiter, lambda_smooth):
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        optimize(
            filename_graphml=graphml,
            initial_radius=80.0,
            sides=sides,
            mesh_scale=0.25,
            coordinates_scale=0.8,
            lambda_curvature=1.0,
            lambda_smooth=lambda_smooth,
            directory_output=directory,
            maxiter=maxiter,
            backend=backend,
        )
    elapsed = time.perf_counter() - start
    return elapsed, json.loads((directory / "output.json").read_text())


def output_difference(legacy, fast):
    """Return the maximum height difference after checking identical inputs."""
    for key in ("network", "routes", "traffic", "initial"):
        if legacy[key] != fast[key]:
            raise ValueError(f"The {key} outputs differ between backends")
    legacy_parameters = {k: v for k, v in legacy["parameters"].items() if k != "backend"}
    fast_parameters = {k: v for k, v in fast["parameters"].items() if k != "backend"}
    if legacy_parameters != fast_parameters:
        raise ValueError("The optimizer inputs differ between backends")
    return float(np.max(np.abs(np.asarray(legacy["final"]) - np.asarray(fast["final"]))))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphml", type=Path, default=DEFAULT_GRAPHML,
                        help="GraphML input with node lat/long and edge ricciCurvature attributes")
    parser.add_argument("--sides", type=int, default=20, help="vertices along each mesh side")
    parser.add_argument("--maxiter", type=int, default=5, help="L-BFGS-B iteration cap")
    parser.add_argument("--repeats", type=int, default=3, help="measured runs per backend")
    parser.add_argument("--lambda-smooth", type=float, default=0.005)
    parser.add_argument("--atol", type=float, default=1e-6,
                        help="maximum allowed absolute difference in final heights")
    args = parser.parse_args(argv)
    if not args.graphml.is_file():
        parser.error(f"GraphML input does not exist: {args.graphml}")
    if args.sides < 3 or args.maxiter < 1 or args.repeats < 1 or args.atol < 0:
        parser.error("sides must be at least 3, maxiter/repeats positive, and atol nonnegative")

    samples = {"legacy": [], "torch": []}
    max_difference = 0.0
    with tempfile.TemporaryDirectory(prefix="manifold-optimizer-benchmark-") as temporary:
        root = Path(temporary)
        # Discard one run of each backend to exclude first-use library setup.
        for backend in ("legacy", "torch"):
            run_once(args.graphml, root / f"warmup-{backend}", backend,
                     args.sides, args.maxiter, args.lambda_smooth)
        for repeat in range(args.repeats):
            outputs = {}
            order = ("legacy", "torch") if repeat % 2 == 0 else ("torch", "legacy")
            for backend in order:
                elapsed, output = run_once(
                    args.graphml, root / f"{backend}-{repeat}", backend,
                    args.sides, args.maxiter, args.lambda_smooth,
                )
                samples[backend].append(elapsed)
                outputs[backend] = output
            max_difference = max(max_difference, output_difference(outputs["legacy"], outputs["torch"]))

    legacy_seconds = statistics.median(samples["legacy"])
    torch_seconds = statistics.median(samples["torch"])
    print(f"Python {platform.python_version()} | {args.sides}x{args.sides} mesh | "
          f"maxiter={args.maxiter} | {args.repeats} measured runs per backend")
    print(f"Legacy median: {legacy_seconds:.4f} s")
    print(f"Torch median:  {torch_seconds:.4f} s")
    print(f"Speedup:       {legacy_seconds / torch_seconds:.2f}x")
    print(f"Maximum final height difference: {max_difference:.3g} "
          f"(allowed {args.atol:.3g})")
    return 0 if max_difference <= args.atol else 1


if __name__ == "__main__":
    raise SystemExit(main())
