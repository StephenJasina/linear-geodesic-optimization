"""
Build surfaces whose curvature along a network's edges matches the
edges' Ricci curvatures.

Two methods are available, selected by the `method` parameter:
* 'construction' (the default): each network edge's Ricci curvature is
  turned into a target Hessian, and the heights are found with a few
  sparse linear solves (see
  `linear_geodesic_optimization/construction/hessian_design.py`). The
  usual loss is evaluated for reporting, and for an optional L-BFGS
  polish (`hessian_polish_maxiter`).
* 'lbfgs': the usual loss is minimized with L-BFGS over every mesh
  height, starting from a sphere cap or from a previous run's output.

Both methods write the same output.json format, read by collation.py.
"""

import argparse
import json
import pathlib
import time
import warnings

import numpy as np

import linear_geodesic_optimization.driver as driver
from linear_geodesic_optimization import pipeline
from linear_geodesic_optimization.construction import hessian_design
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.mesh.rectangle import sphere_cap_heights


# Error on things like division by 0
warnings.simplefilter('error')

def optimize(
    *,  # All parameters are keyword only
    filename_probes=None,
    filename_links=None,
    filename_graphml=None,
    filename_json=None,
    latency_threshold=None,
    clustering_distance=None,
    ricci_curvature_alpha=0.,
    ricci_curvature_reweight=None,
    ricci_curvature_distribution_reweight_scale=0.,
    lambda_curvature=1.,
    lambda_smooth=0.,
    smooth_loss_type='mvs',
    initial_radius=None,
    sides,
    mesh_scale=1.,
    coordinates_scale=0.8,
    network_trim_radius=None,
    directory_output,
    method='construction',
    backend='torch',
    maxiter=None,
    initialization_file_path=None,
    hessian_anisotropy=1.,
    hessian_orientation='hill',
    hessian_regularization=1e-3,
    hessian_tube_radius=None,
    hessian_iterations=5,
    hessian_slope_correction=False,
    hessian_smoothness=0.,
    hessian_polish_maxiter=0,
    hessian_initialization=None,
    **kwargs
):
    if method not in ('construction', 'lbfgs'):
        raise ValueError(f'Unknown method: {method}')
    if method == 'construction' and network_trim_radius is not None \
            and not np.isposinf(network_trim_radius):
        raise ValueError('The construction method does not support network_trim_radius')
    computer_type = pipeline.get_computer_type(backend)

    time_start = time.perf_counter()

    # Construct the mesh
    mesh = RectangleMesh(sides, sides, mesh_scale)

    network, routes, traffic = pipeline.load_network(
        filename_probes=filename_probes,
        filename_links=filename_links,
        filename_graphml=filename_graphml,
        filename_json=filename_json,
        latency_threshold=latency_threshold,
        clustering_distance=clustering_distance,
        ricci_curvature_alpha=ricci_curvature_alpha,
        ricci_curvature_reweight=ricci_curvature_reweight,
        ricci_curvature_distribution_reweight_scale=ricci_curvature_distribution_reweight_scale,
    )
    network_vertices, network_edges, network_curvatures = pipeline.map_network_to_mesh(
        mesh, network, coordinates_scale
    )

    # Setup snapshots
    pipeline.reset_output_directory(directory_output)

    parameters = pipeline.base_parameters(
        latency_threshold=latency_threshold,
        clustering_distance=clustering_distance,
        ricci_curvature_alpha=ricci_curvature_alpha,
        ricci_curvature_reweight=ricci_curvature_reweight,
        ricci_curvature_distribution_reweight_scale=ricci_curvature_distribution_reweight_scale,
        lambda_curvature=lambda_curvature,
        lambda_smooth=lambda_smooth,
        smooth_loss_type=smooth_loss_type,
        initial_radius=initial_radius,
        sides=sides,
        mesh_scale=mesh_scale,
        coordinates_scale=coordinates_scale,
        method=method,
        backend=backend,
        network_trim_radius=network_trim_radius,
    )

    def make_computer():
        return computer_type(
            mesh, network_vertices, network_edges, network_curvatures,
            pipeline.fat_edge_width(sides, mesh_scale),
            lambda_curvature, lambda_smooth,
            directory=directory_output,
            smooth_loss_type=smooth_loss_type,
        )

    if method == 'lbfgs':
        z_0, z, extra = _optimize_lbfgs(
            mesh, network_vertices, network_edges, make_computer,
            sides, initial_radius, network_trim_radius, maxiter,
            initialization_file_path,
        )
    else:
        z_0, z, extra = _construct(
            mesh, network_vertices, network_edges, network_curvatures,
            make_computer, parameters, directory_output, time_start,
            sides, mesh_scale, initial_radius,
            hessian_anisotropy, hessian_orientation, hessian_regularization,
            hessian_tube_radius, hessian_iterations, hessian_slope_correction,
            hessian_smoothness, hessian_polish_maxiter, hessian_initialization,
        )

    pipeline.write_output(
        directory_output,
        parameters=parameters,
        initial=z_0,
        final=z,
        network=network,
        routes=routes,
        traffic=traffic,
        **extra,
    )

def _optimize_lbfgs(
    mesh, network_vertices, network_edges, make_computer,
    sides, initial_radius, network_trim_radius, maxiter,
    initialization_file_path,
):
    """
    Minimize the loss with L-BFGS, starting from a sphere cap of radius
    `initial_radius` or, if given, the final heights in
    `initialization_file_path`.

    Return `(z_0, z, extra)`, where `extra` holds the additional keys
    for output.json.
    """
    # Initialize mesh
    if initialization_file_path is None:
        z_0 = sphere_cap_heights(sides, sides, initial_radius)
    else:
        with open(initialization_file_path, 'r') as f:
            z_0 = np.array(json.load(f)['final'])
    z_0 = mesh.set_parameters(z_0)

    if network_trim_radius is not None and not np.isposinf(network_trim_radius):
        mesh.trim_to_graph(network_vertices, network_edges, network_trim_radius)
        z_0 = mesh.get_parameters()

    computer = make_computer()
    z = pipeline.minimize_lbfgs(
        computer, z_0, None if maxiter is None else {'maxiter': maxiter}
    ).x
    z = mesh.set_parameters(z)

    return z_0, z, {}

def _construct(
    mesh, network_vertices, network_edges, network_curvatures,
    make_computer, parameters, directory_output, time_start,
    sides, mesh_scale, initial_radius,
    hessian_anisotropy, hessian_orientation, hessian_regularization,
    hessian_tube_radius, hessian_iterations, hessian_slope_correction,
    hessian_smoothness, hessian_polish_maxiter, hessian_initialization,
):
    """
    Construct the heights from target Hessians, then optionally polish
    them with L-BFGS. Seeds from the initialization strategy are
    ignored; the initial heights come from `hessian_initialization`.

    Add the construction's parameters to `parameters`, and return
    `(z_initial, z, extra)`, where `extra` holds the additional keys
    for output.json.
    """
    loss_epsilon = pipeline.fat_edge_width(sides, mesh_scale)
    tube_radius = loss_epsilon if hessian_tube_radius is None else hessian_tube_radius

    time_network = time.perf_counter()

    z_constructed = hessian_design.solve_heights(
        mesh, network_vertices, network_edges, network_curvatures,
        tube_radius, hessian_anisotropy, hessian_orientation,
        hessian_regularization, hessian_iterations,
        hessian_slope_correction, hessian_smoothness,
        hessian_initialization, initial_radius,
    )
    z_initial = hessian_design.initial_heights(
        mesh, hessian_initialization, initial_radius
    )

    time_constructed = time.perf_counter()

    # The usual loss, used for reporting and the optional polish
    computer = make_computer()

    def losses(z):
        loss = computer.forward(z)
        return {
            'loss': float(loss),
            'L_curvature': float(computer.curvature_loss.loss),
            'L_smooth': float(computer.smooth_loss.loss),
        }

    losses_constructed = losses(z_constructed)

    # Optional polish with the usual optimizer (None polishes until
    # convergence, lifting scipy's default caps of 15000 iterations and
    # 15000 function evaluations)
    z = z_constructed
    polish = None
    if hessian_polish_maxiter != 0:
        if hessian_polish_maxiter is None:
            options = {'maxiter': 10**7, 'maxfun': 10**7}
        else:
            options = {'maxiter': hessian_polish_maxiter}
        result = pipeline.minimize_lbfgs(computer, z, options, z)
        z = result.x
        polish = {
            'maxiter': None if hessian_polish_maxiter is None else int(hessian_polish_maxiter),
            'iterations': int(result.nit),
            'evaluations': int(result.nfev),
            'message': str(result.message),
            'losses': losses(z),
            'distance_from_constructed': float(np.linalg.norm(z - z_constructed)),
            'time': time.perf_counter() - time_constructed,
        }

    computer.forward(z)
    residuals = hessian_design.edge_residuals(
        computer.kappa_G, mesh.get_coordinates()[:, :2],
        hessian_design.hessian_operators(sides, mesh_scale)[3],
        network_vertices, network_edges, network_curvatures, loss_epsilon
    )
    z = mesh.set_parameters(z)

    parameters.update({
        # Always None, so that collation.py uses 'initial' (the
        # construction's initial heights) as the baseline; the actual
        # radius is recorded as 'hessian_initial_radius'
        'initial_radius': None,
        'hessian_anisotropy': float(hessian_anisotropy),
        'hessian_orientation': hessian_orientation,
        'hessian_regularization': float(hessian_regularization),
        'hessian_tube_radius': float(tube_radius),
        'hessian_iterations': int(hessian_iterations),
        'hessian_slope_correction': bool(hessian_slope_correction),
        'hessian_smoothness': float(hessian_smoothness),
        'hessian_initialization': hessian_initialization,
        'hessian_initial_radius': float(initial_radius) if initial_radius is not None else None,
    })

    construction = {
        'losses': losses_constructed,
        'time_network': time_network - time_start,
        'time_construction': time_constructed - time_network,
        'polish': polish,
        'edges': residuals,
    }
    print(
        f'{directory_output}: '
        f'L_curvature {losses_constructed["L_curvature"]:.6f}, '
        f'construction time {construction["time_construction"]:.2f}s'
    )

    return z_initial, z, {'construction': construction}

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('--dry-run', '-n', action='store_true',
                        help='validate the config and output directories, then exit without optimizing')
    args = parser.parse_args()
    config_file = pathlib.PurePath(args.config_file)

    # Read JSON config file and generate optimization parameters from it
    arguments, arguments_collation, settings, defaults = driver.load_config(config_file)

    # Determine where the outputs are going. Check that the output
    # format has the right types, and that the output directories don't
    # already exist or collide with one another.
    output_format = driver.get_output_format(settings, defaults, validate=True)
    driver.assign_output_directories(arguments, settings, output_format, existence_check='must_not_exist')
    driver.check_no_duplicate_output_directories(arguments)

    # Check the collation settings now rather than after optimizing
    driver.assign_collation_paths(
        arguments, arguments_collation, settings, defaults,
        driver.get_initialization(settings)
    )

    if not arguments or args.dry_run:
        # Exit before producing output
        return

    # Determine the initialization strategy, and then use that to run
    # the optimization, using multiprocessing where possible. Runs using
    # the construction method ignore their seeds, so for them the
    # strategy only determines the run order (and which runs
    # collation.py skips).
    initialization = driver.get_initialization(settings)
    n_cores = settings['n_cores'] if 'n_cores' in settings else None
    driver.assign_initialization_file_paths(arguments, initialization)
    driver.dispatch_optimization_batches(initialization, arguments, optimize, n_cores)

if __name__ == '__main__':
    main()
