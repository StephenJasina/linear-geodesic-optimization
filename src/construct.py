"""
Build surfaces directly from network Ricci curvatures.

This is an alternative to optimization.py that uses the same config
files and writes the same output.json format (so collation.py works
unchanged). Instead of running L-BFGS over every mesh height, each
network edge's Ricci curvature is turned into a target Hessian, and the
heights are found with one sparse linear solve followed by a scalar
amplitude calibration against the usual loss. See
`linear_geodesic_optimization/construction/hessian_design.py`.
"""

import argparse
import json
import os
import pathlib
import shutil
import time
import warnings

import numpy as np
import scipy

import linear_geodesic_optimization.driver as driver
from linear_geodesic_optimization.construction import hessian_design
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization.torch_optimization \
    import Computer as TorchComputer
from optimization import load_network, map_network_to_mesh


# Error on things like division by 0
warnings.simplefilter('error')

def construct(
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
    sides,
    mesh_scale=1.,
    coordinates_scale=0.8,
    network_trim_radius=None,
    directory_output,
    hessian_anisotropy=1.,
    hessian_orientation='hill',
    hessian_regularization=1e-3,
    hessian_tube_radius=None,
    hessian_iterations=5,
    hessian_polish_maxiter=0,
    **kwargs
):
    if network_trim_radius is not None and not np.isposinf(network_trim_radius):
        raise ValueError('construct.py does not support network_trim_radius')

    time_start = time.perf_counter()

    width = height = sides
    mesh = RectangleMesh(width, height, mesh_scale)

    network, routes, traffic = load_network(
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
    network_vertices, network_edges, network_curvatures = map_network_to_mesh(
        mesh, network, coordinates_scale
    )

    if os.path.isdir(directory_output):
        shutil.rmtree(directory_output)
    os.makedirs(directory_output)

    # Same fat edge width as the loss in optimization.py
    loss_epsilon = 1.01 * 2**0.5 * mesh_scale / width
    tube_radius = loss_epsilon if hessian_tube_radius is None else hessian_tube_radius

    time_network = time.perf_counter()

    # Linear Hessian fit
    z_unit = hessian_design.solve_heights(
        mesh, network_vertices, network_edges, network_curvatures,
        tube_radius, hessian_anisotropy, hessian_orientation,
        hessian_regularization, hessian_iterations,
    )

    # Amplitude calibration against the exact loss
    computer = TorchComputer(
        mesh, network_vertices, network_edges, network_curvatures,
        loss_epsilon, lambda_curvature, lambda_smooth,
        directory=directory_output
    )
    scale = hessian_design.calibrate_scale(computer.forward, z_unit)
    z_constructed = scale * z_unit

    time_constructed = time.perf_counter()

    def losses(z):
        loss = computer.forward(z)
        return {
            'loss': loss,
            'L_curvature': computer.curvature_loss.loss,
            'L_smooth': computer.smooth_loss.loss,
        }

    losses_constructed = losses(z_constructed)

    # Optional polish with the usual optimizer
    z = z_constructed
    polish = None
    if hessian_polish_maxiter:
        computer.diagnostics(z)
        z = scipy.optimize.minimize(
            computer.forward, z, method='L-BFGS-B', jac=computer.reverse,
            callback=computer.diagnostics,
            options={'maxiter': hessian_polish_maxiter},
        ).x
        polish = {
            'maxiter': int(hessian_polish_maxiter),
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

    parameters = {
        'epsilon': float(latency_threshold) if latency_threshold is not None else None,
        'clustering_distance': float(clustering_distance) if clustering_distance is not None else None,
        'ricci_curvature_alpha': float(ricci_curvature_alpha) if ricci_curvature_alpha is not None else None,
        'ricci_curvature_reweight': ricci_curvature_reweight,
        'ricci_curvature_distribution_reweight_scale': ricci_curvature_distribution_reweight_scale,
        'lambda_curvature': float(lambda_curvature),
        'lambda_smooth': float(lambda_smooth),
        # No sphere initialization; collation.py then uses 'initial'
        'initial_radius': None,
        'width': int(width),
        'height': int(height),
        'mesh_scale': float(mesh_scale),
        'coordinates_scale': float(coordinates_scale),
        'backend': 'construction',
        'network_trim_radius': None,
        'hessian_anisotropy': float(hessian_anisotropy),
        'hessian_orientation': hessian_orientation,
        'hessian_regularization': float(hessian_regularization),
        'hessian_tube_radius': float(tube_radius),
        'hessian_iterations': int(hessian_iterations),
    }

    construction = {
        'scale': scale,
        'losses': losses_constructed,
        'time_network': time_network - time_start,
        'time_construction': time_constructed - time_network,
        'polish': polish,
        'edges': residuals,
    }
    print(
        f'{directory_output}: scale {scale:.4g}, '
        f'L_curvature {losses_constructed["L_curvature"]:.6f}, '
        f'construction time {construction["time_construction"]:.2f}s'
    )

    with open(directory_output / 'output.json', 'w') as file_output:
        json.dump({
            'parameters': parameters,
            'initial': np.zeros(width * height).tolist(),
            'final': z.tolist(),
            'network': network,
            'routes': routes,
            'traffic': traffic,
            'construction': construction,
        }, file_output, ensure_ascii=False, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('--dry-run', '-n', action='store_true',
                        help='validate the config and output directories, then exit without constructing')
    args = parser.parse_args()
    config_file = pathlib.PurePath(args.config_file)

    arguments, settings, defaults = driver.load_config(config_file)

    output_format = driver.get_output_format(settings, defaults, validate=True)
    driver.assign_output_directories(arguments, settings, output_format, existence_check='must_not_exist')
    driver.check_no_duplicate_output_directories(arguments)

    if not arguments or args.dry_run:
        return

    # Every construction is independent, so seeds from the
    # initialization strategy are ignored. The strategy still determines
    # the run order (and which runs collation.py skips).
    initialization = driver.get_initialization(settings)
    n_cores = settings['n_cores'] if 'n_cores' in settings else None
    driver.dispatch_optimization_batches(initialization, arguments, construct, n_cores)

if __name__ == '__main__':
    main()
