"""
Build surfaces whose curvature along a network's edges matches the
edges' Ricci curvatures.

Three methods are available, selected by the `method` parameter:
* 'construction' (the default): each network edge's Ricci curvature is
  turned into a target Hessian, and the heights are found with a few
  sparse linear solves (see
  `linear_geodesic_optimization/construction/hessian_design.py`). The
  usual loss is evaluated for reporting, and for an optional L-BFGS
  polish (`hessian_polish_maxiter`).
* 'ridges': each network edge becomes a ridge raised above the plane
  z = 0, with node heights set by throughput and each crest bent to
  match the edge's Ricci curvature (see
  `linear_geodesic_optimization/construction/ridges.py`). As with
  'construction', the loss is evaluated for reporting and for an
  optional L-BFGS polish (`ridge_polish_maxiter`).
* 'lbfgs': the usual loss is minimized with L-BFGS over every mesh
  height, starting from a sphere cap or from a previous run's output.

All methods write the same output.json format, read by collation.py.
"""

import argparse
import json
import pathlib
import time
import warnings

import numpy as np

import linear_geodesic_optimization.driver as driver
from linear_geodesic_optimization import geodesic_curvature, pipeline, route_stretch
from linear_geodesic_optimization.construction import hessian_design, ridges
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
    lambda_geodesic=0.,
    geodesic_curve='route',
    lambda_stretch=0.,
    stretch_refresh=100,
    stretch_smoothing=0.035,
    curvature_tube_radius=None,
    curvature_tube_falloff=0.,
    curvature_tube_falloff_weight=False,
    curvature_tube_falloff_ends=True,
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
    hessian_boundary='fixed',
    ridge_width=0.3,
    ridge_height_scale=0.05,
    ridge_smoothing=0.035,
    ridge_height_base=0.25,
    ridge_node_height='max',
    ridge_height_transform='linear',
    ridge_blend_p=8.,
    ridge_polish_maxiter=0,
    ridge_height_normalization='batch',
    ridge_height_reference=None,
    ridge_crest_height='nodes',
    **kwargs
):
    if method not in ('construction', 'ridges', 'lbfgs'):
        raise ValueError(f'Unknown method: {method}')
    if method in ('construction', 'ridges') and network_trim_radius is not None \
            and not np.isposinf(network_trim_radius):
        raise ValueError(f'The {method} method does not support network_trim_radius')
    computer_type = pipeline.get_computer_type(backend)
    if lambda_stretch and method == 'lbfgs':
        raise ValueError('lambda_stretch is supported only by the polish of the construction and ridges methods')

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
        lambda_geodesic=lambda_geodesic,
        geodesic_curve=geodesic_curve,
        lambda_stretch=lambda_stretch,
        stretch_refresh=stretch_refresh,
        stretch_smoothing=stretch_smoothing,
        curvature_tube_radius=curvature_tube_radius,
        curvature_tube_falloff=curvature_tube_falloff,
        curvature_tube_falloff_weight=curvature_tube_falloff_weight,
        curvature_tube_falloff_ends=curvature_tube_falloff_ends,
        initial_radius=initial_radius,
        sides=sides,
        mesh_scale=mesh_scale,
        coordinates_scale=coordinates_scale,
        method=method,
        backend=backend,
        network_trim_radius=network_trim_radius,
    )

    # The geodesic loss: the bending energy of the routes (as natural
    # cubic splines through their waypoints) or of the links they use,
    # weighted by traffic, so that the routes are geodesics
    geodesic_bending = None
    if lambda_geodesic:
        if routes is None or traffic is None:
            raise ValueError('lambda_geodesic needs a JSON network with routes and traffic')
        if geodesic_curve not in ('route', 'link'):
            raise ValueError(f'Unknown geodesic_curve "{geodesic_curve}"')
        curves, weights = geodesic_curvature.route_curves(
            network, routes, traffic, geodesic_curve
        )
        geodesic_bending = geodesic_curvature.CurveBending(
            mesh, [network_vertices[list(curve)] for curve in curves],
            weights, 0.25 * mesh_scale / (sides - 1),
            'spline' if geodesic_curve == 'route' else 'polyline',
        )

    # The route stretch loss: how much longer each route is than the
    # geodesic between its endpoints (see route_stretch.py)
    stretch = None
    if lambda_stretch:
        if routes is None or traffic is None:
            raise ValueError('lambda_stretch needs a JSON network with routes and traffic')
        curves, weights = geodesic_curvature.route_curves(network, routes, traffic, 'route')
        stretch = route_stretch.RouteStretch(
            mesh, [network_vertices[list(curve)] for curve in curves],
            weights, 0.25 * mesh_scale / (sides - 1),
            stretch_smoothing * mesh_scale,
        )
        stretch.labels = [[network[0]['labels'][index] for index in curve] for curve in curves]
        stretch.refresh_every = stretch_refresh

    def make_computer(stage=None):
        computer = make_base_computer(stage)
        if stretch is None:
            return computer
        return route_stretch.WithRouteStretch(computer, stretch, lambda_stretch)

    # The radius of the tube around each edge over which the curvature
    # loss compares kappa_G with the edge's ORC: by default just over one
    # mesh diagonal, or `curvature_tube_radius` times mesh_scale, which
    # doesn't shrink as the mesh is refined. A list of radii polishes in
    # stages (ridges/construction polish only), `*_polish_maxiter`
    # iterations each, each stage starting from the previous one's
    # result; the last radius is the one reported.
    tube_fractions = curvature_tube_radius if isinstance(curvature_tube_radius, (list, tuple)) \
        else [curvature_tube_radius]
    if not tube_fractions:
        raise ValueError('curvature_tube_radius must not be an empty list')
    if len(tube_fractions) > 1 and (method == 'lbfgs' or lambda_stretch):
        raise ValueError('Staged curvature_tube_radius needs the construction or ridges polish, without lambda_stretch')
    tube_stages = [
        pipeline.curvature_tube_radius(sides, mesh_scale, fraction)
        for fraction in tube_fractions
    ]
    loss_tube_radius = tube_stages[-1]
    # The soft edge (see `curvature_loss.tapered_tube`), one value per
    # stage, or one value for every stage
    falloff_stages = curvature_tube_falloff if isinstance(curvature_tube_falloff, (list, tuple)) \
        else [curvature_tube_falloff] * len(tube_stages)
    if len(falloff_stages) != len(tube_stages):
        raise ValueError('curvature_tube_falloff needs one value per curvature_tube_radius stage')

    def make_base_computer(stage=None):
        stage = len(tube_stages) - 1 if stage is None else stage
        tube_radius = tube_stages[stage]
        computer = computer_type(
            mesh, network_vertices, network_edges, network_curvatures,
            tube_radius,
            lambda_curvature, lambda_smooth,
            directory=directory_output,
            smooth_loss_type=smooth_loss_type,
            lambda_geodesic=lambda_geodesic,
            geodesic_bending=geodesic_bending,
            curvature_falloff=falloff_stages[stage] * mesh_scale,
            curvature_falloff_weight=curvature_tube_falloff_weight,
            curvature_falloff_ends=curvature_tube_falloff_ends,
        )
        computer.loss_tube_radius = tube_radius
        return computer
    make_computer.tube_stages = tube_stages

    if method == 'lbfgs':
        z_0, z, extra = _optimize_lbfgs(
            mesh, network_vertices, network_edges, make_computer,
            sides, initial_radius, network_trim_radius, maxiter,
            initialization_file_path,
        )
    elif method == 'ridges':
        z_0, z, extra = _build_ridges(
            mesh, network, network_vertices, network_edges, network_curvatures,
            make_computer, parameters, directory_output, time_start,
            sides, mesh_scale,
            ridge_width, ridge_height_scale, ridge_height_base,
            ridge_node_height, ridge_height_transform, ridge_blend_p,
            ridge_smoothing, ridge_polish_maxiter,
            ridge_height_normalization, ridge_height_reference,
            ridge_crest_height,
        )
    else:
        z_0, z, extra = _construct(
            mesh, network_vertices, network_edges, network_curvatures,
            make_computer, parameters, directory_output, time_start,
            sides, mesh_scale, initial_radius,
            hessian_anisotropy, hessian_orientation, hessian_regularization,
            hessian_tube_radius, hessian_iterations, hessian_slope_correction,
            hessian_smoothness, hessian_polish_maxiter, hessian_initialization,
            hessian_boundary,
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
    hessian_boundary,
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
        boundary=hessian_boundary,
    )
    z_initial = hessian_design.initial_heights(
        mesh, hessian_initialization, initial_radius
    )

    time_constructed = time.perf_counter()

    z, losses_constructed, polish, residuals = _report_and_polish(
        mesh, z_constructed, make_computer, hessian_polish_maxiter,
        sides, mesh_scale, network_vertices, network_edges,
        network_curvatures, time_constructed,
    )

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
        'hessian_boundary': hessian_boundary,
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

def _report_and_polish(
    mesh, z_constructed, make_computer, polish_maxiter,
    sides, mesh_scale, network_vertices, network_edges, network_curvatures,
    time_constructed,
):
    """
    Evaluate the usual loss on constructed heights, optionally polish
    them with L-BFGS, and compare each edge's curvature to its target.

    Return `(z, losses_constructed, polish, residuals)`, where `polish`
    is None when `polish_maxiter` is 0. A `polish_maxiter` of None
    polishes until convergence.
    """
    computer = make_computer()

    stretch = getattr(computer, 'stretch', None)

    def losses(z):
        loss = computer.forward(z)
        result = {
            'loss': float(loss),
            'L_curvature': float(computer.curvature_loss.loss),
            'L_smooth': float(computer.smooth_loss.loss),
            'L_geodesic': float(computer.geodesic_loss.loss),
        }
        if stretch is not None:
            result['L_stretch'] = float(computer.stretch_loss.loss)
        return result

    if stretch is not None:
        stretch.refresh(z_constructed)
    losses_constructed = losses(z_constructed)

    # Optional polish with the usual optimizer (None polishes until
    # convergence, lifting scipy's default caps of 15000 iterations and
    # 15000 function evaluations)
    z = z_constructed
    polish = None
    if polish_maxiter != 0:
        if polish_maxiter is None:
            options = {'maxiter': 10**7, 'maxfun': 10**7}
        else:
            options = {'maxiter': polish_maxiter}
        stages = []
        if stretch is None:
            # Earlier stages of a staged tube radius, each from the
            # previous stage's result
            iterations = evaluations = 0
            for stage, tube_radius in enumerate(getattr(make_computer, 'tube_stages', [None])[:-1]):
                result = pipeline.minimize_lbfgs(make_computer(stage), z, options, z)
                z = result.x
                iterations += result.nit
                evaluations += result.nfev
                stages.append({
                    'tube_radius': float(tube_radius),
                    'iterations': int(result.nit),
                    'message': str(result.message),
                })
            result = pipeline.minimize_lbfgs(computer, z, options, z)
            iterations += result.nit
            evaluations += result.nfev
        else:
            # Alternate between refreshing the competing geodesics and
            # running `stretch.refresh_every` L-BFGS iterations
            iterations = evaluations = 0
            while iterations < options['maxiter']:
                stretch.refresh(z)
                round_options = dict(options, maxiter=min(
                    stretch.refresh_every, options['maxiter'] - iterations
                ))
                result = pipeline.minimize_lbfgs(computer, z, round_options, z)
                z = result.x
                iterations += result.nit
                evaluations += result.nfev
                if result.nit == 0:
                    break
            stretch.refresh(z)
        z = result.x
        polish = {
            'maxiter': None if polish_maxiter is None else int(polish_maxiter),
            'iterations': int(iterations),
            'evaluations': int(evaluations),
            'message': str(result.message),
            'losses': losses(z),
            'distance_from_constructed': float(np.linalg.norm(z - z_constructed)),
            'time': time.perf_counter() - time_constructed,
        }
        if stages:
            polish['stages'] = stages
        if stretch is not None:
            polish['route_stretch'] = [
                {'route': labels, 'weight': float(weight), 'stretch': float(value)}
                for labels, weight, value in zip(
                    stretch.labels, stretch.weights, stretch.stretches(z)
                )
            ]

    computer.forward(z)
    residuals = hessian_design.edge_residuals(
        computer.kappa_G, mesh.get_coordinates()[:, :2],
        hessian_design.hessian_operators(sides, mesh_scale)[3],
        network_vertices, network_edges, network_curvatures,
        computer.loss_tube_radius,
    )
    z = mesh.set_parameters(z)

    return z, losses_constructed, polish, residuals

def _build_ridges(
    mesh, network, network_vertices, network_edges, network_curvatures,
    make_computer, parameters, directory_output, time_start,
    sides, mesh_scale,
    ridge_width, ridge_height_scale, ridge_height_base,
    ridge_node_height, ridge_height_transform, ridge_blend_p,
    ridge_smoothing, ridge_polish_maxiter,
    ridge_height_normalization, ridge_height_reference,
    ridge_crest_height='nodes',
):
    """
    Build the heights from raised ridges along the network's edges,
    then optionally polish them with L-BFGS. Seeds from the
    initialization strategy are ignored; the initial heights are the
    flat ambient plane z = 0.

    `ridge_width` is the ridges' half-width and `ridge_smoothing` the
    width of the final Gaussian smoothing, both as fractions of
    `mesh_scale`, so the surface doesn't depend on `sides`.
    `ridge_height_scale` is the maximum node height as a fraction of
    the half-width (roughly the steepest cross slope, so it controls
    how far the surface is from the small-slope regime).

    `ridge_height_reference` is the node throughput that maps to the
    full height. With `ridge_height_normalization` 'batch',
    `main` sets it to the largest node throughput across the run's
    batch (see `assign_ridge_height_references`), so heights are
    comparable across an animation. If it is None, the run's own
    largest node throughput is used.

    Add the ridge parameters to `parameters`, and return
    `(z_initial, z, extra)`, where `extra` holds the additional keys
    for output.json.
    """
    throughputs = network[2].get('throughput')
    if throughputs is None:
        raise ValueError(
            'The ridges method needs edge throughputs (a JSON network '
            'with traffic, or a links CSV with a throughput column)'
        )

    time_network = time.perf_counter()

    width = ridge_width * mesh_scale
    height = ridge_height_scale * width
    z_constructed, heights, edges = ridges.build_heights(
        mesh.get_coordinates()[:, :2], (sides, sides), network_vertices,
        network_edges, network_curvatures, throughputs, width, height,
        base=ridge_height_base, node_rule=ridge_node_height,
        transform=ridge_height_transform, blend_p=ridge_blend_p,
        smoothing=ridge_smoothing * mesh_scale,
        reference=ridge_height_reference,
        crest=ridge_crest_height,
    )
    z_initial = np.zeros_like(z_constructed)

    time_constructed = time.perf_counter()

    z, losses_constructed, polish, residuals = _report_and_polish(
        mesh, z_constructed, make_computer, ridge_polish_maxiter,
        sides, mesh_scale, network_vertices, network_edges,
        network_curvatures, time_constructed,
    )
    if polish is not None:
        polish['fraction_below_ambient'] = float(np.mean(z < 0.))
        polish['min_height'] = float(np.min(z))

    for residual, edge in zip(residuals, edges):
        residual.update(edge)

    parameters.update({
        # Always None, so that collation.py uses 'initial' (the flat
        # plane) as the baseline
        'initial_radius': None,
        'ridge_width': float(ridge_width),
        'ridge_height_scale': float(ridge_height_scale),
        'ridge_height_base': float(ridge_height_base),
        'ridge_node_height': ridge_node_height,
        'ridge_height_transform': ridge_height_transform,
        'ridge_blend_p': float(ridge_blend_p),
        'ridge_smoothing': float(ridge_smoothing),
        'ridge_polish_maxiter': None if ridge_polish_maxiter is None else int(ridge_polish_maxiter),
        'ridge_height_normalization': ridge_height_normalization,
        'ridge_height_reference': None if ridge_height_reference is None else float(ridge_height_reference),
        'ridge_crest_height': ridge_crest_height,
    })

    construction = {
        'losses': losses_constructed,
        'time_network': time_network - time_start,
        'time_construction': time_constructed - time_network,
        'polish': polish,
        'node_heights': heights.tolist(),
        'edges': residuals,
    }
    print(
        f'{directory_output}: '
        f'L_curvature {losses_constructed["L_curvature"]:.6f}, '
        f'construction time {construction["time_construction"]:.2f}s'
    )

    return z_initial, z, {'construction': construction}

def assign_ridge_height_references(arguments) -> None:
    """
    For each batch, set `ridge_height_reference` on its runs that use
    the ridges method with `ridge_height_normalization` 'batch', to the
    largest node throughput (under each run's `ridge_node_height` rule)
    across those runs. Each run's network is loaded here exactly as
    `optimize` loads it, so the reference matches what the runs see.
    """
    if any(
        argument_dict['method'] == 'ridges'
        and argument_dict['ridge_height_normalization'] not in ('batch', 'run')
        for argument_batch in arguments
        for argument_dict in argument_batch
    ):
        raise ValueError("ridge_height_normalization must be 'batch' or 'run'")
    for argument_batch in arguments:
        runs = [
            argument_dict
            for argument_dict in argument_batch
            if argument_dict['method'] == 'ridges'
            and argument_dict['ridge_height_normalization'] == 'batch'
        ]
        if not runs:
            continue
        reference = 0.
        for argument_dict in runs:
            network, _, _ = pipeline.load_network(**{
                key: argument_dict[key]
                for key in (
                    'filename_probes', 'filename_links', 'filename_graphml',
                    'filename_json', 'latency_threshold', 'clustering_distance',
                    'ricci_curvature_alpha', 'ricci_curvature_reweight',
                    'ricci_curvature_distribution_reweight_scale',
                )
            })
            graph_data, _, edge_data = network
            throughputs = edge_data.get('throughput')
            if throughputs is None:
                raise ValueError(
                    'The ridges method needs edge throughputs (a JSON network '
                    'with traffic, or a links CSV with a throughput column)'
                )
            node_throughputs = ridges.node_throughputs(
                len(graph_data['coordinates']), graph_data['edges'],
                throughputs, argument_dict['ridge_node_height']
            )
            if len(node_throughputs):
                reference = max(reference, float(np.max(node_throughputs)))
        for argument_dict in runs:
            argument_dict['ridge_height_reference'] = reference

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
    assign_ridge_height_references(arguments)
    driver.dispatch_optimization_batches(initialization, arguments, optimize, n_cores)

if __name__ == '__main__':
    main()
