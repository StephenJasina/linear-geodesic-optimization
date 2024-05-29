import concurrent.futures
import datetime
import itertools
import json
import os
# TODO: Convert to plain text
import pickle
import shutil
import time
import warnings

import networkx as nx
import numpy as np
import scipy

from linear_geodesic_optimization.data import input_network
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization import optimization


# Error on things like division by 0
warnings.simplefilter('error')

def get_phi(mesh, mu, sigma):
    return np.exp(-np.linalg.norm((mesh.get_coordinates()[:, :2] - mu) / sigma, axis=1)**2)

def get_perturbation(mesh: RectangleMesh, curvature, phi, target_difference):
    topology = mesh.get_topology()
    p = mesh.get_coordinates()[:, :2]
    curvature.reverse()

    scaling_factor_1 = np.max(np.abs([
        sum([
            dif_kappa_i_j * (phi_j - phi_i)
            for w in v.vertices()
            for j in (w.index,)
            for phi_j, dif_kappa_i_j in ((phi[j], dif_kappa_i[j]),)
        ])
        for v in topology.vertices()
        if not v.is_on_boundary()
        for i in (v.index,)
        for phi_i, dif_kappa_i in ((phi[i], curvature.dif_kappa_1[i]),)
    ]))
    scaling_factor_2 = np.max(np.abs([
        sum([
            dif_kappa_i_j * (phi_j - phi_i)
            for w in v.vertices()
            for j in (w.index,)
            for phi_j, dif_kappa_i_j in ((phi[j], dif_kappa_i[j]),)
        ])
        for v in topology.vertices()
        if not v.is_on_boundary()
        for i in (v.index,)
        for phi_i, dif_kappa_i in ((phi[i], curvature.dif_kappa_2[i]),)
    ]))
    scaling_factor = max(scaling_factor_1, scaling_factor_2)

    if scaling_factor == 0.:
        epsilon = 1.
    else:
        epsilon = target_difference / scaling_factor

    return epsilon * phi

rng = np.random.default_rng()
def generate_neighbor(mesh, curvature, target_difference):
    p = mesh.get_coordinates()[:, :2]
    z = mesh.get_parameters()
    amin = np.amin(p, axis = 0)
    amax = np.amax(p, axis = 0)
    mu = rng.uniform(amin, amax, 2)
    sigma = rng.uniform(0, np.linalg.norm(amax - amin))

    phi = get_phi(mesh, mu, sigma)
    perturbation = get_perturbation(mesh, curvature, phi, target_difference)

    return z + perturbation

def simulated_annealing(mesh: RectangleMesh, computer: optimization.Computer, max_iters):
    z = mesh.get_parameters()
    z_best = np.copy(z)
    loss = computer.forward(z)
    loss_best = loss
    for iteration, temperature in enumerate(np.linspace(1., 0., max_iters, False)):
        mesh.set_parameters(z)
        computer.curvature_loss.forward()
        computer.smooth_loss.forward()
        computer.geodesic_loss.forward()
        print(f'Loss at iteration {iteration} is {loss:0.6f} ({computer.curvature_loss.loss:0.6f}, {computer.smooth_loss.loss:0.6f}, {computer.geodesic_loss.loss:0.6f})')

        z_new = generate_neighbor(mesh, computer.curvature, rng.normal(scale = temperature / 5.))
        loss_new = computer.forward(z_new)
        if loss_new < loss or np.exp(-(loss_new - loss) / temperature * 1000) > rng.random():
            z = z_new
            loss = loss_new

        if loss < loss_best:
            z_best = np.copy(z)
            loss_best = loss

    return z_best

def main(probes_filename, latencies_filename, epsilon, clustering_distance,
         lambda_curvature, lambda_smooth, lambda_geodesic,
         sides, mesh_scale,
         leaveout_proportion=0.,
         maxiter=1000, output_dir_name=os.path.join('..', 'out'),
         graphml_filename=None):
    # Construct a mesh
    width = height = sides
    mesh = RectangleMesh(width, height, mesh_scale)
    z_0 = mesh.get_parameters()

    # Construct the network graph
    if graphml_filename is None:
        probes_file_path = os.path.join('..', 'data', probes_filename)
        latencies_file_path = os.path.join('..', 'data', latencies_filename)
        network, latencies = input_network.get_graph(
            probes_file_path, latencies_file_path,
            epsilon, clustering_distance,
            should_include_latencies=True
        )
    else:
        graphml_file_path = os.path.join('..', 'data', graphml_filename)
        network = nx.read_graphml(graphml_file_path)
        latencies = []
    network = input_network.extract_from_graph(network, latencies)
    network_coordinates, bounding_box, network_edges, network_curvatures, network_latencies = network
    network_vertices = mesh.map_coordinates_to_support(np.array(network_coordinates), np.float64(0.8), bounding_box)
    leaveout_count = int(leaveout_proportion * len(network_latencies))
    leaveout_seed = time.monotonic_ns() % (2**31 - 1)
    if leaveout_count > 0:
        rng = np.random.default_rng(leaveout_seed)
        rng.shuffle(network_latencies)
        network_latencies = network_latencies[:-leaveout_count]

    # Setup snapshots
    directory = os.path.join(
        output_dir_name,
        f'{lambda_curvature}_{lambda_smooth}_{lambda_geodesic}_{width}_{height}_{mesh_scale}'
    )
    if os.path.isdir(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

    parameters = {
        'epsilon': epsilon,
        'clustering_distance': clustering_distance,
        'should_remove_TIVs': False, # TODO: Pass this as a parameter?
        'lambda_curvature': lambda_curvature,
        'lambda_smooth': lambda_smooth,
        'lambda_geodesic': lambda_geodesic,
        'width': width,
        'height': height,
        'mesh_scale': mesh_scale,
        'coordinates_scale': 0.8,
        'leaveout_count': leaveout_count,
        'leaveout_seed': leaveout_seed
    }
    if graphml_filename is None:
        parameters['probes_filename'] = probes_filename
        parameters['latencies_filename'] = latencies_filename
    else:
        parameters['graphml_filename'] = graphml_filename

    with open(os.path.join(directory, 'parameters'), 'wb') as f:
        pickle.dump(parameters, f)

    computer = optimization.Computer(
        mesh, network_vertices, network_edges, network_curvatures,
        network_latencies, 1.01 * 2**0.5 * mesh_scale / width,
        lambda_curvature, lambda_smooth, lambda_geodesic,
        directory)

    z = simulated_annealing(mesh, computer, maxiter)

    with open(os.path.join(directory, 'output'), 'wb') as f:
        pickle.dump({
            'parameters': parameters,
            'initial': optimization.Computer.to_float_list(z_0),
            'final': optimization.Computer.to_float_list(z),
            'network': network, # TODO: JSONify this
        }, f)

if __name__ == '__main__':
    graphml_filenames = [
        os.path.join('toy', 'toy.graphml')
    ]
    probes_filenames = [
        os.path.join('toy', 'toy_probes.graphml')
    ]
    latencies_filenames = [
        os.path.join('toy', 'toy_latencies.graphml')
    ]

    count = len(probes_filenames)

    epsilon = np.inf
    epsilons = [epsilon] * count
    clustering_distances = [None] * count

    lambda_curvatures = [1.] * count
    lambda_smooths = [0.00001] * count
    lambda_geodesics = [0.] * count
    sides = [50] * count
    mesh_scales = [5.] * count

    leaveout_proportions = [1.] * count

    max_iters = [2500] * count

    output_dir_names = [
        os.path.join('..', f'out_simulated_annealing', 'toy')
        for probes_filename in probes_filenames
    ]

    arguments = list(zip(
        probes_filenames, latencies_filenames, epsilons, clustering_distances,
        lambda_curvatures, lambda_smooths, lambda_geodesics,
        sides, mesh_scales,
        leaveout_proportions,
        max_iters,
        output_dir_names,
        graphml_filenames,
    ))
    if len(arguments) > 1:
        # Need to use ProcessPoolExecutor instead of multiprocessing.Pool
        # to allow child processes to spawn their own subprocesses
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for _ in executor.map(main, *zip(*arguments)):
                pass
    else:
        main(*arguments[0])
