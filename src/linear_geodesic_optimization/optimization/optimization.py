import multiprocessing
import os
import pickle

import numpy as np

from linear_geodesic_optimization.optimization \
    import laplacian, geodesic, linear_regression, curvature, curvature_loss, smooth
from linear_geodesic_optimization.optimization.partial_selection \
    import approximate_geodesics_fpi

class DifferentiationHierarchy:
    '''
    Structure for consolidating evaluation and gradient computation of the
    linear geodesic optimization loss functions.
    '''

    def __init__(self, mesh, latencies, network_vertices, network_edges,
                 ricci_curvatures,
                 lambda_geodesic=1., lambda_curvature=1., lambda_smooth=0.01,
                 smooth_strategy='mvs_cross', directory=None, cores=None):
        '''
        Parameters:
        * `mesh`: The mesh to optimize over
        * `latencies`: The measured (real world) latencies. This should be a
                       map from vertex indices to lists of pairs of vertex
                       indices and floats (essentially an annotated adjacency
                       list)
        * `network_vertices`: A list of vertices in coordinate form.
                              Alternatively, a numpy array of the vertices. In
                              particular, these vertices should be embedded
                              into the mesh
        * `network_edges`: A list of edges in the network, where each edge is
                           represented as a pair of indices into
                           `network_vertices`
        * `lamda_geodesic`: The strength lambda of the geodesic loss
        * `lamda_curvature`: The strength lambda of the curvature loss
        * `lamda_smooth`: The strength lambda of the smoothing loss
        * `directory`: Where to save snapshots of the mesh for each iteration
                       of optimization
        * `cores`: The number of CPU cores to use when optimizing
        '''

        self.mesh = mesh

        epsilon = mesh.get_epsilon()

        self.latencies = latencies

        self.lambda_geodesic = lambda_geodesic
        self.lambda_smooth = lambda_smooth
        self.lambda_curvature = lambda_curvature

        # For caching purposes, it's a good idea to keep one copy of the
        # geodesic computation classes for each city.
        self.laplacian_forward = laplacian.Forward(mesh)
        self.geodesic_forwards = \
            {mesh_index: geodesic.Forward(mesh, self.laplacian_forward)
             for mesh_index in latencies}
        self.linear_regression_forward = linear_regression.Forward()
        self.curvature_forward = curvature.Forward(
            mesh, self.laplacian_forward
        )
        self.curvature_loss_forward = curvature_loss.Forward(
            mesh, network_vertices, network_edges, ricci_curvatures, epsilon,
            self.curvature_forward
        )
        self.smooth_forward = smooth.Forward(
            mesh, self.laplacian_forward, self.curvature_forward, smooth_strategy
        )

        self.laplacian_reverses = \
            {mesh_index: laplacian.Reverse(mesh, self.laplacian_forward)
             for mesh_index in latencies}
        self.geodesic_reverses = \
            {mesh_index: geodesic.Reverse(mesh,
                                          self.geodesic_forwards[mesh_index],
                                          self.laplacian_reverses[mesh_index])
             for mesh_index in latencies}
        self.linear_regression_reverse = \
            linear_regression.Reverse(self.linear_regression_forward)
        self.curvature_reverse = curvature.Reverse(
            mesh, self.laplacian_forward, self.curvature_forward,
            next(iter(self.laplacian_reverses.values()))
        )
        self.curvature_loss_reverse = curvature_loss.Reverse(
            mesh, network_vertices, network_edges, ricci_curvatures, epsilon,
            self.curvature_forward, self.curvature_reverse
        )
        self.smooth_reverse = smooth.Reverse(
            mesh, self.laplacian_forward, self.curvature_forward,
            next(iter(self.laplacian_reverses.values())),
            self.curvature_reverse, smooth_strategy
        )

        # Count of iterations for diagnostic purposes
        self.iterations = 0

        # Location where to save iterations of the hierarchy
        self.directory = directory

        self.cores = cores if cores is not None else 1

    @staticmethod
    def _forwards_call(mesh_index, t, geodesic_forward):
        '''
        Internal method used for multiprocessing.

        Given data relevant to a certain city, returns the measured latencies
        and the corresponding geodesic distances on the mesh.
        '''

        mesh_indices, latencies = [], []
        if t:
            mesh_indices, latencies = zip(*t)
        mesh_indices = list(mesh_indices)
        geodesic_forward.calc([mesh_index])
        phi = geodesic_forward.phi[mesh_indices]

        return latencies, phi

    def get_forwards(self):
        '''
        Returns the losses of the current mesh.
        '''

        t = []
        phi = []

        self.laplacian_forward.calc()

        if self.cores > 1:
            # This just calls _forwards_call once for each city
            with multiprocessing.Pool(self.cores) as pool:
                arguments = [(mesh_index, self.latencies[mesh_index],
                              self.geodesic_forwards[mesh_index])
                             for mesh_index in self.latencies]
                ts, phis = zip(
                    *pool.starmap(DifferentiationHierarchy._forwards_call,
                                  arguments))
                for t_part in ts:
                    t.extend(t_part)
                for phi_part in phis:
                    phi.extend(phi_part)
        else:
            for mesh_index in self.latencies:
                t_part, phi_part = DifferentiationHierarchy._forwards_call(
                    mesh_index,
                    self.latencies[mesh_index],
                    self.geodesic_forwards[mesh_index]
                )
                t.extend(t_part)
                phi.extend(phi_part)

        t = np.array(t)
        phi = np.array(phi)

        self.linear_regression_forward.calc(phi, t)
        self.smooth_forward.calc()
        self.curvature_loss_forward.calc()
        return self.linear_regression_forward.lse, \
               self.smooth_forward.L_smooth, \
               self.curvature_loss_forward.L_curvature

    @staticmethod
    def _reverses_call(mesh_index, t, mesh, dif_v,
                      geodesic_forward, geodesic_reverse):
        '''
        Internal method used for multiprocessing.

        Given data relevant to a certain city, returns the measured latencies,
        the corresponding geodesic distances on the mesh, and the partial
        derivatives of the distances.
        '''

        V = len(dif_v)

        mesh_indices, latencies = [], []
        if t:
            mesh_indices, latencies = zip(*t)
        mesh_indices = list(mesh_indices)
        geodesic_forward.calc([mesh_index])
        phi = geodesic_forward.phi[mesh_indices]

        # For efficiency, only compute the relevant partial derivatives (or,
        # at least, try to compute as few irrelevant ones as possible)
        # TODO: Fix this call
        # ls = approximate_geodesics_fpi(mesh, geodesic_forward.phi,
        #                                mesh_indices)
        ls = set(range(V))
        dif_phi = [None for _ in range(V)]
        for l in range(V):
            if l in ls:
                geodesic_reverse.calc([mesh_index], dif_v[l], l)
                dif_phi[l] = geodesic_reverse.dif_phi[mesh_indices]
            else:
                # Vertices not affected by the l-th partial
                dif_phi[l] = [0 for _ in range(len(mesh_indices))]

        return latencies, phi, dif_phi

    def get_reverses(self):
        partials = self.mesh.get_partials()
        V = partials.shape[0]
        dif_v = {l: partials[l,:] for l in range(V)}
        dif_L_geodesic = np.zeros(V)
        dif_L_smooth = np.zeros(V)
        dif_L_curvature = np.zeros(V)

        self.laplacian_forward.calc()

        t = []
        phi = []
        dif_phi = [[] for _ in range(V)]

        if self.lambda_geodesic != 0:
            if self.cores > 1:
                # This just calls _reverses_call once for each city
                with multiprocessing.Pool(self.cores) as pool:
                    arguments = [(mesh_index, self.latencies[mesh_index],
                                  self.mesh, dif_v,
                                  self.geodesic_forwards[mesh_index],
                                  self.geodesic_reverses[mesh_index])
                                 for mesh_index in self.latencies]
                    ts, phis, dif_phis = zip(
                        *pool.starmap(DifferentiationHierarchy._reverses_call,
                                      arguments))
                    for t_part in ts:
                        t.extend(t_part)
                    for phi_part in phis:
                        phi.extend(phi_part)
                    for dif_phi_part in dif_phis:
                        for l, dif_phi_subpart in enumerate(dif_phi_part):
                            dif_phi[l].extend(dif_phi_subpart)
            else:
                for mesh_index in self.latencies:
                    t_part, phi_part, dif_phi_part = \
                        DifferentiationHierarchy._reverses_call(
                            mesh_index,
                            self.latencies[mesh_index],
                            self.mesh,
                            dif_v,
                            self.geodesic_forwards[mesh_index],
                            self.geodesic_reverses[mesh_index]
                        )
                    t.extend(t_part)
                    phi.extend(phi_part)
                    for l, dif_phi_subpart in enumerate(dif_phi_part):
                        dif_phi[l].extend(dif_phi_subpart)

            t = np.array(t)
            phi = np.array(phi)

        for l in range(V):
            if self.lambda_geodesic != 0:
                self.linear_regression_reverse.calc(phi, t, dif_phi[l])
                dif_L_geodesic[l] += self.linear_regression_reverse.dif_lse

            self.smooth_reverse.calc(dif_v[l], l)
            dif_L_smooth[l] += self.smooth_reverse.dif_L_smooth

            self.curvature_loss_reverse.calc(dif_v[l], l)
            dif_L_curvature[l] += self.curvature_loss_reverse.dif_L_curvature

        return dif_L_geodesic, dif_L_smooth, dif_L_curvature

    def get_loss_callback(self):
        '''
        Return a function that takes in mesh parameters and returns the
        corresponding loss (this is "f").
        '''

        def loss(parameters):
            self.mesh.set_parameters(parameters)
            L_geodesic, L_smooth, L_curvature = self.get_forwards()
            return self.lambda_geodesic * L_geodesic \
                + self.lambda_smooth * L_smooth \
                + self.lambda_curvature * L_curvature
        return loss

    def get_dif_loss_callback(self):
        '''
        Return a function that takes in mesh parameters and returns the
        derivative of the corresponding loss (this is "g").
        '''

        def dif_loss(parameters):
            self.mesh.set_parameters(parameters)
            dif_L_geodesic, dif_L_smooth, dif_L_curvature = self.get_reverses()
            return self.lambda_geodesic * dif_L_geodesic \
                + self.lambda_smooth * dif_L_smooth \
                + self.lambda_curvature * dif_L_curvature
        return dif_loss

    def diagnostics(self, _):
        '''
        Save the hierarchy to disk and output some useful information about
        the loss functions.
        '''

        L_geodesic, L_smooth, L_curvature = self.get_forwards()
        loss = self.lambda_geodesic * L_geodesic \
            + self.lambda_smooth * L_smooth \
            + self.lambda_curvature * L_curvature
        print(f'iteration {self.iterations}:')
        print(f'\tL_geodesic: {L_geodesic:.6f}')
        print(f'\tL_smooth: {L_smooth:.6f}')
        print(f'\tL_curvature: {L_curvature:.6f}')
        print(f'\tLoss: {(loss):.6f}\n')

        if self.directory is not None:
            with open(os.path.join(self.directory,
                                   str(self.iterations)), 'wb') as f:

                phis = []
                ts = []
                for si in self.latencies:
                    self.geodesic_forwards[si].calc([si])
                    phi = self.geodesic_forwards[si].phi
                    for sj, tij in self.latencies[si]:
                        phis.append(phi[sj])
                        ts.append(tij)
                phis = np.array(phis)
                ts = np.array(ts)

                pickle.dump({
                    'mesh': self.mesh,
                    'L_geodesic': L_geodesic,
                    'L_smooth': L_smooth,
                    'L_curvature': L_curvature,
                    'lambda_geodesic': self.lambda_geodesic,
                    'lambda_smooth': self.lambda_smooth,
                    'lambda_curvature': self.lambda_curvature,
                    'true_latencies': ts,
                    'estimated_latencies': phis,
                }, f)

        self.iterations += 1
