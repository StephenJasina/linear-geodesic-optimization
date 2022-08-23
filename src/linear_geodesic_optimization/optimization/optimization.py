import multiprocessing
import os
import pickle

import numpy as np

from linear_geodesic_optimization.optimization \
    import laplacian, geodesic, linear_regression, smooth
from linear_geodesic_optimization.optimization.partial_selection \
    import approximate_geodesics_fpi

class DifferentiationHierarchy:
    '''
    Structure for consolidating evaluation and gradient computation of the
    linear geodesic optimization loss functions.
    '''

    def __init__(self, mesh, ts, lam=0.01, directory=None, cores=None):
        '''
        Parameters:
        * `mesh`: The mesh to optimize over
        * `ts`: The measured (real world) latencies
        * `lam`: The strength lambda of the smoothing parameter
        * `directory`: Where to save snapshots of the mesh for each iteration
                       of optimization
        * `cores`: The number of CPU cores to use when optimizing. Defaults to
                   `os.cpu_count() - 2`
        '''

        self.mesh = mesh

        partials = self.mesh.get_partials()
        self.dif_v = {l: partials[l] for l in range(partials.shape[0])}

        self.ts = ts

        self.lam = lam

        # For caching purposes, it's a good idea to keep one copy of the
        # geodesic computation classes for each city.
        self.laplacian_forward = laplacian.Forward(mesh)
        self.geodesic_forwards = \
            {mesh_index: geodesic.Forward(mesh, self.laplacian_forward)
             for mesh_index in ts}
        self.linear_regression_forward = linear_regression.Forward()
        self.smooth_forward = smooth.Forward(mesh, self.laplacian_forward)
        self.laplacian_reverse = laplacian.Reverse(mesh,
                                                   self.laplacian_forward)
        self.geodesic_reverses = \
            {mesh_index: geodesic.Reverse(mesh,
                                          self.geodesic_forwards[mesh_index],
                                          self.laplacian_reverse)
             for mesh_index in ts}
        self.linear_regression_reverse = \
            linear_regression.Reverse(self.linear_regression_forward)
        self.smooth_reverse = smooth.Reverse(mesh, self.laplacian_forward,
                                             self.laplacian_reverse)

        # Count of iterations for diagnostic purposes
        self.iterations = 0

        # Location where to save iterations of the hierarchy
        self.directory = directory

        self.cores = cores if cores is not None else os.cpu_count() - 2

    @staticmethod
    def _forwards_call(mesh_index, t, geodesic_forward):
        '''
        Internal method used for multiprocessing.

        Given data relevant to a certain city, returns the measured latencies
        and the corresponding geodesic distances on the mesh.
        '''

        mesh_indices, latencies = zip(*t)
        mesh_indices = list(mesh_indices)
        geodesic_forward.calc([mesh_index])
        phi = geodesic_forward.phi[mesh_indices]

        return latencies, phi

    def get_forwards(self):
        '''
        Returns the losses of the current mesh.
        '''

        ts = None
        phis = None

        # This just calls _forwards_call once for each city
        with multiprocessing.Pool(self.cores) as pool:
            arguments = [(mesh_index, self.ts[mesh_index],
                          self.geodesic_forwards[mesh_index])
                         for mesh_index in self.ts]
            ts, phis = zip(
                *pool.starmap(DifferentiationHierarchy._forwards_call,
                              arguments))

        # t and phi are parallel arrays

        t = []
        for t_part in ts:
            t.extend(t_part)
        t = np.array(t)

        phi = []
        for phi_part in phis:
            phi.extend(phi_part)
        phi = np.array(phi)

        self.linear_regression_forward.calc(phi, t)
        self.smooth_forward.calc()
        return self.linear_regression_forward.lse, self.smooth_forward.L_smooth

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

        mesh_indices, latencies = zip(*t)
        mesh_indices = list(mesh_indices)
        geodesic_forward.calc([mesh_index])
        phi = geodesic_forward.phi[mesh_indices]

        # For efficiency, only compute the relevant partial derivatives (or,
        # at least, try to compute as few irrelevant ones as possible)
        ls = approximate_geodesics_fpi(mesh, geodesic_forward.phi,
                                       mesh_indices)
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
        V = self.mesh.get_partials().shape[0]
        dif_lse = np.zeros(V)
        dif_L_smooth = np.zeros(V)

        # This just calls _reverses_call once for each city
        with multiprocessing.Pool(self.cores) as pool:
            arguments = [(mesh_index, self.ts[mesh_index],
                          self.mesh, self.dif_v,
                          self.geodesic_forwards[mesh_index],
                          self.geodesic_reverses[mesh_index])
                         for mesh_index in self.ts]
            ts, phis, dif_phis = zip(
                *pool.starmap(DifferentiationHierarchy._reverses_call,
                              arguments))

        # t and phi are parallel arrays. They are also parallel to each element
        # of dif_phi

        t = []
        for t_part in ts:
            t.extend(t_part)
        t = np.array(t)

        phi = []
        for phi_part in phis:
            phi.extend(phi_part)
        phi = np.array(phi)

        dif_phi = [[] for _ in range(V)]
        for dif_phi_part in dif_phis:
            for l, dif_phi_subpart in enumerate(dif_phi_part):
                dif_phi[l].extend(dif_phi_subpart)

        for l in range(V):
            self.linear_regression_reverse.calc(phi, t, dif_phi[l], l)
            dif_lse[l] += self.linear_regression_reverse.dif_lse

            self.smooth_reverse.calc(self.dif_v[l], l)
            dif_L_smooth[l] += self.smooth_reverse.dif_L_smooth

        return dif_lse, dif_L_smooth

    def get_loss_callback(self):
        '''
        Return a function that takes in mesh parameters and returns the
        corresponding loss (this is "f").
        '''

        def loss(parameters):
            self.mesh.set_parameters(parameters)
            lse, L_smooth = self.get_forwards()
            return lse + self.lam * L_smooth
        return loss

    def get_dif_loss_callback(self):
        '''
        Return a function that takes in mesh parameters and returns the
        derivative of the corresponding loss (this is "g").
        '''

        def dif_loss(parameters):
            self.mesh.set_parameters(parameters)
            dif_lse, dif_L_smooth = self.get_reverses()
            return dif_lse + self.lam * dif_L_smooth
        return dif_loss

    def diagnostics(self, _):
        '''
        Save the hierarchy to disk and output some useful information about
        the loss functions.
        '''

        if self.directory is not None:
            with open(os.path.join(self.directory,
                                   str(self.iterations)), 'wb') as f:
                pickle.dump(self, f)

        lse, L_smooth = self.get_forwards()
        print(f'iteration {self.iterations}:')
        print(f'\tlse: {lse:.6f}')
        print(f'\tL_smooth: {L_smooth:.6f}\n')
        print(f'\tLoss: {(lse + self.lam * L_smooth):.6f}')

        self.iterations += 1
