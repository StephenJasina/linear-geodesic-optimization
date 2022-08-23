import multiprocessing
import os
import pickle

import numpy as np

from linear_geodesic_optimization.optimization \
    import laplacian, geodesic, linear_regression, smooth
from linear_geodesic_optimization.optimization.partial_selection \
    import approximate_geodesics_fpi

class DifferentiationHierarchy:
    def __init__(self, mesh, ts, lam=0.01, directory=None):
        self.mesh = mesh
        partials = self.mesh.get_partials()
        self.dif_v = {l: partials[l] for l in range(partials.shape[0])}

        self.ts = ts

        self.lam = lam

        self.laplacian_forward = laplacian.Forward(mesh)
        self.geodesic_forwards = \
            {s_index: geodesic.Forward(mesh, self.laplacian_forward)
             for s_index in ts}
        self.linear_regression_forward = linear_regression.Forward()
        self.smooth_forward = smooth.Forward(mesh, self.laplacian_forward)
        self.laplacian_reverse = \
            laplacian.Reverse(mesh, self.laplacian_forward)
        self.geodesic_reverses = \
            {s_index: geodesic.Reverse(mesh, self.geodesic_forwards[s_index],
                                       self.laplacian_reverse)
             for s_index in ts}
        self.linear_regression_reverse = \
            linear_regression.Reverse(self.linear_regression_forward)
        self.smooth_reverse = smooth.Reverse(mesh, self.laplacian_forward,
                                             self.laplacian_reverse)

        # Count of iterations for diagnostic purposes
        self.iterations = 0

        # Location where to save iterations of the hierarchy
        self.directory = directory

    @staticmethod
    def _forwards_call(s_index, t, geodesic_forward):
        gamma = [s_index]
        connected_s_index, t_s_index = zip(*t)
        connected_s_index = list(connected_s_index)
        geodesic_forward.calc(gamma)
        phi = geodesic_forward.phi[connected_s_index]

        return t_s_index, phi

    def get_forwards(self, s_indices=None):
        if s_indices is None:
            s_indices = self.ts.keys()

        with multiprocessing.Pool(os.cpu_count() // 3) as pool:
            arguments = [(s_index, self.ts[s_index],
                          self.geodesic_forwards[s_index])
                         for s_index in s_indices]
            ts, phis = zip(*pool.starmap(DifferentiationHierarchy._forwards_call, arguments))

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
    def _reverse_call(s_index, t, mesh, geodesic_forward, geodesic_reverse):
        partials = mesh.get_partials()
        V = partials.shape[0]
        dif_v = {l: partials[l] for l in range(partials.shape[0])}

        gamma = [s_index]
        connected_s_index, t_s_index = zip(*t)
        connected_s_index = list(connected_s_index)
        geodesic_forward.calc(gamma)
        phi = geodesic_forward.phi[connected_s_index]

        ls = approximate_geodesics_fpi(mesh, geodesic_forward.phi, connected_s_index)
        dif_phi = [None for l in range(V)]
        for l in range(V):
            if l in ls:
                geodesic_reverse.calc(gamma, dif_v[l], l)
                dif_phi[l] = geodesic_reverse.dif_phi[connected_s_index]
            else:
                dif_phi[l] = [0 for _ in range(len(connected_s_index))]

        return t_s_index, phi, dif_phi

    def get_reverses(self, s_indices=None):
        if s_indices is None:
            s_indices = self.ts.keys()

        V = self.mesh.get_partials().shape[0]
        dif_lse = np.zeros(V)
        dif_L_smooth = np.zeros(V)

        with multiprocessing.Pool(os.cpu_count() // 3) as pool:
            arguments = [(s_index, self.ts[s_index], self.mesh,
                          self.geodesic_forwards[s_index],
                          self.geodesic_reverses[s_index])
                         for s_index in s_indices]
            ts, phis, dif_phis = zip(*pool.starmap(DifferentiationHierarchy._reverse_call, arguments))

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

    def get_loss_callback(self, s_indices=None):
        if s_indices is None:
            s_indices = self.ts.keys()

        def loss(parameters):
            self.mesh.set_parameters(parameters)
            lse, L_smooth = self.get_forwards(s_indices)
            return lse + self.lam * L_smooth
        return loss

    def get_dif_loss_callback(self, s_indices=None):
        if s_indices is None:
            s_indices = self.ts.keys()

        def dif_loss(parameters):
            self.mesh.set_parameters(parameters)
            dif_lse, dif_L_smooth = self.get_reverses(s_indices)
            return dif_lse + self.lam * dif_L_smooth
        return dif_loss

    def diagnostics(self, _):
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
