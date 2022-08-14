import numpy as np

from linear_geodesic_optimization.optimization \
    import laplacian, geodesic, linear_regression, smooth
from linear_geodesic_optimization.optimization.partial_selection \
    import approximate_geodesics_fpi

class DifferentiationHierarchy:
    def __init__(self, mesh, ts, lam=0.01):
        self.mesh = mesh
        directions = self.mesh.get_directions()
        self.dif_v = {l: directions[l] for l in range(directions.shape[0])}

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

    def get_forwards(self, s_indices=None):
        if s_indices is None:
            s_indices = self.ts.keys()

        phis = []
        lse = 0

        for s_index in s_indices:
            gamma = [s_index]
            s_connected, t = zip(*self.ts[s_index])
            s_connected = list(s_connected)
            t = np.array(t)
            self.geodesic_forwards[s_index].calc(gamma)
            phi = self.geodesic_forwards[s_index].phi
            phis.append(phi)
            self.linear_regression_forward.calc(phi[s_connected], t)
            lse += self.linear_regression_forward.lse
        self.smooth_forward.calc()
        L_smooth = self.smooth_forward.L_smooth

        return phis, lse, L_smooth

    def get_reverses(self, s_indices=None):
        if s_indices is None:
            s_indices = self.ts.keys()

        V = self.mesh.get_directions().shape[0]
        dif_lse = np.zeros(V)
        dif_L_smooth = np.zeros(V)

        # Avoid recomputing these values too many times
        phis = {}
        ls = {}
        s_connecteds = {}
        ts = {}
        for s_index in s_indices:
            s_connected, t = zip(*self.ts[s_index])
            s_connected = list(s_connected)
            s_connecteds[s_index] = s_connected
            ts[s_index] = np.array(t)
            self.geodesic_forwards[s_index].calc([s_index])
            phi = self.geodesic_forwards[s_index].phi

            phis[s_index] = phi[s_connected]
            ls[s_index] = approximate_geodesics_fpi(self.mesh, phi, s_connected)

        for l in range(V):
            for s_index in s_indices:
                if l in ls[s_index]:
                    self.geodesic_reverses[s_index].calc([s_index], self.dif_v[l], l)
                    dif_phi = self.geodesic_reverses[s_index].dif_phi[s_connecteds[s_index]]
                    self.linear_regression_reverse.calc(phis[s_index], ts[s_index], dif_phi, l)
                    dif_lse[l] += self.linear_regression_reverse.dif_lse

            self.smooth_reverse.calc(self.dif_v[l], l)
            dif_L_smooth[l] += self.smooth_reverse.dif_L_smooth

        return dif_lse, dif_L_smooth

    def get_loss_callback(self, s_indices=None):
        if s_indices is None:
            s_indices = self.ts.keys()

        def loss(rho):
            if np.min(rho) <= 0.:
                return np.inf
            self.mesh.set_rho(rho)
            _, lse, L_smooth = self.get_forwards(s_indices)
            return lse + self.lam * L_smooth
        return loss

    def get_dif_loss_callback(self, s_indices=None):
        if s_indices is None:
            s_indices = self.ts.keys()

        def dif_loss(rho):
            if min(rho) <= 0.:
                return np.zeros(rho.shape[0])
            self.mesh.set_rho(rho)
            dif_lse, dif_L_smooth = self.get_reverses(s_indices)
            return dif_lse + self.lam * dif_L_smooth
        return dif_loss
