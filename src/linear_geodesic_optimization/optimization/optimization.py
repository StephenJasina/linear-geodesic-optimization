import numpy as np

from linear_geodesic_optimization.optimization \
    import laplacian, geodesic, linear_regression, smooth
from linear_geodesic_optimization.optimization.partial_selection \
    import approximate_geodesics_fpi

class DifferentiationHierarchy:
    def __init__(self, mesh, ts, lam=0.01):
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

    def get_forwards(self, s_indices=None):
        if s_indices is None:
            s_indices = self.ts.keys()

        lse = 0
        phi = []
        t = []

        for s_index in s_indices:
            gamma = [s_index]
            connected_s_index, t_s_index = zip(*self.ts[s_index])
            connected_s_index = list(connected_s_index)
            self.geodesic_forwards[s_index].calc(gamma)
            phi.extend(self.geodesic_forwards[s_index].phi[connected_s_index])
            t.extend(t_s_index)

        phi = np.array(phi)
        t = np.array(t)

        self.linear_regression_forward.calc(phi, t)
        lse = self.linear_regression_forward.lse

        self.smooth_forward.calc()
        L_smooth = self.smooth_forward.L_smooth

        return lse, L_smooth

    def get_reverses(self, s_indices=None):
        if s_indices is None:
            s_indices = self.ts.keys()

        V = self.mesh.get_partials().shape[0]
        dif_lse = np.zeros(V)
        dif_L_smooth = np.zeros(V)

        phi = []
        t = []
        dif_phi = {l: [] for l in range(V)}

        for s_index in s_indices:
            gamma = [s_index]
            connected_s_index, t_s_index = zip(*self.ts[s_index])
            connected_s_index = list(connected_s_index)
            self.geodesic_forwards[s_index].calc(gamma)
            phi.extend(self.geodesic_forwards[s_index].phi[connected_s_index])
            t.extend(t_s_index)

            ls = approximate_geodesics_fpi(self.mesh, self.geodesic_forwards[s_index].phi, connected_s_index)
            for l in range(V):
                if l in ls:
                    self.geodesic_reverses[s_index].calc(gamma, self.dif_v[l], l)
                    dif_phi[l].extend(self.geodesic_reverses[s_index].dif_phi[connected_s_index])
                else:
                    dif_phi[l].extend([0 for _ in range(len(connected_s_index))])

        phi = np.array(phi)
        t = np.array(t)

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
