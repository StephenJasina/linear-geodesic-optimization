import numpy as np

from linear_geodesic_optimization.optimization import laplacian

class Forward:
    '''
    Implementation various curvature functions.
    '''

    def __init__(self, mesh, laplacian_forward=None):
        self._mesh = mesh
        self._updates = self._mesh.updates() - 1
        self._v = None
        self._e = self._mesh.get_edges()
        self._c = self._mesh.get_c()

        self._V = len(self._e)

        self._laplacian_forward = laplacian_forward
        if self._laplacian_forward is None:
            self._laplacian_forward = laplacian.Forward(mesh)

        self.N = None
        self.D_inv = None
        self.cot = None
        self.LC = None

        # A map i -> kappa_G(i)
        self.kappa_G = None

        # A |V| x 3 matrix
        self.vertex_normal = None

        # A |V| x 3 matrix
        self.mean_curvature_normal = None

        # A map i -> kappa_H(i)
        self.kappa_H = None

        # A map i -> kappa_1(i)
        self.kappa_1 = None

        # A map i -> kappa_2(i)
        self.kappa_2 = None

    def _calc_kappa_G(self):
        Dkappa_G = np.full(self._V, 2 * np.pi)
        # On the boundary, use Geodesic curvature instead of Gaussian curvature
        Dkappa_G[list(self._mesh.get_boundary_vertices())] = np.pi
        for (i, j), cot_ij in self.cot.items():
            Dkappa_G[self._c[i, j]] -= np.arccos(cot_ij / (1 + cot_ij**2)**0.5)
        return self.D_inv @ Dkappa_G

    def _calc_vertex_normal(self):
        vertex_normal = np.zeros((self._V, 3))
        for i in range(self._V):
            for j in self._e[i]:
                vertex_normal[i,:] += self.N[i,j] / (1 + self.cot[i,j]**2)**0.5
        return vertex_normal

    def _calc_mean_curvature_normal(self):
        return -self.D_inv @ (self.LC @ self._v) / 2

    def _calc_kappa_H(self):
        left = self.vertex_normal / np.linalg.norm(self.vertex_normal, axis=1).reshape((-1, 1))
        right = self.mean_curvature_normal
        return np.array([left[i,:] @ right[i,:] for i in range(self._V)])

    def _calc_kappa_1(self):
        return self.kappa_H + np.sqrt(np.maximum(0., self.kappa_H**2 - self.kappa_G))

    def _calc_kappa_2(self):
        return self.kappa_H - np.sqrt(np.maximum(0., self.kappa_H**2 - self.kappa_G))

    def calc(self):
        self._laplacian_forward.calc()
        self.N = self._laplacian_forward.N
        self.D_inv = self._laplacian_forward.D_inv
        self.cot = self._laplacian_forward.cot
        self.LC = self._laplacian_forward.LC_neumann

        if self._updates != self._mesh.updates():
            self._updates = self._mesh.updates()
            self._v = self._mesh.get_vertices()

            self.kappa_G = self._calc_kappa_G()
            self.vertex_normal = self._calc_vertex_normal()
            self.mean_curvature_normal = self._calc_mean_curvature_normal()
            self.kappa_H = self._calc_kappa_H()
            self.kappa_1 = self._calc_kappa_1()
            self.kappa_2 = self._calc_kappa_2()

class Reverse:
    '''
    Implementation of the gradient of the curvature loss function on a mesh.
    This implementation assumes the l-th partial affects only the l-th vertex.
    '''

    def __init__(self, mesh, laplacian_forward=None, curvature_forward=None,
                 laplacian_reverse=None):
        self._mesh = mesh
        self._updates = self._mesh.updates() - 1
        self._v = None
        self._e = self._mesh.get_edges()
        self._c = self._mesh.get_c()

        self._V = len(self._e)

        self._dif_v = None
        self._l = None

        self._laplacian_forward = laplacian_forward
        if self._laplacian_forward is None:
            self._laplacian_forward = laplacian.Forward(mesh)

        self._curvature_forward = curvature_forward
        if self._curvature_forward is None:
            self._curvature_forward = Forward(mesh, self._laplacian_forward)

        self._laplacian_reverse = laplacian_reverse
        if self._laplacian_reverse is None:
            self._laplacian_reverse = laplacian.Reverse(mesh)

        self._N = None
        self._D_inv = None
        self._cot = None
        self._LC = None

        self._kappa_G = None
        self._vertex_normal = None
        self._mean_curvature_normal = None

        self._dif_N = None
        self._dif_D = None
        self._dif_cot = None
        self._dif_LC = None

        # Derivatives match the types of what are being differentiated.
        self.dif_kappa_G = None
        self.dif_vertex_normal = None
        self.dif_mean_curvature_normal = None
        self.dif_kappa_H = None
        self.dif_kappa_1 = None
        self.dif_kappa_2 = None

    def _calc_dif_kappa_G(self):
        dif_Dkappa_G = np.zeros(self._V)
        for (i, j), cot_ij in self._cot.items():
            if (i, j) in self._dif_cot:
                dif_Dkappa_G[self._c[i,j]] += self._dif_cot[i,j] / (1 + cot_ij**2)
        return self._D_inv @ (dif_Dkappa_G - self._dif_D @ self._kappa_G)

    def _calc_dif_vertex_normal(self):
        dif_vertex_normal = np.zeros((self._V, 3))
        for i in range(self._V):
            for j in self._e[i]:
                cot = self._cot[i,j]
                N = self._N[i,j]
                dif_cot = self._dif_cot[i,j] if (i, j) in self._dif_cot else 0.
                dif_N = self._dif_N[i, j] if (i, j) in self._dif_N else 0.
                dif_vertex_normal[i,:] += (dif_N - cot * dif_cot / (1 + cot**2) * N) / (1 + cot**2)**0.5
        return dif_vertex_normal

    def _calc_dif_mean_curvature_normal(self):
        dif_v = np.zeros((self._V, 3))
        dif_v[self._l,:] = self._dif_v
        return -self._D_inv @ (
            (self._dif_LC - self._dif_D @ self._D_inv @ self._LC) @ self._v
            + self._LC @ dif_v
        ) / 2

    def _calc_dif_kappa_H(self):
        dif_kappa_H = np.zeros(self._V)
        for i in range(self._V):
            vn = self._vertex_normal[i,:]
            mcn = self._mean_curvature_normal[i,:]
            dif_vn = self.dif_vertex_normal[i,:]
            dif_mcn = self.dif_mean_curvature_normal[i,:]
            dif_kappa_H[i] = (dif_vn.T @ ((np.eye(3) - np.outer(vn, vn) / (vn @ vn)) @ mcn) + vn.T @ dif_mcn) / np.linalg.norm(vn)
        return dif_kappa_H

    def _calc_dif_kappa_1(self):
        return np.divide(
            2 * self._kappa_1 * self.dif_kappa_H - self.dif_kappa_G,
            self._kappa_1 - self._kappa_2,
            np.copy(self.dif_kappa_H),
            where=(self._kappa_1 != self._kappa_2)
        )

    def _calc_dif_kappa_2(self):
        return np.divide(
            self.dif_kappa_G - 2 * self._kappa_2 * self.dif_kappa_H,
            self._kappa_1 - self._kappa_2,
            np.copy(self.dif_kappa_H),
            where=(self._kappa_1 != self._kappa_2)
        )

    def calc(self, dif_v, l):
        self._laplacian_forward.calc()
        self._N = self._laplacian_forward.N
        self._D_inv = self._laplacian_forward.D_inv
        self._cot = self._laplacian_forward.cot
        self._LC = self._laplacian_forward.LC_neumann

        self._curvature_forward.calc()
        self._kappa_G = self._curvature_forward.kappa_G
        self._vertex_normal = self._curvature_forward.vertex_normal
        self._mean_curvature_normal = self._curvature_forward.mean_curvature_normal
        self._kappa_H = self._curvature_forward.kappa_H
        self._kappa_1 = self._curvature_forward.kappa_1
        self._kappa_2 = self._curvature_forward.kappa_2

        self._laplacian_reverse.calc(dif_v, l)
        self._dif_N = self._laplacian_reverse.dif_N
        self._dif_D = self._laplacian_reverse.dif_D
        self._dif_cot = self._laplacian_reverse.dif_cot
        self._dif_LC = self._laplacian_reverse.dif_LC_neumann

        if self._updates != self._mesh.updates() or self._l != l:
            self._updates = self._mesh.updates()
            self._v = self._mesh.get_vertices()
            self._dif_v = dif_v
            self._l = l

            self.dif_kappa_G = self._calc_dif_kappa_G()
            self.dif_vertex_normal = self._calc_dif_vertex_normal()
            self.dif_mean_curvature_normal = self._calc_dif_mean_curvature_normal()
            self.dif_kappa_H = self._calc_dif_kappa_H()
            self.dif_kappa_1 = self._calc_dif_kappa_1()
            self.dif_kappa_2 = self._calc_dif_kappa_2()
