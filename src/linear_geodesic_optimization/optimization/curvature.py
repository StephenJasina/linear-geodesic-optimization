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

        # A map i -> kappa_H(i)
        self.kappa_H = None

        self.L_curvature = None

    def _calc_kappa_G(self):
        kappa_G = np.full(self._V, 2 * np.pi)
        # On the boundary, use Geodesic curvature instead of Gaussian curvature
        kappa_G[list(self._mesh.get_boundary_vertices())] = np.pi
        for (i, j), cot_ij in self.cot.items():
            kappa_G[self._c[i, j]] -= np.arccos(cot_ij / (1 + cot_ij**2)**0.5)
        return kappa_G

    def _calc_kappa_H(self):
        kappa_Hn = -self.D_inv @ (self.LC @ self._v) / 2
        kappa_H = np.linalg.norm(kappa_Hn, axis=1)
        for i in range(self._V):
            if kappa_Hn[i] @ self.N[(i, self._e[i][0])] < 0:
                kappa_H[i] *= -1
        return kappa_H

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
            self.kappa_H = self._calc_kappa_H()

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

        self._cot = None
        self._kappa_G = None

        self._dif_cot = None

        # Derivatives match the types of what are being differentiated.
        self.dif_kappa_G = None
        self.dif_L_curvature = None

    def _calc_dif_kappa_G(self):
        dif_kappa_G = np.zeros(self._V)
        for (i, j), cot_ij in self._cot.items():
            if (i, j) in self._dif_cot:
                dif_kappa_G[self._c[i,j]] += self._dif_cot[i,j] / (1 + cot_ij**2)
        return dif_kappa_G

    def calc(self, dif_v, l):
        self._laplacian_forward.calc()
        self._cot = self._laplacian_forward.cot

        self._curvature_forward.calc()
        self._kappa_G = self._curvature_forward.kappa_G

        self._laplacian_reverse.calc(dif_v, l)
        self._dif_cot = self._laplacian_reverse.dif_cot

        if self._updates != self._mesh.updates() or self._l != l:
            self._updates = self._mesh.updates()
            self._v = self._mesh.get_vertices()
            self._dif_v = dif_v
            self._l = l

            self.dif_kappa_G = self._calc_dif_kappa_G()
