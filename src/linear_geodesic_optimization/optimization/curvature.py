import numpy as np

from linear_geodesic_optimization.optimization import laplacian

class Forward:
    '''
    Implementation of the curvature loss function.
    '''

    def __init__(self, mesh, network_vertices, network_edges, ricci_curvatures,
                 epsilon, laplacian_forward=None):
        self._mesh = mesh
        self._updates = self._mesh.updates() - 1
        self._v = None
        self._e = self._mesh.get_edges()
        self._c = self._mesh.get_c()

        self._V = len(self._e)

        self._network_vertices = network_vertices
        self._network_edges = network_edges
        self._ricci_curvatures = ricci_curvatures
        self._fat_edges = mesh.get_fat_edges(network_vertices, network_edges,
                                             epsilon)

        self._laplacian_forward = laplacian_forward
        if self._laplacian_forward is None:
            self._laplacian_forward = laplacian.Forward(mesh)

        self.cot = None

        # A map i -> kappa(i)
        self.kappa = None
        self.L_curvature = None

    def _calc_kappa(self):
        kappa = np.full(self._V, 2 * np.pi)
        for (i, j), cot_ij in self.cot.items():
            kappa[self._c[i, j]] -= np.arctan(1 / cot_ij) % np.pi
        return kappa

    def _calc_L_curvature(self):
        L_curvature = sum((self.kappa[i] - ricci_curvature)**2
                          for ricci_curvature, fat_edge in zip(self._ricci_curvatures,
                                                               self._fat_edges)
                          for i in fat_edge)
        return sum((self.kappa[i] - ricci_curvature)**2
                   for ricci_curvature, fat_edge in zip(self._ricci_curvatures,
                                                        self._fat_edges)
                   for i in fat_edge)

    def calc(self):
        self._laplacian_forward.calc()
        self.cot = self._laplacian_forward.cot

        if self._updates != self._mesh.updates():
            self._updates = self._mesh.updates()
            self._v = self._mesh.get_vertices()

            self.kappa = self._calc_kappa()
            self.L_curvature = self._calc_L_curvature()

class Reverse:
    '''
    Implementation of the gradient of the curvature loss function on a mesh.
    This implementation assumes the l-th partial affects only the l-th vertex.
    '''

    def __init__(self, mesh, network_vertices, network_edges, ricci_curvatures,
                 epsilon, laplacian_forward=None, curvature_forward = None,
                 laplacian_reverse=None):
        self._mesh = mesh
        self._updates = self._mesh.updates() - 1
        self._v = None
        self._e = self._mesh.get_edges()
        self._c = self._mesh.get_c()

        self._V = len(self._e)

        self._network_vertices = network_vertices
        self._network_edges = network_edges
        self._ricci_curvatures = ricci_curvatures
        self._fat_edges = mesh.get_fat_edges(network_vertices, network_edges,
                                             epsilon)

        self._dif_v = None
        self._l = None

        self._laplacian_forward = laplacian_forward
        if self._laplacian_forward is None:
            self._laplacian_forward = laplacian.Forward(mesh)

        self._curvature_forward = curvature_forward
        if self._curvature_forward is None:
            self._curvature_forward = Forward(mesh)

        self._laplacian_reverse = laplacian_reverse
        if self._laplacian_reverse is None:
            self._laplacian_reverse = laplacian.Reverse(mesh)

        self._cot = None
        self._kappa = None

        self._dif_cot = None

        # Derivatives are stored as maps sending l to the partial with respect
        # to rho_l. The types of the outputs of the maps match the types of
        # what are being differentiated.
        self.dif_kappa = None
        self.dif_L_curvature = None

    def _calc_dif_kappa(self):
        dif_kappa = np.zeros(self._V)
        for (i, j), cot_ij in self._cot.items():
            if (i, j) in self._dif_cot:
                dif_kappa[self._c[i,j]] += self._dif_cot[i,j] / (1 + cot_ij**2)
        return dif_kappa

    def _calc_dif_L_curvature(self):
        return sum(2 * (self._kappa[i] - ricci_curvature) * self.dif_kappa[i]
                   for ricci_curvature, fat_edge in zip(self._ricci_curvatures,
                                                        self._fat_edges)
                   for i in fat_edge)

    def calc(self, dif_v, l):
        self._laplacian_forward.calc()
        self._cot = self._laplacian_forward.cot

        self._curvature_forward.calc()
        self._kappa = self._curvature_forward.kappa

        self._laplacian_reverse.calc(dif_v, l)
        self._dif_cot = self._laplacian_reverse.dif_cot

        if self._updates != self._mesh.updates() or self._l != l:
            self._updates = self._mesh.updates()
            self._v = self._mesh.get_vertices()
            self._dif_v = dif_v
            self._l = l

            self.dif_kappa = self._calc_dif_kappa()
            self.dif_L_curvature = self._calc_dif_L_curvature()
