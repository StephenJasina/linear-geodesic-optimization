import numpy as np

from linear_geodesic_optimization.optimization import curvature

class Forward:
    '''
    Implementation of the scalar approximation loss function. This, in
    particular, is used for curvature loss.
    '''

    def __init__(self, mesh, network_vertices, network_edges, ricci_curvatures,
                 epsilon, curvature_forward=None):
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

        self._curvature_forward = curvature_forward
        if self._curvature_forward is None:
            self._curvature_forward = curvature.Forward(mesh)

        self.kappa_G = None

        self.L_curvature = None

    def calc(self):
        self._curvature_forward.calc()
        self.kappa_G = self._curvature_forward.kappa_G

        if self._updates != self._mesh.updates():
            self._updates = self._mesh.updates()
            self.L_curvature = (
                sum((self.kappa_G[i] - ricci_curvature)**2
                    for ricci_curvature, fat_edge in zip(
                        self._ricci_curvatures,
                        self._fat_edges
                    )
                    for i in fat_edge) / sum(len(fat_edge)
                                             for fat_edge in self._fat_edges)
                if self._ricci_curvatures else 0
            )

class Reverse:
    '''
    Implementation of the gradient of the curvature loss function on a mesh.
    This implementation assumes the l-th partial affects only the l-th vertex.
    '''

    def __init__(self, mesh, network_vertices, network_edges, ricci_curvatures,
                 epsilon, curvature_forward=None, curvature_reverse=None):
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

        self._curvature_forward = curvature_forward
        if self._curvature_forward is None:
            self._curvature_forward = curvature.Forward(mesh)

        self._curvature_reverse = curvature_reverse
        if self._curvature_reverse is None:
            self._curvature_reverse = curvature.Reverse(
                mesh,
                self._curvature_forward._laplacian_forward,
                self._curvature_forward,
            )

        self.dif_L_curvature = None

    def calc(self, dif_v, l):
        self._curvature_forward.calc()
        self.kappa_G = self._curvature_forward.kappa_G

        self._curvature_reverse.calc(dif_v, l)
        self.dif_kappa_G = self._curvature_reverse.dif_kappa_G

        if self._updates != self._mesh.updates() or self._l != l:
            self._updates = self._mesh.updates()
            self._dif_v = dif_v
            self._l = l
            self.dif_L_curvature = (
                sum(2 * (self.kappa_G[i] - ricci_curvature) * self.dif_kappa_G[i]
                    for ricci_curvature, fat_edge in zip(self._ricci_curvatures,
                                                         self._fat_edges)
                    for i in fat_edge) / sum(len(fat_edge)
                                             for fat_edge in self._fat_edges)
                if self._ricci_curvatures else 0
            )
