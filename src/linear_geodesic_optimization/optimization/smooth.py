from linear_geodesic_optimization.optimization import laplacian, curvature

class Forward:
    def __init__(self, mesh, network_vertices, network_edges, ricci_curvatures,
                 epsilon, laplacian_forward=None, curvature_forward=None):
        self._mesh = mesh

        self._updates = self._mesh.updates() - 1

        self._laplacian_forward = laplacian_forward
        if self._laplacian_forward is None:
            self._laplacian_forward = laplacian.Forward(mesh)

        self._curvature_forward = curvature_forward
        if self._curvature_forward is None:
            self._curvature_forward = curvature.Forward(
                mesh, network_vertices, network_edges, ricci_curvatures,
                epsilon, laplacian_forward
            )

        self.LC = None
        self.kappa = None
        self.L_smooth = None

    def calc(self):
        self._laplacian_forward.calc()
        self.LC = self._laplacian_forward.LC_dirichlet

        self._curvature_forward.calc()
        self.kappa = self._curvature_forward.kappa

        if self._updates != self._mesh.updates():
            self._updates = self._mesh.updates()

            self.L_smooth = (
                -self.kappa.T @ (self.LC @ self.kappa)
                / self._mesh.get_support_area()
            )

class Reverse:
    def __init__(self, mesh, network_vertices, network_edges, ricci_curvatures,
                 epsilon, laplacian_forward=None, curvature_forward=None,
                 laplacian_reverse=None, curvature_reverse=None):
        self._mesh = mesh
        self._updates = self._mesh.updates() - 1

        self._dif_v = None
        self._l = None

        self._laplacian_forward = laplacian_forward
        if self._laplacian_forward is None:
            self._laplacian_forward = laplacian.Forward(mesh)

        self._laplacian_reverse = laplacian_reverse
        if self._laplacian_reverse is None:
            self._laplacian_reverse = laplacian.Reverse(mesh,
                                                        laplacian_forward)

        self._curvature_forward = curvature_forward
        if self._curvature_forward is None:
            self._curvature_forward = curvature.Forward(
                mesh, network_vertices, network_edges, ricci_curvatures,
                epsilon, laplacian_forward
            )

        self._curvature_reverse = curvature_reverse
        if self._curvature_reverse is None:
            self._curvature_reverse = curvature.Reverse(
                mesh, network_vertices, network_edges, ricci_curvatures,
                epsilon, laplacian_forward, curvature_forward,
                laplacian_reverse
            )

        self._LC = None
        self._kappa = None

        self.dif_LC = None
        self.dif_kappa = None

        self.dif_L_smooth = None

    def calc(self, dif_v, l):
        self._laplacian_forward.calc()
        self._LC = self._laplacian_forward.LC_dirichlet

        self._curvature_forward.calc()
        self._kappa = self._curvature_forward.kappa

        self._laplacian_reverse.calc(dif_v, l)
        self.dif_LC = self._laplacian_reverse.dif_LC_neumann

        self._curvature_reverse.calc(dif_v, l)
        self.dif_kappa = self._curvature_reverse.dif_kappa

        if self._updates != self._mesh.updates() or self._l != l:
            self._updates = self._mesh.updates()
            self._dif_v = dif_v
            self._l = l

            self.dif_L_smooth =  -(
                self.dif_kappa.T @ (self._LC @ self._kappa)
                + self._kappa.T @ (self.dif_LC @ self._kappa)
                + self._kappa.T @ (self._LC @ self.dif_kappa)
            )
