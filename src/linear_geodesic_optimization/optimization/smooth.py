from linear_geodesic_optimization.optimization import laplacian, curvature

class Forward:
    def __init__(self, mesh, laplacian_forward=None, curvature_forward=None,
                 strategy='mvs_cross'):
        self._mesh = mesh

        self._updates = self._mesh.updates() - 1

        self._laplacian_forward = laplacian_forward
        if self._laplacian_forward is None:
            self._laplacian_forward = laplacian.Forward(mesh)

        self._curvature_forward = curvature_forward
        if self._curvature_forward is None:
            self._curvature_forward = curvature.Forward(
                mesh, laplacian_forward
            )

        self.strategy = strategy

        self.LC = None
        self.kappa_G = None
        self.kappa_H = None
        self.kappa_1 = None
        self.kappa_2 = None
        self.L_smooth = None

    def calc(self):
        self._laplacian_forward.calc()
        self.LC = self._laplacian_forward.LC_dirichlet

        self._curvature_forward.calc()
        self.kappa_G = self._curvature_forward.kappa_G
        self.kappa_H = self._curvature_forward.kappa_H
        self.kappa_1 = self._curvature_forward.kappa_1
        self.kappa_2 = self._curvature_forward.kappa_2

        if self._updates != self._mesh.updates():
            self._updates = self._mesh.updates()

            if self.strategy == 'gaussian':
                print('g')
                self.L_smooth = -self.kappa_G.T @ (self.LC @ self.kappa_G) \
                    / self._mesh.get_support_area()
            elif self.strategy == 'mean':
                print('h')
                self.L_smooth = -self.kappa_H.T @ (self.LC @ self.kappa_H) \
                    / self._mesh.get_support_area()
            else:
                print('m')
                self.L_smooth = (
                    -(self.kappa_1.T @ (self.LC @ self.kappa_1)
                        + self.kappa_2.T @ (self.LC @ self.kappa_2))
                    / self._mesh.get_support_area()
                )

class Reverse:
    def __init__(self, mesh, laplacian_forward=None, curvature_forward=None,
                 laplacian_reverse=None, curvature_reverse=None,
                 strategy='mvs_cross'):
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
                mesh, laplacian_forward
            )

        self._curvature_reverse = curvature_reverse
        if self._curvature_reverse is None:
            self._curvature_reverse = curvature.Reverse(
                mesh, laplacian_forward, curvature_forward, laplacian_reverse
            )

        self.strategy = strategy

        self._LC = None
        self._kappa_G = None
        self._kappa_H = None
        self._kappa_1 = None
        self._kappa_2 = None

        self.dif_LC = None
        self.dif_kappa_G = None
        self.dif_kappa_H = None
        self.dif_kappa_1 = None
        self.dif_kappa_2 = None

        self.dif_L_smooth = None

    def calc(self, dif_v, l):
        self._laplacian_forward.calc()
        self._LC = self._laplacian_forward.LC_dirichlet

        self._curvature_forward.calc()
        self._kappa_G = self._curvature_forward.kappa_G
        self._kappa_H = self._curvature_forward.kappa_H
        self._kappa_1 = self._curvature_forward.kappa_1
        self._kappa_2 = self._curvature_forward.kappa_2

        self._laplacian_reverse.calc(dif_v, l)
        self.dif_LC = self._laplacian_reverse.dif_LC_dirichlet

        self._curvature_reverse.calc(dif_v, l)
        self.dif_kappa_G = self._curvature_reverse.dif_kappa_G
        self.dif_kappa_H = self._curvature_reverse.dif_kappa_H
        self.dif_kappa_1 = self._curvature_reverse.dif_kappa_1
        self.dif_kappa_2 = self._curvature_reverse.dif_kappa_2

        if self._updates != self._mesh.updates() or self._l != l:
            self._updates = self._mesh.updates()
            self._dif_v = dif_v
            self._l = l

            if self.strategy == 'gaussian':
                self.dif_L_smooth = -(
                    self.dif_kappa_G.T @ (self._LC @ self._kappa_G)
                    + self._kappa_G.T @ (self.dif_LC @ self._kappa_G)
                    + self._kappa_G.T @ (self._LC @ self.dif_kappa_G)
                ) / self._mesh.get_support_area()
            elif self.strategy == 'mean':
                self.dif_L_smooth = -(
                    self.dif_kappa_H.T @ (self._LC @ self._kappa_H)
                    + self._kappa_H.T @ (self.dif_LC @ self._kappa_H)
                    + self._kappa_H.T @ (self._LC @ self.dif_kappa_H)
                ) / self._mesh.get_support_area()
            else:
                self.dif_L_smooth = -(
                    self.dif_kappa_1.T @ (self._LC @ self._kappa_1)
                    + self._kappa_1.T @ (self.dif_LC @ self._kappa_1)
                    + self._kappa_1.T @ (self._LC @ self.dif_kappa_1)
                    + self.dif_kappa_2.T @ (self._LC @ self._kappa_2)
                    + self._kappa_2.T @ (self.dif_LC @ self._kappa_2)
                    + self._kappa_2.T @ (self._LC @ self.dif_kappa_2)
                ) / self._mesh.get_support_area()
