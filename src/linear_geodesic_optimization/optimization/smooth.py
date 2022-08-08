from linear_geodesic_optimization.optimization import laplacian

class Forward:
    def __init__(self, mesh, laplacian_forward=None):
        self._mesh = mesh
        self._rho = None

        self._updates = self._mesh.updates() - 1

        self._laplacian_forward = laplacian_forward
        if self._laplacian_forward is None:
            self._laplacian_forward = laplacian.Forward(mesh)

        self.LC = None
        self.L_smooth = None

    def calc(self):
        self._laplacian_forward.calc()
        self.LC = self._laplacian_forward.LC

        if self._updates != self._mesh.updates():
            self._updates = self._mesh.updates()
            self._rho = self._mesh.get_rho()

            self.L_smooth = -self._rho.T @ (self.LC @ self._rho)

class Reverse:
    def __init__(self, mesh, laplacian_forward=None, laplacian_reverse=None):
        self._mesh = mesh
        self._updates = self._mesh.updates() - 1
        self._rho = None

        self._dif_v = None
        self._l = None

        self._laplacian_forward = laplacian_forward
        if self._laplacian_forward is None:
            self._laplacian_forward = laplacian.Forward(mesh)

        self._laplacian_reverse = laplacian_reverse
        if self._laplacian_reverse is None:
            self._laplacian_reverse = laplacian.Reverse(mesh, laplacian_forward)

        self._LC = None
        self.dif_LC = None

        self.dif_L_smooth = None

    def calc(self, dif_v, l):
        self._laplacian_forward.calc()
        self._LC = self._laplacian_forward.LC

        self._laplacian_reverse.calc(dif_v, l)
        self.dif_LC = self._laplacian_reverse.dif_LC

        if self._updates != self._mesh.updates() or self._l != l:
            self._updates = self._mesh.updates()
            self._rho = self._mesh.get_rho()
            self._dif_v = dif_v
            self._l = l

            L_rho = self._LC @ self._rho + self._LC.T @ self._rho
            self.dif_L_smooth = -L_rho[l] - self._rho @ (self.dif_LC @ self._rho)