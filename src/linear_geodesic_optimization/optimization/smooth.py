from linear_geodesic_optimization.optimization import laplacian

class Forward:
    def __init__(self, mesh, laplacian_forward=None):
        self._mesh = mesh
        self._parameters = None

        self._updates = self._mesh.updates() - 1

        self._laplacian_forward = laplacian_forward
        if self._laplacian_forward is None:
            self._laplacian_forward = laplacian.Forward(mesh)

        self.LC = None
        self.L_smooth = None

    def calc(self):
        self._laplacian_forward.calc()
        self.LC = self._laplacian_forward.LC_neumann

        if self._updates != self._mesh.updates():
            self._updates = self._mesh.updates()
            self._parameters = self._mesh.get_parameters()

            self.L_smooth = -self._parameters.T @ (self.LC @ self._parameters)

class Reverse:
    def __init__(self, mesh, laplacian_forward=None, laplacian_reverse=None):
        self._mesh = mesh
        self._updates = self._mesh.updates() - 1
        self._parameters = None

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
        self._LC = self._laplacian_forward.LC_neumann

        self._laplacian_reverse.calc(dif_v, l)
        self.dif_LC = self._laplacian_reverse.dif_LC_neumann

        if self._updates != self._mesh.updates() or self._l != l:
            self._updates = self._mesh.updates()
            self._parameters = self._mesh.get_parameters()
            self._dif_v = dif_v
            self._l = l

            L_parameters = self._LC @ self._parameters + self._LC.T @ self._parameters
            self.dif_L_smooth = -L_parameters[l] - self._parameters @ (self.dif_LC @ self._parameters)
