from . import laplacian

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

    def calc_L_smooth(self):
        self._laplacian_forward.calc_L()
        self.LC = self._laplacian_forward.LC

        if self._updates != self._mesh.updates():
            self._updates = self._mesh.updates()
            self._rho = self._mesh.get_rho()

            self.L_smooth = -self._rho.T @ (self.LC @ self._rho)

        return self.L_smooth

class Reverse:
    def __init__(self, mesh, laplacian_forward=None, laplacian_reverse=None):
        self._mesh = mesh
        self._updates = self._mesh.updates() - 1
        self._rho = None

        self._dif_v = None
        self._ls = None

        self._laplacian_forward = laplacian_forward
        if self._laplacian_forward is None:
            self._laplacian_forward = laplacian.Forward(mesh)

        self._laplacian_reverse = laplacian_reverse
        if self._laplacian_reverse is None:
            self._laplacian_reverse = laplacian.Reverse(mesh, laplacian_forward)

        self._LC = None
        self.dif_LC = None

        self.dif_L_smooth = None

    def calc_dif_L_smooth(self, dif_v, ls=None):
        if self._ls is None:
            self._ls = dif_v.keys()
        if ls is None:
            ls = dif_v.keys()

        self._laplacian_forward.calc_L()
        self._LC = self._laplacian_forward.LC

        self._laplacian_reverse.calc_dif_L(dif_v, ls)
        self.dif_LC = self._laplacian_reverse.dif_LC

        if self._updates != self._mesh.updates() or self._ls != ls:
            self._updates = self._mesh.updates()
            self._rho = self._mesh.get_rho()
            self._dif_v = dif_v
            self._ls = ls

            L_rho = self._LC @ self._rho + self._LC.T @ self._rho
            self.dif_L_smooth = {l: -L_rho[l]
                                    - self._rho @ (self.dif_LC[l] @ self._rho)
                                 for l in self._ls}

        return self.dif_L_smooth