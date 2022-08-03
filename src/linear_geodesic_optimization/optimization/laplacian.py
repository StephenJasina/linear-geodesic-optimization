import numpy as np
from scipy import linalg
from scipy import sparse

class Forward:
    '''
    Implementation of the Laplace-Beltrami operator on a mesh.
    '''

    def __init__(self, mesh):
        self._mesh = mesh
        self._updates = self._mesh.updates() - 1
        self._v = None
        self._e = self._mesh.get_edges()
        self._c = self._mesh.get_c()

        self._V = len(self._e)

        # A map (i, j) -> N_ij
        self.N = None

        # A map (i, j) -> A_ij
        self.A = None

        # A sparse matrix D
        self.D = None

        # A map (i, j) -> cot(theta_ij)
        self.cot = None

        # A sparse matrix L_C
        self.LC = None

        # A sparse matrix L
        self.L = None

    def _calc_N(self):
        v = self._v
        e = self._e
        c = self._c
        return {(i, j): np.cross(v[i] - v[c[i,j]], v[j] - v[c[i,j]])
                for i, es in enumerate(e)
                for j in es}

    def _calc_A(self):
        return {(i, j): linalg.norm(N) / 2
                for (i, j), N in self.N.items()}

    def _calc_D(self):
        e = self._e
        A = self.A
        return sparse.diags([sum(A[i,j] for j in e) / 3
                             for i, e in enumerate(e)])

    def _calc_cot(self):
        v = self._v
        e = self._e
        c = self._c
        A = self.A
        return {(i, j): (v[i] - v[c[i,j]]) @ (v[j] - v[c[i,j]]) / (2 * A[i,j])
                for i, es in enumerate(e)
                for j in es}

    def _calc_LC(self):
        row = []
        col = []
        data = []
        for (i, j), cot_ij in self.cot.items():
            half_cot_ij = cot_ij / 2

            row.append(i)
            col.append(j)
            data.append(half_cot_ij)

            row.append(j)
            col.append(i)
            data.append(half_cot_ij)

            row.append(i)
            col.append(i)
            data.append(-half_cot_ij)

            row.append(j)
            col.append(j)
            data.append(-half_cot_ij)
        return sparse.coo_array((data, (row, col)),
                                 shape=(self._V, self._V)).tocsc()

    def _calc_L(self):
        return self.D_inv @ self.LC

    def calc(self):
        if self._updates != self._mesh.updates():
            self._updates = self._mesh.updates()
            self._v = self._mesh.get_vertices()

            self.N = self._calc_N()
            self.A = self._calc_A()
            self.D = self._calc_D()
            self.D_inv = sparse.diags(1 / self.D.data.flatten())
            self.cot = self._calc_cot()
            self.LC = self._calc_LC()
            self.L = self._calc_L()

class Reverse:
    '''
    Implementation of the gradient of the Laplace-Beltrami operator on a mesh.
    This implementation assumes the l-th partial affects only the l-th vertex.
    '''

    def __init__(self, mesh, laplacian_forward=None):
        self._mesh = mesh
        self._updates = self._mesh.updates() - 1
        self._v = None
        self._e = self._mesh.get_edges()
        self._c = self._mesh.get_c()

        self._V = len(self._e)

        self._dif_v = None
        self._ls = None

        self._laplacian_forward = laplacian_forward
        if self._laplacian_forward is None:
            self._laplacian_forward = Forward(mesh)

        self._N = None
        self._A = None
        self._D = None
        self._cot = None
        self._LC = None
        self._L = None

        # Derivatives are stored as maps sending l to the partial with respect
        # to rho_l. The types of the outputs of the maps match the types of
        # what are being differentiated.
        self.dif_N = None
        self.dif_A = None
        self.dif_D = None
        self.dif_cot = None
        self.dif_LC = None
        self.dif_L = None

    def _calc_dif_N(self, l):
        dif_N = {}
        v = self._v
        e = self._e
        c = self._c
        dif_v = self._dif_v[l]
        for i, es in enumerate(e):
            vi = v[i]
            for j in es:
                k = c[i,j]
                vj = v[j]
                vk = v[k]
                if l == i:
                    dif_N[i,j] = np.cross(vk - vj, dif_v)
                elif l == j:
                    dif_N[i,j] = np.cross(vi - vk, dif_v)
                elif l == k:
                    dif_N[i,j] = np.cross(vj - vi, dif_v)

                # For efficiency, only store the nonzero values
        return dif_N

    def _calc_dif_A(self, l):
        e = self._e
        N = self._N
        A = self._A
        dif_N = self.dif_N[l]
        return {(i, j): (N[i,j] @ dif_N[i,j]) / (4 * A[i,j])
                for i, es in enumerate(e)
                for j in es
                if (i, j) in dif_N}

    def _calc_dif_D(self, l):
        e = self._e
        dif_A = self.dif_A[l]
        return sparse.diags([sum(dif_A[i,j] for j in es if (i, j) in dif_A) / 3
                             for i, es in enumerate(e)])

    def _calc_dif_cot(self, l):
        dif_cot = {}
        v = self._v
        e = self._e
        c = self._c
        dif_v = self._dif_v[l]
        dif_A = self.dif_A[l]
        vl = v[l]
        for j in e[l]:
            k = c[l,j]
            vj = v[j]
            vk = v[k]
            dif_cot[l,j] = (((vj - vk) @ dif_v
                             - 2 * self._cot[l,j] * dif_A[l,j])
                            / (2 * self._A[l,j]))
            dif_cot[k,l] = (((vk - vj) @ dif_v
                             - 2 * self._cot[k,l] * dif_A[k,l])
                            / (2 * self._A[k,l]))
            dif_cot[j,k] = (((2 * vl - vj - vk) @ dif_v
                             - 2 * self._cot[j,k] * dif_A[j,k])
                            / (2 * self._A[j,k]))
        return dif_cot

    def _calc_dif_LC(self, l):
        dif_cot = self.dif_cot[l]
        row = []
        col = []
        data = []
        for (i, j), dif_cot_ij in dif_cot.items():
            half_dif_cot_ij = dif_cot_ij / 2

            row.append(i)
            col.append(j)
            data.append(half_dif_cot_ij)

            row.append(j)
            col.append(i)
            data.append(half_dif_cot_ij)

            row.append(i)
            col.append(i)
            data.append(-half_dif_cot_ij)

            row.append(j)
            col.append(j)
            data.append(-half_dif_cot_ij)
        return sparse.coo_array((data, (row, col)),
                                 shape=(self._V, self._V)).tocsc()

    def _calc_dif_L(self, l):
        return self._D_inv @ (self.dif_LC[l] - self.dif_D[l] @ self._L)

    def calc(self, dif_v, ls=None):
        if self._ls is None:
            self._ls = range(self._V)
        if ls is None:
            ls = range(self._V)

        self._laplacian_forward.calc()
        self._N = self._laplacian_forward.N
        self._A = self._laplacian_forward.A
        self._D = self._laplacian_forward.D
        self._D_inv = self._laplacian_forward.D_inv
        self._cot = self._laplacian_forward.cot
        self._LC = self._laplacian_forward.LC
        self._L = self._laplacian_forward.L

        if self._updates != self._mesh.updates() or self._ls != ls:
            self._updates = self._mesh.updates()
            self._v = self._mesh.get_vertices()
            self._dif_v = dif_v
            self._ls = ls

            self.dif_N = {l: self._calc_dif_N(l) for l in self._ls}
            self.dif_A = {l: self._calc_dif_A(l) for l in self._ls}
            self.dif_D = {l: self._calc_dif_D(l) for l in self._ls}
            self.dif_cot = {l: self._calc_dif_cot(l) for l in self._ls}
            self.dif_LC = {l: self._calc_dif_LC(l) for l in self._ls}
            self.dif_L = {l: self._calc_dif_L(l) for l in self._ls}
