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
        self.LC_neumann = None

        # A sparse matrix L_C, which has different boundary conditions from the
        # one above
        self.LC_dirichlet = None

    def _calc_N(self):
        v = self._v
        e = self._e
        c = self._c
        return {(i, j): np.cross(v[i] - v[c[i,j]], v[j] - v[c[i,j]])
                for i, es in enumerate(e)
                for j in es}

    def _calc_A(self):
        return {(i, j): linalg.norm(N) / 2.
                for (i, j), N in self.N.items()}

    def _calc_D(self):
        e = self._e
        A = self.A
        return sparse.diags([sum(A[i,j] for j in e) / 3.
                             for i, e in enumerate(e)])

    def _calc_cot(self):
        v = self._v
        e = self._e
        c = self._c
        A = self.A
        return {(i, j): (v[i] - v[c[i,j]]) @ (v[j] - v[c[i,j]]) / (2. * A[i,j])
                for i, es in enumerate(e)
                for j in es}

    def _calc_LC(self, neumann=True):
        row = []
        col = []
        data = []
        for (i, j), cot_ij in self.cot.items():
            # Ignore the boundary in the Dirichlet boundary case
            if not neumann and (j, i) not in self.cot:
                continue

            half_cot_ij = cot_ij / 2.

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
                                 shape=(self._V, self._V)).tocsr()

    def _calc_L(self):
        return self.D_inv @ self.LC_neumann

    def calc(self):
        if self._updates != self._mesh.updates():
            self._updates = self._mesh.updates()
            self._v = self._mesh.get_vertices()

            self.N = self._calc_N()
            self.A = self._calc_A()
            self.D = self._calc_D()
            self.D_inv = sparse.diags(1. / self.D.data.flatten())
            self.cot = self._calc_cot()
            self.LC_neumann = self._calc_LC(True)
            if self._mesh.get_boundary_vertices():
                self.LC_dirichlet = self._calc_LC(False)
            else:
                self.LC_dirichlet = None

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
        self._l = None

        self._laplacian_forward = laplacian_forward
        if self._laplacian_forward is None:
            self._laplacian_forward = Forward(mesh)

        self._N = None
        self._A = None
        self._D = None
        self._cot = None
        self._LC_neumann = None
        self._LC_dirichlet = None

        # Derivatives are stored as maps sending l to the partial with respect
        # to rho_l. The types of the outputs of the maps match the types of
        # what are being differentiated.
        self.dif_N = None
        self.dif_A = None
        self.dif_D = None
        self.dif_cot = None
        self.dif_LC_neumann = None
        self.dif_LC_dirichlet = None
        self.dif_L = None

    def _calc_dif_N(self):
        dif_N = {}
        v = self._v
        e = self._e
        c = self._c
        l = self._l
        dif_v = self._dif_v
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

    def _calc_dif_A(self):
        e = self._e
        N = self._N
        A = self._A
        dif_N = self.dif_N
        return {(i, j): (N[i,j] @ dif_N[i,j]) / (4. * A[i,j])
                for i, es in enumerate(e)
                for j in es
                if (i, j) in dif_N}

    def _calc_dif_D(self):
        e = self._e
        dif_A = self.dif_A
        return sparse.diags([sum(dif_A[i,j]
                                 for j in es if (i, j) in dif_A) / 3.
                             for i, es in enumerate(e)])

    def _calc_dif_cot(self):
        dif_cot = {}
        v = self._v
        e = self._e
        c = self._c
        l = self._l
        dif_v = self._dif_v
        dif_A = self.dif_A
        vl = v[l]
        for j in e[l]:
            k = c[l,j]
            vj = v[j]
            vk = v[k]
            dif_cot[l,j] = (((vj - vk) @ dif_v
                             - 2. * self._cot[l,j] * dif_A[l,j])
                            / (2. * self._A[l,j]))
            dif_cot[k,l] = (((vk - vj) @ dif_v
                             - 2. * self._cot[k,l] * dif_A[k,l])
                            / (2. * self._A[k,l]))
            dif_cot[j,k] = (((2 * vl - vj - vk) @ dif_v
                             - 2. * self._cot[j,k] * dif_A[j,k])
                            / (2. * self._A[j,k]))
        return dif_cot

    def _calc_dif_LC(self, neumann=True):
        dif_cot = self.dif_cot
        row = []
        col = []
        data = []
        for (i, j), dif_cot_ij in dif_cot.items():
            # Ignore the boundary in the Dirichlet boundary case
            if not neumann and (j, i) not in self.dif_cot:
                continue

            half_dif_cot_ij = dif_cot_ij / 2.

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

    def calc(self, dif_v, l):
        self._laplacian_forward.calc()
        self._N = self._laplacian_forward.N
        self._A = self._laplacian_forward.A
        self._D = self._laplacian_forward.D
        self._D_inv = self._laplacian_forward.D_inv
        self._cot = self._laplacian_forward.cot
        self._LC_neumann = self._laplacian_forward.LC_neumann
        self._LC_dirichlet = self._laplacian_forward.LC_dirichlet

        if self._updates != self._mesh.updates() or self._l != l:
            self._updates = self._mesh.updates()
            self._v = self._mesh.get_vertices()
            self._dif_v = dif_v
            self._l = l

            self.dif_N = self._calc_dif_N()
            self.dif_A = self._calc_dif_A()
            self.dif_D = self._calc_dif_D()
            self.dif_cot = self._calc_dif_cot()
            self.dif_LC_neumann = self._calc_dif_LC(True)
            if self._LC_dirichlet is not None:
                self.dif_LC_dirichlet = self._calc_dif_LC(False)
            else:
                self.dif_LC_dirichlet = None
