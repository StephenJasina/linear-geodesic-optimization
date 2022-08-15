import numpy as np
from scipy import linalg
from scipy import sparse
from scipy.sparse.linalg import eigsh, splu

from linear_geodesic_optimization.optimization import laplacian

class Forward:
    '''
    Implementation of the heat method for geodesic distance computation.
    '''

    def __init__(self, mesh, laplacian_forward=None):
        '''
        Quantities that can be naturally realized as matrices are stored as such
        (sometimes sparsely if reasonable). All other quantities have a one to
        one correspondence to pairs (v, f), where f is a face containing the
        vertex v. In other words, they are in correspondence to
        `mesh.get_all_faces()`, and are thus stored in the same format. If we
        think of the former of a map from vertices to lists of faces, then the
        latter is a map from vertices to lists of quantities. For efficiency
        reasons, these maps are stored as `list`s.
        '''

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
        self.A = None
        self.D = None
        self.cot = None
        self.LC_neumann = None
        self.LC_dirichlet = None

        # A float
        self.h2 = None

        # Objects with a `.solve` method that takes a vector as input and
        # returns a vector
        self.D_h2LC_neumann_inv = None
        self.D_h2LC_dirichlet_inv = None

        # An object with a `.solve` method that takes a vector as input and
        # returns a vector
        self.LC_neumann_inv = None

        # An iterable containing indices
        self._gamma = None

        # A vector
        self.u = None

        # A vector
        self.q = None

        # A map (i, j) -> m_ij
        self.m = None

        # A map (i, j) -> u_ij
        self.grad_u = None

        # A map (i, j) -> X_ij
        self.X = None

        # A map (i, j) -> p_ij
        self.p = None

        # A matrix
        self.div_X = None

        # A matrix
        self.phi = None

    def _calc_h2(self):
        v = self._v
        e = self._e
        return np.mean([linalg.norm(v[i] - v[j])
                        for i, es in enumerate(e)
                        for j in es])**2

    def _calc_D_h2LC_neumann_inv(self):
        return splu(self.D.tocsc() - self.h2 * self.LC_neumann)

    def _calc_D_h2LC_dirichlet_inv(self):
        return splu(self.D.tocsc() - self.h2 * self.LC_dirichlet)

    def _calc_LC_neumann_inv(self):
        # Need to add a small offset to guarantee that the inverse exists.
        # Since L_C is negative semidefinite, subracting off a small positive
        # multiple of the identity guarantees that the resulting matrix is
        # invertible. For generality's sake, we pick the magnitude relative to
        # the largest eigenvalue of L_C.
        offset_magnitude = eigsh(self.LC_neumann, k=1,
                                 return_eigenvectors=False)[0] * 1.e-10
        return splu(self.LC_neumann - sparse.eye(self._V) * offset_magnitude)

    def _calc_u(self):
        delta = np.zeros((self._V, 1))
        delta[self._gamma] = 1.
        if self.D_h2LC_dirichlet_inv is None:
            return self.D_h2LC_neumann_inv.solve(delta)
        else:
            return (self.D_h2LC_neumann_inv.solve(delta)
                    + self.D_h2LC_dirichlet_inv.solve(delta)) / 2.

    def _calc_q(self):
        v = self._v
        e = self._e
        c = self._c
        u = self.u
        return {(i, j): u[i] * (v[c[i,j]] - v[j])
                for i, es in enumerate(e)
                for j in es}

    def _calc_m(self):
        e = self._e
        c = self._c
        q = self.q
        return {(i, j): q[i,j] + q[j,c[i,j]] + q[c[i,j],i]
                for i, es in enumerate(e)
                for j in es}

    def _calc_grad_u(self):
        e = self._e
        N = self.N
        m = self.m
        return {(i, j): np.cross(N[i,j], m[i,j])
                for i, es in enumerate(e)
                for j in es}

    def _calc_X(self):
        return {(i, j): -grad_u / linalg.norm(grad_u)
                for (i, j), grad_u in self.grad_u.items()}

    def _calc_p(self):
        v = self._v
        e = self._e
        cot = self.cot
        return {(i, j): cot[i,j] * (v[j] -v[i])
                for i, es in enumerate(e)
                for j in es}

    def _calc_div_X(self):
        e = self._e
        c = self._c
        X = self.X
        p = self.p
        return np.array([sum([(p[i,j] - p[c[i,j],i]) @ X[i,j]
                              for j in es]) / 2
                         for i, es in enumerate(e)])

    def _calc_phi(self):
        phi = self.LC_neumann_inv.solve(self.div_X)
        # We subtract off the minimum here so that we satisfy the obvious
        # initial conidition (namely, the distance from gamma to itself
        # should be 0)
        return phi - min(phi)

    def calc(self, gamma):
        self._laplacian_forward.calc()
        self.N = self._laplacian_forward.N
        self.A = self._laplacian_forward.A
        self.D = self._laplacian_forward.D
        self.cot = self._laplacian_forward.cot
        self.LC_neumann = self._laplacian_forward.LC_neumann
        self.LC_dirichlet = self._laplacian_forward.LC_dirichlet

        updated = False
        if self._updates != self._mesh.updates():
            updated = True
            self._updates = self._mesh.updates()
            self._v = self._mesh.get_vertices()

            self.h2 = self._calc_h2()
            self.D_h2LC_neumann_inv = self._calc_D_h2LC_neumann_inv()
            if self.LC_dirichlet is not None:
                self.D_h2LC_dirichlet_inv = self._calc_D_h2LC_dirichlet_inv()
            else:
                self.D_h2LC_dirichlet_inv = None
            self.LC_neumann_inv = self._calc_LC_neumann_inv()

        if (updated or self._updates != self._mesh.updates()
            or self._gamma != gamma):
            if not updated:
                self._updates = self._mesh.updates()
                self._v = self._mesh.get_vertices()
            self._gamma = gamma

            self.u = self._calc_u()
            self.q = self._calc_q()
            self.m = self._calc_m()
            self.grad_u = self._calc_grad_u()
            self.X = self._calc_X()
            self.p = self._calc_p()
            self.div_X = self._calc_div_X()
            self.phi = self._calc_phi()

class Reverse:
    '''
    Implementation of the gradient of the heat method for geodesic distance
    computation. Objects of this class must be initialized with the indices of
    the vertics with respect to which we are differentiating (`l`) as well as
    the partial derivatives of v with repect to rho_l (stored in the same shape
    as `mesh.get_vertices()`).
    '''

    def __init__(self, mesh, geodesic_forward=None, laplacian_reverse=None):
        self._mesh = mesh
        self._updates = self._mesh.updates() - 1
        self._v = None
        self._e = self._mesh.get_edges()
        self._c = self._mesh.get_c()

        self._V = len(self._e)

        self._dif_v = None
        self._l = None

        self._geodesic_forward = geodesic_forward
        if self._geodesic_forward is None:
            self._geodesic_forward = Forward(mesh)

        self._laplacian_reverse = laplacian_reverse
        if self._laplacian_reverse is None:
            self._laplacian_reverse = laplacian.Reverse(mesh)

        self._N = None
        self._A = None
        self._D = None
        self._cot = None
        self._LC_neumann = None
        self._LC_dirichlet = None
        self._h2 = None
        self._D_h2LC_neumann_inv = None
        self._D_h2LC_dirichlet_inv = None
        self._LC_neumann_inv = None

        self._dif_v = None

        self._dif_N = None
        self._dif_A = None
        self._dif_D = None
        self._dif_cot = None
        self._dif_LC_neumann = None
        self._dif_LC_dirichlet = None

        self._gamma = None
        self._u = None
        self._q = None
        self._m = None
        self._grad_u = None
        self._X = None
        self._p = None
        self._div_X = None
        self._phi = None

        # Derivatives are stored as maps sending l to the partial with respect
        # to rho_l. The types of the outputs of the maps match the types of
        # what are being differentiated.
        self.dif_u = None
        self.dif_grad_u = None
        self.dif_X = None
        self.dif_p = None
        self.dif_div_X = None
        self.dif_phi = None

    def _calc_dif_u(self):
        if self._D_h2LC_dirichlet_inv is None:
            return -self._D_h2LC_neumann_inv.solve(
                (self._dif_D.tocsc() - self._h2 * self._dif_LC_neumann) @ self._u)
        else:
            return (
                -self._D_h2LC_neumann_inv.solve(
                    (self._dif_D.tocsc() - self._h2 * self._dif_LC_neumann) @ self._u
                ) - self._D_h2LC_dirichlet_inv.solve(
                    (self._dif_D.tocsc() - self._h2 * self._dif_LC_neumann) @ self._u
                )) / 2.

    def _calc_dif_q(self):
        dif_q = {}
        v = self._v
        e = self._e
        c = self._c
        l = self._l
        dif_v = self._dif_v
        u = self._u
        dif_u = self.dif_u
        for i, es in enumerate(e):
            for j in es:
                k = c[i,j]
                dif_q[i,j] = dif_u[i] * (v[k] - v[j])
                if l == j:
                    dif_q[i,j] -= u[i] * dif_v
                elif l == k:
                    dif_q[i,j] += u[i] * dif_v
        return dif_q

    def _calc_dif_m(self):
        e = self._e
        c = self._c
        dif_q = self.dif_q
        return {(i, j): dif_q[i,j] + dif_q[j,c[i,j]] + dif_q[c[i,j],i]
                for i, es in enumerate(e)
                for j in es}

    def _calc_dif_grad_u(self):
        dif_grad_u = {}
        e = self._e
        N = self._N
        m = self._m
        dif_N = self._dif_N
        dif_m = self.dif_m
        for i, es in enumerate(e):
            for j in es:
                dif_grad_u[i,j] = np.cross(N[i,j], dif_m[i,j])
        for (i, j) in dif_N:
            dif_grad_u[i,j] += np.cross(dif_N[i,j], m[i,j])
        return dif_grad_u

    def _calc_dif_X(self):
        e = self._e
        grad_u = self._grad_u
        X = self._X
        dif_grad_u = self.dif_grad_u
        return {(i, j): ((X[i,j] @ dif_grad_u[i,j]) * X[i,j] - dif_grad_u[i,j])
                        / linalg.norm(grad_u[i,j])
                for i, es in enumerate(e)
                for j in es}

    def _calc_dif_p(self):
        dif_p = {}
        v = self._v
        c = self._c
        l = self._l
        dif_v = self._dif_v
        cot = self._cot
        dif_cot = self._dif_cot
        for i, es in enumerate(self._e):
            for j in es:
                if l == i:
                    dif_p[i,j] = dif_cot[i,j] * (v[j] - v[i]) \
                        - cot[i,j] * dif_v
                elif l == j:
                    dif_p[i,j] = dif_cot[i,j] * (v[j] - v[i]) \
                        + cot[i,j] * dif_v
                elif l == c[i,j]:
                    dif_p[i,j] = dif_cot[i,j] * (v[j] - v[i])
        return dif_p

    def _calc_dif_div_X(self):
        dif_div_X = np.zeros(self._V)
        e = self._e
        c = self._c
        X = self._X
        p = self._p
        dif_X = self.dif_X
        dif_p = self.dif_p
        for i, es in enumerate(e):
            for j in es:
                k = c[i,j]
                dpij = dif_p[i,j] if (i, j) in dif_p else np.zeros(3)
                dpki = dif_p[k,i] if (k, i) in dif_p else np.zeros(3)
                dif_div_X[i] += ((dpij - dpki) @ X[i,j]
                                 + (p[i,j] - p[k,i]) @ dif_X[i,j]) / 2
        return dif_div_X

    def _calc_dif_phi(self):
        return self._LC_neumann_inv.solve(self.dif_div_X - self._LC_neumann @ self._phi)

    def calc(self, gamma, dif_v, l):
        self._geodesic_forward.calc(gamma)
        self._N = self._geodesic_forward.N
        self._A = self._geodesic_forward.A
        self._D = self._geodesic_forward.D
        self._cot = self._geodesic_forward.cot
        self._LC_neumann = self._geodesic_forward.LC_neumann
        self._LC_dirichlet = self._geodesic_forward.LC_dirichlet
        self._h2 = self._geodesic_forward.h2
        self._D_h2LC_neumann_inv = self._geodesic_forward.D_h2LC_neumann_inv
        self._D_h2LC_dirichlet_inv = self._geodesic_forward.D_h2LC_dirichlet_inv
        self._LC_neumann_inv = self._geodesic_forward.LC_neumann_inv
        self._u = self._geodesic_forward.u
        self._q = self._geodesic_forward.q
        self._m = self._geodesic_forward.m
        self._grad_u = self._geodesic_forward.grad_u
        self._X = self._geodesic_forward.X
        self._p = self._geodesic_forward.p
        self._div_X = self._geodesic_forward.div_X
        self._phi = self._geodesic_forward.phi

        self._laplacian_reverse.calc(dif_v, l)
        self._dif_N = self._laplacian_reverse.dif_N
        self._dif_A = self._laplacian_reverse.dif_A
        self._dif_D = self._laplacian_reverse.dif_D
        self._dif_cot = self._laplacian_reverse.dif_cot
        self._dif_LC_neumann = self._laplacian_reverse.dif_LC_neumann
        self._dif_LC_dirichlet = self._laplacian_reverse.dif_LC_dirichlet

        if (self._updates != self._mesh.updates() or self._gamma != gamma
            or self._l != l):
            self._updates = self._mesh.updates()
            self._v = self._mesh.get_vertices()
            self._gamma = gamma
            self._dif_v = dif_v
            self._l = l

            self.dif_u = self._calc_dif_u()
            self.dif_q = self._calc_dif_q()
            self.dif_m = self._calc_dif_m()
            self.dif_grad_u = self._calc_dif_grad_u()
            self.dif_X = self._calc_dif_X()
            self.dif_p = self._calc_dif_p()
            self.dif_div_X = self._calc_dif_div_X()
            self.dif_phi = self._calc_dif_phi()
