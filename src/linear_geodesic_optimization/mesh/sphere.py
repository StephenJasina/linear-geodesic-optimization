import numpy as np
from scipy import linalg

from linear_geodesic_optimization.mesh import mesh

class Mesh(mesh.Mesh):
    '''
    Representation of a mesh that is "approximately" spherical. In particular,
    projecting the vertices of the mesh onto the unit sphere will yield a
    geodesic polyhedron with icosahedral symmetry.
    '''

    def __init__(self, frequency):
        self._partials, self._edges, self._faces, self._c \
            = self._initial_mesh(frequency)
        self._rho = np.ones(self._partials.shape[0])
        self._updates = 0

    def _initial_mesh(self, frequency):
        '''
        Compute a geodesic polyhedron with icosahedral symmetry.
        '''

        # Start by defining a simple icosahedron.
        p = (2 / (5 + 5**0.5))**0.5
        q = (2 / (5 - 5**0.5))**0.5
        icosahedron_vertices = np.array([
            [0, p, q], [0, p, -q], [0, -p, q], [0, -p, -q],
            [q, 0, p], [-q, 0, p], [q, 0, -p], [-q, 0, -p],
            [p, q, 0], [p, -q, 0], [-p, q, 0], [-p, -q, 0],
        ])

        # The following generates the edges
        #
        # for i in range(12):
        #     for j in range(i + 1, 12):
        #         vi = icosahedron_vertices[i,:]
        #         vj = icosahedron_vertices[j,:]
        #         if np.isclose(linalg.norm(vi - vj), 2 * p):
        #             print(f'({i}, {j}),')
        icosahedron_edges = [
            (0, 2), (0, 4), (0, 5), (0, 8), (0, 10),
            (1, 3), (1, 6), (1, 7), (1, 8), (1, 10),
            (2, 4), (2, 5), (2, 9), (2, 11), (3, 6),
            (3, 7), (3, 9), (3, 11), (4, 6), (4, 8),
            (4, 9), (5, 7), (5, 10), (5, 11), (6, 8),
            (6, 9), (7, 10), (7, 11), (8, 10), (9, 11),
        ]

        # The following generates the faces (oriented counterclockwise)
        #
        # for i in range(12):
        #     for j in range(i + 1, 12):
        #         for k in range(i + 1, 12):
        #             vi = icosahedron_vertices[i,:]
        #             vj = icosahedron_vertices[j,:]
        #             vk = icosahedron_vertices[k,:]
        #             v_sum = vi + vj + vk
        #             v_cross = np.cross(vj - vi, vk - vi)
        #             if (np.isclose(linalg.norm(vi - vj), 2 * p) and
        #                     np.isclose(linalg.norm(vj - vk), 2 * p) and
        #                     np.isclose(linalg.norm(vk - vi), 2 * p) and
        #                     v_sum @ v_cross > 0):
        #                 print(f'({i}, {j}, {k}),')
        icosahedron_faces = [
            (0, 2, 4), (0, 4, 8), (0, 5, 2), (0, 8, 10), (0, 10, 5),
            (1, 3, 7), (1, 6, 3), (1, 7, 10), (1, 8, 6), (1, 10, 8),
            (2, 5, 11), (2, 9, 4), (2, 11, 9), (3, 6, 9), (3, 9, 11),
            (3, 11, 7), (4, 6, 8), (4, 9, 6), (5, 7, 11), (5, 10, 7),
        ]

        # Now that we have defined the icosahedron, we can subdivide the faces
        # and then project them to the unit sphere.

        # The vertices are organized by
        #     [
        #         icosahedron vertices,
        #         vertices from the icosahedron edges,
        #         vertices from the icosahedron faces,
        #     ].
        vertices = np.zeros((10 * frequency**2 + 2, 3))

        # Edges are organized in an adjacency list fashion.
        edges = [[] for _ in range(vertices.shape[0])]

        # Faces are represented by (i, j, k) tuples, where i -> j -> k is
        # oriented counterclockwise.
        faces = []

        # Vertices from icosahedron vertices
        vertices[:12, :] = icosahedron_vertices
        index = 12

        # Vertices from icosahedron edges
        edge_index_offset = index
        for e in icosahedron_edges:
            v0 = icosahedron_vertices[e[0], :]
            v1 = icosahedron_vertices[e[1], :]
            d = v1 - v0

            for i in range(1, frequency):
                # Recall that interpolation between v0 and v1 can be written as
                # v = v0 + lambda * (v1 - v0) = v0 + lambda * d. Then we just
                # have to normalize v.
                # We use the name `lam` here since `lambda` is reserved.
                lam = i / frequency
                v = v0 + lam * d
                vertices[index, :] = v / linalg.norm(v)
                index += 1

        # Vertices from icosahedron faces. At the same time, keep track of the
        # edges and the faces.
        for f in icosahedron_faces:
            v0 = icosahedron_vertices[f[0], :]
            v1 = icosahedron_vertices[f[1], :]
            d1 = v1 - v0
            v2 = icosahedron_vertices[f[2], :]
            d2 = v2 - v0

            # Vertices
            face_vertices = [[] for _ in range(frequency + 1)]
            face_vertices[0].append(f[0])
            if f[1] > f[0]:
                edge_index = edge_index_offset \
                    + icosahedron_edges.index((f[0], f[1])) * (frequency - 1)
                for i in range(frequency - 1):
                    face_vertices[0].append(edge_index + i)
            else:
                edge_index = edge_index_offset \
                    + icosahedron_edges.index((f[1], f[0])) * (frequency - 1)
                for i in range(frequency - 1):
                    face_vertices[0].append(edge_index + frequency - 2 - i)
            face_vertices[0].append(f[1])
            if f[2] > f[0]:
                edge_index = edge_index_offset \
                    + icosahedron_edges.index((f[0], f[2])) * (frequency - 1)
                for i in range(frequency - 1):
                    face_vertices[i + 1].append(edge_index + i)
            else:
                edge_index = edge_index_offset \
                    + icosahedron_edges.index((f[2], f[0])) * (frequency - 1)
                for i in range(frequency - 1):
                    face_vertices[i + 1].append(edge_index + frequency - 2 - i)
            face_vertices[frequency].append(f[2])

            for i in range(1, frequency - 1):
                lam_i = i / frequency
                v_prime = v0 + lam_i * d2
                for j in range(1, frequency - i):
                    lam_j = j / frequency
                    v = v_prime + lam_j * d1
                    vertices[index, :] = v / linalg.norm(v)
                    face_vertices[i].append(index)
                    index += 1

            if f[2] > f[1]:
                edge_index = edge_index_offset \
                    + icosahedron_edges.index((f[1], f[2])) * (frequency - 1)
                for i in range(frequency - 1):
                    face_vertices[i + 1].append(edge_index + i)
            else:
                edge_index = edge_index_offset \
                    + icosahedron_edges.index((f[2], f[1])) * (frequency - 1)
                for i in range(frequency - 1):
                    face_vertices[i + 1].append(edge_index + frequency - 2 - i)

            # Edges and faces
            for i in range(frequency):
                for j in range(frequency - i):
                    s = face_vertices[i][j]
                    t = face_vertices[i][j + 1]
                    u = face_vertices[i + 1][j]
                    edges[s].append(t)
                    edges[t].append(u)
                    edges[u].append(s)
                    faces.append((s, t, u))
            for i in range(1, frequency):
                for j in range(frequency - i):
                    s = face_vertices[i][j]
                    t = face_vertices[i - 1][j + 1]
                    u = face_vertices[i][j + 1]
                    edges[s].append(t)
                    edges[t].append(u)
                    edges[u].append(s)
                    faces.append((s, t, u))

        c = {}
        for i, j, k in faces:
            c[i,j] = k
            c[j,k] = i
            c[k,i] = j

        return vertices, edges, faces, c

    def get_partials(self):
        return self._partials

    def get_vertices(self):
        return np.multiply(self._partials, np.reshape(self._rho, (-1, 1)))

    def get_edges(self):
        return self._edges

    def get_faces(self):
        return self._faces

    def get_boundary_vertices(self):
        return set()

    def get_boundary_edges(self):
        return set()

    def get_c(self):
        return self._c

    def get_parameters(self):
        return np.copy(self._rho)

    def set_parameters(self, rho):
        rho_max = np.max(np.abs(rho))
        rho = np.maximum(rho, rho_max / 100)
        rho = rho / np.average([linalg.norm(rho[l])
                                for l in range(rho.shape[0])])
        if not np.allclose(self._rho, rho):
            self._rho = np.copy(rho)
            self._updates += 1
        return rho

    def updates(self):
        return self._updates

    def nearest_vertex_index(self, direction):
        '''
        Find the index of the vertex whose direction is closest to the input
        direction.
        '''

        return np.argmax(self._partials @ direction)

    @staticmethod
    def latitude_longitude_to_direction(latitude, longitude):
        '''
        Takes in latitude (in [-90, 90]) and longitude (in (-180, 180]) and
        returns the corresponding direction on the unit sphere in R^3. For
        efficiency, the ranges of the inputs are not checked.
        '''

        latitude = np.radians(latitude)
        longitude = np.radians(longitude)

        return np.array([
            np.cos(longitude) * np.cos(latitude),
            np.sin(longitude) * np.cos(latitude),
            np.sin(latitude)
        ])