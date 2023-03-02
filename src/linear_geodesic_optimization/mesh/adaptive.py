import numpy as np

from linear_geodesic_optimization.mesh import mesh

class Mesh(mesh.Mesh):
    '''
    Representation of a mesh that refines itself so that, given a set of
    points, each cell contains at most one point.
    '''

    def __init__(self, points, epsilon):
        self._points = points
        self._epsilon = epsilon

        # Internally store vertices in 2 dimensions
        self._vertices, self._edges, self._faces, self._nxt \
            = self._get_initial_mesh(points, epsilon)
        self._parameters = np.zeros(self._vertices.shape[0])
        self._partials = np.zeros((self._vertices.shape[0], 3))
        self._partials[:,2] = 1.
        self._updates = 0

    @staticmethod
    def _split_face(edge, vertices, nxt):
        '''
        Given an oriented edge, repeatedly split the triangle containing that
        edge until the edge has been bisected. The splits are done by finding
        the largest angle in the triangle and bisecting the opposing edge.
        '''

        i, j = edge
        k = nxt[i,j]
        u = vertices[i]
        v = vertices[j]
        x = (u + v) / 2

        if (j, i) in nxt:
            # In this case, the triangle we want to split is adjacent to
            # another triangle
            c2 = (v - u) @ (v - u)

            # First, split the adjacent triangle if necessary
            while True:
                k_prime = nxt[j,i]
                w_prime = vertices[k_prime]

                a2 = (w_prime - v) @ (w_prime - v)
                b2 = (u - w_prime) @ (u - w_prime)

                if c2 >= a2 and c2 >= b2:
                    break

                if a2 >= b2:
                    Mesh._split_face((k_prime, j), vertices, nxt)
                else:
                    Mesh._split_face((i, k_prime), vertices, nxt)

            # Now split the adjacent triangle
            l = len(vertices)
            del nxt[j,i]
            nxt[j,l] = k_prime
            nxt[l,k_prime] = j
            nxt[k_prime,j] = l
            nxt[i,k_prime] = l
            nxt[k_prime,l] = i
            nxt[l,i] = k_prime

        # Split the original triangle
        l = len(vertices)
        del nxt[i,j]
        nxt[i,l] = k
        nxt[l,k] = i
        nxt[k,i] = l
        nxt[j,k] = l
        nxt[k,l] = j
        nxt[l,j] = k

        # Keep track of the added vertex
        vertices.append(x)

    def _get_initial_mesh(self, points, epsilon):
        vertices = [
            np.array([0., 0.]),
            np.array([1., 1.]),
            np.array([1., 0.]),
            np.array([0., 1.]),
        ]
        nxt = {
            (0, 1): 3, (1, 3): 0, (3, 0): 1,
            (0, 2): 1, (2, 1): 0, (1, 0): 2,
        }

        edges = [[] for _ in vertices]
        faces = []
        for (i, j), k in nxt.items():
            edges[i].append(j)
            if i < j and i < k:
                faces.append((i, j, k))

        return np.array(vertices), edges, faces, nxt

    def get_partials(self):
        return np.copy(self._partials)

    def get_vertices(self):
         return np.concatenate((self._vertices,
                               np.reshape(self._parameters, (-1, 1))), axis=1)

    def get_edges(self):
        return self._edges

    def get_faces(self):
        return self._faces

    def get_boundary_vertices(self):
        return set(i for i, _ in self.get_boundary_edges())

    def get_boundary_edges(self):
        nxt = self.get_nxt()
        return set((j, i) for i, j in nxt if (j, i) not in nxt)

    def get_nxt(self):
        return self._nxt

    def get_parameters(self):
        return np.copy(self._parameters)

    def set_parameters(self, parameters):
        if not np.array_equal(self._parameters, parameters):
            self._parameters = np.copy(parameters)
            self._updates += 1
        return np.copy(parameters)

    def updates(self):
        return self._updates

    def get_fat_edges(self, vertices, edges, epsilon):
        raise NotImplementedError

    def get_epsilon(self):
        return self._epsilon

    def get_support_area(self):
        return 1.

    def map_coordinates_to_support(self, coordinates, scale_factor=0.45):
        '''
        Convert a list of (x, y, z) triples into a list of new coordinates that
        have been scaled to lie centered in the unit square. The `scale_factor`
        parameter determines what proportion of the unit square is used (0.45
        means 45% of the width and 45% of the height is used).
        '''

        # Need this check to avoid out-of-bounds errors if coordinates is empty
        if not coordinates:
            return []

        coordinate_min = coordinates[0][0]
        coordinate_max = coordinates[0][0]

        # Find scaling factors so that the x range and y range are both [0, 1].
        # If no such scaling factor exists (i.e., the x-coordinate or
        # y-coordinate is constant), set it to some arbitrary value (in this
        # case, just set it to 1.)
        for x, y in coordinates:
            coordinate_min = min(coordinate_min, x, y)
            coordinate_max = max(coordinate_max, x, y)
        divisor = coordinate_max - coordinate_min
        divisor = 1. if divisor == 0. else divisor

        return [np.array([((x - coordinate_min) / divisor - 0.5) * scale_factor,
                          ((y - coordinate_min) / divisor - 0.5) * scale_factor,
                          0])
                for x, y in coordinates]
