import numpy as np

from linear_geodesic_optimization.mesh import mesh

class Mesh(mesh.Mesh):
    '''
    Representation of a mesh that refines itself so that, given a set of
    points, each cell contains at most one point.
    '''

    def __init__(self, points, epsilon):
        '''
        Initialize an adaptive mesh to a set of points. This function assumes
        the points all lie in [-0.5, 0.5] x [-0.5, 0.5]. This can be done by
        calling Mesh.map_coordinates_to_support.
        '''

        self._points = points
        self._epsilon = epsilon

        # Internally store vertices in 2 dimensions
        self._vertices, self._edges, self._faces, self._nxt \
            = Mesh._get_initial_mesh(points, epsilon)
        self._parameters = np.zeros(self._vertices.shape[0])
        self._partials = np.zeros((self._vertices.shape[0], 3))
        self._partials[:,2] = 1.
        self._updates = 0

    @staticmethod
    def map_coordinates_to_support(coordinates, scale_factor=0.45):
        '''
        Convert a list of (x, y) pairs into a list of new coordinates that
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

        return [np.array([
            ((x - coordinate_min) / divisor - 0.5) * scale_factor,
            ((y - coordinate_min) / divisor - 0.5) * scale_factor,
        ]) for x, y in coordinates]

    @staticmethod
    def _get_face_id(edge, vertices, nxt):
        '''
        Return the longest edge of the face that the input edge is incident to.
        '''

        i, j = edge
        k = nxt[i,j]
        u = vertices[i]
        v = vertices[j]
        w = vertices[k]
        a2 = (w - v) @ (w - v)
        b2 = (u - w) @ (u - w)
        c2 = (v - u) @ (v - u)

        if c2 >= a2 and c2 >= b2:
            return (i, j)
        elif a2 >= b2:
            return (j, k)
        else:
            return (k, i)

    def _split_face_naive(edge, l, vertices, nxt, faces_to_points):
        '''
        Just insert l on the edge without considering upkeep, except for the
        faces_to_points structure.
        '''

        i, j = edge
        k = nxt[i,j]

        # Update nxt
        del nxt[i,j]
        nxt[i,l] = k
        nxt[l,k] = i
        nxt[k,i] = l
        nxt[l,j] = k
        nxt[j,k] = l
        nxt[k,l] = j

        # Update faces_to_points
        contained_points = []
        if (i, j) in faces_to_points:
            contained_points = faces_to_points[i,j]
            del faces_to_points[i,j]
        left = Mesh._get_face_id((i, l), vertices, nxt)
        right = Mesh._get_face_id((l, j), vertices, nxt)
        contained_points_left = []
        contained_points_right = []
        for point in contained_points:
            vk = vertices[k]
            vl = vertices[l]
            if (point - vl) @ np.array([[0., -1.], [1., 0.]]) @ (vk - vl) >= 0:
                contained_points_left.append(point)
            else:
                contained_points_right.append(point)
        faces_to_points[left] = contained_points_left
        faces_to_points[right] = contained_points_right


    @staticmethod
    def _split_face(edge, vertices, nxt, faces_to_points):
        '''
        Given an oriented edge, repeatedly split the triangle containing that
        edge until the edge has been bisected. The splits are done by finding
        the largest angle in the triangle and bisecting the opposing edge.

        The faces_to_points parameter is maintained, in that it is a dictionary
        mapping "faces" (stored as the length of the longest edge on the face)
        to lists of points that the corresponding face contains.
        '''

        i, j = Mesh._get_face_id(edge, vertices, nxt)
        k = nxt[i,j]
        l = len(vertices)

        # Add the new vertex in
        vertices.append((vertices[i] + vertices[j]) / 2)

        if (j, i) in nxt:
            # In this case, the triangle we want to split is adjacent to
            # another triangle

            # First, split the adjacent triangle if necessary (that is, if the
            # shared edge is not the longest one)
            while True:
                k_prime = nxt[j,i]
                adjacent_face_id = Mesh._get_face_id((j, i), vertices, nxt)

                if (j, i) == adjacent_face_id:
                    break

                Mesh._split_face(adjacent_face_id, vertices, nxt, faces_to_points)

            # Now split the adjacent triangle
            Mesh._split_face_naive((j, i), l, vertices, nxt, faces_to_points)

        # Split the original triangle
        Mesh._split_face_naive((i, j), l, vertices, nxt, faces_to_points)

    @staticmethod
    def _get_initial_mesh(points, epsilon):
        vertices = [
            np.array([-0.5, -0.5]),
            np.array([0.5, 0.5]),
            np.array([0.5, -0.5]),
            np.array([-0.5, 0.5]),
        ]
        nxt = {
            (0, 1): 3, (1, 3): 0, (3, 0): 1,
            (1, 0): 2, (0, 2): 1, (2, 1): 0,
        }

        faces_to_points = {
            (0, 1): [point for point in points if point[1] >= point[0]],
            (1, 0): [point for point in points if point[1] < point[0]],
        }
        while faces_to_points:
            edge, contained_points = next(iter(faces_to_points.items()))
            edge_vector = vertices[edge[1]] - vertices[edge[0]]

            if edge_vector @ edge_vector < epsilon**2 and len(contained_points) <= 1:
                del faces_to_points[edge]
                continue

            Mesh._split_face(edge, vertices, nxt, faces_to_points)

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
