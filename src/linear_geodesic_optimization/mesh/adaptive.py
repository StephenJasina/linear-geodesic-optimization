import numpy as np

from linear_geodesic_optimization.mesh import mesh

class Mesh(mesh.Mesh):
    '''
    Representation of a mesh that refines itself so that, given a set of
    points, each cell contains at most one point.

    Frankly, this implementation is a bit inefficient, but it's fine since the
    initialization is only run once.
    '''

    def __init__(self, width, height, points = []):
        '''
        Initialize an adaptive rectangular mesh to a set of points. This
        function assumes the points all lie in [-0.5, 0.5]^2. This can be done
        by calling Mesh.map_coordinates_to_support.
        '''

        self._points = points

        # Internally store vertices in 2 dimensions
        self._vertices, self._edges, self._faces, self._nxt \
            = Mesh._get_initial_mesh(width, height, points)
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

        coordinate_min = np.inf
        coordinate_max = -np.inf

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
    def _get_grid(width, height):
        # The vertices are ordered lexicographically as (x, y) and are
        # supported in [-0.5, 0.5]^2
        vertices = []
        for i in range(width):
            for j in range(height):
                vertices.append(np.array([
                    i / (width - 1) - 0.5,
                    j / (height - 1) - 0.5
                ]))

        # Add edges and faces cell-by-cell
        nxt = {}
        for i in range(width - 1):
            for j in range(height - 1):
                v00 = i * height + j           # Bottom-left
                v01 = i * height + j + 1       # Top-left
                v10 = (i + 1) * height + j     # Bottom-right
                v11 = (i + 1) * height + j + 1 # Top-right

                # Add the top-left triangle
                nxt[v00,v11] = v01
                nxt[v11,v01] = v00
                nxt[v01,v00] = v11

                # Add the bottom-right triangle
                nxt[v00,v10] = v11
                nxt[v10,v11] = v00
                nxt[v11,v00] = v10

        return vertices, nxt

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
    def _get_edges_and_faces(vertices, nxt):
        edges = [[] for _ in vertices]
        faces = []
        for (i, j), k in nxt.items():
            edges[i].append(j)
            if i < j and i < k:
                faces.append((i, j, k))

        return edges, faces

    @staticmethod
    def _get_initial_mesh(width, height, points):
        vertices, nxt = Mesh._get_grid(width, height)

        faces_to_points = {Mesh._get_face_id(edge, vertices, nxt): [] for edge in nxt}
        for point in points:
            x, y = point
            scaled_x = (x + 0.5) * (width - 1)
            scaled_y = (y + 0.5) * (height - 1)
            first = height * min(int(scaled_x), width - 2) \
                 + min(int(scaled_y), height - 2)
            second = first + height + 1
            if scaled_x % 1 > scaled_y % 1:
                first, second = second, first
            faces_to_points[first,second].append(point)

        while faces_to_points:
            edge, contained_points = next(iter(faces_to_points.items()))
            edge_vector = vertices[edge[1]] - vertices[edge[0]]

            if len(contained_points) <= 1:
                del faces_to_points[edge]
                continue

            Mesh._split_face(edge, vertices, nxt, faces_to_points)

        edges, faces = Mesh._get_edges_and_faces(vertices, nxt)

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

    @staticmethod
    def _segments_cross(a, b):
        '''
        Determine whether two line segments (defined by pairs of 2-vectors)
        intersect
        '''
        p, q = a
        u, v = b
        rot90 = np.array([[0., -1.], [1., 0.]])
        qp = rot90 @ (q - p)
        vu = rot90 @ (v - u)
        s = (qp @ (u - p)) * (qp @ (v - p))
        t = (vu @ (p - u)) * (vu @ (q - u))
        return (s <= 0 and t < 0) or (s < 0 and t <= 0) \
            or (s == 0 and t == 0 and ((p - u) @ (q - u) <= 0 or (p - v) @ (q - v) <= 0))

    @staticmethod
    def _is_in_face(point, face):
        '''
        Determine whether a point is inside a face, where the face is
        represented as an ordered triple.
        '''
        u, v, w = face
        rot90 = np.array([[0., -1.], [1., 0.]])
        vu = rot90 @ (v - u)
        wv = rot90 @ (w - v)
        uw = rot90 @ (u - w)
        # Take advantage of the fact that a triangle is the intersection of
        # three half-planes
        return (vu @ (w - u)) * (vu @ (point - u)) >= 0 \
            and (wv @ (u - v)) * (wv @ (point - v)) >= 0 \
            and (uw @ (v - w)) * (uw @ (point - w)) >= 0

    def get_fat_edges(self, vertices, edges, epsilon):
        fat_edges = []
        for edge in edges:
            fat_edge = set()
            p = vertices[edge[0]]
            q = vertices[edge[1]]

            for i, js in enumerate(self._edges):
                u = self._vertices[i]
                for j in js:
                    k = self._nxt[i,j]
                    v = self._vertices[j]
                    w = self._vertices[k]
                    face = (u, v, w)

                    if Mesh._is_in_face(p, face) or Mesh._is_in_face(q, face) \
                        or Mesh._segments_cross((p, q), (u, v)) \
                        or Mesh._segments_cross((p, q), (v, w)) \
                        or Mesh._segments_cross((p, q), (w, u)):
                        fat_edge.add(i)
                        fat_edge.add(j)
                        fat_edge.add(k)

            fat_edges.append(list(fat_edge))
        return fat_edges

    def restrict_to_fat_edges(self, fat_edges):
        fat_vertices = set.union(*(set(fat_edge) for fat_edge in fat_edges))
        included_vertices = set.union(*(
            {i, j, k}
            for (i, j), k in self._nxt.items()
            if i in fat_vertices or j in fat_vertices or k in fat_vertices
        ))
        sorted_included_vertices = sorted(included_vertices)
        vertex_map = {old_i: new_i for new_i, old_i in enumerate(sorted_included_vertices)}
        self._vertices = self._vertices[sorted_included_vertices,:]
        self._parameters = self._parameters[sorted_included_vertices]
        self._partials = self._partials[sorted_included_vertices,:]
        self._nxt = {
            (vertex_map[i], vertex_map[j]): vertex_map[k]
            for (i, j), k in self._nxt.items()
            if i in fat_vertices or j in fat_vertices or k in fat_vertices
        }
        self._edges, self._faces = Mesh._get_edges_and_faces(self._vertices, self._nxt)

    def get_epsilon(self):
        return max(np.linalg.norm(self._vertices[u,:] - self._vertices[v,:])
                   for u, vs in enumerate(self._edges) for v in vs)

    def get_support_area(self):
        return 1.

    def nearest_vertex_index(self, v):
        '''
        Find the index of the vertex whose (x, y) coordinate pair is closest to
        the input coordinate pair. We assume the input lies in
        [-0.5, 0.5] x [-0.5, 0.5].
        '''

        return np.argmin(np.linalg.norm(self._vertices - v, axis=1))
