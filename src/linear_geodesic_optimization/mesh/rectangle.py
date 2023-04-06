import numpy as np

from linear_geodesic_optimization.mesh import mesh

class Mesh(mesh.Mesh):
    '''
    Representation of a mesh that is "approximately" a rectangle. In
    particular, projecting the vertices of the mesh so that their z-coordinates
    are 0 will yield a rectangle. The mesh itself looks like a grid where each
    cell has been cut by its major diagonal.
    '''

    def __init__(self, width, height):
        self._width = width
        self._height = height
        self._grid, self._edges, self._faces, self._nxt \
            = self._initial_mesh(width, height)
        self._partials = np.zeros((self._grid.shape[0], 3))
        self._partials[:,2] = 1.
        self._z = np.zeros(self._grid.shape[0])
        self._updates = 0

    def _initial_mesh(self, width, height):
        grid = np.zeros((width * height, 2))

        # The vertices are ordered lexicographically as (x, y)
        for i in range(width):
            for j in range(height):
                grid[i*height+j:] = np.array([i, j])

        # Normalize the vertices so that the mesh is supported on [-0.5, 0.5]^2
        grid[:,0] /= (width - 1)
        grid[:,1] /= (height - 1)
        grid -= 0.5

        edges = [[] for _ in range(width * height)]
        faces = []
        nxt = {}

        # Add edges and faces cell-by-cell
        for i in range(width - 1):
            for j in range(height - 1):
                v00 = i * height + j           # Bottom-left
                v01 = i * height + j + 1       # Top-left
                v10 = (i + 1) * height + j     # Bottom-right
                v11 = (i + 1) * height + j + 1 # Top-right

                # Add the top-left triangle
                edges[v00].append(v11)
                edges[v11].append(v01)
                edges[v01].append(v00)
                faces.append((v00, v11, v01))
                nxt[v00,v11] = v01
                nxt[v11,v01] = v00
                nxt[v01,v00] = v11

                # Add the bottom-right triangle
                edges[v00].append(v10)
                edges[v10].append(v11)
                edges[v11].append(v00)
                faces.append((v00, v10, v11))
                nxt[v00,v10] = v11
                nxt[v10,v11] = v00
                nxt[v11,v00] = v10

        return grid, edges, faces, nxt

    def get_partials(self):
        return self._partials

    def get_vertices(self):
        return np.concatenate((self._grid,
                               np.reshape(self._z, (-1, 1))), axis=1)

    def get_edges(self):
        return self._edges

    def get_faces(self):
        return self._faces

    def get_boundary_vertices(self):
        boundary_vertices = set()

        # Left edge
        boundary_vertices.update(range(self._height))

        # Right edge
        boundary_vertices.update(range(self._height * (self._width - 1),
                                       self._height * self._width))

        # Bottom edge
        boundary_vertices.update(range(self._height,
                                       self._height * (self._width - 1),
                                       self._height))

        # Top edge
        boundary_vertices.update(range(2 * self._height - 1,
                                       self._height * self._width - 1,
                                       self._height))

        return boundary_vertices

    def get_boundary_edges(self):
        # Boundary edges are those that appear as a half-edge in one direction,
        # but not in the opposite direction.
        boundary_edges = set()
        for i, j in self._nxt:
            if (j, i) not in self._nxt:
                boundary_edges.add((i, j))
        return boundary_edges

    def get_nxt(self):
        return self._nxt

    def get_parameters(self):
        return np.copy(self._z)

    def set_parameters(self, z):
        if not np.array_equal(self._z, z):
            self._z = np.copy(z)
            self._updates += 1
        return np.copy(z)

    def updates(self):
        return self._updates

    def get_fat_edges(self, vertices, edges, epsilon):
        def is_on_fat_edge(u, v, r, epsilon):
            # Only care about the first two coordinates
            u = u[:2]
            v = v[:2]
            r = r[:2]

            ru = r - u
            rv = r - v
            uv = u - v

            if ru @ uv <= 0 and rv @ uv >= 0:
                if uv @ uv == 0:
                    return ru @ ru < epsilon**2
                return rv @ rv - (uv @ rv)**2 / (uv @ uv) < epsilon**2
            else:
                return ru @ ru < epsilon**2 or rv @ rv < epsilon**2

        return [[i for i in range(self._grid.shape[0])
                 if is_on_fat_edge(vertices[e1], vertices[e2],
                                   self._grid[i,:], epsilon)]
                for (e1, e2) in edges]

    def get_epsilon(self):
        return max(np.linalg.norm(self._grid[u,:] - self._grid[v,:])
                   for u, vs in enumerate(self._edges) for v in vs)

    def nearest_vertex_index(self, v):
        '''
        Find the index of the vertex whose (x, y) coordinate pair is closest to
        the input coordinate pair. We assume the input lies in
        [-0.5, 0.5] x [-0.5, 0.5].
        '''

        i = round((v[0] + 0.5) * (self._width - 1))
        j = round((v[1] + 0.5) * (self._height - 1))
        return i * self._height + j

    @staticmethod
    def map_coordinates_to_support(coordinates, scale_factor=0.45):
        '''
        Convert a list of (x, y, z) triples into a list of new coordinates that
        have been scaled to lie centered in the unit square. The `scale_factor`
        parameter determines what proportion of the unit square is used (0.45
        means 45% of the width and 45% of the height is used).
        '''

        # Need this check to avoid out-of-bounds errors if coordinates is empty
        if not coordinates:
            return []

        coordinates = np.array(coordinates)[:,:2]

        coordinates_min = np.amin(coordinates, axis=0)
        coordinates_max = np.amax(coordinates, axis=0)
        divisor = coordinates_max - coordinates_min
        divisor[np.where(divisor == 0.)] = 1.
        divisor = np.amax(divisor)

        coordinates = coordinates - (coordinates_min + coordinates_max) / 2
        coordinates = coordinates / divisor

        return list(scale_factor * coordinates)

    def get_support_area(self):
        return 1.

    # Legacy functions
    def triangles_of_vertex(self):
        return {i: [face for face in self._faces if i in face]
                for i in range(self._grid.shape[0])}

    def coord_to_ndx(self, i, j):
        return i * self._height + j

    def ndx_to_coord(self, b):
        return [(b // self._height), (b % self._height)]
