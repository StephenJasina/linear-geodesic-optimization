from scipy import sparse

class Mesh:
    def get_partials(self):
        '''
        We assume the mesh is parameterized by |V| scalars, each of which
        affects exactly one vertex. This function returns the non-zero partial
        derivatives.

        For efficiency, this returns a |V| by 3 array. The output of this
        function should be treated as "read only."
        '''

        raise NotImplementedError

    def get_vertices(self):
        '''
        Return the vertices of the mesh.

        For efficiency, this returns a |V| by 3 array.
        '''

        raise NotImplementedError

    def get_edges(self):
        '''
        Return the edges of the mesh adjacency list format.

        The output of this function should be treated as "read only."
        '''

        raise NotImplementedError

    def get_faces(self):
        '''
        Return a list of triples (i, j, k) representing the (indices of the)
        faces of the mesh. Each i -> j -> k is oriented counterclockwise.

        The output of this function should be treated as "read only."
        '''

        raise NotImplementedError

    def get_boundary_vertices(self):
        '''
        Return a set of the indices of the vertices on the boundary of the
        mesh.
        '''

        raise NotImplementedError

    def get_boundary_edges(self):
        '''
        Return a set of the edges on the boundary of the mesh. By convention,
        these edges will be oriented in a "counterclockwise" fashion.
        '''

        raise NotImplementedError

    def get_c(self):
        '''
        Return a map from pairs of indices to indices where (i, j) maps to k
        precisely when i -> j -> k is a face oriented counterclockwise.

        The output of this function should be treated as "read only."
        '''

        raise NotImplementedError

    def get_parameters(self):
        '''
        Return the parameters of the vertices of this mesh, ordered in the same
        way as the output of `get_vertices`.
        '''

        raise NotImplementedError

    def set_parameters(self, parameters):
        '''
        Set the parameters of the vertices of this mesh, ordered in the same
        way as the output of `get_vertices`.
        '''

        raise NotImplementedError

    def updates(self):
        '''
        Return the number of times `set_rho` has been called. This function is
        an easy (O(1)) way to determine whether the mesh has been updated.
        '''

        raise NotImplementedError

    def get_fat_edges(self, vertices, edges, epsilon):
        '''
        For a list of edges in a graph embedded in our mesh (represented as
        pairs of indices into `vertices`) and a width epsilon > 0, return a
        list of lists of vertices in our mesh. Each list of vertices
        corresponds to a fattened edge.
        '''

        raise NotImplementedError

    def get_epsilon(self):
        '''
        Return the maximum space between vertices if the mesh were to have a
        default value of parameters (e.g., a unit sphere or a flat plane).
        '''

        raise NotImplementedError

    def get_support_area(self):
        '''
        Return the area of the support of the mesh. For example, if the support
        is the unit sphere, then this will return 4*pi.
        '''

        raise NotImplementedError

    def get_graph_laplacian(self):
        '''
        Return the graph Laplacian of the mesh as a CSR sparse matrix.
        '''
        size = self.get_vertices().shape[0]
        boundary_edges = self.get_boundary_edges()
        row = []
        col = []
        data = []
        for u, vs in enumerate(self.get_edges()):
            for v in vs:
                row.append(u)
                col.append(v)
                data.append(-1)

                row.append(u)
                col.append(u)
                data.append(1)

                # Need to check the boundary so that the returned matix is
                # symmetric
                if (u, v) in boundary_edges:
                    row.append(v)
                    col.append(u)
                    data.append(-1)

                    row.append(v)
                    col.append(v)
                    data.append(1)
        return sparse.coo_array((data, (row, col)),
                                shape=(size, size)).tocsr()
