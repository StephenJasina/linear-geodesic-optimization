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
        Return a set of the indices of the edges on the boundary of the mesh.
        By convention, these edges will be oriented in a "counterclockwise"
        fashion.
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
