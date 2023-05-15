class Mesh:
    def get_topology(self):
        '''
        Return the topology of the mesh.

        This should be a `dcelmesh.Mesh` object.
        '''

    def get_partials(self):
        '''
        We assume the mesh is parameterized by |V| scalars, each of which
        affects exactly one vertex. This function returns the non-zero partial
        derivatives.

        For efficiency, this returns a |V| by 3 array. The output of this
        function should be treated as "read only."
        '''

        raise NotImplementedError

    def get_coordinates(self):
        '''
        Return the coordinates of the vertices of the mesh.

        For efficiency, this returns a |V| by 3 array.
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

    def get_support_area(self):
        '''
        Return the area of the support of the mesh. For example, if the support
        is the unit sphere, then this will return 4*pi.
        '''

        raise NotImplementedError
