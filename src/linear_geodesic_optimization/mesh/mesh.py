import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go

class Mesh:
    def get_directions(self):
        '''
        Return the (normalized) directions of the mesh. This is useful for
        computing partial derivatives later on. The output of this function
        should be treated as "read only."
        '''

        raise NotImplementedError

    def get_vertices(self):
        '''
        Return the vertices of the mesh.
        '''

        raise NotImplementedError

    def get_edges(self):
        '''
        Return the edges of the mesh adjacency list format. The output of this
        function should be treated as "read only."
        '''

        raise NotImplementedError

    def get_faces(self):
        '''
        Return a list of triples (i, j, k) representing the (indices of the)
        faces of the mesh. Each i -> j -> k is oriented counterclockwise. The
        output of this function should be treated as "read only."
        '''

        raise NotImplementedError

    def get_c(self):
        '''
        Return a map from pairs of indices to indices where (i, j) maps to k
        precisely when i -> j -> k is a face oriented counterclockwise. The
        output of this function should be treated as "read only."
        '''

        raise NotImplementedError

    def get_rho(self):
        '''
        Return the magnitudes of the vertices of this mesh, ordered in the same
        way as the output of `get_vertices`.
        '''

        raise NotImplementedError

    def set_rho(self, rho):
        '''
        Set the magnitudes of the vertices of this mesh, ordered in the same
        way as the output of `get_vertices`.
        '''

        raise NotImplementedError

    def updates(self):
        '''
        Return the number of times `set_rho` has been called. This function is
        an easy (O(1)) way to determine whether the mesh has been updated.
        '''

        raise NotImplementedError

class Animation3D:
    '''
    Animator for meshes. Frames should be added one-by-one via `add_frame`, and
    then the Plotly animation figure is returned via `get_fig`.
    '''

    def __init__(self):
        self.clear_frames()

    def clear_frames(self):
        '''
        Reset the animation.
        '''

        self._frames = []
        self._radius = 0

    def add_frame(self, mesh):
        '''
        Add a frame that contains `mesh`.
        '''

        vertices = mesh.get_vertices()
        faces = mesh.get_faces()
        data = ff.create_trisurf(
            vertices[:,0], vertices[:,1], vertices[:,2],
            faces, colormap='#1f77b4',
            show_colorbar=False, plot_edges=True,
            aspectratio=dict(x=1, y=1, z=1)
        ).data
        self._radius = max(self._radius, np.max(vertices))
        frame = go.Frame(data=data)
        frame['name'] = str(len(self._frames))
        self._frames.append(frame)

    def get_fig(self, duration=0):
        '''
        Return a Plotly figure that can be displayed by calling its `.show()`
        method. The `duration` parameter controls the speed of the animation
        (lower is faster; must be non-negative).
        '''

        fig_dict = {
            'data': self._frames[0].data,
            'layout': {},
            'frames': self._frames,
        }

        fig_dict['layout']['title'] = {
            'text': 'Sphere Optimization',
        }
        fig_dict['layout']['width'] = 600
        fig_dict['layout']['height'] = 600

        axis_format = {
            'color': '#ffffff',
            'range': [-self._radius, self._radius],
            'showaxeslabels': False,
            'showticklabels': False,
            'showbackground': False,
            'showspikes': False,
        }
        fig_dict['layout']['scene'] = {}
        fig_dict['layout']['scene']['xaxis'] = axis_format
        fig_dict['layout']['scene']['yaxis'] = axis_format
        fig_dict['layout']['scene']['zaxis'] = axis_format
        fig_dict['layout']['scene']['hovermode'] = False

        fig_dict['layout']['updatemenus'] = [
            {
                'type': 'buttons',
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': duration},
                                        'fromcurrent': True,
                                        'transition': {'duration': 0}}],
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {'frame': {'duration': 0},
                                          'mode': 'immediate',
                                          'transition': {'duration': 0}}],
                    },
                ],
            }
        ]

        sliders_dict = {
            'currentvalue': {
                'prefix': 'batch: ',
                'xanchor': 'right',
            },
            'transition': {'duration': 0},
            'steps': [
                {
                    'args': [
                        [str(i)],
                        {
                            'mode': 'immediate',
                            'frame': {'duration': duration},
                            'transition': {'duration': 0},
                        },
                    ],
                    'label': str(i),
                    'method': 'animate',
                }
                for i, _ in enumerate(self._frames)
            ]
        }
        fig_dict['layout']['sliders'] = [sliders_dict]
        return go.Figure(fig_dict)
