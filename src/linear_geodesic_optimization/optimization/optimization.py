import os
import pickle
import typing

import numpy as np
import numpy.typing as npt

from linear_geodesic_optimization import data
from linear_geodesic_optimization.mesh.mesh import Mesh
from linear_geodesic_optimization.optimization.curvature \
    import Computer as Curvature
from linear_geodesic_optimization.optimization.curvature_loss \
    import Computer as CurvatureLoss
from linear_geodesic_optimization.optimization.geodesic \
    import Computer as Geodesic
from linear_geodesic_optimization.optimization.geodesic_loss \
    import Computer as GeodesicLoss
from linear_geodesic_optimization.optimization.laplacian \
    import Computer as Laplacian
from linear_geodesic_optimization.optimization.smooth_loss \
    import Computer as SmoothLoss


class Computer:
    """
    Structure for consolidating evaluation and gradient computation of the
    linear geodesic optimization loss functions.
    """

    def __init__(
        self,
        mesh: Mesh,
        network_vertices: npt.NDArray[np.float64],
        network_edges: typing.List[typing.Tuple[int, int]],
        network_curvatures: typing.List[np.float64],
        network_latencies: typing.List[typing.List[typing.Tuple[int,
                                                                np.float64]]],
        epsilon: np.float64,
        lambda_curvature: np.float64 = np.float64(1.),
        lambda_smooth: np.float64 = np.float64(0.01),
        lambda_geodesic: np.float64 = np.float64(1.),
        directory: typing.Optional[str] = None):
        """
        Parameters:
        * `mesh`: The mesh to optimize over
        * `network_vertices`: A list of vertices in coordinate form.
          Alternatively, a numpy array of the vertices. In particular,
          these vertices should be embedded into the mesh
        * `network_edges`: A list of edges in the network, where each
          edge is represented as a pair of indices into
          `network_vertices`
        * `network_curvatures`: A list of curvatures assigned to each
          element of `network_edges`.
        * `network_latencies`: The measured (real world) latencies. This
          should be a list of lists of pairs of vertex indices and
          floats (essentially an annotated adjacency list)
        * `epsilon`: The thickness of the fat edges
        * `lamda_curvature`: The strength of the curvature loss
        * `lamda_smooth`: The strength of the smoothing loss
        * `lamda_geodesic`: The strength of the geodesic loss
        * `directory`: Where to save snapshots of the mesh for each
          iteration of optimization
        """
        self.mesh = mesh

        self.latencies = data.map_latencies_to_mesh(mesh, network_vertices,
                                                    network_latencies)

        self.lambda_curvature = lambda_curvature
        self.lambda_smooth = lambda_smooth
        self.lambda_geodesic = lambda_geodesic

        self.directory = directory

        self.laplacian = Laplacian(mesh)
        self.curvature = Curvature(mesh, self.laplacian)
        self.curvature_loss = CurvatureLoss(mesh, network_vertices,
                                            network_edges, network_curvatures,
                                            epsilon, self.curvature)
        self.smooth = SmoothLoss(mesh, self.laplacian, self.curvature)
        self.geodesics = [
            Geodesic(mesh, u, v)
            for (u, v), _ in self.latencies
        ]
        self.geodesic_loss = GeodesicLoss(self.geodesics, np.array([
            latency
            for _, latency in self.latencies
        ]))

        # Count of iterations for diagnostic purposes
        self.iterations = 0

    def forward(self, z: typing.Optional[npt.NDArray[np.float64]] = None):
        if z is not None:
            self.mesh.set_parameters(z)
        self.curvature_loss.forward()
        self.smooth.forward()
        self.geodesic_loss.forward()
        return self.lambda_curvature * self.curvature_loss.loss \
            + self.lambda_smooth * self.smooth.loss \
            + self.lambda_geodesic * self.geodesic_loss.loss

    def reverse(self, z: typing.Optional[npt.NDArray[np.float64]] = None):
        if z is not None:
            self.mesh.set_parameters(z)
        self.smooth.reverse()
        self.curvature_loss.reverse()
        self.geodesic_loss.reverse()
        dif_geodesic_loss = np.zeros(self.mesh.get_topology().n_vertices())
        for index, loss in self.geodesic_loss.dif_loss.items():
            dif_geodesic_loss[index] = loss
        return self.lambda_curvature * self.curvature_loss.dif_loss \
            + self.lambda_smooth * self.smooth.dif_loss \
            + self.lambda_geodesic * dif_geodesic_loss

    @staticmethod
    def to_float_list(array: npt.NDArray[np.float64]):
        return [float(item) for item in array]

    def diagnostics(self, _):
        """
        Save the hierarchy to disk and output some useful information about
        the loss functions.
        """
        loss = self.forward()
        print(f'iteration {self.iterations}:')
        print(f'\tL_curvature: {self.curvature_loss.loss:.6f}')
        print(f'\tL_smooth: {self.smooth.loss:.6f}')
        print(f'\tL_geodesic: {self.geodesic_loss.loss:.6f}')
        print(f'\tLoss: {loss:.6f}\n')

        if self.directory is not None:
            with open(os.path.join(self.directory,
                                   str(self.iterations)), 'wb') as f:
                pickle.dump({
                    'mesh_parameters': Computer.to_float_list(self.mesh.get_parameters()),
                    'L_curvature': float(self.curvature_loss.loss),
                    'L_smooth': float(self.smooth.loss),
                    'L_geodesic': float(self.geodesic_loss.loss),
                    'true_latencies': Computer.to_float_list(self.geodesic_loss.t),
                    'estimated_latencies': Computer.to_float_list(self.geodesic_loss.phi),
                }, f)

        self.iterations += 1
