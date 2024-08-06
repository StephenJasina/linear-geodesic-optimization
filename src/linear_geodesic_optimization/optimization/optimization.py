import os
# TODO: Convert to plain text
import pickle
import typing

import numpy as np
import numpy.typing as npt

from linear_geodesic_optimization.mesh.mesh import Mesh
from linear_geodesic_optimization.optimization.curvature \
    import Computer as Curvature
from linear_geodesic_optimization.optimization.curvature_loss \
    import Computer as CurvatureLoss
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
        network_edges: typing.List[typing.List[typing.Tuple[int, int]]],
        network_curvatures: typing.List[typing.List[np.float64]],
        epsilon: np.float64,
        lambda_curvature: np.float64 = np.float64(1.),
        lambda_smooth: np.float64 = np.float64(0.01),
        network_weights: typing.Optional[typing.List[np.float64]]=None,
        directory: typing.Optional[str] = None
    ):
        """
        Parameters:
        * `mesh`: The mesh to optimize over
        * `network_vertices`: A list of vertices in coordinate form.
          Alternatively, a numpy array of the vertices. In particular,
          these vertices should be embedded into the mesh
        * `network_edges`: A list of lists of edges in the network,
          where each edge is represented as a pair of indices into
          `network_vertices`
        * `network_curvatures`: A list of lists of curvatures assigned
          to each element of `network_edges`.
        * `epsilon`: The thickness of the fat edges
        * `lamda_curvature`: The strength of the curvature loss
        * `lamda_smooth`: The strength of the smoothing loss
        * `directory`: Where to save snapshots of the mesh for each
          iteration of optimization
        """
        self.mesh = mesh

        self.lambda_curvature = lambda_curvature
        self.lambda_smooth = lambda_smooth
        if network_weights is None:
            network_weights = [1. / len(network_edges)] * len(network_edges)
        self.network_weights = network_weights

        self.directory = directory

        self.laplacian = Laplacian(mesh)
        self.curvature = Curvature(mesh, self.laplacian)
        self.curvature_losses = [
            CurvatureLoss(
                mesh, network_vertices,
                network_edges_, network_curvatures_,
                epsilon, self.curvature
            )
            for network_edges_, network_curvatures_ in zip(
                network_edges, network_curvatures
            )
        ]
        self.smooth_loss = SmoothLoss(mesh, self.laplacian, self.curvature)

        # Count of iterations for diagnostic purposes
        self.iterations = 0

    def forward(self, z: typing.Optional[npt.NDArray[np.float64]] = None):
        if z is not None:
            self.mesh.set_parameters(z)
        for curvature_loss in self.curvature_losses:
            curvature_loss.forward()
        self.smooth_loss.forward()
        return self.lambda_curvature * sum(
            curvature_loss.loss * network_weight
            for curvature_loss, network_weight in zip(self.curvature_losses, self.network_weights)
        ) + self.lambda_smooth * self.smooth_loss.loss

    def reverse(self, z: typing.Optional[npt.NDArray[np.float64]] = None):
        if z is not None:
            self.mesh.set_parameters(z)
        self.smooth_loss.reverse()
        for curvature_loss in self.curvature_losses:
            curvature_loss.reverse()
        return self.lambda_curvature * sum(
            curvature_loss.dif_loss * network_weight
            for curvature_loss, network_weight in zip(self.curvature_losses, self.network_weights)
        ) + self.lambda_smooth * self.smooth_loss.dif_loss

    @staticmethod
    def to_float_list(array: npt.NDArray[np.float64]):
        return [float(item) for item in array]

    def diagnostics(self, x = None, f = None, context = None):
        """
        Save the hierarchy to disk and output some useful information about
        the loss functions.
        """
        loss = self.forward()
        curvature_loss = sum(
            curvature_loss.loss * network_weight
            for curvature_loss, network_weight in zip(self.curvature_losses, self.network_weights)
        )
        print(
            f'iteration {self.iterations}:\n'
            + f'\tL_curvature: {curvature_loss:.6f}\n'
            + f'\tL_smooth: {self.smooth_loss.loss:.6f}\n'
            + f'\tLoss: {loss:.6f}\n'
        )

        if self.directory is not None:
            with open(os.path.join(self.directory,
                                   str(self.iterations)), 'wb') as f:
                pickle.dump({
                    'mesh_parameters': Computer.to_float_list(self.mesh.get_parameters()),
                    'L_curvature': float(curvature_loss),
                    'L_smooth': float(self.smooth_loss.loss),
                }, f)

        self.iterations += 1
