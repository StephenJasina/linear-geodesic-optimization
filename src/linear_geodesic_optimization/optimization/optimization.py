import json
import pathlib
import types
import typing

import numpy as np
import numpy.typing as npt

from linear_geodesic_optimization.mesh.mesh import Mesh
from linear_geodesic_optimization.optimization.curvature \
    import Computer as Curvature
# from linear_geodesic_optimization.optimization.curvature_loss_gridded \
#     import Computer as CurvatureLoss
from linear_geodesic_optimization.optimization.curvature_loss \
    import Computer as CurvatureLoss
from linear_geodesic_optimization.optimization.laplacian \
    import Computer as Laplacian
from linear_geodesic_optimization.optimization.smooth_loss \
    import Computer as SmoothLoss
from linear_geodesic_optimization.optimization.smooth_loss_tensor \
    import Computer as SmoothLossTensor


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
        epsilon: np.float64,
        lambda_curvature: np.float64 = np.float64(1.),
        lambda_smooth: np.float64 = np.float64(0.01),
        edge_weights: typing.Optional[typing.List[np.float64]]=None,
        directory: typing.Optional[str] = None,
        smooth_loss_type: typing.Literal['mvs', 'tensor'] = 'mvs',
        lambda_geodesic: np.float64 = np.float64(0.),
        geodesic_bending=None,
        curvature_falloff: float = 0.,
        curvature_falloff_weight: bool = False,
        curvature_falloff_ends: bool = True,
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
        * `smooth_loss_type`: Either `'mvs'` (the MVS-cross energy of
          the principal curvatures) or `'tensor'` (the Dirichlet energy
          of the shape operator; requires an untrimmed rectangle mesh)
        * `lambda_geodesic`: The strength of the geodesic loss, the
          bending energy of the routes
        * `geodesic_bending`: A `geodesic_curvature.CurveBending`
          giving the geodesic loss (needed when `lambda_geodesic` is
          nonzero)
        * `curvature_falloff`, `curvature_falloff_weight`: a soft edge
          for the curvature loss's tube (see
          `curvature_loss.tapered_tube`)
        """
        self.mesh = mesh

        self.lambda_curvature = lambda_curvature
        self.lambda_smooth = lambda_smooth
        self.edge_weights = edge_weights

        self.directory = directory

        if lambda_geodesic and geodesic_bending is None:
            raise ValueError('lambda_geodesic needs geodesic_bending')
        self.lambda_geodesic = lambda_geodesic
        self.geodesic_bending = geodesic_bending
        self.geodesic_loss = types.SimpleNamespace(loss=0., dif_loss=0.)

        self.laplacian = Laplacian(mesh)
        self.curvature = Curvature(mesh, self.laplacian)
        self.curvature_loss = CurvatureLoss(
            mesh, network_vertices,
            network_edges, network_curvatures,
            epsilon, self.curvature,
            # edge_weights
            falloff=curvature_falloff,
            falloff_weight=curvature_falloff_weight,
            falloff_ends=curvature_falloff_ends,
        )
        if smooth_loss_type == 'mvs':
            self.smooth_loss = SmoothLoss(
                mesh, self.laplacian, self.curvature
            )
        elif smooth_loss_type == 'tensor':
            self.smooth_loss = SmoothLossTensor(mesh, self.laplacian)
        else:
            raise ValueError(
                f'Unknown smooth loss type "{smooth_loss_type}"'
            )

        # Count of iterations for diagnostic purposes
        self.iterations = 0

    def forward(self, z: typing.Optional[npt.NDArray[np.float64]] = None):
        if z is not None:
            self.mesh.set_parameters(z)
        self.curvature_loss.forward()
        self.smooth_loss.forward()
        self._geodesic()
        return self.lambda_curvature * self.curvature_loss.loss \
            + self.lambda_smooth * self.smooth_loss.loss \
            + self.lambda_geodesic * self.geodesic_loss.loss

    def reverse(self, z: typing.Optional[npt.NDArray[np.float64]] = None):
        if z is not None:
            self.mesh.set_parameters(z)
        self.smooth_loss.reverse()
        self.curvature_loss.reverse()
        self._geodesic()
        return self.lambda_curvature * self.curvature_loss.dif_loss \
            + self.lambda_smooth * self.smooth_loss.dif_loss \
            + self.lambda_geodesic * self.geodesic_loss.dif_loss

    def _geodesic(self):
        """Evaluate the geodesic loss and its gradient, if it is used."""
        if self.lambda_geodesic:
            loss, gradient = self.geodesic_bending.evaluate(
                np.asarray(self.mesh.get_parameters(), dtype=np.float64)
            )
            self.geodesic_loss.loss = loss
            self.geodesic_loss.dif_loss = gradient

    @property
    def kappa_G(self) -> npt.NDArray[np.float64]:
        """
        The Gaussian curvature at each vertex as of the last call to
        `forward`, matching the torch backend's `Computer.kappa_G`.
        """
        return np.asarray(self.curvature.kappa_G, dtype=np.float64)

    @staticmethod
    def to_float_list(array: npt.NDArray[np.float64]):
        return [float(item) for item in array]

    def diagnostics(self, x = None, f = None, context = None):
        """
        Save the hierarchy to disk and output some useful information about
        the loss functions.
        """
        loss = self.forward()
        print(
            f'iteration {self.iterations}:\n'
            + f'\tL_curvature: {self.curvature_loss.loss:.6f}\n'
            + f'\tL_smooth: {self.smooth_loss.loss:.6f}\n'
            + f'\tL_geodesic: {self.geodesic_loss.loss:.6f}\n'
            + f'\tLoss: {loss:.6f}\n'
        )

        if self.directory is not None and self.iterations % 100 == 0:
            with open(self.directory / f'{self.iterations}.json', 'w') as file_output:
                json.dump({
                    'mesh_parameters': Computer.to_float_list(self.mesh.get_parameters()),
                    'L_curvature': float(self.curvature_loss.loss),
                    'L_smooth': float(self.smooth_loss.loss),
                    'L_geodesic': float(self.geodesic_loss.loss),
                }, file_output)

        self.iterations += 1
