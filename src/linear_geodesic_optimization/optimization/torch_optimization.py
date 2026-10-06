"""Array-based optimization of a rectangular mesh using PyTorch autodiff.

The mesh topology and fat-edge membership are fixed during an optimization.
They are converted to integer arrays once; evaluations operate on whole arrays
instead of building dictionaries of local partial derivatives.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch


class Computer:
    """Drop-in objective and gradient for the legacy optimization computer."""

    def __init__(
        self,
        mesh,
        network_vertices,
        network_edges,
        network_curvatures,
        epsilon,
        lambda_curvature=1.0,
        lambda_smooth=0.01,
        directory: Optional[Path] = None,
    ):
        # Batch jobs run in separate processes; one thread per process avoids
        # oversubscribing CPU cores when many optimizations run concurrently.
        torch.set_num_threads(1)
        self.mesh = mesh
        self.lambda_curvature = lambda_curvature
        self.lambda_smooth = lambda_smooth
        self.directory = directory
        self.iterations = 0
        self.curvature_loss = SimpleNamespace(loss=0.0)
        self.smooth_loss = SimpleNamespace(loss=0.0)
        self._cached_parameters = None
        self._cached_loss = None
        self._cached_gradient = None

        topology = mesh.get_topology()
        faces = np.array(
            [[vertex.index for vertex in face.vertices()] for face in topology.faces()],
            dtype=np.int64,
        )
        halfedges = list(topology.halfedges())
        edge_pairs = np.array(
            [[vertex.index for vertex in edge.vertices()] for edge in topology.edges()],
            dtype=np.int64,
        )
        boundary = np.array(
            [vertex.is_on_boundary() for vertex in topology.vertices()], dtype=bool
        )
        self._faces = torch.as_tensor(faces)
        self._h_u = torch.tensor([edge.origin.index for edge in halfedges])
        self._h_v = torch.tensor([edge.destination.index for edge in halfedges])
        self._h_w = torch.tensor([edge.previous.origin.index for edge in halfedges])
        self._h_face = torch.tensor([edge.face.index for edge in halfedges])
        self._h_edge = torch.tensor([edge.edge.index for edge in halfedges])
        self._e_u = torch.as_tensor(edge_pairs[:, 0])
        self._e_v = torch.as_tensor(edge_pairs[:, 1])
        self._interior = torch.as_tensor(~boundary)
        self._h_interior = self._interior[self._h_u] & self._interior[self._h_v]
        self._xy = torch.as_tensor(
            mesh.get_coordinates()[:, :2].copy(), dtype=torch.float64
        )
        self._n_vertices = topology.n_vertices
        self._n_edges = topology.n_edges

        # A graph edge contributes its normalized length divided over mesh
        # vertices within epsilon of its projected line segment.
        network_vertices = np.asarray(network_vertices, dtype=np.float64)
        lengths = np.array([
            np.linalg.norm(network_vertices[i, :2] - network_vertices[j, :2])
            for i, j in network_edges
        ])
        edge_weights = lengths / lengths.sum()
        mesh_points = self._xy.numpy()
        fat_indices = []
        targets = []
        weights = []
        for (i, j), target, edge_weight in zip(
            network_edges, network_curvatures, edge_weights
        ):
            u = network_vertices[i, :2]
            direction = network_vertices[j, :2] - u
            length_squared = direction @ direction
            if length_squared:
                fraction = np.clip(
                    ((mesh_points - u) @ direction) / length_squared, 0.0, 1.0
                )
            else:
                fraction = np.zeros(len(mesh_points))
            distance_squared = np.sum(
                (mesh_points - (u + fraction[:, None] * direction)) ** 2,
                axis=1,
            )
            members = np.flatnonzero(distance_squared < epsilon**2)
            if target is None or len(members) == 0:
                continue
            fat_indices.extend(members)
            targets.extend([target] * len(members))
            weights.extend([edge_weight / len(members)] * len(members))
        self._fat_indices = torch.tensor(fat_indices, dtype=torch.long)
        self._targets = torch.tensor(targets, dtype=torch.float64)
        self._weights = torch.tensor(weights, dtype=torch.float64)

    def _evaluate(self, parameters):
        parameters = np.asarray(parameters, dtype=np.float64)
        self.mesh.set_parameters(parameters)
        if self._cached_parameters is not None and np.array_equal(
            parameters, self._cached_parameters
        ):
            return
        heights = torch.tensor(parameters, dtype=torch.float64, requires_grad=True)
        points = torch.cat((self._xy, heights[:, None]), dim=1)
        faces = self._faces

        normals = torch.cross(
            points[faces[:, 0]] - points[faces[:, 2]],
            points[faces[:, 1]] - points[faces[:, 2]],
            dim=1,
        )
        areas = torch.linalg.vector_norm(normals, dim=1) / 2
        vertex_areas = torch.zeros(self._n_vertices, dtype=torch.float64).index_add(
            0, self._h_u, areas[self._h_face] / 3
        )

        pu, pv, pw = points[self._h_u], points[self._h_v], points[self._h_w]
        cotangents = ((pu - pw) * (pv - pw)).sum(dim=1) / (
            2 * areas[self._h_face]
        )
        half_cotangents = cotangents / 2
        edge_laplacian = torch.zeros(self._n_edges, dtype=torch.float64).index_add(
            0, self._h_edge, half_cotangents
        )
        vertex_laplacian = torch.zeros(
            self._n_vertices, dtype=torch.float64
        ).index_add(0, self._h_u, -half_cotangents).index_add(
            0, self._h_v, -half_cotangents
        )
        interior_cotangents = torch.where(
            self._h_interior, half_cotangents, 0.0
        )
        interior_edges = torch.zeros(
            self._n_edges, dtype=torch.float64
        ).index_add(0, self._h_edge, interior_cotangents)
        interior_vertices = torch.zeros(
            self._n_vertices, dtype=torch.float64
        ).index_add(0, self._h_u, -interior_cotangents).index_add(
            0, self._h_v, -interior_cotangents
        )

        angles = torch.acos(torch.clamp(
            cotangents / torch.sqrt(1 + cotangents**2), -1, 1
        ))
        angle_sum = torch.zeros(
            self._n_vertices, dtype=torch.float64
        ).index_add(0, self._h_w, angles)
        gaussian = torch.where(
            self._interior, (2 * np.pi - angle_sum) / vertex_areas, 0.0
        )
        curvature_loss = (
            (gaussian[self._fat_indices] - self._targets) ** 2 * self._weights
        ).sum()

        vertex_normals = torch.zeros((self._n_vertices, 3), dtype=torch.float64)
        for corner in range(3):
            vertex_normals = vertex_normals.index_add(
                0, faces[:, corner], normals
            )
        mean = torch.zeros((self._n_vertices, 3), dtype=torch.float64)
        mean = mean.index_add(
            0, self._e_u, edge_laplacian[:, None] * points[self._e_v]
        ).index_add(
            0, self._e_v, edge_laplacian[:, None] * points[self._e_u]
        )
        mean = -(
            mean + vertex_laplacian[:, None] * points
        ) / (2 * vertex_areas[:, None])
        mean_curvature = torch.where(
            self._interior,
            torch.linalg.vector_norm(mean, dim=1)
            * torch.sign((mean * vertex_normals).sum(dim=1)),
            0.0,
        )
        radicand = mean_curvature**2 - gaussian
        offset = torch.where(
            radicand > 0, torch.sqrt(torch.clamp_min(radicand, 1e-30)), 0.0
        )
        principal_1 = mean_curvature + offset
        principal_2 = mean_curvature - offset
        smooth_loss = -2 * (
            interior_edges
            * (
                principal_1[self._e_u] * principal_1[self._e_v]
                + principal_2[self._e_u] * principal_2[self._e_v]
            )
        ).sum()
        smooth_loss -= (
            interior_vertices * (principal_1**2 + principal_2**2)
        ).sum()
        smooth_loss *= areas.sum()

        loss = (
            self.lambda_curvature * curvature_loss
            + self.lambda_smooth * smooth_loss
        )
        gradient, = torch.autograd.grad(loss, heights)
        self._cached_parameters = parameters.copy()
        self._cached_loss = float(loss.detach())
        self._cached_gradient = gradient.detach().numpy().copy()
        self.curvature_loss.loss = float(curvature_loss.detach())
        self.kappa_G = gaussian.detach().numpy().copy()
        self.smooth_loss.loss = float(smooth_loss.detach())

    def forward(self, parameters=None):
        self._evaluate(self.mesh.get_parameters() if parameters is None else parameters)
        return self._cached_loss

    def reverse(self, parameters=None):
        self._evaluate(self.mesh.get_parameters() if parameters is None else parameters)
        return self._cached_gradient.copy()

    def diagnostics(self, parameters=None, f=None, context=None):
        loss = self.forward(parameters)
        print(
            f'iteration {self.iterations}:\n'
            f'\tL_curvature: {self.curvature_loss.loss:.6f}\n'
            f'\tL_smooth: {self.smooth_loss.loss:.6f}\n'
            f'\tLoss: {loss:.6f}\n'
        )
        if self.directory is not None and self.iterations % 100 == 0:
            with open(self.directory / f'{self.iterations}.json', 'w') as output:
                json.dump({
                    'mesh_parameters': self.mesh.get_parameters().tolist(),
                    'L_curvature': self.curvature_loss.loss,
                    'L_smooth': self.smooth_loss.loss,
                }, output)
        self.iterations += 1
