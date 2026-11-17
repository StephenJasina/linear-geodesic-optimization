"""Tests for the shared helpers in `linear_geodesic_optimization.pipeline`."""

import numpy as np

from linear_geodesic_optimization import pipeline
from linear_geodesic_optimization.mesh.rectangle import Mesh
from linear_geodesic_optimization.optimization.torch_optimization import Computer


def test_computer_from_output_reproduces_its_loss():
    sides = 21
    mesh = Mesh(sides, sides, 1.)
    xy = mesh.get_coordinates()[:, :2]
    z = 0.1 * np.exp(-((xy[:, 0] - 0.5)**2 + (xy[:, 1] - 0.5)**2) / 0.05)
    output = {
        'parameters': {
            'width': sides, 'height': sides, 'mesh_scale': 1.,
            'coordinates_scale': 0.8, 'lambda_curvature': 1., 'lambda_smooth': 0.01,
            'smooth_loss_type': 'mvs', 'backend': 'torch',
        },
        'network': (
            {'labels': ['a', 'b', 'c'], 'coordinates': [[0., 0.], [1., 0.], [0., 1.]],
             'edges': [[0, 1], [1, 2], [0, 2]], 'bounding_box': None},
            {},
            {'ricciCurvature': [-0.5, 0.2, 0.1]},
        ),
    }
    computer = pipeline.computer_from_output(output, mesh)
    network_vertices, network_edges, network_curvatures = pipeline.map_network_to_mesh(
        mesh, output['network'], 0.8
    )
    reference = Computer(
        mesh, network_vertices, network_edges, network_curvatures,
        pipeline.fat_edge_width(sides, 1.), 1., 0.01,
    )
    assert np.isclose(computer.forward(z), reference.forward(z))
