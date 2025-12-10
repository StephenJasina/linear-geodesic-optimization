import json
import pathlib

import networkx as nx
import numpy as np

from linear_geodesic_optimization.data import input_network
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh

def write_output_from_function(
    function,
    path,
    width=50,
    height=50
):
    mesh = RectangleMesh(width, height)
    z_0 = [0] * mesh.get_topology().n_vertices
    z = [
        function(x, y)
        for x, y, _ in mesh.get_coordinates()
    ]
    parameters = {
        'filename_probes': None,
        'filename_links': None,
        'filename_graphml': None,
        'epsilon': None,
        'clustering_distance': None,
        'should_remove_TIVs': False,
        'ricci_curvature_alpha': 0.,
        'lambda_curvature': 0.,
        'lambda_smooth': 0.,
        'initial_radius': 20.,
        'width': width,
        'height': height,
        'mesh_scale': 1.,
        'coordinates_scale': 0.8,
        'network_trim_radius': None,
    }
    with open(path, 'w') as file_output:
        json.dump({
            'parameters': parameters,
            'initial': z_0,
            'final': z,
            'network': input_network.get_network_data(nx.Graph()),
        }, file_output, ensure_ascii=False)

if __name__ == '__main__':
    def f(x, y):
        return (x - 0.5)**2 + (y - 0.5)**2
    def g(x, y):
        return (x - 0.5)**2 - (y - 0.5)**2
    write_output_from_function(
        f,
        pathlib.PurePath('positive.json'),
    )
    write_output_from_function(
        g,
        pathlib.PurePath('negative.json'),
    )
