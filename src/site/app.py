#!/usr/bin/python3
import csv
import json
import os
import pickle
import sys

import dcelmesh
import flask
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import numpy as np
from flask import request
from flask import Response
import networkx as nx
from networkx.readwrite import json_graph

# Assume we're running from src/
sys.path.append('.')
from linear_geodesic_optimization.data import input_mesh, utility
from linear_geodesic_optimization.mesh.basic import Mesh as BasicMesh
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization.geodesic import Computer as Geodesic

app = flask.Flask(__name__, static_folder='')
retval = None

@app.route('/calc-curvature', methods=['POST'])
def calc_curvature():
    data = request.json
    G = json_graph.node_link_graph(data, multigraph=False)
    G = OllivierRicci(G, alpha=0).compute_ricci_curvature()
    return json_graph.node_link_data(G)

@app.route('/calc-distance', methods=['POST'])
def calc_distance():
    data = request.json
    verts = np.array(data['verts'])
    tris = np.array(data['faces'])
    edges = np.array(data['edges'])
    mesh = BasicMesh(dcelmesh.Mesh(len(verts), tris), verts)
    geodesics = [
        Geodesic(mesh, u, v)
        for (u, v) in edges
    ]
    for geodesic in geodesics:
        geodesic.forward()
    ret = {
        'distances': [geodesic.distance for geodesic in geodesics],
        'paths': [geodesic.path_coordinates for geodesic in geodesics]
    }
    return json.dumps(ret)

@app.route('/unpickle', methods=['POST'])
def unpickle():
    unpickled_data = pickle.loads(request.get_data())
    if (
        'parameters' in unpickled_data
        and 'initial' in unpickled_data
        and 'final' in unpickled_data
        and 'network' in unpickled_data
    ):
        # Get height data
        z = unpickled_data['final']
        z_0 = unpickled_data['initial']

        parameters = unpickled_data['parameters']
        width = parameters['width']
        height = parameters['height']
        coordinates_scale = parameters['coordinates_scale']

        network = unpickled_data['network']

        mesh = input_mesh.get_mesh(z, width, height, network, coordinates_scale, True, z_0)
        z = mesh.get_parameters()
        z = np.flip(z.reshape((width, height)), axis=1).T.reshape((-1))
        z = z - np.amin(z)
        z = z * 0.15 / np.amax(z)
        z = z - np.amax(z)
        z = z.tolist()

        # Get map and network data
        coordinates, bounding_box, network_edges, network_curvatures, network_latencies = network
        coordinates = np.array(coordinates)
        network_vertices = mesh.map_coordinates_to_support(coordinates, coordinates_scale, bounding_box)
        vertices = [
            (network_vertex[1] * 20., network_vertex[0] * 20.)
            for network_vertex in network_vertices
        ]
        edges = [
            [str(edge[0]), str(edge[1]), curvature]
            for edge, curvature in zip(network_edges, network_curvatures)
        ]

        center_xy = (np.amin(coordinates, axis=0) + np.amax(coordinates, axis=0)) / 2.
        center = utility.inverse_mercator(*center_xy)
        left, _ = utility.inverse_mercator(np.amin(coordinates[:,0]), 0)
        right, _ = utility.inverse_mercator(np.amax(coordinates[:,0]), 0)
        zoom_factor = coordinates_scale * 360. / (right - left)

        to_return = {
            'heights': z,
            'vertices': vertices,
            'edges': edges,
            'mapCenter': center,
            'mapZoomFactor': zoom_factor,
        }
    else:
        to_return = {}

    return Response(json.dumps(to_return), mimetype='application/json')

@app.route('/')
def static_proxy():
    return app.send_static_file('index.html')

if __name__=='__main__':
    app.run()
