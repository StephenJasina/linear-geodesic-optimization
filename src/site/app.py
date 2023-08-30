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
from linear_geodesic_optimization import data
from linear_geodesic_optimization.mesh.basic import Mesh as BasicMesh
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization.geodesic import Computer as Geodesic

directory = os.path.join('site', 'example_output')
max_iterations = 10000
vertical_scale = 0.15

sys.path.append(r'python/surface/src')

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

@app.route('/calc-surface', methods=['POST'])
def calc_surface():
    json_data = request.json

    # TODO: Actually run the optimization algorithm with this data
    smooth_pen = int(json_data['smooth_pen'])
    niter = int(json_data['niter'])
    hmap = json_data['map']
    G = json_graph.node_link_graph(json_data['graph'])
    H = nx.Graph(G)

    mesh = data.get_mesh_output(directory, max_iterations, True)
    z = mesh.get_parameters()
    width = mesh.get_width()
    height = mesh.get_height()

    z = np.flip(z.reshape((width, height)), axis=1).T.reshape((-1))
    z = z - np.amin(z)
    z = z * vertical_scale / np.amax(z)
    z = z - np.amax(z)
    z = z.tolist()
    return Response(json.dumps(z), mimetype='text/plain')

@app.route('/')
def static_proxy():
    return app.send_static_file('index.html')

if __name__=='__main__':
    app.run()
