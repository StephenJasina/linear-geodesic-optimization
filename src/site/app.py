#!/usr/bin/python3
import json
import sys

import dcelmesh
import flask
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import numpy as np
from networkx.readwrite import json_graph

# Assume we're running from src/
sys.path.append('.')
from linear_geodesic_optimization.mesh.basic import Mesh as BasicMesh
from linear_geodesic_optimization.optimization.geodesic import Computer as Geodesic

app = flask.Flask(__name__, static_folder='')
retval = None

@app.route('/calc-curvature', methods=['POST'])
def calc_curvature():
    data = flask.request.json
    G = json_graph.node_link_graph(data, multigraph=False)
    G = OllivierRicci(G, alpha=0).compute_ricci_curvature()
    return json_graph.node_link_data(G)

@app.route('/calc-distance', methods=['POST'])
def calc_distance():
    data = flask.request.json
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

@app.route('/')
def static_proxy():
    return app.send_static_file('index.html')

if __name__=='__main__':
    app.run()
