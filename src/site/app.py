#!/usr/bin/python3
import json
import flask
import numpy as np
import csv
from flask import request
from flask import Response
import networkx as nx
from networkx.readwrite import json_graph
from OllivierRicci import ricciCurvature
import potpourri3d as pp3d
from python.surface import BasicDemo as bd
from python.surface.src.generating_tessalation import generating_tessalation_2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import concurrent.futures
import sys
import io
import time
import pickle
from python.geodesic import GeodesicDistanceComputation

# Assume we're running from src/
sys.path.append('.')
from linear_geodesic_optimization import data
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh

directory = '../out_US/graph16/mvs_cross/0.0_1.0_0.0002_16.0_40_40/'
max_iterations = 1000
scale = 0.5

sys.path.append(r'python/surface/src')

app = flask.Flask(__name__, static_folder='')
retval = None

@app.route('/calc-curvature', methods=['POST'])
def calc_curvature():
    data = request.json
    # print(data)
    G = json_graph.node_link_graph(data)
    Gf = ricciCurvature(G,alpha=0,verbose=True)
    Gr = json_graph.node_link_data(Gf)

    return Gr

@app.route('/calc-distance', methods=['POST'])
def calc_distance():
    data = request.json
    # print(data)
    verts = np.array(data['verts'])
    tris = np.array(data['faces'])
    nodes = np.array(data['nodes'])
    edges = np.array(data['edges'])
    # print(verts.shape)
    compute_distance = GeodesicDistanceComputation(verts, tris)
    distances = []
    grads = []
    paths = []
    for node in nodes:
        dist = compute_distance(node)
        # print(dist.shape)
        distances.append(dist.tolist())
        #TODO check reshape for extended plane
        # grad = np.gradient(dist.reshape(100, 50))
        grad = np.gradient(dist)
        grads.append(grad.tolist())
    path_solver = pp3d.EdgeFlipGeodesicSolver(verts, tris)
    for edge in edges:
        if edge[0] != edge[1]:
            paths.append(path_solver.find_geodesic_path(v_start=edge[0], v_end=edge[1]).tolist())
        else:
            paths.append([[0, 0, 0]])
    # print(path_pts)
    ret = {}
    ret['distances'] = distances
    ret['grads'] = grads
    ret['paths'] = paths
    # dist = np.trunc(distances[0]).reshape((50, 50))
    # dist = compute_distance(0)
    # dist = dist.reshape(50, 50)
    # print(np.gradient(dist))
    # print(dist.size)
    # with np.printoptions(threshold=np.inf):
    #     print(dist)
    return json.dumps(ret)

@app.route('/refine', methods=["GET"])
def refine():
    send_data = {}
    with open("FinalResult.csv", mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for index, row in enumerate(csv_reader):
            if index > 0:
                cities = row['Cities'].strip("()").split(',')
                cities[0] = cities[0].strip('\'')
                cities[1] = cities[1].strip(' \'')
                if row['Latency'] == "":
                    continue
                diff = float(row['Latency']) - float(row['GeodesicPredic'])
                # print(cities[0] + " to " + cities[1] + ": " + str(diff))
                if cities[0] < cities[1]:
                    if not cities[0] in send_data:
                        send_data[cities[0]] = {}
                    send_data[cities[0]][cities[1]] = diff

    print(send_data.keys())
    return json.dumps(send_data)

@app.route('/calc-surface', methods=['POST'])
def calc_surface():
    global retval
    json_data = request.json
    # print(data)
    # print("\n\n")

    smooth_pen = int(json_data['smooth_pen'])
    niter = int(json_data['niter'])
    hmap = json_data['map']
    G = json_graph.node_link_graph(json_data['graph'])
    H = nx.Graph(G)
    # print(type(H))
    # print(G.edges(data=True))
    # print("\n\n")
    # nx.write_graphml(H, "newgraph.graphml")
    ### takes a graph as input and generates a tessellation on a 2D plane.
    if (retval == None):
        # TODO: This should be changed to a different initialization strategy
        # ret = generating_tessalation_2(H)
        ret = None
        retval = ret
    else:
        ret = retval

    def generate():
        cur_time = time.time()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # TODO: This is where we should call a different optimization strategy
            ### take as input a heatmap, smooth penality and number of iterations
            future = executor.submit(bd.main, ret, smooth_pen, niter, hmap)
            while future.running():
                time.sleep(5)
                if time.time() - cur_time > 10:
                    cur_time = time.time()
                    yield json.dumps('')
            result = future.result()
            yield json.dumps(result.tolist())
    #     for i in range(10):
    #         print(i+10)
    #         yield json.dumps(i+10)
    #         import time
    #         time.sleep(5)
    mesh = data.get_postprocessed_output(directory, max_iterations)
    z = mesh.get_parameters()
    z = z - np.amin(z)
    z = z * scale / np.amax(z)
    z = z - np.amax(z)
    z = z.tolist()
    return Response(json.dumps(z), mimetype='text/plain')

    # Generator test
    # for val in bd.main(ret):
    #     print(val)

    # plot = bd.get_heatmap(ret)
    # output = io.BytesIO()
    #print("\nplot\n")
    # FigureCanvas(plot).print_png(output)
    # return Response(output.getvalue(), mimetype='image/png')

    # zf = bd.main(ret)
    # return json.dumps(zf.tolist())

@app.route('/')
def static_proxy():
    return app.send_static_file('index.html')

if __name__=='__main__':
    app.run()
