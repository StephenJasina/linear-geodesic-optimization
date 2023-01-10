#!/usr/bin/python3
import json
import flask
import numpy as np
import potpourri3d as pp3d
import csv
from flask import request
from flask import Response
import networkx as nx
from networkx.readwrite import json_graph
from OllivierRicci import ricciCurvature
from python.surface import BasicDemo as bd
from python.surface.src.generating_tessalation import generating_tessalation_2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import concurrent.futures
import sys
import io
import time
import pickle
from python.geodesic import GeodesicDistanceComputation

sys.path.append('..')
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh

mesh_path = '/home/jasina/Desktop/out/20230105_125532a/1695'

sys.path.append(r'python/surface/src')

app = flask.Flask(__name__, static_folder='')
retval = None

@app.route('/dummy', methods=['POST'])
def dummy():
    data = request.json
    # print(data)

    return data

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

def get_z_from_path(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        mesh = data['mesh']
        vertices = mesh.get_vertices()
        x = list(sorted(set(vertices[:,0])))
        y = list(sorted(set(vertices[:,1])))
        z = np.rot90(vertices[:,2].reshape(len(x), len(y))).reshape((-1,))
        return z.tolist()

@app.route('/calc-surface', methods=['POST'])
def calc_surface():
    global retval
    print('start')
    data = request.json
    # print(data)
    # print("\n\n")

    smooth_pen = int(data['smooth_pen'])
    niter = int(data['niter'])
    hmap = data['map']
    print(hmap)
    ### @Stephen: this is here where we have to update the hmap! ###
    G = json_graph.node_link_graph(data['graph'])
    H = nx.Graph(G)
    # print(type(H))
    # print(G.edges(data=True))
    # print("\n\n")
    print("Output graph")
    nx.write_graphml(H, "newgraph.graphml")
    ### takes a graph as input and generates a tessellation on a 2D plane.
    if (retval == None):
        # TODO: This should be changed to a different initialization strategy
        ret = generating_tessalation_2(H)
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
    return Response(json.dumps(get_z_from_path(mesh_path)), mimetype='text/plain')

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
