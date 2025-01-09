import csv
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from linear_geodesic_optimization.data import input_network


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Expected a single directory as input")
        sys.exit(1)
    directory = sys.argv[1]
    latencies = {}
    with open(os.path.join(directory, 'parameters'), 'rb') as f:
        parameters = pickle.load(f)
        latencies_file_path = os.path.join('..', 'data', parameters['latencies_filename'])
        probes_file_path = os.path.join('..', 'data', parameters['probes_filename'])
        epsilon = parameters['epsilon']
        clustering_distance = parameters['clustering_distance']

        graph = input_network.get_graph_from_paths(
            probes_file_path, latencies_file_path,
            epsilon=epsilon, 
            clustering_distance=clustering_distance
        )
    with open(latencies_file_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['rtt']:
                latencies[(row['source_id'], row['target_id'])] = float(row['rtt'])
    geodesics = {}
    with open(os.path.join(directory, 'geodesics.csv')) as f:
        reader = csv.DictReader(f)
        for row in reader:
            geodesics[(row['source'], row['destination'])] = float(row['geodesic_distance'])

    id_to_node = {}
    for node, data in graph.nodes(data=True):
        for id in data['elements']:
            id_to_node[id] = node
    # for node, data in graph.nodes(data=True):
    #     id_to_node[node] = node

    valid_keys = [
        (source_id, target_id)
        for (source_id, target_id) in latencies
        if source_id in id_to_node and target_id in id_to_node
    ]
    x = [
        latencies[(source_id, target_id)]
        for (source_id, target_id) in valid_keys
    ]
    y = [
        geodesics[(id_to_node[source_id], id_to_node[target_id])]
        for (source_id, target_id) in valid_keys
    ]

    print(stats.linregress(x, y))

    plt.plot(x, y, 'b.')
    plt.show()
