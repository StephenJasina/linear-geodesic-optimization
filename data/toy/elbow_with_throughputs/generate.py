import csv
import itertools
import os
import pathlib
import sys

import networkx as nx
import numpy as np

sys.path.append(str(pathlib.PurePath('..', '..', '..', 'src')))
from linear_geodesic_optimization.data import input_network


edges_to_remove = [
]

output_directory = pathlib.PurePath('_'.join(itertools.chain(['removed'], [''.join(pair) for pair in edges_to_remove])))

# Create the network in NetworkX
network = input_network.get_graph_from_csvs(
    pathlib.PurePath('elbow_probes.csv'),
    pathlib.PurePath('elbow_links.csv'),
)

# Remove the given edge and recompute the Ricci curvature
for edge in edges_to_remove:
    if (edge[0], edge[1]) in network.edges:
        network.remove_edge(edge[0], edge[1])
network = input_network.compute_ricci_curvatures(network)

# Add throughputs to the network
shortest_paths = dict(nx.all_pairs_shortest_path(network))
for _, _, edge_data in network.edges(data=True):
    edge_data['throughput'] = 0.
for u, shortest_paths_u in shortest_paths.items():
    for v, path_uv in shortest_paths_u.items():
        for (s, t) in itertools.pairwise(path_uv):
            network.edges[s, t]['throughput'] += 1. / len(path_uv)
throughputs = [edge_data['throughput'] for _, _, edge_data in network.edges(data=True)]
throughput_max = max(throughputs)
for _, _, edge_data in network.edges(data=True):
    edge_data['throughput'] *= 0.4 / throughput_max

# Save the network
os.makedirs(output_directory, exist_ok=True)
with open(output_directory / 'probes.csv', 'w') as file_probes:
    writer = csv.DictWriter(file_probes, ['id', 'city', 'country', 'latitude', 'longitude'])
    writer.writeheader()
    for node, node_data in network.nodes(data=True):
        writer.writerow({
            'id': node,
            'city': node_data['city'],
            'country': node_data['country'],
            'latitude': node_data['lat'],
            'longitude': node_data['long'],
        })
with open(output_directory / 'links.csv', 'w') as file_links:
    writer = csv.DictWriter(file_links, ['source_id', 'target_id', 'rtt', 'gcl', 'throughput'])
    writer.writeheader()
    for source, target, edge_data in network.edges(data=True):
        writer.writerow({
            'source_id': source,
            'target_id': target,
            'rtt': edge_data['rtt'],
            'gcl': edge_data['gcl'],
            'throughput': edge_data['throughput']
        })
nx.write_graphml(network, output_directory / 'graph.graphml')
