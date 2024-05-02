import csv
import os
import sys

from GraphRicciCurvature.OllivierRicci import OllivierRicci
import networkx as nx

sys.path.append(os.path.join('..', '..', 'src'))
from linear_geodesic_optimization.data import utility

name = 'elbow'
graph = nx.read_graphml(f'{name}.graphml')

orc = OllivierRicci(graph, weight='weight', alpha=0.)
graph = orc.compute_ricci_curvature()

# Delete extraneous edge data
for _, _, d in graph.edges(data=True):
    del d['weight']

for source_id, target_id, d in graph.edges(data=True):
    source = graph.nodes[source_id]
    target = graph.nodes[target_id]
    d['rtt'] = utility.get_GCL(
        (source['lat'], source['long']),
        (target['lat'], target['long'])
    )
rtts = nx.floyd_warshall(graph, 'rtt')

with open(f'{name}_probes.csv', 'w') as f:
    writer = csv.DictWriter(f, ['id', 'city', 'country', 'latitude', 'longitude'])
    writer.writeheader()

    for node, d in graph.nodes(data=True):
        writer.writerow({
            'id': node,
            'city': f'city {node}',
            'country': f'country {node}',
            'latitude': d['lat'],
            'longitude': d['long'],
        })

with open(f'{name}_latencies.csv', 'w') as f:
    writer = csv.DictWriter(f, ['source_id', 'target_id', 'rtt'])
    writer.writeheader()

    for source_node in graph.nodes:
        source = graph.nodes[source_node]
        for target_node in graph.nodes:
            target = graph.nodes[target_node]
            writer.writerow({
                'source_id': source_node,
                'target_id': target_node,
                'rtt': float(rtts[source_node][target_node])
            })

nx.write_graphml(graph, f'{name}.graphml')
