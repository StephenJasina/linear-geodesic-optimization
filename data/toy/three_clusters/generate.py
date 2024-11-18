import csv
import os
import pathlib
import sys

import networkx as nx
import numpy as np

sys.path.append(str(pathlib.PurePath('..', '..', '..', 'src')))
from linear_geodesic_optimization.data import curvature, utility


graph = nx.Graph()
n = 6
cluster_radius = 0.06

# Generate the first cluster
cluster_center = np.array([1., 0.]) / 4
for i in range(n):
    xy = cluster_center + cluster_radius * np.array([np.sin(2 * np.pi * i / n), np.cos(2 * np.pi * i / n)])
    long, lat = utility.inverse_mercator(xy[0], xy[1])
    graph.add_node(
        f'a{i}',
        city = f'a{i}',
        country = '',
        lat = lat,
        long = long
    )
for i in range(n - 1):
    for j in range(i + 1, n):
        graph.add_edge(
            f'a{i}', f'a{j}',
            throughput = 1.
        )

# Generate the second cluster
cluster_center = np.array([-0.5, np.sqrt(3.) / 2.]) / 4.
for i in range(n):
    xy = cluster_center + cluster_radius * np.array([np.sin(2 * np.pi * i / n), np.cos(2 * np.pi * i / n)])
    long, lat = utility.inverse_mercator(xy[0], xy[1])
    graph.add_node(
        f'b{i}',
        city = f'b{i}',
        country = '',
        lat = lat,
        long = long
    )
for i in range(n - 1):
    for j in range(i + 1, n):
        graph.add_edge(
            f'b{i}', f'b{j}',
            throughput = 1.
        )

# Generate the third cluster
cluster_center = np.array([-0.5, -np.sqrt(3.) / 2.]) / 4.
for i in range(n):
    xy = cluster_center + cluster_radius * np.array([np.sin(2 * np.pi * i / n), np.cos(2 * np.pi * i / n)])
    long, lat = utility.inverse_mercator(xy[0], xy[1])
    graph.add_node(
        f'c{i}',
        city = f'c{i}',
        country = '',
        lat = lat,
        long = long
    )
for i in range(n - 1):
    for j in range(i + 1, n):
        graph.add_edge(
            f'c{i}', f'c{j}',
            throughput = 1.
        )

# Join the clusters
graph.add_edge('a5', 'b2', throughput = 1.)
graph.add_edge('a4', 'c1', throughput = 1.)

# Write the probes
with open('probes.csv', 'w') as f:
    writer = csv.DictWriter(f, [
        'id',
        'city',
        'country',
        'latitude',
        'longitude',
    ])
    writer.writeheader()

    for node, data in sorted(graph.nodes(data=True)):
        writer.writerow({
            'id': node,
            'city': data['city'],
            'country': data['country'],
            'latitude': data['lat'],
            'longitude': data['long'],
        })

# Write the throughputs and graphml
os.makedirs('throughputs', exist_ok=True)
os.makedirs('graphml', exist_ok=True)
# for index, center_throughput in enumerate(np.linspace(2., 0., 25)):
#     graph.edges['a0', 'b0']['throughput'] = float(center_throughput)
for index, _ in enumerate([None]):
    ricci_curvatures = curvature.ricci_curvature_optimal_transport(graph, edge_weight_label = 'throughput')
    for (source, target), ricci_curvature in ricci_curvatures.items():
        graph.edges[source, target]['ricciCurvature'] = ricci_curvature

    with open(pathlib.PurePath('throughputs', f'{index}.csv'), 'w') as f:
        writer = csv.DictWriter(f, [
            'source_id',
            'target_id',
            'throughput',
        ])
        writer.writeheader()

        for source, target, data in graph.edges(data=True):
            writer.writerow({
                'source_id': source,
                'target_id': target,
                'throughput': data['throughput'],
            })

    nx.write_graphml(graph, pathlib.PurePath('graphml', f'{index}.graphml'))
