"""Convert the graph{N}.graphml latency-threshold sweep into a single
graph.json in the esnet JSON format.
"""

import json
import pathlib
import re
import sys

import networkx as nx

DIRECTORY = pathlib.Path(__file__).resolve().parent
sys.path.append(str(DIRECTORY.parents[3] / 'src'))
from linear_geodesic_optimization.data import utility


def _get_threshold(path):
    match = re.search(r'graph(\d+)\.graphml$', path.name)
    return int(match.group(1))


def main():
    paths = sorted(DIRECTORY.glob('graph*.graphml'), key=_get_threshold)

    nodes = {}
    edge_thresholds = {}
    for path in paths:
        threshold = _get_threshold(path)
        graph = nx.read_graphml(path)

        for node, data in graph.nodes(data=True):
            if node not in nodes:
                nodes[node] = {
                    'id': node,
                    'latitude': data['lat'],
                    'longitude': data['long'],
                }

        for u, v in graph.edges():
            key = (min(u, v), max(u, v))
            if key not in edge_thresholds:
                edge_thresholds[key] = threshold

    links = []
    delays = []
    for (u, v), threshold in sorted(
        edge_thresholds.items(), key=lambda kv: (int(kv[0][0]), int(kv[0][1]))
    ):
        rtt = utility.get_GCL(
            (nodes[u]['latitude'], nodes[u]['longitude']),
            (nodes[v]['latitude'], nodes[v]['longitude']),
        ) + threshold
        for source, target in ((u, v), (v, u)):
            links.append({'source_id': source, 'target_id': target})
            delays.append({
                'source_id': source,
                'target_id': target,
                'rtt': rtt,
            })

    data = {
        'nodes': sorted(nodes.values(), key=lambda node: int(node['id'])),
        'links': links,
        'delays': delays,
    }

    with open(DIRECTORY / 'graph.json', 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == '__main__':
    main()
