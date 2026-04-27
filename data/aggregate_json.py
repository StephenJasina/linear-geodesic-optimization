"""
Utility to aggregate a set of JSON files (essentially summing them).
"""

import argparse
import collections
import json


def aggregate_json_blobs(files, path_output):
    nodes = {}
    links = {}
    traffic = collections.defaultdict(float)

    for path in files:
        with open(path, 'r') as f:
            blob = json.load(f)

        for node in blob['nodes']:
            if node['id'] not in nodes:
                nodes[node['id']] = node

        for link in blob['links']:
            if (link['source_id'], link['target_id']) not in links:
                links[link['source_id'], link['target_id']] = link

        for flow in blob['traffic']:
            traffic[tuple(flow['route'])] += flow['volume']

    with open(path_output, 'w') as f:
        json.dump({
            'nodes': [value for _, value in sorted(nodes.items())],
            'links': [value for _, value in sorted(links.items())],
            'traffic': [
                {
                    'route': route,
                    'volume': volume,
                }
                for route, volume in traffic.items()
            ]
        }, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')
    parser.add_argument('-o', '--output', required=True)

    args = parser.parse_args()
    aggregate_json_blobs(args.files, args.output)
