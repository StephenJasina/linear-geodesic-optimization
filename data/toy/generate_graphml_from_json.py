import csv
import json
import math
import os
import sys
import typing

import networkx as nx


def inverse_mercator(x: float, y: float) -> typing.Tuple[float]:
    longitude = x * 360.
    latitude = math.atan(math.exp(y * 2. * math.pi)) * 360. / math.pi - 90.
    return (longitude, latitude)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.stderr.write(f'Usage: python {sys.argv[0]} <JSON file>\n')
        sys.exit(0)

    with open(sys.argv[1], 'r') as json_file:
        json_data = json.loads(json_file.read())

    positions = json_data['position']
    curvatures = json_data['curvature']
    latencies = json_data['latency']

    xs, ys = zip(*positions.values())
    divisor = 2 * (max(*xs, *ys) - min(*xs, *ys))
    mu_x = sum(xs) / len(xs)
    mu_y = sum(ys) / len(ys)

    network = nx.Graph()
    for vertex_label, position in positions.items():
        longitude, latitude = inverse_mercator(
            float((position[0] - mu_x) / divisor),
            float((position[1] - mu_y) / divisor)
        )
        network.add_node(vertex_label, long=longitude, lat=latitude)
    for edge_label, curvature in curvatures.items():
        u_label, v_label = tuple(edge_label)
        network.add_edge(u_label, v_label, ricciCurvature=curvature)

    name = os.path.splitext(os.path.basename(sys.argv[1]))[0]
    nx.graphml.write_graphml(network, f'{name}.graphml')

    with open(f'{name}_latencies.csv', 'w') as latencies_csv:
        latencies_writer = csv.writer(latencies_csv)
        for edge_label, latency in latencies.items():
            u_label, v_label = tuple(edge_label)
            latencies_writer.writerow([u_label, v_label, latency])
