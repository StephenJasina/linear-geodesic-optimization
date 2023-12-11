import argparse
import csv
import os
import sys

import csv_to_graphml


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--probes-file', '-p', type=str, required=True,
                        dest='probes_filename', metavar='<filename>',
                        help='Input file containing probes information')
    parser.add_argument('--latencies-file', '-l', type=str, required=True,
                        dest='latencies_filename', metavar='<filename>',
                        help='Input file containing latency information')
    parser.add_argument('--epsilon', '-e', type=float, required=False,
                        dest='epsilon', metavar='<epsilon>',
                        help='Residual threshold')
    args = parser.parse_args()
    probes_filename = args.probes_filename
    latencies_filename = args.latencies_filename
    epsilon = args.epsilon
    if epsilon is None:
        epsilon = float('inf')

    graph = csv_to_graphml.get_graph(
        probes_filename, latencies_filename, epsilon, 500000, 2
    )

    severities = []
    triangle_count = 0
    tiv_count = 0
    tiv_edges = set()
    for u, v, data in graph.edges(data=True):
        rtt_uv = data['rtt']
        for w in graph.nodes:
            if (u, w) in graph.edges and (v, w) in graph.edges:
                triangle_count += 1
                rtt_uw = graph.edges[u,w]['rtt']
                rtt_vw = graph.edges[v,w]['rtt']

                rtt_max = max(rtt_uv, rtt_uw, rtt_vw)
                rtt_sum = rtt_uv + rtt_uw + rtt_vw
                if 2 * rtt_max > rtt_sum:
                    if rtt_uv > rtt_uw:
                        if rtt_uv > rtt_vw:
                            tiv_edges.add((min(u, v), max(u, v)))
                        else:
                            tiv_edges.add((min(v, w), max(v, w)))
                    else:
                        if rtt_uw > rtt_vw:
                            tiv_edges.add((min(u, w), max(u, w)))
                        else:
                            tiv_edges.add((min(v, w), max(v, w)))


                    tiv_count += 1
                    severity = rtt_max / (rtt_sum - rtt_max)
                    severities.append(severity)
                    print(f'TIV with severity {severity:.4f}')
    print(f'Proportion of TIVs (triangles): {(tiv_count / triangle_count):.4f}')
    print(f'Proportion of TIVs (edges): {(len(tiv_edges) / len(graph.edges)):.4f}')
    print(f'Worst severity: {max(severities):.4f}')
