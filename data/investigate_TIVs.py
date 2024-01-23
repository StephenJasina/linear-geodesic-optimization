import argparse
import csv
import collections
import os
import sys
import typing

import matplotlib.pyplot as plt
import networkx as nx

import csv_to_graphml


def compute_triangles(graph: nx.Graph, weight: str = 'rtt') \
        -> typing.Dict[typing.Tuple[str, str, str],
                       typing.Tuple[float, float, float]]:
    triangles: typing.Dict[typing.Tuple[str, str, str],
                           typing.Tuple[float, float, float]] = {}
    for u, v, data in graph.edges(data=True):
        weight_uv = data[weight]
        u, v = min(u, v), max(u, v)
        for w in graph.nodes:
            if w <= v:
                continue

            if (u, w) in graph.edges and (v, w) in graph.edges:
                weight_uw = graph.edges[u,w][weight]
                weight_vw = graph.edges[v,w][weight]

                triangles[u,v,w] = (weight_vw, weight_uw, weight_uv)

    return triangles

def get_long_edges(
    graph: nx.Graph,
    triangles: typing.Dict[typing.Tuple[str, str, str],
                           typing.Tuple[float, float, float]]
) -> typing.Set[typing.Tuple[str, str]]:
    long_edges = set()
    for u, v, w in triangles:
        weight_uv = graph.edges[u,v]['rtt']
        weight_uw = graph.edges[u,w]['rtt']
        weight_vw = graph.edges[v,w]['rtt']

        if weight_uv > weight_uw:
            if weight_uv > weight_vw:
                long_edges.add((u, v))
            else:
                long_edges.add((v, w))
        else:
            if weight_uw > weight_vw:
                long_edges.add((u, w))
            else:
                long_edges.add((v, w))
    return long_edges


def compute_goodnesses(
    triangles: typing.Dict[typing.Tuple[str, str, str],
                           typing.Tuple[float, float, float]],
    use_r: bool = False
) -> typing.Dict[typing.Tuple[str, str, str], float]:
    goodnesses: typing.Dict[typing.Tuple[str, str, str], float] = {}
    for triangle, (weight_uv, weight_uw, weight_vw) in triangles.items():
        weight_max = max(weight_uv, weight_uw, weight_vw)
        weight_sum = weight_uv + weight_uw + weight_vw
        goodness = weight_max / (weight_sum - weight_max)
        if use_r:
            goodness *= (1 + 2 * weight_max - weight_sum)
        goodnesses[triangle] = goodness

    return goodnesses

def plot_goodnesses(goodnesses):
    fig = plt.figure()
    ax_1, ax_2 = fig.subplots(1, 2, sharey=True)
    if goodnesses:
        ax_1.ecdf(goodnesses.values())
    ax_1.plot([1, 1], [0, 1], 'r-.')
    ax_1.set_title('Goodness (All Triangles)')
    ax_1.set_xlabel('Goodness (lower = better)')
    ax_1.set_ylabel('CDF')

    goodness_violators = [goodness for goodness in goodnesses.values() if goodness > 1]
    if goodness_violators:
        ax_2.ecdf(goodness_violators)
    ax_2.set_title('Goodness (Violators Only)')
    ax_2.set_xlabel('Goodness (lower = better)')
    ax_2.set_ylabel('CDF')
    plt.show()

def compute_greedy_set_cover(sets: typing.Dict[typing.Hashable,
                                               typing.Set[typing.Any]]) \
        -> typing.List[typing.Hashable]:
    current_cover = set()
    candidate_sets = dict(sets)
    set_cover: typing.List[typing.Hashable] = []
    while True:
        # Find the candidate with the most uncovered elements
        best_candidate = None
        best_candidate_set = None
        best_candidate_size = 0
        removable_candidates = []
        for candidate, candidate_set in candidate_sets.items():
            # Small optimization to avoid unnecessary difference computations
            if len(candidate_set) <= best_candidate_size:
                continue

            candidate_size = len(candidate_set.difference(current_cover))

            # Remove sets that are already covered
            if candidate_size == 0:
                removable_candidates.append(candidate)
                continue

            if candidate_size > best_candidate_size:
                best_candidate = candidate
                best_candidate_set = candidate_set
                best_candidate_size = candidate_size

        # We're done if no elements are uncovered
        if best_candidate is None:
            return set_cover

        for candidate in removable_candidates:
            del candidate_sets[candidate]

        set_cover.append(best_candidate)
        current_cover = current_cover.union(best_candidate_set)

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
        probes_filename, latencies_filename, epsilon, 500000
    )

    triangles = compute_triangles(graph)
    goodnesses = compute_goodnesses(triangles, use_r=False)

    tivs = [
        triangle
        for triangle, goodness in goodnesses.items()
        if goodness > 1.
    ]

    long_edges = get_long_edges(graph, tivs)

    tiv_edge_sets = collections.defaultdict(set)
    for i, (u, v, w) in enumerate(tivs):
        tiv_edge_sets[u,v].add(i)
        tiv_edge_sets[u,w].add(i)
        tiv_edge_sets[v,w].add(i)
    tiv_edge_set_cover = compute_greedy_set_cover(tiv_edge_sets)

    tiv_vertex_sets = collections.defaultdict(set)
    for i, (u, v, w) in enumerate(tivs):
        tiv_vertex_sets[u].add(i)
        tiv_vertex_sets[v].add(i)
        tiv_vertex_sets[w].add(i)
    tiv_vertex_set_cover = compute_greedy_set_cover(tiv_vertex_sets)

    print(f'Proportion of TIVs (triangles): {(len(tivs) / len(goodnesses)):.4f}')
    print(f'Proportion of long edges: {len(long_edges) / len(graph.edges):.4f}')
    print('Approximate proportion of TIVs (edges): '
          + f'{len(tiv_edge_set_cover) / len(graph.edges):.4f}')
    print('Approximate proportion of TIVs (vertices): '
          + f'{len(tiv_vertex_set_cover) / len(graph.nodes):.4f}')
    print(f'Worst goodness: {max(goodnesses.values()):.4f}')

    plot_goodnesses(goodnesses)

    thresholds = list(sorted(
        threshold
        for _, _, d in graph.edges(data=True)
        for threshold in (d['rtt'] - d['gcl'],)
    ))
    edge_counts = []
    long_edge_counts = []
    for i in range(0, len(thresholds), len(thresholds) // 100):
        threshold = thresholds[i]
        graph = csv_to_graphml.get_graph(
            probes_filename, latencies_filename, threshold, 500000
        )

        triangles = compute_triangles(graph)
        goodnesses = compute_goodnesses(triangles, use_r=False)

        tivs = [
            triangle
            for triangle, goodness in goodnesses.items()
            if goodness > 1.05
        ]

        long_edges = get_long_edges(graph, tivs)

        edge_counts.append(len(graph.edges))
        long_edge_counts.append(len(long_edges))
        print(len(graph.edges), len(long_edges), len(graph.edges) / max(1, len(long_edges)))

    plt.plot(edge_counts, long_edge_counts)
    plt.xlabel('Number of Total Edges')
    plt.ylabel('Number of Violating Edges')
    plt.title('Proportion of Violating Edges across Thresholds')
    plt.show()
