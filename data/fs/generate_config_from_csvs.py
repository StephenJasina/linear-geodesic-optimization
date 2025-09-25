import argparse
import collections
import csv
import itertools
import json
import os
import pathlib
import sys

import networkx as nx
import numpy as np
from scipy import optimize, sparse

sys.path.insert(0, str(pathlib.PurePath('..', '..', 'src')))
from linear_geodesic_optimization.data import utility


def convert_string_to_indetifier(s: str):
    return ('_' if s is not None and '0' <= s[0] <= '9' else '') + ''.join(
        c if c != ' ' else '_'
        for c in s
        if 'a' <= c <= 'z' or 'A' <= c <= 'Z' or '0' <= 'c' <= '9' or c == '_' or c == ' '
    )

def generate_traffic_matrix(graph: nx.Graph):
    indices_to_link_ids = []
    link_ids_to_indices = {}
    traffic_out_per_node = collections.defaultdict(float)
    traffic_in_per_node = collections.defaultdict(float)
    traffic_total = 0.
    for index, (id_source, id_target, data) in enumerate(graph.edges(data=True)):
        throughput = data['throughput']

        indices_to_link_ids.append((id_source, id_target))
        link_ids_to_indices[(id_source, id_target)] = index

        traffic_out_per_node[id_source] += throughput
        traffic_in_per_node[id_target] += throughput
        traffic_total += throughput
    # Some obvious error prevention
    traffic_out_per_node = dict(traffic_out_per_node)
    traffic_in_per_node = dict(traffic_in_per_node)

    traffic_counts = np.array([
        graph.edges[source_id, target_id]['throughput']
        for source_id, target_id in indices_to_link_ids
    ])
    sources, targets = zip(*[
        (source, target)
        for source in graph.nodes
        for target in graph.nodes
        if (
            source != target
            and traffic_out_per_node[source] > 0.
            and traffic_in_per_node[target] > 0.
        )
    ])

    # TODO: Make this more efficient?
    traffic_matrix_data = []
    traffic_matrix_row_ind = []
    traffic_matrix_col_ind = []
    for index, (source, target) in enumerate(zip(sources, targets)):
        route = nx.dijkstra_path(graph, source, target, 'latency')
        for link_id in itertools.pairwise(route):
            traffic_matrix_data.append(1)
            traffic_matrix_row_ind.append(link_ids_to_indices[link_id])
            traffic_matrix_col_ind.append(index)

    traffic_matrix = sparse.csr_matrix(
        (traffic_matrix_data, (traffic_matrix_row_ind, traffic_matrix_col_ind)),
        shape=(len(indices_to_link_ids), len(sources))
    )
    traffic_matrix_transpose = sparse.csr_matrix(
        (traffic_matrix_data, (traffic_matrix_col_ind, traffic_matrix_row_ind)),
        shape=(len(sources), len(indices_to_link_ids))
    )

    # Objective function
    def f(xs, lam=0.01):
        errors = traffic_counts - traffic_total * (traffic_matrix @ xs)
        accuracy = errors @ errors

        penalty = 0.
        if lam != 0.:
            for x, source, target in zip(xs, sources, targets):
                n_s = traffic_out_per_node[source]
                n_d = traffic_in_per_node[target]
                if n_s > 0. and n_d > 0. and x != 0.:
                    penalty += x * np.log2(x * traffic_total**2 / (n_s * n_d))

        if np.isinf(lam):
            return penalty

        return accuracy / traffic_total**2 + lam**2 * penalty

    # Gradient of objective function
    def g(xs, lam=0.01):
        errors = traffic_counts - traffic_total * (traffic_matrix @ xs)
        dif_accuracy = -2 * traffic_total * traffic_matrix_transpose @ errors

        if lam != 0.:
            dif_penalty = np.array([
                np.log2(x * traffic_total**2 / (traffic_source * traffic_target)) + 1 / np.log(2)
                if traffic_source > 0. and traffic_target > 0. else 0.
                for x, source, target in zip(xs, sources, targets)
                for traffic_source in (traffic_out_per_node[source],)
                for traffic_target in (traffic_in_per_node[target],)
            ])
        else:
            dif_penalty = np.zeros(xs.shape)

        return dif_accuracy / traffic_total**2 + lam**2 * dif_penalty

    # Use a simple gravity model for initialization. This shouldn't
    # actually matter all that much
    x_0 = np.array([
        traffic_source * traffic_target / traffic_total**2
        for source, target in zip(sources, targets)
        for traffic_source in (traffic_out_per_node[source],)
        for traffic_target in (traffic_in_per_node[target],)
    ])

    xs_opt, _, _ = optimize.fmin_l_bfgs_b(
        f, x_0, fprime=g, args=[0.01],
        # factr=1e1, pgtol=1e-12,
        bounds=[(1e-12, 1.) for _ in x_0]
    )

    # return {
    #     (source, target): traffic_total * x
    #     for x, source, target in zip(xs_opt, sources, targets)
    # }

    traffic_matrix = {}
    for x, source, target in zip(xs_opt, sources, targets):
        if source not in traffic_matrix:
            traffic_matrix[source] = {}
        # traffic_matrix[source][target] = traffic_total * x
        traffic_matrix[source][target] = x
    return traffic_matrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--probes', '-p', metavar='probes-file', dest='probes_file_path', type=str, required=True)
    parser.add_argument('--links', '-l', metavar='latencies_file', dest='links_file_path', type=str, required=True)
    parser.add_argument('--output', '-o', metavar='output_file', dest='output_file_path', type=str, required=True)
    args = parser.parse_args()

    probes_file_path = pathlib.PurePath(args.probes_file_path)
    links_file_path = pathlib.PurePath(args.links_file_path)
    output_file_path = pathlib.PurePath(args.output_file_path)

    if not os.path.exists(probes_file_path):
        sys.stderr.write('Probes file does not exist')
        sys.exit(0)
    if not os.path.exists(links_file_path):
        sys.stderr.write('Links file does not exist')
        sys.exit(0)

    graph = nx.DiGraph()

    probe_ids_to_indices = {}
    probe_indices_to_ids = []
    with open(probes_file_path, 'r') as f:
        reader = csv.DictReader(f)
        for i, line in enumerate(reader):
            id_probe = line['id']
            latitude = float(line['latitude'])
            longitude = float(line['latitude'])

            graph.add_node(
                id_probe,
                latitude=latitude,
                longitude=longitude,
                ip=f'10.{1 + i}.0.0/16',
                city=line['city'].replace(' ', '_')
            )

            probe_ids_to_indices[id_probe] = i
            probe_indices_to_ids.append(id_probe)

    with open(links_file_path, 'r') as f:
        reader = csv.DictReader(f)
        for line in reader:
            id_source = line['source_id']
            id_target = line['target_id']
            latency = None
            if 'rtt' in line and line['rtt']:
                latency = float(line['rtt'])
            else:
                probe_source = graph.nodes[id_source]
                probe_target = graph.nodes[id_target]
                latency = utility.get_GCL(
                    (probe_source['latitude'], probe_source['longitude']),
                    (probe_target['latitude'], probe_target['longitude'])
                )

            throughput = None
            if 'throughput' in line and line['throughput']:
                throughput = float(line['throughput'])
            else:
                throughput = 400000000000 / 10  # TODO: make this more realistic if needed

            graph.add_edge(
                id_source, id_target,
                latency=latency,
                throughput=throughput
            )

    traffic_matrix = generate_traffic_matrix(graph)

    def generate_node_dict(id_probe):
        probe = graph.nodes[id_probe]

        if id_probe in traffic_matrix:
            modulators = {
                f'm{i}': f'modulator start=0.0 generator=s{i} profile=((300,),(1,))'
                for i, (target, throughput) in enumerate(traffic_matrix[id_probe].items())
            }
            sources = {
                f's{i}': f'harpoon ipsrc={graph.nodes[id_probe]["ip"]} ipdst={graph.nodes[target]["ip"]} sport=randomchoice(22,80,443) dport=randomunifint(1024,65535) flowsize=pareto({(throughput * 1e8 * (1.2 - 1) / 1.2):.4f},1.2) flowstart=exponential(1.0) mss=randomchoice(1460) lossrate=randomchoice(0.001) tcpmodel=msmo97'
                for i, (target, throughput) in enumerate(traffic_matrix[id_probe].items())
            }
        else:
            modulators = {}
            sources = {}

        node_dict = {
            'id': f'{id_probe}_{probe["city"]}',
            'ipdests': probe['ip'],
            'autoack': False,
            'defaultroute': True,
            'traffic': ' '.join(modulators.keys())
        }
        for id_modulator, modulator in modulators.items():
            node_dict[id_modulator] = modulator
        for id_source, source in sources.items():
            node_dict[id_source] = source

        return node_dict

    with open(output_file_path, 'w') as f:
        configuration = {
            'directed': False,
            'multigraph': True,
            'graph': [
                ['node', {}],
                ['graph', {'flowexport': 'text'}],
                ['edge', {}],
                ['name', 'test'],
            ],
            'nodes': [
                generate_node_dict(id_probe)
                for id_probe in graph.nodes
            ],
            'links': [
                {
                    'source': probe_ids_to_indices[id_source],
                    'target': probe_ids_to_indices[id_target],
                    'capacity': 400000000000,
                    'delay': latency,
                    'weight': latency,
                }
                for id_source, id_target, latency in graph.edges.data('latency')
                for probe_source in (graph.nodes[id_source],)
                for probe_target in (graph.nodes[id_target],)
            ],
        }
        json.dump(configuration, f, indent=4)
