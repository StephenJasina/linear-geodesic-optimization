import argparse
import collections
import csv
import os
import pathlib
import sys

import networkx as nx

sys.path.insert(0, str(pathlib.PurePath('..', '..', 'src')))
from linear_geodesic_optimization.data import utility


def ip_to_bitstring(ip, slash=32):
    octets = [int(octet) for octet in ip.split('.')]
    if len(octets) != 4:
        raise ValueError()
    for octet in octets:
        if not (0 <= octet < 256):
            raise ValueError()
    return ''.join([f'{octet:08b}' for octet in octets])[:slash]

def generate_routing_tables(graph):
    sources, targets = zip(*[
        (source, target)
        for source in graph.nodes
        for target in graph.nodes
    ])

    # "Slow," but this should be fine since there won't be too many nodes
    routing_tables = {}
    for source in graph.nodes:
        next_hop_from_source = {}
        for target in graph.nodes:
            if source != target:
                path = nx.dijkstra_path(graph, source, target, 'latency')
                next_hop_from_source[target] = path[1]
        routing_tables[source] = next_hop_from_source

    return routing_tables

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pobes', '-p', metavar='probes-file', dest='probes_file_path', type=str, required=True)
    parser.add_argument('--links', '-l', metavar='latencies_file', dest='links_file_path', type=str, required=True)
    parser.add_argument('--fs-output', '-f', metavar='fs_output_directory', dest='fs_output_directory_path', type=str, required=True)
    parser.add_argument('--output', '-o', metavar='output_file', dest='output_file_path', type=str, required=True)
    args = parser.parse_args()

    probes_file_path = pathlib.PurePath(args.probes_file_path)
    links_file_path = pathlib.PurePath(args.links_file_path)
    fs_output_directory_path = pathlib.PurePath(args.fs_output_directory_path)
    output_file_path = pathlib.PurePath(args.output_file_path)

    if not os.path.exists(probes_file_path):
        sys.stderr.write('Probes file does not exist')
        sys.exit(0)
    if not os.path.exists(links_file_path):
        sys.stderr.write('Links file does not exist')
        sys.exit(0)

    graph = nx.DiGraph()

    with open(probes_file_path, 'r') as f:
        reader = csv.DictReader(f)
        for i, line in enumerate(reader):
            id_probe = line['id']
            latitude = float(line['latitude'])
            longitude = float(line['latitude'])

            ip = f'10.{1 + i}.0.0'
            slash=16

            graph.add_node(
                id_probe,
                latitude=latitude,
                longitude=longitude,
                ip=f'{ip}/{slash}',
                bitstring=ip_to_bitstring(ip, slash),
                city = line['city'].replace(' ', '_')
            )

    def address_to_name(address: str):
        # Remove port if necessary
        ip = address.split(':')[0]

        bitstring = ip_to_bitstring(ip)
        for id_probe, bitstring_probe in graph.nodes.data('bitstring'):
            if bitstring_probe == bitstring[:len(bitstring_probe)]:
                return id_probe
        return None

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

    routing_tables = generate_routing_tables(graph)

    link_counts = collections.defaultdict(int)
    for fs_output_file_name in sorted(os.listdir(fs_output_directory_path)):
        fs_output_file_path = fs_output_directory_path / fs_output_file_name
        id_current = fs_output_file_name.split('_')[0]
        with open(fs_output_file_path, 'r') as f:
            for line in f.readlines():
                line_split = line.rstrip().split(' ')

                ip_source, ip_target = line_split[5].split('->')
                id_source = address_to_name(ip_source)
                id_target = address_to_name(ip_target)

                if id_current != id_target:
                    id_next_hop = routing_tables[id_current][id_target]
                    link_counts[(id_current, id_next_hop)] += int(line_split[10])

    with open(output_file_path, 'w') as f:
        writer = csv.DictWriter(f, ['source_id', 'target_id', 'throughput'])
        writer.writeheader()
        for (id_source, id_target), link_count in sorted(link_counts.items()):
            writer.writerow({
                'source_id': id_source,
                'target_id': id_target,
                'throughput': link_count,
            })
