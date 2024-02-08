import collections
import csv
import os
import re
import shutil
import subprocess
import sys
import urllib.request

from GraphRicciCurvature.OllivierRicci import OllivierRicci
import networkx as nx

sys.path.append(os.path.join('..', 'src'))
from linear_geodesic_optimization.data import utility


url_base = "https://networks.skewed.de/net/internet_top_pop/files"

directory = 'topology_zoo'

with open(os.path.join(directory, 'dataset_names.txt'), 'r') as f_in:
    for line in f_in.readlines():
        name = line.rstrip()
        print(name)
        name_archive = os.path.join(directory, name + '.xml.zst')
        name_xml = os.path.join(directory, name + '.xml')
        name_graphml = os.path.join(directory, name + '.graphml')
        name_probes = os.path.join(directory, name + '_probes.csv')
        name_latencies = os.path.join(directory, name + '_latencies.csv')

        # Don't bog down the servers unnecessarily
        if not os.path.exists(name_archive):
            with open(name_archive, 'wb') as f_out:
                f_out.write(
                    urllib.request.urlopen(url_base + f'/{name}.xml.zst').read()
                )

        subprocess.run(['zstd', '-dq', name_archive])
        subprocess.run(['mv', name_xml, name_graphml])
        subprocess.run(['sed', '-i',
                        's/"vector_[^"]\+"/"string"/g;s/"short"/"int"/g',
                        name_graphml])

        graph = nx.read_graphml(name_graphml)
        graph = nx.Graph(graph)

        # Do a silly check for whether we have location data
        if 'Latitude' not in next(iter(graph.nodes(data=True)))[1]:
            print('\tNo location data')
            subprocess.run(['rm', name_graphml])
            continue

        # Check whether we have any missing nodes
        all_valid = True
        for node, data in graph.nodes(data=True):
            if data['label'] == 'None':
                print('\tInvalid node')
                all_valid = False
                break
        if not all_valid:
            subprocess.run(['rm', name_graphml])
            continue

        # Some cities correspond to multiple nodes, so we need to do
        # some deduplication
        partition = collections.defaultdict(set)
        for node, data in graph.nodes(data=True):
            city = re.sub('[ \\d]*$', '', data['label'])
            data['label'] = city
            partition[city].add(node)
        relabel_mapping = {}
        def node_data(b):
            index = min([int(b_i[1:]) for b_i in b])
            relabel_mapping[b] = index
            data = graph.nodes['n' + str(index)]
            return {
                'city': data['label'],
                'country': data['Country'] if 'Country' in data else '',
                'lat': data['Latitude'],
                'long': data['Longitude'],
            }
        def edge_data(b, c):
            return {'weight': 1.}
        graph = nx.quotient_graph(graph, partition,
                                  node_data=node_data, edge_data=edge_data)
        nx.relabel_nodes(graph, relabel_mapping, False)

        with open(name_probes, 'w') as f:
            writer = csv.DictWriter(f, ['id', 'city', 'country', 'latitude', 'longitude'])
            writer.writeheader()

            for node, data in graph.nodes(data=True):
                writer.writerow({
                    'id': node,
                    'city': data['city'],
                    'country': data['country'],
                    'latitude': data['lat'],
                    'longitude': data['long'],
                })

        # Approximate RTTs using GCLs and shortest paths
        for source_id, target_id, data in graph.edges(data=True):
            source = graph.nodes[source_id]
            target = graph.nodes[target_id]
            data['rtt'] = utility.get_GCL(
                (source['lat'], source['long']),
                (target['lat'], target['long'])
            )
        rtts = nx.floyd_warshall(graph, 'rtt')

        with open(name_latencies, 'w') as f:
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

        orc = OllivierRicci(graph, weight='weight', alpha=0.)
        graph = orc.compute_ricci_curvature()
        for _, _, d in graph.edges(data=True):
            del d['weight']
        nx.write_graphml(graph, name_graphml)
