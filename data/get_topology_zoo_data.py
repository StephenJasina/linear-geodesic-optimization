import csv
import os
import re
import shutil
import subprocess
import urllib.request

import networkx as nx

import csv_to_graphml
import utility


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

        # Do a silly check for whether we have location data
        if 'Latitude' not in next(iter(graph.nodes(data=True)))[1]:
            print('\tNo location data')
            continue

        # Check whether we have any missing nodes
        all_valid = True
        for node, data in graph.nodes(data=True):
            if data['label'] == 'None':
                print('\tInvalid node')
                all_valid = False
                break
        if not all_valid:
            continue

        # Some cities correspond to multiple nodes, so we need to do
        # some deduplication
        cities_to_nodes = dict()
        unique_nodes = []

        with open(name_probes, 'w') as f:
            writer = csv.DictWriter(f, ['id', 'city', 'country', 'latitude', 'longitude'])
            writer.writeheader()

            for node, data in graph.nodes(data=True):
                city = re.sub('[ \\d]*$', '', data['label'])
                if city in cities_to_nodes:
                    graph.add_edge(node, cities_to_nodes[city])
                    continue
                cities_to_nodes[city] = node
                unique_nodes.append(node)

                writer.writerow({
                    'id': data['id'],
                    'city': city,
                    'country': data['Country'] if 'Country' in data else '',
                    'latitude': data['Latitude'],
                    'longitude': data['Longitude'],
                })

        # Approximate RTTs using GCLs and shortest paths
        for source_id, target_id, data in graph.edges(data=True):
            source = graph.nodes[source_id]
            target = graph.nodes[target_id]
            data.clear()
            data['GCL'] = utility.get_GCD_latency(
                (source['Latitude'], source['Longitude']),
                (target['Latitude'], target['Longitude'])
            )
        rtts = nx.floyd_warshall(graph, 'GCL')

        with open(name_latencies, 'w') as f:
            writer = csv.DictWriter(f, ['source_id', 'target_id', 'rtt'])
            writer.writeheader()

            for source_node in unique_nodes:
                source = graph.nodes[source_node]
                for target_node in unique_nodes:
                    target = graph.nodes[target_node]
                    writer.writerow({
                        'source_id': source['id'],
                        'target_id': target['id'],
                        'rtt': utility.get_GCD_latency(
                            (source['Latitude'], source['Longitude']),
                            (target['Latitude'], target['Longitude'])
                        )
                    })


        graph = csv_to_graphml.get_graph(name_probes, name_latencies)
        nx.write_graphml(graph, name_graphml)
