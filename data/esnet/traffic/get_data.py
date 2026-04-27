
import collections
import csv
import datetime
import json
import os
import pathlib
import urllib.request
import sys
import zoneinfo

import networkx as nx
import numpy as np
import scipy.interpolate

sys.path.insert(0, str(pathlib.PurePath('..', '..')))
sys.path.insert(0, str(pathlib.PurePath('..', '..', '..', 'src')))
from linear_geodesic_optimization.data import input_network, tomography
import csv_to_json


# ESnet's Elasticsearch URL
url_elasticsearch = 'https://el.gc1.prod.stardust.es.net:9200/stardust_perfsonar_*/_search'

def get_throughput_query(time_inital, time_final):
    return {
        'size': 10000,
        'query': {
            'bool': {
                'filter': [
                    {
                        'range': {
                            'pscheduler.start_time': {
                                'gte': time_inital.isoformat(),
                                'lt': time_final.isoformat(),
                            },
                        },
                    },
                    {
                        'query_string': {
                            'analyze_wildcard': True,
                            'query': 'test.type.keyword: throughput AND meta.source.hostname.keyword: *\\-tp.es.net AND meta.destination.hostname.keyword: *\\-tp.es.net',
                        },
                    },
                ],
            },
        },
        'fields': [
            'test.spec.source.keyword',
            'test.spec.dest.keyword',
            'meta.ip_version',
            'pscheduler.start_time',
            'result.throughput',
        ],
        '_source': False,
    }

def get_latency_query(time_inital, time_final):
    return {
        'size': 0,
        'aggs': {
            'source': {
                'terms': {
                    'field': 'meta.source.hostname.keyword',
                    'size': 10000,
                },
                'aggs': {
                    'dest': {
                        'terms': {
                            'field': 'meta.destination.hostname.keyword',
                            'size': 10000,
                        },
                        'aggs': {
                            'ip_version': {
                                'terms': {
                                    'field': 'meta.ip_version',
                                    'size': 10000,
                                },
                                'aggs': {
                                    'min_latency': {
                                        'min': {
                                            'field': 'result.latency.min',
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
        'query': {
            'bool': {
                'filter': [
                    {
                        'range': {
                            'pscheduler.start_time': {
                                'gte': time_inital.isoformat(),
                                'lt': time_final.isoformat(),
                            },
                        },
                    },
                    {
                        'query_string': {
                            'analyze_wildcard': True,
                            'query': 'test.type.keyword: latencybg AND meta.source.hostname.keyword: *\\-lat.es.net AND meta.destination.hostname.keyword: *\\-lat.es.net',
                        },
                    },
                ],
            },
        },
    }

def get_from_elasticsearch(query, path_output):
    """
    Gab data from ESnet's PerfSONAR data.

    Given a JSON query (as the object `query`) and an output location,
    query ESnet's Elasticsearch instance and stored the response.
    Additionally return the response as a Python object for convenience.
    """
    request = urllib.request.Request(
        url_elasticsearch, json.dumps(query).encode('ascii'),
        method='POST',
        headers={
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
    )
    with urllib.request.urlopen(request) as response:
        data = response.read()

    blob = json.loads(data)
    with open(path_output, 'w') as f:
        json.dump(blob, f)
    return blob

def group_throughput_data(blobs, path_output, time_initial, time_final, time_step):
    """
    Group throughput data by IE pair.

    Data from ESnet is returned as a list of measurements. This function
    turns these measurements into time series for each ingress-egress
    pair. Results are additionally interpolated so that the data is
    ultimately a series of snapshots at each hour (default). The outputs
    are finally stored as timestamped CSVs.
    """
    if not os.path.exists(path_output):
        os.makedirs(path_output, exist_ok=True)

    # Group together tests by their source-destination pairs and IP
    # version
    key_to_tests = collections.defaultdict(list)
    for blob in blobs:
        for hit in blob['hits']['hits']:
            fields = hit['fields']
            source = fields['test.spec.source.keyword'][0]
            destination = fields['test.spec.dest.keyword'][0]
            ip_version = fields['meta.ip_version'][0]
            throughput = fields['result.throughput'][0]
            timestamp = datetime.datetime.fromisoformat(fields['pscheduler.start_time'][0])

            key_to_tests[source, destination, ip_version].append({
                'throughput': throughput,
                'timestamp': timestamp,
            })

    # Sort lists by their timestamps (needed for our later interpolation)
    keys = list(key_to_tests.keys())
    for key in keys:
        key_to_tests[key] = list(sorted(key_to_tests[key], key=(lambda test: test['timestamp'])))

    # Compute interpolators
    key_to_spline = {}
    for key, tests in key_to_tests.items():
        timestamp_initial = tests[0]['timestamp']
        timestamp_final = tests[-1]['timestamp']
        x = [(test['timestamp'] - timestamp_initial).total_seconds() for test in tests]
        y = [test['throughput'] for test in tests]

        if (timestamp_initial - time_initial).total_seconds() != 0.:
            x.insert(0, 0.)
            y.insert(0, tests[0]['throughput'])
        if (time_final - timestamp_final).total_seconds() != 0.:
            x.append((time_final - time_initial).total_seconds())
            y.append(tests[-1]['throughput'])

        if len(x) == 1:
            key_to_spline[key] = lambda t: y[0]
        else:
            # ie_pair_to_spline[ie_pair] = scipy.interpolate.CubicSpline(x, y)
            key_to_spline[key] = scipy.interpolate.interp1d(x, y, kind='linear')

    time = time_initial
    while time <= time_final:
        time_delta = time - time_initial

        # Compute the interpolations
        links_to_write = {}
        for key, spline in key_to_spline.items():
            source = key[0].split('-')[0].upper()
            destination = key[1].split('-')[0].upper()
            links_to_write[source, destination, key[2]] = spline(time_delta.total_seconds())

        # Write everything to disk
        with open(path_output / time.strftime('%Y%m%d%H%M%S.csv'), 'w') as f:
            writer = csv.DictWriter(f, ['source_id', 'target_id', 'ip_version', 'throughput'])
            writer.writeheader()
            for key in sorted(links_to_write.keys()):
                writer.writerow({
                    'source_id': key[0],
                    'target_id': key[1],
                    'ip_version': key[2],
                    'throughput': links_to_write[key],
                })

        time = time + time_step

    return [
        (
            source.split('-')[0].upper(),
            destination.split('-')[0].upper()
        )
        for source, destination, _ in keys
    ]

def group_latency_data(blobs, path_output, time_initial, time_final, time_step):
    """
    Group throughput data by IE pair.

    This function mostly exists to mimic the interface of
    `group_throughput_data`. Latency data is automatically aggregated by
    the Elasticsearch query.
    """
    time = time_initial
    for blob in blobs:
        key_to_tests = collections.defaultdict(list)
        for source_bucket in blob['aggregations']['source']['buckets']:
            source = source_bucket['key'].split('-')[0].upper()
            for destination_bucket in source_bucket['dest']['buckets']:
                destination = destination_bucket['key'].split('-')[0].upper()
                for ip_version_bucket in destination_bucket['ip_version']['buckets']:
                    ip_version = ip_version_bucket['key']
                    latency = ip_version_bucket['min_latency']['value']

                    key_to_tests[source, destination, ip_version].append({
                        'latency': latency,
                    })

        with open(path_output / time.strftime('%Y%m%d%H%M%S.csv'), 'w') as f:
            writer = csv.DictWriter(f, ['source_id', 'target_id', 'ip_version', 'latency'])
            writer.writeheader()
            for key in sorted(key_to_tests.keys()):
                writer.writerow({
                    'source_id': key[0],
                    'target_id': key[1],
                    'ip_version': key[2],
                    'latency': min(test['latency'] for test in key_to_tests[key]),
                })

        time = time + time_step

def complete_traffic_matrix(ie_pair_to_traffic):
    def iterative_completion(M, n_iter=20):
        """Fill NaNs, do truncated SVD, re-fill, repeat."""
        mask = ~np.isnan(M)
        X = M.copy()
        col_means = np.nanmean(X, axis=0)
        X[~mask] = np.take(col_means, np.where(~mask)[1])

        for _ in range(n_iter):
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            rank, M_approx = fill_values_gavish_donoho(X)  # or use a fixed rank
            X[~mask] = M_approx[~mask]  # only update missing entries

        return X

    def fill_values_gavish_donoho(M):
        # Estimate noise from median singular value
        n = M.shape[0]
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        noise_sigma = np.median(s) / (2.858 * np.sqrt(n))

        # Optimal threshold
        lam = (4 / np.sqrt(3)) * np.sqrt(n) * noise_sigma
        rank = np.sum(s > lam)
        return rank, (U[:, :rank] * s[:rank]) @ Vt[:rank]

    index_to_node = list(set(source for source, _ in ie_pair_to_traffic.keys()) | set(destination for _, destination in ie_pair_to_traffic.keys()))
    node_to_index = {node: index for index, node in enumerate(index_to_node)}
    n_nodes = len(index_to_node)

    traffic_matrix = np.full((n_nodes, n_nodes), np.nan)
    for (source, destination), traffic in ie_pair_to_traffic.items():
        traffic_matrix[node_to_index[source], node_to_index[destination]] = traffic

    _, traffic_matrix = fill_values_gavish_donoho(iterative_completion(traffic_matrix))

    return {
        (source, destination): traffic
        for index_source, source in enumerate(index_to_node)
        for index_destination, destination in enumerate(index_to_node)
        for traffic in (traffic_matrix[index_source, index_destination],)
        if traffic != 0.
    }

def write_json(network: nx.Graph, ie_pair_to_route, probe_to_cluster_representative, path_input, path_output):
    """
    Convert a CSV file of traffic into a JSON output.
    """
    ie_pair_to_traffic = collections.defaultdict(float)
    with open(path_input, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['source_id'] not in probe_to_cluster_representative or row['target_id'] not in probe_to_cluster_representative:
                continue
            source = probe_to_cluster_representative[row['source_id']]
            destination = probe_to_cluster_representative[row['target_id']]
            ie_pair_to_traffic[source, destination] += float(row['throughput'])

    ie_pair_to_traffic = complete_traffic_matrix(ie_pair_to_traffic)

    ie_pairs = list(sorted(ie_pair_to_traffic.keys()))
    csv_to_json.write_graph(
        network,
        [ie_pair_to_route[ie_pair] for ie_pair in ie_pairs],
        [ie_pair_to_traffic[ie_pair] for ie_pair in ie_pairs],
        path_output
    )

def main():
    path_query = pathlib.PurePath('queries', 'throughput.json')
    directory_output_search_throughput = pathlib.PurePath('outputs', 'throughput_dumps')
    directory_output_search_latency = pathlib.PurePath('outputs', 'latency_dumps')
    directory_output_group_throughputs = pathlib.PurePath('outputs', 'links')
    directory_output_group_latencies = pathlib.PurePath('outputs', 'links_latencies')
    path_output_json = pathlib.PurePath('json')
    path_probes = pathlib.PurePath('probes.csv')
    path_links = pathlib.PurePath('links.csv')
    time_initial = datetime.datetime(2026, 4, 1, 0, 0, 0, 0, zoneinfo.ZoneInfo('America/New_York'))
    time_final = datetime.datetime(2026, 4, 15, 0, 0, 0, 0, zoneinfo.ZoneInfo('America/New_York'))
    # An event happens in this range
    # time_initial = datetime.datetime(2026, 4, 7, 6, 0, 0, 0, zoneinfo.ZoneInfo('America/New_York'))
    # time_final = datetime.datetime(2026, 4, 8, 18, 0, 0, 0, zoneinfo.ZoneInfo('America/New_York'))
    time_step = datetime.timedelta(seconds=3600.)
    clustering_distance = 500000.

    # Get ESnet data
    blobs_throughput = []
    blobs_latency = []
    time = time_initial
    while time + time_step <= time_final:
        path_output_search_throughput = directory_output_search_throughput / (time.strftime('%Y%m%d%H%M%S.json'))
        path_output_search_latency = directory_output_search_latency / (time.strftime('%Y%m%d%H%M%S.json'))

        if os.path.exists(path_output_search_throughput):
            print(f'using cached dump for {path_output_search_throughput}')
            with open(path_output_search_throughput, 'r') as f:
                blob_throughput = json.load(f)
        else:
            print(f'downloading from ESnet for {path_output_search_throughput}')
            blob_throughput = get_from_elasticsearch(get_throughput_query(time, time + time_step), path_output_search_throughput)
        blobs_throughput.append(blob_throughput)

        if os.path.exists(path_output_search_latency):
            print(f'using cached dump for {path_output_search_latency}')
            with open(path_output_search_latency, 'r') as f:
                blob_latency = json.load(f)
        else:
            print(f'downloading from ESnet for {path_output_search_latency}')
            blob_latency = get_from_elasticsearch(get_latency_query(time, time + time_step), path_output_search_latency)
        blobs_latency.append(blob_latency)

        time = time + time_step
    ie_pairs = group_throughput_data(blobs_throughput, directory_output_group_throughputs, time_initial, time_final, time_step)
    group_latency_data(blobs_latency, directory_output_group_latencies, time_initial, time_final, time_step)

    # Write JSON
    network = input_network.get_graph_from_csvs(
        path_probes, path_links,
        clustering_distance=clustering_distance,
        should_compute_curvatures=False,
        directed=True
    )
    probe_to_cluster_representative = {
        element: node
        for node, data in network.nodes(data=True)
        for element in (data['elements'] if 'elements' in data else [node])
    }
    ie_pairs = [
        (probe_to_cluster_representative[source], probe_to_cluster_representative[destination])
        for source, destination in ie_pairs
        if source in probe_to_cluster_representative and destination in probe_to_cluster_representative
    ]
    routes = tomography.get_shortest_routes(network, 'gcl')
    ie_pair_to_route = {
        (route[0], route[-1]): route
        for route in routes
    }
    os.makedirs(path_output_json, exist_ok=True)

    time = time_initial
    while time <= time_final:
        path_grouped_file = time.strftime('%Y%m%d%H%M%S.csv')
        path_input = directory_output_group_throughputs / path_grouped_file
        write_json(network, ie_pair_to_route, probe_to_cluster_representative, path_input, path_output_json / f'{path_input.stem}.json')

        time = time + time_step

if __name__ == '__main__':
    main()
