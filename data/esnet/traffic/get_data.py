
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
import scipy.interpolate

sys.path.insert(0, str(pathlib.PurePath('..', '..')))
sys.path.insert(0, str(pathlib.PurePath('..', '..', '..', 'src')))
from linear_geodesic_optimization.data import input_network, tomography
import csv_to_json


# ESnet's Elasticsearch URL
url_elasticsearch = 'https://el.gc1.prod.stardust.es.net:9200/stardust_perfsonar_*/_search'

def get_from_elasticsearch(path_query, path_output):
    """
    Gab data from ESnet's PerfSONAR data.

    Given a JSON query (stored at `path_query`) and an output location,
    query ESnet's Elasticsearch instance and stored the response.
    Additionally return the response as a Python object for convenience.
    """
    with open(path_query) as f:
        query = json.dumps(json.load(f)).encode('ascii')

    request = urllib.request.Request(
        url_elasticsearch, query,
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

def group_data(
    blob, path_output,
    time_initial, time_final, time_step = datetime.timedelta(seconds=3600.)
):
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
    keys = [
        ie_pair
        for ie_pair, tests in key_to_tests.items()
    ]
    for key in keys:
        key_to_tests[key] = list(sorted(key_to_tests[key], key=(lambda test: test['timestamp'])))

    # Compute splines
    key_to_spline = {}
    for key, tests in key_to_tests.items():
        timestamp_initial = tests[0]['timestamp']
        timestamp_final = tests[-1]['timestamp']
        x = [(test['timestamp'] - timestamp_initial).total_seconds() for test in tests]
        y = [test['throughput'] for test in tests]

        if (timestamp_initial - time_initial).total_seconds() == 0.:
            x.insert(0, 0.)
            y.insert(0, tests[0]['throughput'])
        if (time_final - timestamp_final) != 0.:
            x.append((time_final - time_initial).total_seconds())
            y.append(tests[-1]['throughput'])

        if len(x) == 1:
            key_to_spline[key] = lambda t: y[0]
        else:
            # ie_pair_to_spline[ie_pair] = scipy.interpolate.CubicSpline(x, y)
            key_to_spline[key] = scipy.interpolate.interp1d(x, y, kind='linear')

    time_current = time_initial
    while time_current <= time_final:
        time_delta = time_current - time_initial

        # Compute the interpolations
        links_to_write = {}
        for key, spline in key_to_spline.items():
            source = key[0].split('-')[0].upper()
            destination = key[1].split('-')[0].upper()
            links_to_write[source, destination, key[2]] = spline(time_delta.total_seconds())

        # Write everything to disk
        with open(path_output / time_current.strftime('%Y%m%d%H%M%S.csv'), 'w') as f:
            writer = csv.DictWriter(f, ['source_id', 'target_id', 'ip_version', 'throughput'])
            writer.writeheader()
            for key in sorted(links_to_write.keys()):
                writer.writerow({
                    'source_id': key[0],
                    'target_id': key[1],
                    'ip_version': key[2],
                    'throughput': links_to_write[key],
                })

        time_current = time_current + time_step

    return [
        (
            source.split('-')[0].upper(),
            destination.split('-')[0].upper()
        )
        for source, destination, _ in keys
    ]

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

    ie_pairs = list(sorted(ie_pair_to_traffic.keys()))
    csv_to_json.write_graph(
        network,
        [ie_pair_to_route[ie_pair] for ie_pair in ie_pairs],
        [ie_pair_to_traffic[ie_pair] for ie_pair in ie_pairs],
        path_output
    )

def main():
    path_query = pathlib.PurePath('queries', 'throughput.json')
    path_output_search = pathlib.PurePath('outputs', 'throughput_dump.json')
    path_output_group = pathlib.PurePath('outputs', 'links')
    path_output_json = pathlib.PurePath('json')
    path_probes = pathlib.PurePath('probes.csv')
    path_links = pathlib.PurePath('links.csv')
    time_initial = datetime.datetime(2026, 4, 1, 0, 0, 0, 0, zoneinfo.ZoneInfo('America/New_York'))
    time_final = datetime.datetime(2026, 4, 2, 0, 0, 0, 0, zoneinfo.ZoneInfo('America/New_York'))
    clustering_distance = 500000.

    # Get ESnet data
    if os.path.exists(path_output_search):
        print('using cached dump')
        with open(path_output_search, 'r') as f:
            blob = json.load(f)
    else:
        print('downloading from ESnet')
        blob = get_from_elasticsearch(path_query, path_output_search)
    ie_pairs = group_data(blob, path_output_group, time_initial, time_final)

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
    routes = tomography.get_shortest_routes(network, 'gcl', ie_pairs)
    ie_pair_to_route = {
        (route[0], route[-1]): route
        for route in routes
    }
    os.makedirs(path_output_json, exist_ok=True)
    for path_grouped_file in sorted(os.listdir(path_output_group)):
        path_input = path_output_group / path_grouped_file
        write_json(network, ie_pair_to_route, probe_to_cluster_representative, path_input, path_output_json / f'{path_input.stem}.json')

if __name__ == '__main__':
    main()
