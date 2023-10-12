import pandas as pd
import urllib.request
from ripe.atlas.sagan import Result
import json
import os
import datetime
from collections import defaultdict
from tqdm import tqdm
from utils_for_traceroutes import *
import warnings
warnings.simplefilter("ignore")
# Global Variables
ANCHOR_DATA = {}
UNKNOWN_SOURCES = []
ip_type = 'ipv4'
ip_type_num = {'ipv4' : 4, 'ipv6' : 6}


def loading_metadata():
    '''
    Metadata
    '''
    print('Loading metadata...')
    ip2as = load_ip_to_asn()
    peers_per_asn = load_peers()
    customer_cone_per_asn, _ = load_customer_cone()
    ripe_probes = load_ripe_probes()
    # rdns_per_ip = load_rdns_names()
    _, asn_per_ixp_ip = load_ixp_ip(ixp_name_filter=None)
    country_by_ip, city_by_ip, geo_by_ip, score_by_ip = load_ripe_ipmap('../data/ripeipmap/')
    org_per_asn = load_org_per_asn()
    '''
    '''
    trie = prefix_granularity()
    return peers_per_asn, customer_cone_per_asn, ripe_probes, ip2as, asn_per_ixp_ip, city_by_ip, org_per_asn



def fetch_data_from_url(url):
    """Fetches data from the specified URL and returns it as JSON."""
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def infer_anchors(page_num, start_time, end_time, protocol='IPv4'):
    """Fetch and process anchor measurements from the RIPE Atlas API."""
    date_start = datetime.datetime.strptime(date_start_str, '%Y-%m-%d')
    # date_end = datetime.datetime.strptime(date_end_str, '%Y-%m-%d')
    date_end = date_start + datetime.timedelta(hours=1)
    start_time = datetime.datetime.timestamp(date_start)
    end_time = datetime.datetime.timestamp(date_end)
    print(start_time)
    print(end_time)
    file_name = f'../data/{ip_type}/anchor_meshes_traceroutes/anchor_meshes{page_num}.json'
    if os.path.exists(file_name):
        print('File already exists')
        return None  # return a flag indicating no results
    max_ttl = 256
    anchor_data = fetch_data_from_url(
        f"https://atlas.ripe.net/api/v2/anchor-measurements/?format=json&page={page_num}")
    res = defaultdict(list)
    if not anchor_data['results']:
        return None  # return a flag indicating no results
    peers_per_asn, customer_cone_per_asn, ripe_probes, ip2asn, asn_per_ixp_ip, city_by_ip, org_per_asn = loading_metadata()
    for measurement in tqdm(anchor_data['results']):
        if measurement['type'] == 'traceroute' and measurement['is_mesh']:
            target_data = fetch_data_from_url(measurement['target'])
            source = target_data['probe']
            with urllib.request.urlopen(measurement['measurement']) as results:
                for result in results.readlines():
                    measurement_data = json.loads(result.decode("utf-8"))
                    if measurement_data['af'] != ip_type_num[ip_type]:
                        continue
                    split_result = measurement_data['result'].split('?')
                    print(split_result)
                    # timebound_url = split_result[0] + f"?start_time={int(start_time)}&?stop_time={int(end_time)}format=txt"
                    timebound_url = split_result[0] + f"?&start={int(start_time)}&stop={int(end_time)}&format=txt"
                    loc_res = {}
                    # Fetch and process atlas results
                    with urllib.request.urlopen(timebound_url) as atlas_res:
                        if protocol in measurement_data['description']:
                            for (i, atlas_result) in tqdm(enumerate(atlas_res.readlines())):
                                atlas_data = Result.get(atlas_result.decode("utf-8"))
                                print(atlas_data)
                                # map the dst address to the probe
                                prb_id, ip_path, src_address,dst_address, hop_path = \
                                    atlas_data.probe_id, atlas_data.ip_path, atlas_data.source_address, atlas_data.destination_address, atlas_data.hops
                                # hop_path = [hop[0] for hop in hop_path]
                                list_of_ttl = ['*' for i in range(0, max_ttl)]
                                list_of_rtt = ['*' for i in range(0, max_ttl)]
                                list_of_ip = ['*' for i in range(0, max_ttl)]
                                for hop in hop_path:
                                    print(hop.packets[0].raw_data)
                                    if 'x' in hop.packets[0].raw_data:
                                        continue
                                    ip = hop.packets[0].raw_data['from']
                                    rtt = hop.packets[0].raw_data['rtt']
                                    ttl = hop.packets[0].raw_data['ttl']
                                    list_of_ttl[ttl - 1] = ttl
                                    list_of_rtt[ttl - 1] = rtt
                                    list_of_ip[ttl - 1] = ip
                                    # for packets in hop.packets[0]:
                                    #     ip = packets.raw_data['from']
                                ### delete all the last ttl until the first one that is not *
                                for i in range(len(list_of_ttl) - 1, -1, -1):
                                    if list_of_ttl[i] == '*':
                                        del list_of_ttl[i]
                                        del list_of_rtt[i]
                                        del list_of_ip[i]
                                    else:
                                        break
                                if dst_address not in ripe_probes.keys():
                                    continue
                                dst_id = ripe_probes[dst_address]['id']
                                # for i in probes['address_v4']:
                                #     print(i)
                                #     print(trie.search_best(i).prefix)
                                ip_path = [ip[0] for ip in ip_path]
                                loc_res['ip_path'] = ip_path
                                as_path = []
                                org_path = []
                                geolocated_path = []
                                if len(ip_path) != 1:
                                    print('?!')
                                for ip in ip_path:
                                    if ip in asn_per_ixp_ip.keys():
                                        as_path.append(asn_per_ixp_ip[ip][0])
                                        geolocated_path.append(asn_per_ixp_ip[ip][1])
                                    else:
                                        if ip is not None:
                                            as_path.append(ip2asn.lookup(ip)[0])
                                        else:
                                            as_path.append(None)
                                        if ip in city_by_ip.keys():
                                            geolocated_path.append(city_by_ip[ip])
                                        else:
                                            geolocated_path.append(None)
                                for asn in as_path:
                                    if asn in org_per_asn.keys():
                                        org_path.append(org_per_asn[asn])
                                    else:
                                        org_path.append(asn)
                                loc_res['as_path'] = as_path
                                loc_res['org_path'] = org_path
                                loc_res['geolocated_path'] = geolocated_path
                                loc_res['destination_ip'] = dst_address
                                loc_res['end_time'] = atlas_data['end_time']
                                asn_per_ttl = {}
                                res[str(prb_id)+'_'+str(dst_id)].append(loc_res)
                                # Add the source asn
                                # asn_per_ttl[0] = source_asn, (IP_TYPE.OTHER, None)
                                # ip_per_ttl, asn_per_ttl_ = extract_as_path_from_traceroute_db(hops, asn_per_ixp_ip,
                                #                                                               ip2asn)
                                # asn_per_ttl.update(asn_per_ttl_)
                                ### org_path
                                org_path = []
                                for asn in as_path:
                                    if asn in org_per_asn.keys():
                                        org_path.append(org_per_asn[asn])
                                    else:
                                        org_path.append(asn)
                                ### replace the -1 from the as_path to None
                                # as_path = [None if asn == -1 else asn for asn in as_path]
                                # loop through the AS numbers in as_path and check if they belong to the same org
                                # if they do, replace the AS number with the first AS number that appears in the path
                                for i in range(1, len(as_path)):
                                    if as_path[i] in org_per_asn.keys() and as_path[i - 1] in org_per_asn.keys():
                                        if org_per_asn[as_path[i]] == org_per_asn[as_path[i - 1]]:
                                            as_path[i] = as_path[i - 1]
    # # Save the global dictionary as a JSON file
    with open(f'../data/{ip_type}/anchor_meshes_traceroutes/anchor_meshes{page_num}.json',
              'w') as outfile:
        json.dump(res, outfile)


def convert_to_latencymatrix(input_path, output_path):
    """Converts measurement data into a latency matrix and saves it as CSV."""

    measurements = {}

    # Read all JSON files in the directory
    for root, _, files in os.walk(input_path):
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(root, file)) as fp:
                    measurements.update(json.load(fp))

    df = pd.DataFrame(measurements)
    df.to_csv(output_path)
    return df


if __name__ == '__main__':
    page = 1
    date_start_str = '2023-10-06'
    date_end_str = '2023-10-07'
    # for i in range(0,24):
    while True:
        result = infer_anchors(page,date_start_str, date_end_str,'IPv4')
        if result is None:  # if no results are returned, break the loop
            break
        page += 1
        break

    # Convert to latency matrix
    # convert_to_latencymatrix('Datasets/AnchorMeasurements/', 'path_to_output.csv')