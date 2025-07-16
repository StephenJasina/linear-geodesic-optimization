import os
import pyasn
import json
import jsonlines
import radix
import csv
import gzip


def load_ixp_ip(ixp_name_filter=None):
    ixp_file = '../data/traceroute_meta/merged-members-gen-20230621.txt'
    ips_per_asn_per_ixp = {}
    asn_per_ixp_ip = {}
    with open(ixp_file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            line = line.strip("\n")
            tokens = line.split("\t")
            ip, asn, ixp_name = tokens
            if asn == " Telx Atlanta" or asn == ' Telx New York':
                continue
            if ixp_name_filter is not None and ixp_name != ixp_name_filter:
                continue
            if asn != "None":
                ips_per_asn_per_ixp.setdefault(ixp_name, {})[int(asn)] = ip
                asn_per_ixp_ip[ip] = (int(asn), ixp_name)
            else:
                asn_per_ixp_ip[ip] = (-1, ixp_name)

    return ips_per_asn_per_ixp, asn_per_ixp_ip
#
# def load_ixp_ip():
#     IXPs = {}
#     with open('../data/traceroute_meta/merged-members-gen-20230621.txt', 'r') as f:
#         for row in f.readlines():
#             if row.startswith('#'):
#                 continue
#             IXPs[row.split('\t')[0]] = row.split('\t')[-1].strip()
#     return IXPs

def load_peers():
    peers_file =  "../data/traceroute_meta/202310.as-rel.txt"
    peers_per_asn = {}
    with open(peers_file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            asn_1, asn_2, rel = [int(t) for t in line.split("|")]
            if rel == 0:
                peers_per_asn.setdefault(asn_1, set()).add(asn_2)
                peers_per_asn.setdefault(asn_2, set()).add(asn_1)
    return peers_per_asn

def load_customer_cone():
    customer_cone_file = "../data/traceroute_meta/customer_cone_per_as_2023_05.json"
    with open(customer_cone_file) as f:
        customer_cone_per_asn = json.load(f)
        customer_cone_per_asn = {int(x): set(customer_cone_per_asn[x]) for x in customer_cone_per_asn}
        providers_per_asn = {}
        for asn, customers in customer_cone_per_asn.items():
            providers_per_asn.setdefault(asn, set())
            for customer in customers:
                providers_per_asn.setdefault(customer, set()).add(asn)

        return customer_cone_per_asn, providers_per_asn

def load_ripe_probes():
    anchors_file = "../data/traceroute_meta/ripe_anchors.json"
    probes_file = "../data/traceroute_meta/ripe_probes.json"
    # probes = json.load(open(probes_file,'r'))
    anchors = json.load(open(anchors_file,'r'))
    # probes.extend(anchors)
    probes = {p["address_v4"]: p for p in anchors}
    return probes

def load_ip_to_asn():
    ip_to_asn_file = '../data/traceroute_meta/ipasns_2023_10.txt'
    ip2asn = pyasn.pyasn(ip_to_asn_file)
    return ip2asn

def load_ripe_ipmap(ripe_dir, is_only_single_radius=False):
    # ripe_dir = "resources/ripeipmap"
    ripe_files = os.listdir(ripe_dir)
    ripe_files = [f"{ripe_dir}/{rf}" for rf in ripe_files if rf.endswith(".csv")]

    city_by_ip_candidate = {}
    country_by_ip = {}
    geo_by_ip = {}
    score_by_ip = {}
    min_localizations = len(ripe_files) / 2
    min_score = 0
    for ripe_file in sorted(ripe_files):
        with open(ripe_file) as f:
            lines = csv.reader(f, quotechar='"', delimiter=',',
                               quoting=csv.QUOTE_ALL, skipinitialspace=True)
            next(lines)
            for tokens in lines:

                if len(tokens) > 10:
                    # Bug with WASHINGTON

                    if tokens[1] == "WASHINGTON":
                        continue
                        tokens.remove("D.C.")
                        tokens.remove("D.C.")
                        # Remove the , after Washington
                        # line = line.replace("Washington,", "Washington")
                        # line = line.replace("WASHINGTON,", "WASHINGTON")
                        # tokens = line.split(",")
                    else:
                        # print(line)
                        continue
                ip, city_code, city, state, country, country_code_iso2, country_code_iso3, lat, long, score = tokens
                country_by_ip[ip] = country_code_iso2
                if city != "":
                    lat, long = float(lat), float(long)
                    if score == "\n":
                        continue
                    score = score.strip("\n")
                    if score == "":
                        score = 0
                    else:
                        score = float(score)

                    if score >= min_score:
                        ip = ip.split("/")[0]
                        city_by_ip_candidate.setdefault(ip, []).append(city)
                        geo_by_ip[ip] = (lat, long)
                        score_by_ip[ip] = score
    city_by_ip = {}
    for ip, cities in city_by_ip_candidate.items():
        if len(cities) > min_localizations and len(set(cities)) == 1:
            city_by_ip[ip] = cities[0]
        else:
            del geo_by_ip[ip]
            del score_by_ip[ip]

    return country_by_ip, city_by_ip, geo_by_ip, score_by_ip

def load_org_per_asn():
    YEAR = '2023'
    MONTH = '05'
    DAY = '22'
    asn_path = f'ASNS-{YEAR}-{MONTH}-{DAY}.json'
    with open(f'../data/traceroute_meta/{asn_path}', 'r') as f:
        asn_per_org = {}
        for line in jsonlines.Reader(f):
            if not (line['organization'] is None):
                if 'orgName' in line['organization']:
                    asn_per_org[int(line['asn'])] = line['organization']['orgName']
                else:
                    asn_per_org[int(line['asn'])] = line['asn']
            else:
                asn_per_org[int(line['asn'])] = line['asn']
    return asn_per_org


def prefix_granularity():
    ### read a file
    dico = {}
    trie = radix.Radix()

    with gzip.open(f'../data/traceroute_meta/routeviews-rv2-20231009-1000.pfx2as.gz', 'r') as fin:
        for index,line in enumerate(fin):
            if index == 0:
                continue
            line = line.decode("utf-8")
            line = line.strip('\n')
            tokens = line.split('\t')
            prefix = tokens[0] + '/' + tokens[1]
            asn = tokens[2]
            dico[prefix] = asn
            trie.add(prefix)
    return trie

print(load_ixp_ip())