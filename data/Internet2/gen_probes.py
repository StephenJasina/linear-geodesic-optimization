import csv

def get_nodes(link_name: str) -> tuple[str, str]:
    parts = link_name.split('-')
    return parts[1], parts[2]

nodes = set()
with open('links.txt', 'r') as f:
    for line in f.readlines():
        line = line.rstrip()
        if not line:
            continue

        node_1, node_2 = get_nodes(line)
        nodes.add(node_1)
        nodes.add(node_2)

with open('probes.csv', 'w') as f:
    writer = csv.DictWriter(f, ['id', 'city', 'country', 'latitude', 'longitude'])
    writer.writeheader()
    for node in sorted(nodes):
        writer.writerow({
            'id': node,
            'country': 'United States of America',
        })
