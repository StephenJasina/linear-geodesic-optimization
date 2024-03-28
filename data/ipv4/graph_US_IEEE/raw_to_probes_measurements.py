import csv
import os

def add_probe(probes, id, city, latitude, longitude):
    if id not in probes:
        probes[id] = {
            'city': city,
            'country': 'United States of America',
            'latitude': latitude,
            'longitude': longitude,
        }

def convert_raw_to_probes_and_latencies(
    raw_file_path,
    probes_file_path,
    latencies_file_path
):
    with open(raw_file_path, 'r') as f:
        reader = csv.DictReader(f)
        probes = {}
        latencies = {}
        for line in reader:
            # add_probe(
            #     probes,
            #     line['source_id'],
            #     line['city_source'],
            #     line['lat_source'],
            #     line['long_source'],
            # )
            add_probe(
                probes,
                line['target_id'],
                line['city_target'],
                line['lat_target'],
                line['long_target'],
            )

            latencies[(line['source_id'], line['target_id'])] = line['rtt']

    with open(probes_file_path, 'w') as f:
        writer = csv.DictWriter(f, ['id', 'city', 'country', 'latitude', 'longitude'])
        writer.writeheader()
        for id, data in sorted(probes.items()):
            writer.writerow({
                'id': id,
                'city': data['city'],
                'country': data['country'],
                'latitude': data['latitude'],
                'longitude': data['longitude'],
            })

    with open(latencies_file_path, 'w') as f:
        writer = csv.DictWriter(f, ['source_id', 'target_id', 'rtt'])
        writer.writeheader()
        for (source_id, target_id), rtt in sorted(latencies.items()):
            writer.writerow({
                'source_id': source_id,
                'target_id': target_id,
                'rtt': rtt,
            })

if __name__ == '__main__':
    input_directory = 'raw'
    output_directory = '.'
    for raw_filename in os.listdir(input_directory):
        raw_file_path = os.path.join(input_directory, raw_filename)
        if os.path.isfile(raw_file_path) and raw_file_path.startswith('raw'):
            probes_file_path = os.path.join(output_directory, 'probes' + raw_filename[3:])
            latencies_file_path = os.path.join(output_directory, 'latencies' + raw_filename[3:])
            convert_raw_to_probes_and_latencies(
                raw_file_path,
                probes_file_path,
                latencies_file_path
            )
