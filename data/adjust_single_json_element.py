import json
import os
import pathlib

path_input = pathlib.PurePath('Internet2', 'json', '20240804190000.json')
directory_output = pathlib.PurePath('Internet2', 'faked_single_route_change')
source = 'SEAT'
destination = 'SALT'
proportion_range = (0., 10.)
count = 20

with open(path_input, 'r') as f:
    json_data = json.load(f)

traffic_index = None
for index, traffic_dict in enumerate(json_data['traffic']):
    route = traffic_dict['route']
    if route[0] == source and route[-1] == destination:
        traffic_index = index
        break

print(f'Using route {" -> ".join(json_data["traffic"][traffic_index]["route"])} with volume {json_data["traffic"][traffic_index]["volume"]:.6f}')

os.makedirs(directory_output, exist_ok=True)

volume_original = json_data['traffic'][traffic_index]['volume']
for i in range(count):
    volume = volume_original * (proportion_range[0] * (count - 1 - i) + proportion_range[1] * i) / (count - 1)
    json_data['traffic'][traffic_index]['volume'] = volume
    with open(directory_output / f'{i}.json', 'w') as f:
        json.dump(json_data, f, indent=4)
