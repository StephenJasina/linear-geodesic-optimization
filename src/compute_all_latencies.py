import json
import math
import os

directory = os.path.join('..', 'data', 'toy')

initial_latencies = None

with open(os.path.join(directory, 'latency.json')) as latency_file:
    initial_latencies = {
        (key[0], key[1]): value
        for key, value in json.load(latency_file).items()
    }

vertices = list(sorted(set([v for (v, _) in initial_latencies]) | set([v for (_, v) in initial_latencies])))
vertex_to_index = {v: i for i, v in enumerate(vertices)}
distances = [[math.inf] * len(vertices) for _ in range(len(vertices))]

for (u, v), weight in initial_latencies.items():
    distances[vertex_to_index[u]][vertex_to_index[v]] = weight
    distances[vertex_to_index[v]][vertex_to_index[u]] = weight

for i in range(len(vertices)):
    distances[i][i] = 0.

for k in range(len(vertices)):
    for i in range(len(vertices)):
        for j in range(len(vertices)):
            if distances[i][j] > distances[i][k] + distances[k][j]:
                distances[i][j] = distances[i][k] + distances[k][j]

new_latencies = {
    (vertices[i] + vertices[j]): distances[i][j]
    for i in range(len(vertices))
    for j in range(i + 1, len(vertices))
}
print(json.dumps(new_latencies))
