import csv
import heapq
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join('..', 'src'))
from linear_geodesic_optimization.data import utility

def get_k_nearest(points, p, k):
    """Find the k nearest points in points to p."""

    heap = []
    # This heap contains elements of the form (-distance, point index).
    # Since this is a min-heap, the farthest point is at the top.
    for index, point in enumerate(points):
        heapq.heappush(heap,
                       (-utility.get_spherical_distance(p, point), index))
        if len(heap) > k:
            heapq.heappop(heap)

    k_nearest = []
    while heap:
        k_nearest.append(heapq.heappop(heap)[1])
    return list(reversed(k_nearest))

if __name__ == '__main__':
    points = []
    with open(os.path.join('graph_Europe', 'probes.csv'), 'r') as probes_file:
        probes_reader = csv.DictReader(probes_file)
        points = [
            utility.get_sphere_point((float(row['latitude']),
                                      float(row['longitude'])))
            for row in probes_reader
        ]

    k = 3
    distances = sorted([
        utility.get_spherical_distance(p,
                                       points[get_k_nearest(points, p, k)[-1]])
        for p in points
    ], reverse=True)

    plt.plot(distances, 'k.')
    plt.show()
