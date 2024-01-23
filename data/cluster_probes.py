import networkx as nx
import numpy as np
import sklearn.cluster

import utility

def get_cluster_center(cluster):
    """Find the point in the cluster nearest to its centroid."""
    sphere_points = [
        utility.get_sphere_point((data['lat'], data['long']))
        for _, data in cluster
    ]
    centroid_direction = sum(sphere_points)
    center_index = np.argmax([
        centroid_direction @ sphere_point
        for sphere_point in sphere_points
    ])

    return cluster[center_index]

def cluster_graph(graph, distance_threshold):
    """
    Simplify a graph using agglomerative clustering with single linkage.

    As input, take a graph whose vertices have `lat` (latitude) and
    `long` (longitude) attributes. The edges must also have associated
    `rtt` (round trip time, or RTT) attributes.

    For the given graph, cluster the vertices according to the input
    parameter. Then, for each cluster, pick a representative node.
    Generate a new graph on these representative nodes, where an edge
    exists between two representative nodes if there is an edge between
    their corresponding clusters. If there are multiple such edges
    between clusters, the associated RTT is the minimal one.
    """
    nodes = list(graph.nodes(data=True))

    clustering = sklearn.cluster.AgglomerativeClustering(
        n_clusters=None, distance_threshold=distance_threshold,
        linkage='single', metric='precomputed'
    )
    sphere_points = [
        utility.get_sphere_point((data['lat'], data['long']))
        for _, data in nodes
    ]
    spherical_distances = [
        [
            utility.get_spherical_distance(a, b)
            for a in sphere_points
        ]
        for b in sphere_points
    ]
    cluster_labels = clustering.fit_predict(spherical_distances)
    cluster_count = max(cluster_labels) + 1

    new_graph = nx.Graph()

    # Copy nodes to new graph
    clusters = [[] for _ in range(cluster_count)]
    cluster_centers = []
    for (node, data), label in zip(nodes, cluster_labels):
        clusters[label].append((node, data))
    for cluster in clusters:
        node, data = get_cluster_center(cluster)
        cluster_centers.append(node)
        new_graph.add_node(node, **data)

    # Copy edges to new graph
    for i, cluster_i in enumerate(clusters):
        for j, cluster_j in enumerate(clusters[i+1:], start=i+1):
            rtt = min([
                graph.edges[node_i,node_j]['rtt']
                for node_i, _ in cluster_i
                for node_j, _ in cluster_j
                if (node_i, node_j) in graph.edges
            ], default=np.inf)
            if rtt != np.inf:
                id_source = cluster_centers[i]
                id_target = cluster_centers[j]
                lat_source = graph.nodes[id_source]['lat']
                long_source = graph.nodes[id_source]['long']
                lat_target = graph.nodes[id_target]['lat']
                long_target = graph.nodes[id_target]['long']
                gcl = utility.get_GCL(
                    [lat_source, long_source],
                    [lat_target, long_target]
                )
                new_graph.add_edge(id_source, id_target,
                                   rtt=rtt, gcl=gcl)

    # Copy other attributes
    new_graph.graph = dict(graph.graph)

    return new_graph
