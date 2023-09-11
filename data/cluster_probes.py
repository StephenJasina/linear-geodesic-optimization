import networkx as nx
import numpy as np
import sklearn.cluster

import utility

def get_cluster_center(cluster):
    p = sum([
        utility.get_sphere_point((float(data['lat']), float(data['long'])))
        for _, data in cluster
    ])
    p = p / np.linalg.norm(p)
    return utility.get_latlong(p)

def cluster_graph(graph, eps=None, min_samples=None):
    nodes = list(graph.nodes(data=True))
    latlongs = np.array([
        (float(data['lat']), float(data['long']))
        for _, data in nodes
    ])

    # Compute clusters. Ensure that each cluster has a unique label
    # (rather than clusters of size 1 being labeled with -1).
    if 'cluster' in nodes[0][1]:
        cluster_labels = [data['cluster'] for _, data in nodes]
        cluster_count = max(cluster_labels) + 1
    else:
        clustering = sklearn.cluster.DBSCAN(
            eps=eps, min_samples=min_samples,
            metric=utility.get_spherical_distance
        )
        clustering.fit([
            utility.get_sphere_point(latlong)
            for latlong in latlongs
        ])
        cluster_labels = list(clustering.labels_)
        cluster_count = max(cluster_labels) + 1
        for i in range(len(cluster_labels)):
            if cluster_labels[i] == -1:
                cluster_labels[i] = cluster_count
                cluster_count += 1

    new_graph = nx.Graph()

    # Copy nodes to new graph
    clusters = [[] for _ in range(cluster_count)]
    for node, label in zip(nodes, cluster_labels):
        clusters[label].append(node)
    # TODO: maybe don't seed the RNG, or even use it
    rng = np.random.default_rng(0)
    for cluster in clusters:
        rng.shuffle(cluster)
    for cluster in clusters:
        node, data = cluster[0]
        data['lat'], data['long'] = get_cluster_center(cluster)
        new_graph.add_node(node, **data)

    # Copy edges to new graph
    for i, cluster_i in enumerate(clusters):
        for cluster_j in clusters[i+1:]:
            rtt = min([
                graph.edges[node_i,node_j]['rtt']
                for node_i, _ in cluster_i
                for node_j, _ in cluster_j
                if (node_i, node_j) in graph.edges
            ], default=np.inf)
            if rtt != np.inf:
                new_graph.add_edge(cluster_i[0][0], cluster_j[0][0], rtt=rtt)

    return new_graph
