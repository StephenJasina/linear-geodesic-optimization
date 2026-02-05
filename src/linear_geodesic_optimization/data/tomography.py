import collections
import heapq
import itertools
import typing

import networkx as nx
import numpy as np
from scipy import optimize, sparse


class PriorityQueue:
    """
    A simple wrapper for Python's heapq utilities.

    Based on https://docs.python.org/3/library/heapq.html#priority-queue-implementation-notes
    """
    def __init__(self):
        self.pq = []
        self.entry_finder = {}
        self.counter = itertools.count()

    def add(self, value: str, priority: float):
        'Add a new task or update the priority of an existing task'
        if value in self.entry_finder:
            self.remove(value)
        id = next(self.counter)
        entry = [priority, value, id, True]
        self.entry_finder[value] = entry
        heapq.heappush(self.pq, entry)

    def remove(self, value: str):
        entry = self.entry_finder.pop(value)
        entry[3] = False

    def pop(self) -> typing.Tuple[str, float]:
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while self.pq:
            priority, value, _, is_valid = heapq.heappop(self.pq)
            if is_valid:
                del self.entry_finder[value]
                return value, priority
        raise KeyError('pop from an empty priority queue')

def get_shortest_routes(graph: nx.Graph, edge_distance_label: typing.Optional[str]=None):
    """
    Run Dijkstra's algorithm.

    This is necessary ensure tie-breaking is always done in the same
    way.
    """
    routes = {}
    for source in graph.nodes:
        tree = {
            node: {
                'distance': np.inf,
                'predecessor': None
            }
            for node in graph.nodes
        }
        tree[source]['distance'] = 0.
        order_visited = []

        queue = PriorityQueue()
        for node in graph.nodes:
            queue.add(node, 0. if node == source else np.inf)
        while len(order_visited) < graph.number_of_nodes():
            node, distance = queue.pop()
            if np.isposinf(distance):
                break
            order_visited.append(node)
            for successor in graph.neighbors(node):
                d_node_successor = graph.edges[node, successor][edge_distance_label] if edge_distance_label is not None else 1.
                distance_candidate = tree[node]['distance'] + d_node_successor
                if distance_candidate < tree[successor]['distance']:
                    tree[successor] = {
                        'distance': distance_candidate,
                        'predecessor': node,
                    }
                    queue.add(successor, distance_candidate)

        routes_from_source = {}
        for node in order_visited:
            predecessor = tree[node]['predecessor']
            if predecessor is None:
                routes_from_source[node] = [node]
            else:
                routes_from_source[node] = routes_from_source[predecessor] + [node]
        routes[source] = routes_from_source
    return routes

def get_random_routes(graph: nx.Graph, seed=None):
    rng = np.random.default_rng(seed)
    routes = {}

    # Create a low-memory copy of the graph
    if isinstance(graph, nx.DiGraph):
        graph_copy = nx.DiGraph()
    else:
        graph_copy = nx.Graph()
    for node in graph.nodes:
        graph_copy.add_node(node)
    for u, v, d in graph.edges(data=True):
        distance = rng.random()
        graph_copy.add_edge(u, v, weight=distance)

    return get_shortest_routes(graph_copy, 'weight')

def compute_traffic_matrix(graph, routes, edge_weight_label='throughput'):
    index_to_link_id = []
    link_id_to_index = {}
    traffic_out_per_node = collections.defaultdict(float)
    traffic_in_per_node = collections.defaultdict(float)
    traffic_total = 0.
    for index, (id_source, id_target, throughput) in enumerate(graph.edges.data(edge_weight_label)):
        index_to_link_id.append((id_source, id_target))
        link_id_to_index[(id_source, id_target)] = index

        traffic_out_per_node[id_source] += throughput
        traffic_in_per_node[id_target] += throughput
        traffic_total += throughput

    # This is a scaling of y by the total traffic in the system
    traffic_counts = np.array([
        graph.edges[source_id, target_id][edge_weight_label]
        for source_id, target_id in index_to_link_id
    ])

    # Determine the ordering of the columns of A, ignoring
    # source-destination pairs with no possible traffic (either due to
    # traffic measurements or lack of possible routes)
    sources, destinations = zip(*[
        (source, destination)
        for source in graph.nodes
        for destination in graph.nodes
        if (
            source != destination
            and traffic_out_per_node[source] > 0.
            and traffic_in_per_node[destination] > 0.
            and source in routes
            and destination in routes[source]
        )
    ])

    # Also have a mapping to go from source-destination pairs to indices
    source_destination_to_index = {
        (source, destination): index
        for index, (source, destination) in enumerate(zip(sources, destinations))
    }

    # Create the (sparse) traffic matrix and its transpose
    traffic_matrix_data = []
    traffic_matrix_row_ind = []
    traffic_matrix_col_ind = []
    for index, (source, destination) in enumerate(zip(sources, destinations)):
        route = routes[source][destination]
        for link_id in itertools.pairwise(route):
            traffic_matrix_data.append(1)
            traffic_matrix_row_ind.append(link_id_to_index[link_id])
            traffic_matrix_col_ind.append(index)

    traffic_matrix = sparse.csr_matrix(
        (traffic_matrix_data, (traffic_matrix_row_ind, traffic_matrix_col_ind)),
        shape=(len(index_to_link_id), len(sources))
    )
    traffic_matrix_transpose = sparse.csr_matrix(
        (traffic_matrix_data, (traffic_matrix_col_ind, traffic_matrix_row_ind)),
        shape=(len(sources), len(index_to_link_id))
    )

    def loss(xs, lam=0.01):
        errors = traffic_counts - traffic_total * (traffic_matrix @ xs)
        accuracy = errors @ errors

        penalty = 0.
        if lam != 0.:
            for x, source, destination in zip(xs, sources, destinations):
                n_s = traffic_out_per_node[source]
                n_d = traffic_in_per_node[destination]
                if n_s > 0. and n_d > 0. and x != 0.:
                    penalty += x * np.log2(x * traffic_total**2 / (n_s * n_d))

        if np.isinf(lam):
            return penalty

        return accuracy / traffic_total**2 + lam**2 * penalty

    def dif_loss(xs, lam=0.01):
        errors = traffic_counts - traffic_total * (traffic_matrix @ xs)
        dif_accuracy = -2 * traffic_total * traffic_matrix_transpose @ errors

        if lam != 0.:
            dif_penalty = np.array([
                np.log2(x * traffic_total**2 / (n_s * n_d)) + 1 / np.log(2)
                if n_s > 0. and n_d > 0. else 0.
                for x, source, destination in zip(xs, sources, destinations)
                for n_s in (traffic_out_per_node[source],)
                for n_d in (traffic_in_per_node[destination],)
            ])
        else:
            dif_penalty = np.zeros(xs.shape)

        return dif_accuracy / traffic_total**2 + lam**2 * dif_penalty

    # Gravity model
    x_0 = np.array([
        n_s * n_d / traffic_total**2
        for source, destination in zip(sources, destinations)
        for n_s in (traffic_out_per_node[source],)
        for n_d in (traffic_in_per_node[destination],)
    ])
    x_0 = x_0 / np.sum(x_0)

    x_opt, _, _ = optimize.fmin_l_bfgs_b(
        loss, x_0, fprime=dif_loss, args=[0.01],
        # factr=1e1, pgtol=1e-12,  # Tuning parameters for the optimizer
        bounds=[(1e-12, 1.) for _ in x_0]
    )
    x_opt = x_opt / np.sum(x_opt)

    return {
        (source, destination): x
        for source, destination, x in zip(sources, destinations, x_opt)
    }
