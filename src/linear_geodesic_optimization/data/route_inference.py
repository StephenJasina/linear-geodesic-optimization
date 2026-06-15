import itertools
import typing

import networkx as nx
import numpy as np
from scipy.optimize import lsq_linear


def dicts_to_parallel_lists(*dictionaries: dict[tuple[int, int], float]) -> tuple[list[float], ...]:
    """
    Convert dictionaries into parallel lists by key.

    Keys that are absent from any of the dictionaries are skipped.
    """
    if not dictionaries:
        # Guard against empty parameter list
        return ()
    return tuple(list(l)
        for l in zip(
            *[
                tuple(dictionary[key] for dictionary in dictionaries)
                for key in dictionaries[0]
                if all([key in dictionary for dictionary in dictionaries])
            ]
        )
    )

def compute_rtts_from_inferred_routes(graph: nx.Graph, link_latencies: dict[tuple[int, int], float], routes: dict[tuple[int, int], list[int]]) -> dict[tuple[int, int], float]:
    return {
        (source, destination): sum(
            link_latencies[min(node_a, node_b), max(node_a, node_b)]
            for node_a, node_b in itertools.pairwise(route)
        )
        for (source, destination), route in routes.items()
    }

def compute_rtts_from_inferred_routes(graph: nx.Graph, link_latencies: dict[tuple[int, int], float], routes: dict[tuple[int, int], list[int]]) -> dict[tuple[int, int], float]:
    return {
        (source, destination): sum(
            link_latencies[min(node_a, node_b), max(node_a, node_b)]
            for node_a, node_b in itertools.pairwise(route)
        )
        for (source, destination), route in routes.items()
    }

def infer_routes(graph: nx.Graph, rtts: dict[tuple[int, int], float], max_iter: typing.Optional[int] = None, all_pairs: bool = False, verbose: bool = False) -> dict[tuple[int, int], list[int]]:
    def estimate_link_latencies(routes: dict[tuple[int, int], list[int]]) -> dict[tuple[int, int], float]:
        """
        Estimate the latencies for each link (edge) in the graph.

        This associates a value `latency` to each edge present in
        `graph` using only connectivity and the passed in `rtts`. The
        result is stored as a dictionary `latencies` mapping node pairs
        to their corresponding `latency`. `latencies` is chosen such
        that, for each `route` in `routes` and its corresponding `rtt`
        (selected to be `rtts[route[0], route[-1]]`), the sum of
        `latencies[route[i], route[i + 1]]` is approximately (in the
        least squares sense) `rtt`.

        Furthermore, the values are constrained so that
        `latencies[a, b]` is at least `graph.edges[a, b]['gcl']`.
        """
        edges = list(graph.edges())
        edge_to_col = {(min(a, b), max(a, b)): j for j, (a, b) in enumerate(edges)}

        rows_A, rows_b = [], []
        for (source, destination), route in routes.items():
            if (source, destination) not in rtts or len(route) < 2:
                continue
            row = [0] * len(edges)
            for a, b in itertools.pairwise(route):
                row[edge_to_col[min(a, b), max(a, b)]] += 1
            rows_A.append(row)
            rows_b.append(rtts[source, destination])

        if not rows_A:
            return {(min(a, b), max(a, b)): graph.edges[a, b]['gcl'] for a, b in edges}

        A = np.array(rows_A, dtype=float)
        b = np.array(rows_b, dtype=float)
        lb = np.array([graph.edges[a, b]['gcl'] for a, b in edges], dtype=float)

        result = lsq_linear(A, b, bounds=(lb, np.inf))
        return {(min(a, b), max(a, b)): float(result.x[j]) for j, (a, b) in enumerate(edges)}

    graph_copy = graph.copy()

    def estimate_routes(link_latencies: dict[tuple[int, int], float]) -> dict[tuple[int, int], list[int]]:
        """
        Estimate routes assuming the given link latencies.

        Given latencies for each edge in `graph`, construct routes whose
        weights (sum of latencies along the route) most closely match
        `rtts`. Only non-backtracking paths are considered: at each hop,
        only neighbours strictly closer (in latency shortest-path distance)
        to the destination are eligible.
        """
        for a, b, data in graph_copy.edges(data=True):
            data['latency'] = link_latencies[min(a, b), max(a, b)]

        new_routes = {}
        for (source, destination) in rtts:
            if source == destination:
                new_routes[source, destination] = [source]
                continue

            target = rtts[source, destination]
            # Shortest latency distance from every node to destination.
            # A neighbour is "closer" iff its entry here is strictly smaller
            # than the current node's, guaranteeing the search is acyclic.
            dist_to_dest = nx.single_source_dijkstra_path_length(
                graph_copy, destination, weight='latency'
            )

            best_path: list[int] | None = None
            best_diff = float('inf')

            def dfs(node: int, latency: float, path: list[int]) -> None:
                nonlocal best_path, best_diff
                if node == destination:
                    diff = abs(latency - target)
                    if diff < best_diff:
                        best_diff, best_path = diff, list(path)
                    return
                # Prune: already further above target than our best match.
                if latency - target >= best_diff:
                    return
                d = dist_to_dest.get(node, float('inf'))
                for neighbor in graph_copy.neighbors(node):
                    if dist_to_dest.get(neighbor, float('inf')) < d:
                        path.append(neighbor)
                        dfs(neighbor, latency + graph_copy.edges[node, neighbor]['latency'], path)
                        path.pop()

            dfs(source, 0.0, [source])
            new_routes[source, destination] = best_path if best_path is not None else [source]

        return new_routes

    link_latencies = {
        (min(a, b), max(a, b)): data['gcl']
        for a, b, data in graph.edges(data=True)
    }
    routes_best = None
    link_latencies_best = None
    r2_best = -1.
    for iteration in itertools.count():
        link_latencies_previous = link_latencies

        if verbose:
            print('Estimating routes')
        routes = estimate_routes(link_latencies)
        if verbose:
            print('Estimating link latencies')
        link_latencies = estimate_link_latencies(routes)

        rtts_inferred = compute_rtts_from_inferred_routes(graph, link_latencies, routes)
        x, y = dicts_to_parallel_lists(rtts, rtts_inferred)
        r2 = np.corrcoef(x, y)[0, 1]**2
        if verbose:
            print(f'r^2: {r2}')

        if r2 > r2_best:
            routes_best = routes
            link_latencies_best = link_latencies
            r2_best = r2

        if link_latencies == link_latencies_previous or (max_iter is not None and iteration > max_iter):
            if verbose:
                print(f'finished inference in {iteration} iterations')
            if all_pairs and link_latencies_best is not None:
                def latency_weight(u, v, data):
                    return link_latencies_best.get((u, v),
                           link_latencies_best.get((v, u), data['gcl']))
                for src, paths in nx.all_pairs_dijkstra_path(graph, weight=latency_weight):
                    for dst, path in paths.items():
                        if (src, dst) not in routes_best:
                            routes_best[src, dst] = path
            return routes_best
