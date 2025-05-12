import typing

import networkx as nx


def compute_triangles(graph: nx.Graph, weight: str = 'rtt') \
        -> typing.Dict[typing.Tuple[str, str, str],
                       typing.Tuple[float, float, float]]:
    triangles: typing.Dict[typing.Tuple[str, str, str],
                           typing.Tuple[float, float, float]] = {}
    for u, v, data in graph.edges(data=True):
        weight_uv = data[weight]
        u, v = min(u, v), max(u, v)
        for w in graph.nodes:
            if w <= v:
                continue

            if (u, w) in graph.edges and (v, w) in graph.edges:
                weight_uw = graph.edges[u,w][weight]
                weight_vw = graph.edges[v,w][weight]

                triangles[u,v,w] = (weight_vw, weight_uw, weight_uv)

    return triangles

def get_long_edges(
    graph: nx.Graph,
    triangles: typing.Dict[typing.Tuple[str, str, str],
                           typing.Tuple[float, float, float]]
) -> typing.Set[typing.Tuple[str, str]]:
    long_edges = set()
    for u, v, w in triangles:
        weight_uv = graph.edges[u,v]['rtt']
        weight_uw = graph.edges[u,w]['rtt']
        weight_vw = graph.edges[v,w]['rtt']

        if weight_uv > weight_uw:
            if weight_uv > weight_vw:
                long_edges.add((u, v))
            else:
                long_edges.add((v, w))
        else:
            if weight_uw > weight_vw:
                long_edges.add((u, w))
            else:
                long_edges.add((v, w))
    return long_edges


def compute_goodnesses(
    triangles: typing.Dict[typing.Tuple[str, str, str],
                           typing.Tuple[float, float, float]],
    use_r: bool = False
) -> typing.Dict[typing.Tuple[str, str, str], float]:
    goodnesses: typing.Dict[typing.Tuple[str, str, str], float] = {}
    for triangle, (weight_uv, weight_uw, weight_vw) in triangles.items():
        weight_max = max(weight_uv, weight_uw, weight_vw)
        weight_sum = weight_uv + weight_uw + weight_vw
        goodness = weight_max / (weight_sum - weight_max)
        if use_r:
            goodness *= (1 + 2 * weight_max - weight_sum)
        goodnesses[triangle] = goodness

    return goodnesses

def compute_greedy_set_cover(sets: typing.Dict[typing.Hashable,
                                               typing.Set[typing.Any]]) \
        -> typing.List[typing.Hashable]:
    current_cover = set()
    candidate_sets = dict(sets)
    set_cover: typing.List[typing.Hashable] = []
    while True:
        # Find the candidate with the most uncovered elements
        best_candidate = None
        best_candidate_set = None
        best_candidate_size = 0
        removable_candidates = []
        for candidate, candidate_set in candidate_sets.items():
            # Small optimization to avoid unnecessary difference computations
            if len(candidate_set) <= best_candidate_size:
                continue

            candidate_size = len(candidate_set.difference(current_cover))

            # Remove sets that are already covered
            if candidate_size == 0:
                removable_candidates.append(candidate)
                continue

            if candidate_size > best_candidate_size:
                best_candidate = candidate
                best_candidate_set = candidate_set
                best_candidate_size = candidate_size

        # We're done if no elements are uncovered
        if best_candidate is None:
            return set_cover

        for candidate in removable_candidates:
            del candidate_sets[candidate]

        set_cover.append(best_candidate)
        current_cover = current_cover.union(best_candidate_set)
