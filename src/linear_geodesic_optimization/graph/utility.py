import typing

import networkx as nx
from scipy import sparse


def csc_matrix_from_attribute(
    graph: nx.Graph,
    attribute_name: typing.Optional[typing.Any] = None,
    default_value: float = 0.
) -> sparse.csc_matrix:
    """
    Compute a sparse adjacency matrix from a graph.

    Entries in the matrix will be populated with 1 by default. If an
    attribute name is provided, then entries will be taken from the
    graph's edges' data, using the default value if the attribute is
    not stored in the edge's data dictionary.

    The returned matrix is in compressed sparse column format.

    This function is close to NetworkX's to_scipy_sparse_array, but with
    better support for default values.
    """
    index_to_node = [node for node in graph.nodes]
    node_to_index = {node: index for index, node in enumerate(index_to_node)}
    n_nodes = len(index_to_node)

    data = []
    row_ind = []
    col_ind = []
    for u, v, edge_data in graph.edges(data=True):
        attribute = 1. if attribute_name is None else \
            edge_data[attribute_name] if attribute_name in edge_data else \
            default_value
        u_index = node_to_index[u]
        v_index = node_to_index[v]

        data.append(attribute)
        data.append(attribute)

        row_ind.append(u_index)
        row_ind.append(v_index)

        col_ind.append(v_index)
        col_ind.append(u_index)

    return sparse.csc_matrix((data, (row_ind, col_ind)), (n_nodes, n_nodes))
