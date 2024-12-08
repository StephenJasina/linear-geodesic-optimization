import typing

import dcelmesh


def compute_connected_components_faces(topology: dcelmesh.Mesh, face_indices) -> typing.List[int]:
    """
    Given a topology and faces in the topology, compute the connected
    components of the faces.
    """
    connected_components = []
    unused_face_indices = set(face_indices)
    while unused_face_indices:
        stack = [next(iter(unused_face_indices))]
        unused_face_indices.remove(stack[0])
        connected_component = set(stack)
        while stack:
            face = topology.get_face(stack.pop())
            for neighbor in face.faces():
                if neighbor.index in unused_face_indices:
                    stack.append(neighbor.index)
                    unused_face_indices.remove(neighbor.index)
                    connected_component.add(neighbor.index)
        connected_components.append(connected_component)

    return connected_components

def compute_holes_faces(topology: dcelmesh.Mesh, vertex_indices) -> typing.List[typing.List[int]]:
    """
    Given a topology and vertices in the topology, compute the
    connected components of the faces not entirely in the set of
    vertices.
    """
    vertex_indices = set(vertex_indices)
    face_indices = []
    for face in topology.faces():
        for vertex in face.vertices():
            if vertex.index not in vertex_indices:
                face_indices.append(face.index)
                break
    return compute_connected_components_faces(
        topology, face_indices
    )

def compute_holes_vertices(
    topology: dcelmesh.Mesh, vertex_indices
) -> typing.Tuple[typing.List[typing.Set[int]], typing.List[typing.Set[int]]]:
    """
    Given a list of vertices, group the vertices not given.

    In particular, we find the islands left over when removing the given
    vertices from the mesh. These are returned as a list of each island,
    where the island is stored as set of vertex indices. Also returned
    are the vertex indices of the vertices on the boundaries of each hole.
    """
    holes_faces = compute_holes_faces(topology, vertex_indices)
    holes = []
    boundaries = []
    for hole_faces in holes_faces:
        halfedge_indices = set(
            halfedge.index
            for face_index in hole_faces
            for halfedge in topology.get_face(face_index).halfedges()
        )
        boundary_halfedge_indices = [
            halfedge_index
            for halfedge_index in halfedge_indices
            if topology.get_halfedge(halfedge_index).is_on_boundary()
        ]
        boundary_vertex_indices = set(
            topology.get_halfedge(boundary_halfedge_index).origin.index
            for boundary_halfedge_index in boundary_halfedge_indices
        )
        hole_vertex_indices = set(
            vertex.index
            for face_index in hole_faces
            for vertex in topology.get_face(face_index).vertices()
        )
        holes.append(hole_vertex_indices)
        boundaries.append(boundary_vertex_indices)
    return holes, boundaries
