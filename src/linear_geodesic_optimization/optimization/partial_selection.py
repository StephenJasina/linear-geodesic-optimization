def approximate_geodesics_fpi(mesh, phi, initial_vertices):
    e = mesh.get_edges()
    nxt = mesh.get_nxt()
    vertices = set()
    to_process = []
    processed = set()
    for b in initial_vertices:
        for i in e[b]:
            cbi = nxt[b,i]
            if phi[b] > phi[i] or phi[b] > phi[cbi]:
                to_process.append((b, i, cbi))

    while to_process:
        (i, j, k) = to_process[-1]
        del to_process[-1]
        if j < i and j < k:
            j, k, i = i, j, k
        elif k < i and k < j:
            k, i, j = i, j, k
        if (i, j, k) in processed:
            continue

        vertices.add(i)
        vertices.add(j)
        vertices.add(k)

        if (j, i) in nxt:
            nxt_ji = nxt[j,i]
            if phi[nxt_ji] < phi[i] or phi[nxt_ji] < phi[j]:
                to_process.append((j, i, nxt_ji))

        if (k, j) in nxt:
            nxt_kj = nxt[k,j]
            if phi[nxt_kj] < phi[j] or phi[nxt_kj] < phi[k]:
                to_process.append((k, j, nxt_kj))

        if (i, k) in nxt:
            nxt_ik = nxt[i,k]
            if phi[nxt_ik] < phi[k] or phi[nxt_ik] < phi[i]:
                to_process.append((i, k, nxt_ik))

        processed.add((i, j, k))

    return vertices

def approximate_geodesics_ti():
    # TODO: Write this
    raise NotImplementedError()
