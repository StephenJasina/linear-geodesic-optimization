def approximate_geodesics_fpi(mesh, phi, initial_vertices):
    e = mesh.get_edges()
    c = mesh.get_c()
    vertices = set()
    to_process = []
    processed = set()
    for b in initial_vertices:
        for i in e[b]:
            cbi = c[b,i]
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

        cji = c[j,i]
        if phi[cji] < phi[i] or phi[cji] < phi[j]:
            to_process.append((j, i, cji))

        ckj = c[k,j]
        if phi[ckj] < phi[j] or phi[ckj] < phi[k]:
            to_process.append((k, j, ckj))

        cik = c[i,k]
        if phi[cik] < phi[k] or phi[cik] < phi[i]:
            to_process.append((i, k, cik))

        processed.add((i, j, k))

    return vertices

def approximate_geodesics_ti():
    # TODO: Write this
    pass
