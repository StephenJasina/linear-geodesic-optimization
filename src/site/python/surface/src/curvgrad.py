import numpy as np

# see the documentation for the definitions of symbols and purposes
# of these functions

def df_dz (n, a, apos, b, bpos, c, cpos):
    '''
    partial f / partial z
    f = (c-b)^T (a-b)
    apos is the index of a into the 1D grid vector
    a is [a_x, a_y, a_z]
    '''
    gr = np.zeros(n)
    gr[apos] = c[2] - b[2]
    gr[bpos] = (2 * b[2]) - a[2] - c[2]
    gr[cpos] = a[2] - b[2]
    return gr

def dh_dz(n, a, apos, b, bpos):
    '''
    partial h / partial z
    f = (a-b)^T (a-b)
    apos is the index of a into the 1D grid vector
    a is [a_x, a_y, a_z]
    '''
    gr = np.zeros(n)
    gr[apos] = 2 * (a[2] - b[2])
    gr[bpos] = 2 * (b[2] - a[2])
    return gr

def fz(a, b, c):
    return (c - b).T @ (a - b)

def pz(a, b, c):
    return np.sqrt((c - b).T @ (c - b))

def qz(a, b, c):
    return np.sqrt((a - b).T @ (a - b))

def gz(a, b, c):
    return pz(a, b, c) * qz(a, b, c)

def wz(a, b, c):
    return fz(a, b, c) / gz(a, b, c)

def dq_dz(n, a, apos, b, bpos, c, cpos):
    return dh_dz(n, a, apos, b, bpos) / (2 * qz(a, b, c))

def dp_dz(n, a, apos, b, bpos, c, cpos):
    return dh_dz(n, c, cpos, b, bpos) / (2 * pz(a, b, c))

def dg_dz(n, a, apos, b, bpos, c, cpos):
    '''
    partial g / partial z
    g = sqrt((c-b)^(c-b)) * sqrt((a-b)^(a-b))
    apos is the index of a into the 1D grid vector
    a is [a_x, a_y, a_z]
    '''
    gr = pz(a, b, c) * dq_dz(n, a, apos, b, bpos, c, cpos) + qz(a, b, c) * dp_dz(n, a, apos, b, bpos, c, cpos)
    return gr

def dw_dz(n, a, apos, b, bpos, c, cpos):
    return ((gz(a, b, c) * df_dz(n, a, apos, b, bpos, c, cpos))
            - (fz(a, b, c) * dg_dz(n, a, apos, b, bpos, c, cpos))) / (gz(a, b, c) * gz(a, b, c))

def dalpha_dz(n, a, apos, b, bpos, c, cpos):
    '''
    partial alpha / partial z
    alpha is angle at vertex b defined by triangle (a, b, c)
    '''
    wzval = wz(a, b, c)
    return - (1/np.sqrt(1 - (wzval * wzval))) * dw_dz(n, a, apos, b, bpos, c, cpos)

def net_dalpha_dz(n, b, triangs, vertices):
    '''
    for a vertex b which is an index into vertices
    and the triangs that contain it
    return the net dalpha_dz vector
    '''
    res = np.zeros(n)
    for T in triangs:
        others = [t for t in T if t != b]
        res += dalpha_dz(n, vertices[others[0]], others[0], vertices[b], b, vertices[others[1]], others[1])
    # experimental - only return gradient for point in question
    newres = np.zeros(n)
    newres[b] = res[b]
    return newres

def L_grad_indiv(n, kappa, c, v, t_of_v, vertices):
    '''
    compute the gradient for an individual data point v
    '''
    if np.isnan(c[v]):
        return np.zeros(n)
    elif np.isnan(kappa[v]):
        return np.zeros(n)
    else:
        nda = net_dalpha_dz(n, v, t_of_v[v], vertices)
        # if (kappa[v] < -0.1):
        #     print("kappa[v]: {}, k-c: {}, nda: {}".format(kappa[v], kappa[v] - c[v], sum(nda)))
        return 2 * (kappa[v] - c[v]) * net_dalpha_dz(n, v, t_of_v[v], vertices)

def L_grad(n, kappa, c, t_of_v, vertices):
    '''
    compute the gradient at all points (full gradient)
    '''
    grad = np.zeros(n)
    for v in range(len(vertices)):
        grad += L_grad_indiv(n, kappa, c, v, t_of_v, vertices)
    return grad

def L_grad_penalized(lam, pen_fn, n, kappa, c, t_of_v, vertices):
    '''
    compute a penalized gradient, applied with weight lam
    '''
    basic_grad = L_grad(n, kappa, c, t_of_v, vertices)
    return basic_grad + lam * np.ravel(pen_fn)

def L_grad_penalized_with_convexity(lam_cvx, L, zf, lam, pen_fn, n, kappa, c, t_of_v, vertices):
    '''
    compute a mangled gradient intended to enforce convexity
    not currently used
    '''
    pen_grad = L_grad_penalized(lam, pen_fn, n, kappa, c, t_of_v, vertices)
    penalty = - np.ravel(L @ zf)
    kappa_copy = kappa.copy()
    kappa_copy[np.isnan(kappa_copy)] = 0
    kappa_copy[kappa_copy < 0] = 0
    penalty = penalty * kappa_copy
    penalty[penalty < 0] = 0
    return pen_grad - lam_cvx * penalty, penalty

def L_grad_penalized_positized(lam, pen_fn, n, kappa, c, t_of_v, vertices):
    '''
    compute a mangled gradient intended to enforce convexity
    rule: if goal curvature is positive, current curvature is less than goal,
    and gradient is positive (ie, moving to a more non-convex direction)
    zero out the gradient here
    not currently used
    '''
    pen_grad = L_grad_penalized(lam, pen_fn, n, kappa, c, t_of_v, vertices)
    for i in range(len(pen_grad)):
        if (kappa[i] > 0) and (c[i] < kappa[i]) and (pen_grad[i] > 0):
            pen_grad[i] = 0
    return pen_grad
