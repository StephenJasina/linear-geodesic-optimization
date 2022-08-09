# Computation

Some notation first. If $i$ and $j$ are two indices vertices for which $(v_i, v_j)$ is an edge, let $c(i, j)$ be the index such that $v_i \to v_j \to v_{c(i, j)}$ traces a triangle counterclockwise. Note that this index exists and is unique assuming we have a mesh without boundary.

We have the following (standard) definition of the Laplace-Beltrami operator on a mesh:

$$\begin{aligned}
    N_{i, j} &\triangleq (v_i - v_{c(i, j)}) \times (v_j - v_{c(i, j)}), & \text{Outward normal of triangle $v_i \to v_j \to v_{c(i, j)}$} \\
    A_{i, j} &\triangleq \frac{1}{2}\|N_{i, j}\|_2, & \text{Area of triangle $v_i \to v_j \to v_{c(i, j)}$} \\
    D_{i, j} &\triangleq \begin{cases}
        \displaystyle\frac{1}{3}\sum_{\substack{k \\ \text{$(v_i, v_k)$ is an edge}}}A_{i, k} & \text{if $i = j$}, \\
        0 & \text{otherwise},
    \end{cases} & \text{Vertex triangle areas; diagonal} \\
    \cot(\theta_{i, j}) &= \frac{(v_i - v_{c(i, j)}) \cdot (v_j - v_{c(i, j)})}{2A_{i, j}}, & \text{Cotangent of $\angle v_iv_{c(i, j)}v_j$} \\
    (L_C)_{i, j} &\triangleq \begin{cases}
        \displaystyle\frac{1}{2}(\cot(\theta_{i, j}) + \cot(\theta_{j, i})) & \text{if $(i, j)$ is an edge}, \\
        \displaystyle-\frac{1}{2}\sum_{\substack{k \\ \text{$(v_i, v_k)$ is an edge}}}(\cot(\theta_{i, k}) + \cot(\theta_{k, i})) & \text{if $i = j$}, \\
        0 & \text{otherwise},
    \end{cases} & \text{Cotangent operator; sparse} \\
    L &\triangleq D^{-1}L_C.
\end{aligned}$$

# Gradient Computation

For the ease of notation (and the avoidance of edge cases), assume that we are using the spherical setup, so $v_\ell = \rho_\ell s_\ell$.

We compute

$$\begin{aligned}
    \frac{\partial v_i}{\partial \rho_\ell} &= \begin{cases}
        s_i & \text{if $\ell = i$}, \\
        0 & \text{otherwise},
    \end{cases} \\
    \frac{\partial N_{i, j}}{\partial \rho_\ell} &= \begin{cases}
        (v_{c(i, j)} - v_j) \times \frac{\partial v_\ell}{\partial \rho_\ell} & \text{if $\ell = i$}, \\
        (v_i - v_{c(i, j)}) \times \frac{\partial v_\ell}{\partial \rho_\ell} & \text{if $\ell = j$}, \\
        (v_j - v_i) \times \frac{\partial v_\ell}{\partial \rho_\ell} & \text{if $\ell = c(i, j)$}, \\
        0 & \text{otherwise},
    \end{cases} \\
    \frac{\partial A_{i, j}}{\partial \rho_\ell} &= \frac{1}{4A_{i, j}}N_{i, j} \cdot \frac{\partial N_{i, j}}{\partial \rho_\ell}, \\
    \left(\frac{\partial D}{\partial \rho_\ell}\right)_{i, j} &= \begin{cases}
        \displaystyle\frac{1}{3}\sum_{\substack{k \\ \text{$(v_i, v_k)$ is an edge}}}\frac{\partial A_{i, k}}{\partial \rho_\ell} & \text{if $i = j$}, \\
        0 & \text{otherwise},
    \end{cases} \\
    \frac{\partial}{\partial \rho_\ell}\cot(\theta_{i, j}) &= \begin{cases}
        \displaystyle\frac{(v_j - v_{c(i, j)}) \cdot \frac{\partial v_\ell}{\partial \rho_\ell} - 2\cot(\theta_{i, j})\frac{\partial A_{i, j}}{\partial \rho_\ell}}{2A_{i, j}} & \text{if $\ell = i$}, \\
        \displaystyle\frac{(v_i - v_{c(i, j)}) \cdot \frac{\partial v_\ell}{\partial \rho_\ell} - 2\cot(\theta_{i, j})\frac{\partial A_{i, j}}{\partial \rho_\ell}}{2A_{i, j}} & \text{if $\ell = j$}, \\
        \displaystyle\frac{(2v_{c(i, j)} - v_i - v_j) \cdot \frac{\partial v_\ell}{\partial \rho_\ell} - 2\cot(\theta_{i, j})\frac{\partial A_{i, j}}{\partial \rho_\ell}}{2A_{i, j}} & \text{if $\ell = c(i, j)$}, \\
        0 & \text{otherwise},
    \end{cases} \\
    \left(\frac{\partial L_C}{\partial \rho_\ell}\right)_{i, j} &= \begin{cases}
        \displaystyle\frac{1}{2}\left(\frac{\partial}{\partial \rho_\ell}\cot(\theta_{i, j}) + \frac{\partial}{\partial \rho_\ell}\cot(\theta_{j, i})\right) & \text{if $(i, j)$ is an edge}, \\
        \displaystyle-\frac{1}{2}\sum_{\substack{k \\ \text{$(v_i, v_k)$ is an edge}}}\left(\frac{\partial}{\partial \rho_\ell}\cot(\theta_{i, k}) + \frac{\partial}{\partial \rho_\ell}\cot(\theta_{k, i})\right) & \text{if $i = j$}, \\
        0 & \text{otherwise},
    \end{cases} \\
    \frac{\partial L}{\partial \rho_\ell} &= D^{-1}\left(\frac{\partial L_C}{\partial \rho_\ell} - \frac{\partial D}{\partial \rho_\ell}L\right).
\end{aligned}$$
