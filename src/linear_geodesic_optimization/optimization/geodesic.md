Note: This document assumes the notation found in `../main.md` and `laplacian.md`.

# Computation

Say we want to find the geodesic distances to a set of points $\gamma$. Following the [Crane et al's heat method](https://www.cs.cmu.edu/~kmcrane/Projects/HeatMethod/), we use the (approximate) heat flow $u^\gamma$, where

$$\begin{aligned}
    t &\triangleq (\text{mean spacing between mesh points})^2, & \text{Adjustable parameter} \\
    \delta^\gamma &\triangleq \begin{cases}
        1 & \text{if $v_i \in \gamma$}, \\
        0 & \text{if $v_i \not\in \gamma$},
    \end{cases} & \text{Heat source} \\
    u^\gamma &\triangleq (D - tL_C)^{-1}\delta^\gamma & \text{Heat flow}
\end{aligned}$$

Similar to when computing the Laplacian, we need to be careful about computing these values on a mesh with boundary. Following Crane et al's advice, we compute $u^\gamma$ using both the zero Neumann and zero Dirichlet boundary conditions, and then average them.

With $u^\gamma$ in hand, we can then compute

$$\begin{aligned}
    q_{i, j} &\triangleq u^\gamma_i(v_{c(i, j)} - v_j), \\
    m_{i, j} &\triangleq q_{i, j} + q_{j, c(i, j)} + q_{c(i, j), i}, \\
    (\widetilde{\nabla} u^\gamma)_{i, j} &\triangleq N_{i, j} \times m_{i, j}, \\
    X^\gamma_{i, j} &\triangleq -\frac{(\widetilde{\nabla} u^\gamma)_{i, j}}{\|(\widetilde{\nabla} u^\gamma)_{i, j}\|_2}, \\
    p_{i, j} &\triangleq \cot(\theta_{i, j})(v_j - v_i), \\
    (\nabla \cdot X^\gamma)_i &= \frac{1}{2}\sum_{\substack{k \\ \text{$(v_i, v_k)$ is an edge}}}(p_{i, k} - p_{c(i, k), i}) \cdot X^\gamma_{i, k}, \\
    \phi^\gamma &= L_C^+ \cdot (\nabla \cdot X^\gamma).
\end{aligned}$$

Here, $L_C^+$ is the [pseudoinverse](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse) of $L_C$ (as it is singular). Note that the integrated divergence can be thought of as taking a sum over triangles $v_i \to v_k \to v_{c(i, k)}$.

Note that we're being careful about which pieces have a dependence on $\gamma$, as we can reuse certain computations if we want to compute distances from multiple sources. Abusing notation, we can get the distance matrix (that is, get rid of the $\gamma$ dependence) from

$$\phi_{i, j} = \left(\phi^{\{v_j\}}\right)_i.$$

# Gradient Computation

Note that $c(i, c(j, i)) = j$. This is helpful for reindexing some sums (in particular, the one for $\nabla \cdot X$).

We then have the following partial derivatives:

$$\begin{aligned}
    \frac{\partial u^\gamma}{\partial \rho_\ell} &= -(D - tL_C)^{-1}\left(\frac{\partial D}{\partial \rho_\ell} - t\frac{\partial L_C}{\partial \rho_\ell}\right)u^\gamma, \\
    \frac{\partial q_{i, j}}{\partial \rho_\ell} &= \begin{cases}
        \displaystyle\frac{\partial u^\gamma_i}{\rho_\ell}(v_{c(i, j)} - v_j) - u^\gamma_i\frac{\partial v_\ell}{\rho_\ell} & \text{if $\ell = j$}, \\
        \displaystyle\frac{\partial u^\gamma_i}{\rho_\ell}(v_{c(i, j)} - v_j) + u^\gamma_i\frac{\partial v_\ell}{\partial \rho_\ell} & \text{if $\ell = c(i, j)$}, \\
        \displaystyle\frac{\partial u^\gamma_i}{\rho_\ell}(v_{c(i, j)} - v_j) & \text{otherwise},
    \end{cases} \\
    \frac{\partial m_{i, j}}{\partial \rho_\ell} &= \frac{\partial q_{i, j}}{\partial \rho_\ell} + \frac{\partial q_{j, c(i, j)}}{\partial \rho_\ell} + \frac{\partial q_{c(i, j), i}}{\partial \rho_\ell}, \\
    \frac{\partial (\widetilde{\nabla} u^\gamma)_{i, j}}{\partial \rho_\ell} &= \frac{\partial N_{i, j}}{\partial \rho_\ell} \times m_{i, j} + N_{i, j} \times \frac{\partial m_{i, j}}{\partial \rho_\ell}, \\
    \frac{\partial X^\gamma_{i, j}}{\partial \rho_\ell} &= -\frac{1}{\|(\widetilde{\nabla} u^\gamma)_{i, j}\|_2}(I - X^\gamma_{i, j}(X^\gamma_{i, j})^\intercal)\frac{\partial (\widetilde{\nabla} u^\gamma)_{i, j}}{\partial \rho_\ell}, \\
    \frac{\partial p_{i, j}}{\partial \rho} &= \begin{cases}
        \displaystyle\left(\frac{\partial}{\partial \rho_\ell}\cot(\theta_{i, j})\right)(v_j - v_i) - \cot(\theta_{i, j})\frac{\partial v_\ell}{\rho_\ell} & \text{if $\ell = i$}, \\
        \displaystyle\left(\frac{\partial}{\partial \rho_\ell}\cot(\theta_{i, j})\right)(v_j - v_i) + \cot(\theta_{i, j})\frac{\partial v_\ell}{\rho_\ell} & \text{if $\ell = j$}, \\
        \displaystyle\left(\frac{\partial}{\partial \rho_\ell}\cot(\theta_{i, j})\right)(v_j - v_i) & \text{if $\ell = c(i, j)$}, \\
        0 & \text{otherwise},
    \end{cases} \\
    \frac{\partial (\nabla \cdot X^\gamma)_i}{\partial \rho_\ell} &= \frac{1}{2}\sum_{\substack{k \\ \text{$(v_i, v_k)$ is an edge}}}\left(\left(\frac{\partial p_{i, k}}{\partial \rho_\ell} - \frac{\partial p_{c(i, k), i}}{\partial \rho_\ell}\right) \cdot X^\gamma_{i, k} + (p_{i, k} - p_{c(i, k), i}) \cdot \frac{\partial X^\gamma_{i, k}}{\partial \rho_\ell}\right) \\
    \frac{\partial \phi^\gamma}{\partial \rho_\ell} &= L_C^+\left(\frac{\partial (\nabla \cdot X^\gamma)}{\partial \rho_\ell} - \frac{\partial L_C}{\partial \rho_\ell}\phi^\gamma\right).
\end{aligned}$$
