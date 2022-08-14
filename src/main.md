# Problem Setup

As input, we are given a graph $G = (V_G, E_G)$, where each vertex is a geographic position $s_i \in S^2$, and each edge $(i, j)$ has an associated (Olivier-Ricci) curvature $\kappa_{i, j} \in (-2, 1)$ and an associated latency $t_{i, j} \in \mathbb{R}_{\ge 0}$.

Intuitively, we want to return a surface in $\mathbb{R}^3$ that is the graph of a function $f : S^2 \to \mathbb{R}_{> 0}$ whose geodesics $g_{i, j}$ between $s_i$ and $s_j$ (and their missing $\rho$-coordinates) have length $\phi_{i, j}$ that is in a linear relationship with the latency.

The strategy to realize this intuition is to create a mesh $M = (V_M, E_M)$ supported on a subset of $S^2$ that contains our input positions. Let $P$ be the support. Then for each $s_i \in P$, we want to assign a $\rho_i \in \mathbb{R}_{> 0}$, which in turn gives a point $v_i = (s_i, \rho_i) \in V$. This setup is made explicit in `mesh/sphere.py`.

# Objective/Loss Functions

To enforce that the mesh approximates our desired surface, we define the objective functions

$$\begin{aligned}
    \mathcal{L}_{\mathrm{geodesic}}(M) &\triangleq \sum_{e \in E_G} (\text{least squares residual of edge \(e\)})^2, \\
    \mathcal{L}_{\mathrm{smooth}}(M) &\triangleq -\rho^\intercal L_C\rho, \\
    \mathcal{L}(M) &\triangleq \mathcal{L}_{\mathrm{geodesic}}(M) + \lambda\mathcal{L}_{\mathrm{smooth}}(M),
\end{aligned}$$

where $L_C$ is the Laplacian of the mesh scaled by vertex area, and $\lambda$ is a tunable hyperparameter. Our goal is then to minimize $\mathcal{L}(M)$.

Note that the loss functions (particularly the geodesic and total ones) also have a dependence on the measured latencies. We omit that as a written parameter because they are treated as fixed (we are really optimizing over the manifold, not over the measured latencies).

For additional details on these loss functions, see the notebooks in `optimization`.

# Putting It All Together with Minibatch Gradient Descent

We can rewrite

$$\mathcal{L}_{\text{geodesic}}(M) = \sum_{v \in V_G} \sum_{\substack{v' \in V_G \\ (v, v') \in E_G}} \frac{1}{2}(\text{least squares residual of edge \((v, v')\)})^2.$$

From here, we can take the standard approach of batching the gradient computation by vertices. This idea fits well with the heat method, since the heat method necessarily computes all geodesic distances (and their gradients) from a single source (represented by $v$ in the above decomposition).

Motivated by this, define

$$\mathcal{L}_{\text{geodesic}, v}(M) = \sum_{\substack{v' \in V_G \\ (v, v') \in E_G}} \frac{1}{2}(\text{least squares residual of edge \((v, v')\)})^2.$$

TODO:
* Try nonuniform meshes (e.g., more detail in America, less in the oceans)
* Is it possible to first optimize over geodesics, and then optimize over smoothness? (My gut instinct is no, but maybe it could be possible under certain circumstances)
* Other smoothness functions
* Same with $\lambda$
* Later step is to plot the graph representing the network on the sphere
* Come up with some example graphs and try to run them through the code. Use them as "canonical" proof of concepts
* Change scale of scatter plot axes to be distance vs distance
* Make canonical datasets
* Maybe use data from the topology zoo
* Run multiple combinations on different servers
* Check out [Cloud Lab](https://cloudlab.us/)

General guidelines
* In general, think $|G| \approx 500$
* We really don't care as much about runtime/scalability compared to getting high quality results with low losses (it's okay if a run takes time on the order of days)
* There is access to powerful servers if I need it