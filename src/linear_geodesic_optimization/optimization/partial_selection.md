Computing the gradient of $\mathcal{L}_{\text{geodesic}, v}(M)$ is quite expensive. We can make the computation more efficient by approximately reconstructing the geodesics from $v$. Importantly, for vertices not on the faces through which the geodesics pass, the partial derivatives will be $0$.

# Fixed Point Iteration Strategy

Consider the set of geodesics between $v_a$ and $v_b$ for $b \in B$, where the geodesic distance is computed from $v_a$ (as in a single source problem). A conservative estimate of these faces incident to these geodesics can be constructed iteratively by finding a minimal fixed point $S_{a, B}$ of $$S \mapsto \left\{(v_i \to v_j \to v_{c(i, j)}) : \begin{array}{c}\text{$(v_i \to v_j \to v_{c(i, j)})$ is adjacent to a face $(v_j \to v_i \to v_{c(j, i)}) \in S$} \\ \text{and $\phi_{a, c(i, j)} < \max(\phi_{a, i}, \phi_{a, j})$}\end{array}\right\}$$ such that $$\{(v_b \to v_i \to v_{c(b, i)}) : \phi_{a, b} > \min(\phi_{a, i}, \phi_{a, c(b, i)}), b \in B\} \subseteq S_{a, B}.$$ In other words, $S_{a, B}$ is (at least) the set of faces through which paths that always move towards $v_a$ can pass through.

With this computed, the vertices we care about are those incident to the faces in $S_{a, B}$.

This strategy is implemented in `approximate_geodesics_fpi`.

# Triangle Inequality Strategy

Suppose we have the same setup as before, where we want to find the faces incident to the geodesic between $v_a$ and $v_b$. Let's focus in on a face $f = (v_i \to v_j \to v_{c(i, j)})$. Note that geodesic distance satisfies the triangle inequality. Define

$$\begin{aligned}
    \widetilde{a} \in \argmin_{\ell \in \{i, j, c(i, j)\}}\left(\phi_{a, \ell}\right), \\
    \widetilde{b} \in \argmin_{\ell \in \{i, j, c(i, j)\}}\left(\phi_{b, \ell}\right).
\end{aligned}$$

From the triangle inequality, we have $$\phi_{a, b} \le \phi_{a, \widetilde{a}} + \|v_{\widetilde{a}} - v_{\widetilde{b}}\|_2 + \phi_{b, \widetilde{b}}.$$ Intuitively, this inequality is saying that $f$ cannot be too far away from $v_a$ and $v_b$.

Note that any $f$ that satisfies the above also automatically satisfies the conditions of the fixed point strategy (the "monotonic" path is the concatenation of the geodesic from $v_b$ to $v_{\widetilde{b}}$, the edge from $v_{\widetilde{b}}$ to $v_{\widetilde{a}}$ (or no edge if $\widetilde{b} = \widetilde{a}$), and the geodesic from $v_{\widetilde{a}}$ to $v_a$). Therefore, this strategy is a refinement of the previous one. Unfortunately, (using the notation from `geodesic.md`) the previous strategy required thinking only of $\phi^{\{v_a\}}$, whereas this strategy requires considering additionally $\phi^{\{v_b\}}$ for each $b$ such that $(a, b) \in E_G$.
