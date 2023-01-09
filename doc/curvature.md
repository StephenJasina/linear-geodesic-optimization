# Computation
Recall that the discretized Gaussian curvature $\kappa(v_i)$ at a vertex $v_i$ is given by $2\pi$ minus the sum of the angles at that vertex. In other words,

$$\kappa(v_i) = 2\pi - \sum_{\substack{k \\ \text{$(v_i, v_k)$ is an edge}}} \theta_{k, c(i, k)}.$$

From the computation of the Laplacian, we already have $\cot(\theta)$ for any $\theta$ we want; all that remains is to take the arccotangent. To ensure that it remains in the correct range, we modulo by $\pi$:

$$\theta = \arctan\left(\frac{1}{\cot(\theta)}\right) \bmod \pi.$$

# Gradient Computation
Again, from the Laplacian computations, we have the gradient of the cotangents of the angles. So, we can compute

$$\begin{aligned}
    \frac{\partial \theta}{\partial \rho} &= \frac{\partial \cot(\theta)}{\partial \rho} \cdot \left(\frac{\partial \cot(\theta)}{\partial \theta}\right)^{-1} \\
        &= -\frac{\partial \cot(\theta)}{\partial \rho} \cdot \sin^2(\theta) \\
        &= -\frac{\partial \cot(\theta)}{\partial \rho} \cdot \frac{1}{1 + \cot^2(\theta)}.
\end{aligned}$$
