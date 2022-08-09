# Computation

Recall the one-dimensional least squares setup. Here, we wish to relate $t_e$ to $d_e$ by $t_e \approx \beta_0 + \beta_1d_e$, where $d$ is some function of $\phi$. The goal is to minimize $$\sum_{e \in E_G}(t_e - \beta_0 - \beta_1d_e)^2$$ with respect to $\beta_0$ and $\beta_1$. Differentiating with respect to $\beta_0$ and $\beta_1$ gives the following relations: $$\begin{aligned}
    0 &= -2\sum_{e \in E_G}(t_e - \beta_0 - \beta_1d_e), \\
    0 &= -2\sum_{e \in E_G}d_e(t_e - \beta_0 - \beta_1d_e).
\end{aligned}$$

Rearranging, $$\begin{aligned}
    \beta_1\sum_{e \in E_G}d_e &= \sum_{e \in E_G}t_e - \beta_0|E_G|, \\
    \beta_1\sum_{e \in E_G}d_e^2 &= \sum_{e \in E_G}d_et_e - \beta_0\sum_{e \in E_G}d_e.
\end{aligned}$$ Scaling and equating, we have $$\begin{aligned}
    \left(\sum_{e \in E_G}d_e^2\right)\left(\sum_{e \in E_G}t_e\right) - \beta_0|E_G|\left(\sum_{e \in E_G}d_e^2\right) &= \left(\sum_{e \in E_G}d_e\right)\left(\sum_{e \in E_G}d_et_e\right) - \beta_0\left(\sum_{e \in E_G}d_e\right)^2 \\
    \beta_0 &= \frac{\left(\sum_{e \in E_G}d_e^2\right)\left(\sum_{e \in E_G}t_e\right) - \left(\sum_{e \in E_G}d_e\right)\left(\sum_{e \in E_G}d_et_e\right)}{|E_G|\left(\sum_{e \in E_G}d_e^2\right) - \left(\sum_{e \in E_G}d_e\right)^2}.
\end{aligned}$$

We similarly find $$\begin{aligned}
    \beta_0|E_G| &= \sum_{e \in E_G}t_e - \beta_1\sum_{e \in E_G}d_e, \\
    \beta_0\sum_{e \in E_G}d_e &= \sum_{e \in E_G}d_et_e - \beta_1\sum_{e \in E_G}d_e^2,
\end{aligned}$$ so $$\begin{aligned}
    \left(\sum_{e \in E_G}d_e\right)\left(\sum_{e \in E_G}t_e\right) - \beta_1\left(\sum_{e \in E_G}d_e\right)^2 &= |E_G|\sum_{e \in E_G}d_et_e - \beta_1|E_G|\sum_{e \in E_G}d_e^2 \\
    \beta_1 &= \frac{|E_G|\sum_{e \in E_G}d_et_e - \left(\sum_{e \in E_G}d_e\right)\left(\sum_{e \in E_G}t_e\right)}{|E_G|\left(\sum_{e \in E_G}d_e^2\right) - \left(\sum_{e \in E_G}d_e\right)^2}.
\end{aligned}$$

These expressions are quite bad, but we can simplify them somewhat if we make the following assumptions: $$\begin{aligned}
    1 &= \frac{1}{|E_G|}\sum_{e \in E_G}d_e^2 , \\
    0 &= \sum_{e \in E_G}d_e.
\end{aligned}$$ Then $$\begin{aligned}
    \beta_0 &= \frac{1}{|E_G|}\sum_{e \in E_G}t_e, \\
    \beta_1 &= \frac{1}{|E_G|}\sum_{e \in E_G}d_et_e.
\end{aligned}$$ The assumptions hold if we have $$\begin{aligned}
    \widetilde{d}_e &\triangleq \phi_e - \frac{1}{|E_G|}\sum_{e' \in E_G}\phi_{e'}, \\
    d_e &\triangleq \frac{\widetilde{d}_e}{\sqrt{\frac{1}{|E_G|}\sum_{e' \in E_G}\widetilde{d}_{e'}^2}}.
\end{aligned}$$ Importantly, these are just affine transformations.

Then $$\mathcal{L}_{\mathrm{geodesic}}(M) \triangleq \sum_{e \in E_G}\left(t_e - \frac{1}{|E_G|}\sum_{e' \in E_G}t_{e'} - \frac{d_e}{|E_G|}\sum_{e' \in E_G}d_{e'}t_{e'}\right)^2$$

# Gradient Computation

From the above, we get, $$\begin{aligned}
    \frac{\partial}{\partial \rho_\ell}\mathcal{L}_{\mathrm{geodesic}}(M) &= -\frac{2}{|E_G|}\sum_{e \in E_G}\left(t_e - \frac{1}{|E_G|}\sum_{e' \in E_G}t_{e'} - \frac{d_e}{|E_G|}\sum_{e' \in E_G}d_{e'}t_{e'}\right) \\
        &\hspace{8em}\cdot \left(\frac{\partial d_e}{\partial \rho_\ell}\sum_{e' \in E_G}d_{e'}t_{e'} + d_e\sum_{e' \in E_G}\frac{\partial d_{e'}}{\partial \rho_\ell}t_{e'}\right), \\
    \frac{\partial d_e}{\partial \rho_\ell} &= \frac{\sqrt{\sum_{e' \in E_G}\widetilde{d}_{e'}^2} \cdot \frac{\partial \widetilde{d}_e}{\partial \rho_\ell} - d_e\sum_{e' \in E_G}\left(\widetilde{d}_{e'} \cdot \frac{\partial \widetilde{d}_{e'}}{\partial \rho_\ell}\right)}{\sum_{e' \in E_G}\widetilde{d}_{e'}^2}, \\
        &= \frac{1}{\sqrt{\sum_{e' \in E_G}\widetilde{d}_{e'}^2}}\left(\frac{\partial \widetilde{d}_e}{\partial \rho_\ell} - d_e\sum_{e' \in E_G}\left(d_{e'} \cdot \frac{\partial \widetilde{d}_{e'}}{\partial \rho_\ell}\right)\right) \\
    \frac{\partial \widetilde{d}_e}{\partial \rho_\ell} &= \frac{\partial \phi_e}{\partial \rho_\ell} - \frac{1}{|E_G|}\sum_{e' \in E_G}\frac{\partial \phi_{e'}}{\partial \rho_\ell}.
\end{aligned}$$
