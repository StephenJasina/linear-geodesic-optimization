# Computation

It would be nice to have something like [SI-MVS](https://www2.eecs.berkeley.edu/Pubs/TechRpts/1993/CSD-93-732.pdf#page=142) (see [this](https://www2.eecs.berkeley.edu/Pubs/TechRpts/1992/CSD-92-664.pdf) too). [Willmore Energy](https://en.wikipedia.org/wiki/Willmore_energy) is another possibility.

[This](https://www.cad-journal.net/files/vol_4/CAD_4(5)_2007_607-617.pdf) gives a good overview.

For now, we use $$\mathcal{L}_{\text{smooth}}(M) \triangleq -\rho^\intercal L_C\rho.$$

# Gradient Computation

We compute the derivatives

$$\begin{aligned}
    \frac{\partial}{\partial \rho_\ell}\mathcal{L}_{\text{smooth}}(M) &= -e_\ell^\intercal L_C\rho - \rho^\intercal\frac{\partial L_C}{\partial \rho_\ell}\rho - \rho^\intercal L_Ce_\ell.
\end{aligned}$$
