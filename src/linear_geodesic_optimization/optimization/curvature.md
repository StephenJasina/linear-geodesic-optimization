Consider $u$, $v$, and $r$ all on the unit sphere. Assume that $u$ and $v$ are not antipodal. We can find the point $\pi(r)$ nearest to $r$ on the great circle passing through $u$ and $v$ by projecting $r$ onto the plane spanned by $u$ and $v$ and then normalizing the result. The strategy for this is to just use Graham-Schmidt to get an orthonormal basis $\{v, w\}$ of the plane. Once we have $\pi(r)$, we can find the distance from $r$ to the great circle by taking advantage of the dot product.

$$\begin{aligned}
    \|u - (u \cdot v)v\|_2^2 &= \|u\|_2^2 + (u \cdot v)^2\|v\|_2^2 - 2(u \cdot v)^2 \\
        &= 1 + (u \cdot v)^2 - 2(u \cdot v)^2 \\
        &= 1 - (u \cdot v)^2, \\
    w &\triangleq \frac{u - (u \cdot v)v}{\|u - (u \cdot v)v\|_2}, \\
    \pi(r) &= \frac{(r \cdot v)v + (r \cdot w)w}{\|(r \cdot v)v + (r \cdot w)w\|_2}, \\
    c &\triangleq \|(r \cdot v)v + (r \cdot w)w\|_2 \\
        &= \sqrt{(r \cdot v)^2 + (r \cdot w)^2} \\
        &= \sqrt{(r \cdot v)^2 + \left(r \cdot \frac{u - (u \cdot v)v}{\|u - (u \cdot v)v\|_2}\right)^2} \\
        &= \sqrt{(r \cdot v)^2 + \frac{(r \cdot u - (r \cdot v)(u \cdot v))^2}{1 - (u \cdot v)^2}} \\
        &= \sqrt{\frac{(r \cdot v)^2 - (r \cdot v)^2(u \cdot v)^2}{1 - (u \cdot v)^2} + \frac{(r \cdot u)^2 + (r \cdot v)^2(u \cdot v)^2 - 2(r \cdot u)(r \cdot v)(u \cdot v)}{1 - (u \cdot v)^2}} \\
        &= \sqrt{\frac{(r \cdot u)^2 + (r \cdot v)^2 - 2(r \cdot u)(r \cdot v)(u \cdot v)}{1 - (u \cdot v)^2}} \\
     \cos(\theta) &= r \cdot \pi(r) \\
        &= \frac{(r \cdot v)^2 + (r \cdot w)^2}{\|(r \cdot v)v + (r \cdot w)w\|_2} \\
        &= \|(r \cdot v)v + (r \cdot w)w\|_2 & \text{(by orthonormality)} \\
        &= c \\
    \theta &= \arccos(c).
\end{aligned}$$

Our real question is whether $r$ is "close" (within distance $\epsilon$) to the shortest path from $u$ to $v$ on the unit sphere. There are two cases to consider. The first is that $r$ is very close to $u$ or $v$. This can be determined by checking $\arccos(r \cdot u) < \epsilon$ or $\arccos(r \cdot v) < \epsilon$ (if either of these is the case, then $r$ is close).

The second case is that $r$ is close to some point that isn't one of the endpoints (this has some overlap with the previous case, but the previous case is easier to check, so we check it first). The trick here is to use the long computation seen above. $r$ is close to the great circle passing through $u$ and $v$ when

$$\arccos(c) < \epsilon.$$

Being a bit more refined, we actually want the angles between $\pi(r)$ and each of $u$ and $v$ to be at most the angle between $u$ and $v$. In other words,

$$\begin{aligned}
    \max(\arccos(\pi(r) \cdot u), \arccos(\pi(r) \cdot v)) &\le \arccos(u \cdot v) \\
    \min(\pi(r) \cdot u, \pi(r) \cdot v) &\ge u \cdot v \\
\end{aligned}$$

For the left hand side, we can compute
$$\begin{aligned}
    \pi(r) \cdot v &= \frac{(r \cdot v)v + (r \cdot w)w}{\|(r \cdot v)v + (r \cdot w)w\|_2} \cdot v \\
        &= \frac{r \cdot v}{\|(r \cdot v)v + (r \cdot w)w\|_2} \\
        &= \frac{r \cdot v}{c}. \\
    \pi(r) \cdot u &= \frac{r \cdot u}{c}. & \text{(by symmetry)}
\end{aligned}$$

Putting this together, $r$ is close if one of the following is true:
* $r \cdot u > \cos(\epsilon)$;
* $r \cdot v > \cos(\epsilon)$;
* $c > \cos(\epsilon)$ and $\min(r \cdot u, r \cdot v) \ge c\,(u \cdot v)$.
