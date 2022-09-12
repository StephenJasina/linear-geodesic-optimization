Consider $u$, $v$, and $r$ all on the unit sphere. Assume that $u$ and $v$ are not antipodal. We can find the point $\pi(r)$ nearest to $r$ on the great circle passing through $u$ and $v$ by projecting $r$ onto the plane spanned by $u$ and $v$ and then normalizing the result. The strategy for this is to just use Graham-Schmidt to get an orthonormal basis $\{v, w\}$ of the plane. Once we have $\pi(r)$, we can find the distance from $r$ to the great circle by taking advantage of the dot product.

$$\begin{aligned}
    \|u - (u \cdot v)v\|_2^2 &= \|u\|_2^2 + (u \cdot v)^2\|v\|_2^2 - 2(u \cdot v)^2 \\
        &= 1 + (u \cdot v)^2 - 2(u \cdot v)^2 \\
        &= 1 - (u \cdot v)^2, \\
    w &\triangleq \frac{u - (u \cdot v)v}{\|u - (u \cdot v)v\|_2}, \\
    \pi(r) &= \frac{(r \cdot v)v + (r \cdot w)w}{\|(r \cdot v)v + (r \cdot w)w\|_2}, \\
    \cos(\theta) &= r \cdot \pi(r) \\
        &= \frac{(r \cdot v)^2 + (r \cdot w)^2}{\|(r \cdot v)v + (r \cdot w)w\|_2} \\
        &= \sqrt{(r \cdot v)^2 + (r \cdot w)^2} \\
        &= \sqrt{(r \cdot v)^2 + \left(r \cdot \frac{u - (u \cdot v)v}{\|u - (u \cdot v)v\|_2}\right)^2} \\
        &= \sqrt{(r \cdot v)^2 + \frac{((r \cdot u) - (u \cdot v)(r \cdot v))^2}{1 - (u \cdot v)^2}} \\
        &= \sqrt{\frac{(r \cdot v)^2(1 - (u \cdot v)^2) + ((r \cdot u) - (u \cdot v)(r \cdot v))^2}{1 - (u \cdot v)^2}} \\
        &= \sqrt{\frac{(r \cdot u)^2 + (r \cdot v)^2 - 2(r \cdot u)(r \cdot v)(u \cdot v)}{1 - (u \cdot v)^2}}, \\
    \theta &= \arccos\left(\sqrt{\frac{(r \cdot u)^2 + (r \cdot v)^2 - 2(r \cdot u)(r \cdot v)(u \cdot v)}{1 - (u \cdot v)^2}}\right).
\end{aligned}$$

Our real question is whether $r$ is "close" (within distance $\epsilon$) to the shortest path from $u$ to $v$ on the unit sphere. There are two cases to consider. The first is that $r$ is very close to $u$ or $v$. This can be determined by checking $\arccos(r \cdot u) < \epsilon$ or $\arccos(r \cdot v) < \epsilon$ (if either of these is the case, then $r$ is close).

The second case is that $r$ is close to some point that isn't one of the endpoints (this has some overlap with the previous case, but the previous case is easier to check, so we check it first). The trick here is to use the long computation seen above. $r$ is close to the great circle passing through $u$ and $v$ when

$$\arccos\left(\sqrt{\frac{(r \cdot u)^2 + (r \cdot v)^2 - 2(r \cdot u)(r \cdot v)(u \cdot v)}{1 - (u \cdot v)^2}}\right) < \epsilon.$$

The midpoint of the shortest path connecting $u$ to $v$ is $\frac{u + v}{\|u + v\|_2}$. $\pi(r)$ lies on this shortest path when it and the midpoint point in the approximate same direction (that is, they have non-negative dot product). In other words, we want

$$\begin{aligned}
    0 &\le ((r \cdot v)v + (r \cdot w)w) \cdot (u + v) \\
        &= (r \cdot v)(u \cdot v) + (r \cdot v)(v \cdot v) + (r \cdot w)(u \cdot w) + (r \cdot w)(v \cdot w) \\
        &= (r \cdot v)(u \cdot v) + (r \cdot v) + (r \cdot w)(u \cdot w) \\
        &= (r \cdot v)(u \cdot v) + (r \cdot v) + \left(r \cdot \frac{u - (u \cdot v)v}{\|u - (u \cdot v)v\|_2}\right)\left(u \cdot \frac{u - (u \cdot v)v}{\|u - (u \cdot v)v\|_2}\right) \\
        &= (r \cdot v)(u \cdot v) + (r \cdot v) + \frac{((r \cdot u) - (u \cdot v)(r \cdot v))((u \cdot u) - (u \cdot v)^2)}{1 - (u \cdot v)^2} \\
        &= (r \cdot v)(u \cdot v) + (r \cdot v) + \frac{((r \cdot u) - (u \cdot v)(r \cdot v))(1 - (u \cdot v)^2)}{1 - (u \cdot v)^2} \\
        &= (r \cdot v)(u \cdot v) + (r \cdot v) + (r \cdot u) - (u \cdot v)(r \cdot v) \\
        &= (r \cdot v) + (r \cdot u) \\
        &= r \cdot (u + v).
\end{aligned}$$

Putting this together, $r$ is close if one of the following is true:
* $r \cdot u < \cos(\epsilon)$;
* $r \cdot v < \cos(\epsilon)$;
* $\sqrt{\frac{(r \cdot u)^2 + (r \cdot v)^2 - 2(r \cdot u)(r \cdot v)(u \cdot v)}{1 - (u \cdot v)^2}} < \cos(\epsilon)$ and $0 \le r \cdot (u + v)$.
