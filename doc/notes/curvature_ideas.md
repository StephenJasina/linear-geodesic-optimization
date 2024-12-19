We want there to be a correspondence between visual effects due to curvature and events in the network. Some simple events could be:
* A link goes down. In this case, its throughput goes to 0, so we should expect the curvature there to increase, potentially resulting in a hill or valley during the optimization.
* A particular route suddenly sees heavy traffic. We expect this to lead to a saddle along that path. If there was already a saddle, then the saddle should deepen.

Are these the case with our current curvature model?

Consider when we have two clusters of sizes $n_1$ and $n_2$ connected by a single edge. Let say the weight on the single edge is $a$, and the weights on the edges in the clusters are $b_1$ and $b_2$. The curvature of the middle edge is always going to be $-2$. Let's look at the edges in cluster $1$. There are two cases:
* Edges that are not incident to the central edge. It turns out that these are too far away from the central edge, so that their curvature is the same as it is in $K_{n_1}$.
* Edges that are incident to the central edge. Some experimentation shows that these edges curvatures decrease as $a$ increases. Similarly, they increase as $b_1$ increases. They are bounded above by the curvatures of the edges not incident to the central edge. They are bounded below by some other (negative) value that is difficult to compute by hand...

Interesting observations:
* Curvature is still bounded below by -2 and above by 1. Negative curvatures are much more prevalent than before (requiring a cluster of size greater than $4$ to become positive).
* The dependence on $\alpha$ in the limiting process seems to be practically absent. In fact, in a random graph with random weights, less than five percent of edges have their curvature changed by more than $0.04$. I'm not sure whether this is due to floating point error, or if it's due to true changes. This seems like it warrants investigation, but it is certainly not a priority.
* Although adding an edge with no throughput doesn't change the weighting scheme, it does change the graph distances. This causes a change in curvature.
