# Ideas for how we want to visualize features

## Link Outage
In terms of shape, it would be nice to have anywhere without a link be flat (more intuitive from a simplicity perspective) or have positive curvature (traffic should not flow through these areas). For correctness, the latter option seems to be more reasonable. This can be enforced with the additive smoothing idea I've mentioned before (add a term to the optimization that just matches each mesh point to a positive curvature).

We can also rely on postprocessing to cut out an area around the link, especially if a link is in a geographically sparse area.

Due to overlapping links, showing an outage with curvature or postprocessing alone might not be possible, so I think we need to supplement it with color. Ensuring the link is drawn "on top" in a prominent color is reasonable. Importantly, we might need to do some sort of postprocessing here where we highlight a link if its throughput falls below a certain threshold rather than going all the way to 0.

If we're drawing (approximate) geodesics, it should hopefully be obvious due to a significant path change.

## Heavy Traffic on a Link
In line with our curvature viewpoint, a link with relatively heavy throughput should have a correspondingly deep saddle. One thing we're not optimizing currently is saddle _shape_: some saddles can be very symmetric while others can be narrow (but deep). From a visualization perspective, it would be nice if our saddles either all were symmetric or all had the same "width", but this is very challenging.

We can augment the manifold shape with color, but this seems very challenging to do nicely. One of the reasons we aren't relying solely on the graph approach is the overlapping edges.

## Node Outage
This should be extremely rare.

The easiest way to display this would be to just use the link outage ideas for each of the incident links. This could be supplemented by cutting out a small area around the node with postprocessing. (This is slightly different from our current convex hull/fat edge postprocessing strategy.)

## High Traffic from a Node
We could just display suddenly heavy traffic from a node by using the ideas from the corresponding section about links.

In terms of visualization, it would be nice to emphasize nodes with heavy traffic, so maybe we could ensure that they have a higher position. (This is actually pretty reasonable to implement in the optimization process!) This is a converse idea to our convex hull postprocessing.

## Overall Traffic Change Throughout the Day
An intuitive way to display this is to change the manifold's vertical scaling (perhaps a special case of high traffic through a node). If we do this in postprocessing, this has the issue of altering curvatures and geodesics. Because this has a more global effect and can be represented essentially as a single time series, we could just display an additional graph to the side of the manifold.

## Major Routing Changes
We hopefully shouldn't need to do anything special here, since our optimization should handle it.