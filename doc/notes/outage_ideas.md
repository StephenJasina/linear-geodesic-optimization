> Milestone: Decide how to display link outages in a very visible way.

# Rely on postprocessing
We can just rely on our postprocessing to set the height of a link that goes down to be low. The idea is that once a link goes down, we will see a major change in the manifold automatically.

For advantages, this is already implemented, and it is a very simple idea.

For disadvantages, this method suffers severely from possible overlap of links. It also might not draw the user's attention as much as other changes to the manifold's shape.

# Show a message in a separate context (e.g., a popup or a message box to the side)
This idea here is to draw the user's attention by not using the manifold at all.

For advantages, this draws the user's attention necessarily. It also gives very detailed, unambiguous information.

For disadvantages, this adds complexity to the design, and it somewhat violates our layered design principle.

# Have a special layer for edges representing link outages
This idea is the most promising and most customizable. There are several options:

## Color the edge on the manifold
For advantages, this is easy to implement (practically done already) and can be made extremely visible if colors are chosen carefully.

For disadvantages, this might suffer from occlusion issues (fixable by drawing edges representing outages on top), and it might not be immediately obvious (addressable by drawing the line in a different style and/or wiht a heavier stroke)

## Draw an edge as an arc over the manifold
For advantages, again this is already basically implemented, and it can be made visible if we're careful about colors and width.

For disadvantages, it doesn't really make sense (at least to me) to draw an arc high up if there is no throughput, but this would be necessary to do if we want the arc to be visible.

## Color an area on the manifold
For advantages, this would make it obvious roughly where there is an issue (generally, edges are not so dense as to cover an entire area).

For disadvantages, avoiding occlusion is not 100% foolproof. We could address this by coloring the area in the foreground with a certain transparency effect, but this idea has its own issues (it changes the colors of the other edges, etc.) It would also be hard to isolate exactly where there is an outage.

# Conclusion
To me, it looks like the best option is to use a combination of postprocessing, coloring the an edge on the manifold, and a supplementary message box to the side. This will draw the user's attention in multiple ways. Furthermore, it will allow the user to both get a quick heuristic view (by the colored edge) and precise understanding (by the message box) of outages. For implementation, the major challenge will be to select better colors so that the color for outages will stand out. Red would be an ideal color for this, but this is currently overloaded by both negative curvature and high throughput.
