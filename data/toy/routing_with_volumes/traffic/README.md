Toy examples where traffic volumes change.
* `two_clusters`: The traffic between two clusters increases. This effectively significantly increases the traffic along one particularly critical link. In terms of numbers, the traffic starts as a uniform distribution, and the traffic between the two clusters triples across the time span.
* `two_clusters_extreme`: This is identical to the `two_clusters` example, but the traffic sextuples instead.
* `die_out`: All traffic along a link dies out. This is a naive example of network behavior (compare to `../reroute/link_outage`)
* `two_clusters_die_out`: All traffic between two clusters (the same as in `two_clusters`) goes to $0$.
