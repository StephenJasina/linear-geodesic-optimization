# Linear Geodesic Optimization
Tool for constructing and viewing a manifold whose geodesic distances are approximately linearly related to some input data. Some documentation can be found in the `doc/` directory.

## Requirements
### System Packages
As most of the optimization code is written in [Python](https://www.python.org/), a working installation (at least version 3.10) must be installed. For viewing the outputs, a modern browser is needed, along with [npm](https://www.npmjs.com/).

### Python Packages
Most of the packages listed below can be installed via `pip install <package name>`. For those that cannot, additional instructions are included.

You'll need the following to get the optimization routine running:
* `networkx`
* `numpy`
* `POT`
* `potpourri3d`
* `pygeodesic` for exact route preferences and the route-comparison example
* `python-dcel-mesh`. To install this, clone [this repo](https://github.com/StephenJasina/python-dcel-mesh) and run `pip install .` from its root directory.
* `scikit-learn`
* `scipy`

Additionally helpful packages for viewing the data are
* `adjustText`
* `basemap`. For this one, make sure Python is version at most 3.12

### Node Packages
These dependencies are controlled by the file `src/site/package.json`. To install them, simply run `npm install` from the `src/site` directory.

## Usage
From the `src` directory, run `python optimization.py config_file.json` followed
by `python collation.py config_file.json`. Configurations live under `data/`;
for example, `../data/toy/routing_with_volumes/config/traffic_two_clusters.json`
when running from `src`.

A config's `arguments_optimization` describe the optimization runs. Its
optional `arguments_collation` lists postprocessing variants (for example,
`"postprocess": [true, false]`). Collation reads each batch's optimizer outputs
once and writes one animation file per variant, named by
`settings.output_format_collation`.

The optional `method` argument selects how the surface is built. The default,
`construction`, turns each edge's Ricci curvature into a target Hessian and
solves for the heights with a few sparse linear solves, optionally followed by
`hessian_polish_maxiter` iterations of L-BFGS on the usual loss. The `lbfgs`
method minimizes the loss with L-BFGS over every mesh height, starting from a
sphere cap (or from a previous run, depending on `settings.initialization`).
Configs that run L-BFGS must set `"method": ["lbfgs"]`.

The optional `backend` argument selects the loss implementation, which is used
by L-BFGS, by the construction's polish, and for the construction's loss report.
The default, `torch`, uses array operations and automatic differentiation and
requires PyTorch (`pip install torch`). The `legacy` backend uses the original
Python derivative code. Both backends use the same objective and output format.
The torch backend uses one CPU thread per optimization process to avoid thread
oversubscription.

To run the webapp, run `npx vite` from the `src/site` directory. This will start the server locally and display a link to the page in the console. The animation can be viewed by dragging and dropping the JSON file produced by the collation process.

## Experimental route-conditioned manifolds

The route-comparison experiment fits four surfaces for the same C → J pair.
It first optimizes the original C → F → G → J routing. Each of three changed
traffic routes then starts from **that same original surface** and runs a
separate L-BFGS refit. The network topology, mesh, and other traffic stay fixed.
The browser page shows the four resulting manifolds and computes an unrestricted
geodesic on each one; the path is not drawn to match the route by the viewer.

For a route constraint, the optional loss compares the shortest surface path
between its endpoints with a path restricted to a corridor around its requested
polyline. Let those lengths be `L_free` and `L_corridor`, and let `D` be the
endpoints' planar distance. The penalty is
`weight × max(0, route_margin + (L_corridor − L_free) / D)²`, averaged over
constraints and multiplied by `lambda_route`. Reducing the length gap makes the
requested corridor competitive for the manifold's free geodesic. Curvature and
smoothness losses remain in the objective, so route fit is a tradeoff.

The `continuous` metric uses `pygeodesic` to find exact geodesics on the current
triangulated surface and inside the corridor. It differentiates path lengths
while holding each selected path fixed during one loss evaluation; paths may
switch between evaluations, so this is an experimental piecewise-smooth
objective. The cheaper `mesh_edges` metric uses shortest paths along mesh edges
as a surrogate. It can disagree with continuous geodesics.

Route preferences require `"method": ["lbfgs"]` and `"backend": ["torch"]`.
For example, add `"lambda_route": [100000]`,
`"route_metric": ["continuous"]`,
`"route_constraints": [[{"route": ["C", "D", "E", "H", "I", "J"]}]]`,
`"route_corridor_width": [0.055]`, and `"route_margin": [0.0]` to a config.
The `output.json` file records the chosen route loss, fit metrics, optimizer
status, and initial and final heights. A low route loss does not guarantee that
the free geodesic lies entirely in the corridor; check the exported path and
fit metrics.

To regenerate the four-variant example from `src`, run:

```sh
python site/scripts/reproduce_route_comparison.py \
  --output-dir site/examples/route-change
```

This writes the four `output.json` files and
`site/public/route-comparison.json`. To use your own alternatives, pass
`--changed-route` one to three times; each consecutive waypoint pair must be a
network link, and every route must have the original endpoints. Then run
`npm install` and `npx vite` from `src/site`, and open `/compare.html`.
The page has synchronized 3D cameras, top views, and animated geodesic traces.
It uses one shared height/color scale and displays height at 6× for visibility;
the path calculations use the actual, unscaled mesh. The **Load comparison
JSON** control lets you inspect another exported set.

The checked-in example uses a 1,500-iteration original surface and 200-iteration
refits. Two alternatives raise exact-geodesic corridor coverage from 69.9% to
81.7% and from 64.2% to 72.7%; their mean route deviations also fall. The
fourth raises coverage only from 67.2% to 68.7% and increases mean deviation.
All runs stop at their iteration caps, and curvature/smoothness losses rise on
the refits. This demonstrates a controllable tradeoff, not a guarantee that any
requested route will become the shortest path.
