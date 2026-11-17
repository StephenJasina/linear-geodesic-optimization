# Configuration parameters

This is a reference for every parameter you can set in a config file, with its
default value, what it does, and some intuition for choosing it.

A config file is a JSON file (usually under `data/.../config/`) that drives the
whole pipeline:

1. `src/optimization.py <config>` builds one surface per *run* and writes
   `../outputs/<...>/output.json` for each run.
2. `src/collation.py <config>` joins the runs of each *batch* (usually a
   sequence of time steps) into an `animation.json` for the web viewer.

Both scripts read the same config file. Run them from `src/`.

> **Where the defaults come from.** Optimization defaults live in
> `driver.base_defaults()` and collation defaults in `driver.collation_defaults()`
> (`src/linear_geodesic_optimization/driver.py`). If this document and the code
> ever disagree, the code wins.

> **About the numbers in "how to choose".** Many recommendations below are
> empirical findings from the toy `routing_with_volumes` datasets (mostly the
> `reroute_link_outage` and `two_clusters` traffic scenarios). Treat them as good
> starting points, not as laws.

---

## Contents

1. [Quick start](#1-quick-start)
2. [How a config file is structured](#2-how-a-config-file-is-structured)
3. [Settings](#3-settings)
4. [Optimization parameters](#4-optimization-parameters)
   - [4.1 Input network](#41-input-network)
   - [4.2 Mesh and network placement](#42-mesh-and-network-placement)
   - [4.3 Choosing a method](#43-choosing-a-method)
   - [4.4 Loss weights](#44-loss-weights)
   - [4.5 The curvature tube](#45-the-curvature-tube)
   - [4.6 Route losses](#46-route-losses)
   - [4.7 Construction method (`hessian_*`)](#47-construction-method-hessian_)
   - [4.8 Ridges method (`ridge_*`)](#48-ridges-method-ridge_)
   - [4.9 Internal parameters](#49-internal-parameters)
5. [Collation parameters](#5-collation-parameters)
6. [Recipes and rules of thumb](#6-recipes-and-rules-of-thumb)
7. [Common errors](#7-common-errors)

---

## 1. Quick start

A minimal config, which builds one surface per graph with the default
`construction` method and turns the sequence into one animation:

```json
{
    "arguments_optimization": {
        "filename_json": ["graph_0.json", "graph_1.json", "graph_2.json"],
        "sides": [50],
        "mesh_scale": [1.0]
    },
    "groups_optimization": [
        ["filename_json"]
    ],
    "settings": {
        "directory_data": ["..", "traffic", "two_clusters"],
        "directory_output": ["toy", "my_experiment"],
        "order_by": "filename_json"
    }
}
```

Every parameter you leave out takes its default. Check that a config is valid,
without running anything, with:

```sh
cd src
python optimization.py ../data/.../config/my_experiment.json --dry-run
```

---

## 2. How a config file is structured

A config file has up to five top-level keys:

| Key | Required | Purpose |
|---|---|---|
| `arguments_optimization` | yes | Values for the parameters of the surface builder (section 4). |
| `groups_optimization` | no | How those values are combined into runs. |
| `arguments_collation` | no | Values for the animation options (section 5). |
| `groups_collation` | no | How those values are combined into collation *variants*. |
| `settings` | no | Everything else: output paths, batching, seeding (section 3). |

The old key names `arguments` and `groups`, and the old settings
`postprocess` / `postprocess_subtract_initialization`, are rejected with an
error that tells you the new name.

### Every value is a list

Each parameter in `arguments_optimization` maps to a **list** of values, even if
there is only one value: `"sides": [50]`, not `"sides": 50`. A parameter whose
value is itself a list (such as a staged `curvature_tube_radius`) is therefore a
list of lists: `"curvature_tube_radius": [[null, 0.025]]`.

### Groups: zip within, multiply across

`groups_optimization` partitions the parameter names into groups.

- **Within a group**, the lists are read *in parallel* (zipped), so they must all
  have the same length. Use this for parameters that should change together,
  such as `mesh_scale` and the `lambda_smooth` that suits it.
- **Across groups**, every combination is taken (a Cartesian product).
- Every parameter you don't put into a group goes into one extra implicit group.
  This means that, without any groups, *all* your lists are zipped together,
  and must all have the same length.

In practice: give every parameter with more than one value a group (alone or
with its partners), and leave the single-valued ones ungrouped. That is why the
quick-start example puts `filename_json` in a group of its own.

Example:

```json
"arguments_optimization": {
    "filename_json": ["g0.json", "g1.json", "g2.json"],
    "mesh_scale":    [1.0, 4.0],
    "lambda_smooth": [0.005, 0.0003125]
},
"groups_optimization": [
    ["filename_json"],
    ["mesh_scale", "lambda_smooth"]
]
```

This gives 3 × 2 = 6 runs. `mesh_scale` 1.0 always goes with `lambda_smooth`
0.005, and 4.0 always goes with 0.0003125.

### Batches

`settings.order_by` names one parameter. The group containing it becomes the
"time axis": runs that differ *only* in that group form one **batch**, in the
order the values are listed. Each batch becomes one animation. In the example
above, with `"order_by": "filename_json"`, there are 2 batches (one per mesh
scale), each holding 3 frames.

Without `order_by`, every run is its own batch, and each animation has a single
frame.

### Collation variants

`arguments_collation` and `groups_collation` work the same way, but are never
batched. Each combination is a *variant*, and every batch is rendered once per
variant. A batch's data is loaded once and reused for all its variants, so
trying several display options is cheap. Display options belong here, never in
`arguments_optimization`.

---

## 3. Settings

| Setting | Default | Summary |
|---|---|---|
| `directory_output` | (required) | Output directory under `../outputs`, as a list of path parts. |
| `output_format` | `["filename_json"]` | Which parameter values name each run's subdirectory. |
| `order_by` | none | The parameter whose group forms a batch (one animation). |
| `initialization` | `"sphere"` | How runs are seeded: `sphere`, `sequential`, or `first`. |
| `directory_data` | none | Extra path parts prepended to the input file names. |
| `directory_data_type` | `"config"` | What `directory_data` is relative to: `config`, `data`, or `script`. |
| `output_format_collation` | `[]` | Which values name the animation files. |
| `directory_collation` | none | Put all animations in this directory instead of each batch's first run directory. |
| `n_cores` | number of cores − 2 | Number of runs computed in parallel. |
| `pass_arguments` | `{}` | Advanced: feed one run's parameter value into the next run's parameter. |

### `directory_output` and `output_format`

Each run is written to

```
../outputs/<directory_output parts>/<output_format parts>/
```

`output_format` is a list. Each entry is either a parameter name, or a list of
parameter names whose values are joined with `_`. Each entry becomes one level
of directories. For example,

```json
"directory_output": ["toy", "reroute_ridges"],
"output_format": [["ridge_width", "ridge_height_scale"], ["index", "filename_json"]]
```

writes runs to `../outputs/toy/reroute_ridges/0.08_12.0/3_graph_3/`.

- File-name parameters appear without their directory and extension
  (`graph_3`).
- List values are joined with `-`, with `null` written as `default`
  (`[null, 0.025]` becomes `default-0.025`).
- `index` is a counter that is unique across every run in the config. Adding it
  is the easiest way to guarantee unique directory names.

**How to choose:** include every parameter you are sweeping, or two runs will
want the same directory, which is an error. The default (`filename_json` only) is
fine as long as you only vary the input file.

### `order_by`

Name the parameter that varies over time, almost always `filename_json` (or
`filename_links`). The other parameters in its group step along with it.

### `initialization`

How each run's starting surface is chosen. This only matters for the `lbfgs`
method: `construction` and `ridges` ignore seeds, so for them it only affects the
run order and which runs are collated.

| Value | Meaning |
|---|---|
| `sphere` | Every run starts from a sphere cap (see `initial_radius`). All runs are independent and run fully in parallel. |
| `sequential` | Each run starts from the previous run's result in the same batch. Frames vary smoothly, but a batch runs serially. |
| `first` | The first run of each batch is only a *seed*: it is computed first, every other run starts from it, and it is **left out** of the animation. |

**How to choose:** use `sphere` for `construction` and `ridges`. For `lbfgs`,
`sequential` gives the smoothest animations (each frame is a small change of the
previous one). `first` is useful when you want a common, well-converged starting
point, e.g. a first entry that runs for many more iterations.

### `directory_data` and `directory_data_type`

Input file names (`filename_*`) are resolved as

```
<base>/<directory_data parts>/<file name>
```

where `<base>` is the directory of the config file (`config`, the default), the
repository's `data/` directory (`data`), or the current directory, i.e. `src/`
(`script`). A file name can also be given as a list of path parts.

### `output_format_collation` and `directory_collation`

Animations are called `animation.json` by default. When you have several
collation variants, give them distinct names with `output_format_collation`
(same syntax as `output_format`). The parts may name collation parameters or
optimization parameters (taken from the batch's first run), and produce names
like `animation_False_min.json`. Duplicate animation paths are an error.

Animations are written into each batch's first run directory, or into
`../outputs/<directory_collation parts>/` if that is set.

### `n_cores`

The number of worker processes. With many runs of a large mesh, setting it below
the core count, and pinning BLAS/OpenMP threads to one per process
(`OMP_NUM_THREADS=1` and friends), can make runs dramatically faster by avoiding
thread oversubscription.

### `pass_arguments`

A map `{"source_parameter": "destination_parameter"}`: within its group, each
run receives the *previous* entry's value of the source as the destination (and
`null` for the first). This is an advanced feature; the seeding strategies above
cover the usual need.

---

## 4. Optimization parameters

Units matter throughout, and three different ones appear:

- **mesh units**: the mesh is a square `mesh_scale` wide;
- **fractions of `mesh_scale`**: most widths and radii, which keeps them the same
  physical size if you change `mesh_scale` or `sides`;
- **the unit square**: only `initial_radius`.

### 4.1 Input network

| Parameter | Default | Used by |
|---|---|---|
| `filename_json` | `null` | all |
| `filename_graphml` | `null` | all |
| `filename_probes` | `null` | all |
| `filename_links` | `null` | all |
| `latency_threshold` | `null` | JSON, CSV |
| `clustering_distance` | `null` | JSON, CSV |
| `ricci_curvature_alpha` | `0.0` | networks without traffic |
| `ricci_curvature_reweight` | `null` | JSON with traffic |
| `ricci_curvature_distribution_reweight_scale` | `0.0` | JSON with traffic |

#### The input file: `filename_json`, `filename_graphml`, `filename_probes` + `filename_links`

Give exactly one source per run, checked in this order:

1. **`filename_graphml`**: a GraphML graph, read as is. Curvatures are expected to
   be in the file, so the curvature and filtering parameters below are ignored.
2. **`filename_json`**: a JSON network. It can carry **routes and traffic**, and
   then curvatures are computed from the traffic, edges get throughputs, and the
   route losses, the `ridges` method and route-following geodesics all become
   available. This is the usual choice.
3. **`filename_probes` and `filename_links`**: a CSV of nodes (probes) and a CSV
   of links (with latencies). Curvatures are computed from the topology.

Set the unused ones to `null` (or leave them out).

#### `latency_threshold` (default `null`, i.e. keep every edge)

Drops every edge whose measured round-trip time exceeds its great-circle (speed
of light in fiber) latency by more than this many milliseconds. Edges with no
latency are always kept.

**How to choose:** this is a cleanup of measurement data. Large excess latency
usually means an indirect path or congestion rather than a physical link. Values
of a few milliseconds are typical (the toy configs use `4`). `null` keeps
everything.

#### `clustering_distance` (default `null`, i.e. no clustering)

Merges nodes that are geographically closer than this distance, in **metres**,
into one node. Links and routes are remapped to the merged nodes, and links that
become self-loops are dropped.

**How to choose:** use it to merge points of presence in the same metro area,
which would otherwise pile up on top of each other on the mesh. The toy configs
use `500000` (500 km).

#### `ricci_curvature_alpha` (default `0.0`)

The "laziness" of the random walk in Ollivier-Ricci curvature: the probability
mass a node keeps for itself, with the rest spread over its neighbors. It is only
used when curvature is computed from the topology (CSV input, or JSON without
traffic). With traffic, curvature comes from the traffic instead.

**How to choose:** `0` is standard. Larger values (up to about `0.5`) shrink all
curvatures towards zero and make them less sensitive to single links.

#### `ricci_curvature_reweight` (default `null`)

A modification of the traffic-based curvature (JSON input with traffic):

- `null`: the plain curvature, κ = 1 − W/d (W is the earth-mover distance and d
  the edge length).
- `"distribution"`: scales the earth-mover cost of each edge up with the traffic
  on it, κ = 1 − (1 + r)·W/d, where r grows from 0 (no traffic) to the scale
  below (the busiest link) on a log scale. **Busier links become more negatively
  curved**, so traffic shows up as saddles.
- `"metric"`: changes the metric used for every distance in the computation
  instead, so the result is still a proper curvature.

#### `ricci_curvature_distribution_reweight_scale` (default `0.0`)

The strength `s` of the `"distribution"` reweighting: the busiest link's cost is
multiplied by `1 + s`, and others by `1 + log(1 + (t/t_max)(eˢ − 1))`, which
lies between 1 and `1 + s`.

**How to choose:** note that with the default `0` the reweighting has **no
effect**. `1.0` is a moderate setting and `2.0` a strong one. Larger values make
traffic more visible, but give larger curvatures, which are harder to fit (for
example, `hessian_slope_correction` diverges earlier).

### 4.2 Mesh and network placement

| Parameter | Default | Unit |
|---|---|---|
| `sides` | `50` | vertices per side |
| `mesh_scale` | `1.0` | mesh units |
| `coordinates_scale` | `0.8` | fraction of the mesh |
| `initial_radius` | `null` (flat) | unit square |
| `network_trim_radius` | `null` (no trim) | mesh units |

#### `sides` (default `50`)

The mesh is a `sides` × `sides` grid of vertices, each square cut into two
triangles.

**How to choose:** the cost grows roughly with `sides²` (or worse for
L-BFGS). `40`–`50` is good for exploring, `80`–`120` for final figures. Narrow
features (ridges, tubes) need enough vertices across them: a ridge or tube
should be at least a few cells wide, where a cell is `mesh_scale / (sides − 1)`.
For example, a `ridge_width` of 0.05 shows diagonal aliasing even at 120 sides.

Things that are measured in mesh cells (the default curvature tube, and
`hessian_tube_radius` by default) shrink when you refine the mesh. Things
measured as fractions of `mesh_scale` (most others) don't.

#### `mesh_scale` (default `1.0`)

The physical width of the mesh. The target curvatures are fixed numbers, so this
sets how strongly the surface must bend: on a wider mesh, the same curvature
produces relief that is taller relative to the mesh width (for `construction`,
heights grow like `mesh_scale²`, so relative relief grows like `mesh_scale`).

**How to choose:** surfaces at `mesh_scale` 0.05–0.25 look flat purely for this
reason. Use about `1` for a gentle surface and about `4` for tall, dramatic
relief. Several other parameters should change with it, notably
`lambda_smooth` (roughly ∝ 1/`mesh_scale²`) and `hessian_smoothness`. Put them in
the same group so they move together.

#### `coordinates_scale` (default `0.8`)

The fraction of the mesh's width and height that the network's bounding box
occupies, centered. Node coordinates are projected with Mercator first.

**How to choose:** `0.8` leaves a margin so that features near the outer nodes
aren't cut off by the mesh boundary. Use less (0.6–0.7) if you see boundary
effects, especially with `hessian_boundary: "fixed"` or wide ridges.

#### `initial_radius` (default `null`)

The starting surface for `lbfgs` (and for `construction` with
`hessian_initialization: "sphere"`): a sphere cap of this radius, measured as if
the mesh were the unit square, shifted so its lowest point is 0. `null` means an
infinite radius, i.e. a flat start.

**How to choose:** a flat start is a critical point of the L-BFGS loss and tends
to give flat or creased results. A broad cap such as `1`–`2` usually gets a
lower loss. Ignored by `ridges`.

#### `network_trim_radius` (default `null`)

`lbfgs` only (the other methods raise an error if it is set). Deletes every mesh
vertex farther than this from the network's edges, so only a band around the
network is optimized.

**How to choose:** it speeds up L-BFGS on sparse networks, at the cost of a
ragged surface boundary. Usually leave it `null`.

### 4.3 Choosing a method

| Parameter | Default | Values |
|---|---|---|
| `method` | `"construction"` | `construction`, `ridges`, `lbfgs` |
| `backend` | `"torch"` | `torch`, `legacy` |
| `maxiter` | `null` (scipy's default cap) | `lbfgs` only |
| `initialization_file_path` | `null` | set automatically |

#### `method`

How the surface heights are produced. All three methods report the same loss
(sections 4.4–4.6), so their results are directly comparable.

- **`construction`** turns each edge's curvature into a target Hessian (a dome
  for positive curvature, a saddle for negative) and solves for the heights with
  a few sparse linear solves. It is fast (well under a second) and interpretable:
  every feature traces back to an edge. It recovers most of what L-BFGS achieves
  on the curvature loss, and an optional L-BFGS polish (`hessian_polish_maxiter`)
  closes the rest. Parameters: section 4.7.
- **`ridges`** raises a ridge along each edge, with heights set by **traffic
  throughput**, so everything is above the plane z = 0 and busy parts of the
  network are high ground. The raw ridges are only a starting point; the L-BFGS
  polish (`ridge_polish_maxiter`) does the actual curvature fitting, while
  keeping the throughput ordering. This is the method to use when you want
  geodesics on the surface to follow the network's routes. Needs edge
  throughputs (JSON with traffic). Parameters: section 4.8.
- **`lbfgs`** minimizes the loss directly with L-BFGS-B from a sphere cap or a
  seed run. It is the original method and the baseline. It is slow (often
  thousands of iterations) and can get stuck, but it makes no structural
  assumptions.

**How to choose:** start with `construction` for a quick look. Use `ridges` with
a polish when traffic and routes matter. Use `lbfgs` for comparisons with
earlier results. Configs that run L-BFGS must say `"method": ["lbfgs"]`
explicitly.

#### `backend` (default `"torch"`)

Which implementation of the loss and its gradient to use: `torch` (autodiff;
faster) or `legacy` (hand-derived gradients in numpy). Both give the same numbers
to within floating point error. Use `legacy` only to cross-check.

#### `maxiter` (default `null`)

The maximum number of L-BFGS iterations for `method: "lbfgs"`. `null` uses
scipy's default limit (15000). The polishes of the other methods have their own
`*_polish_maxiter` parameters.

**How to choose:** L-BFGS usually runs to several thousand iterations. A few
hundred gives a rough picture.

#### `initialization_file_path` (default `null`)

The `output.json` whose final heights seed an `lbfgs` run. It is **filled in
automatically** from `settings.initialization`, so you don't normally set it.

### 4.4 Loss weights

The loss that every method reports (and that L-BFGS and the polishes minimize)
is

```
loss = lambda_curvature · L_curvature + lambda_smooth · L_smooth
       + lambda_geodesic · L_geodesic + lambda_stretch · L_stretch
```

| Parameter | Default |
|---|---|
| `lambda_curvature` | `1.0` |
| `lambda_smooth` | `0.0` |
| `smooth_loss_type` | `"mvs"` |

#### `lambda_curvature` (default `1.0`)

The weight of the curvature loss: the mismatch between each edge's target
(Ricci) curvature and the surface's Gaussian curvature near the edge (see
section 4.5 for what "near" means). Edges are weighted by their length.

**How to choose:** leave it at `1` and tune the other weights relative to it.

#### `lambda_smooth` (default `0.0`)

The weight of a smoothness penalty, which discourages curvature from changing
abruptly across the surface (without it, the optimizer is free to create spikes
and grid-scale noise between the edges).

**How to choose:** the smoothness term grows with `mesh_scale²`, so scale the
weight as about `1/mesh_scale²`. Empirically:

- `0.005 / mesh_scale²` gives smooth surfaces, but on tall surfaces it
  stops short, strongly curved edges from reaching their targets (some negative
  edges stay domes).
- About 100× smaller (e.g. `3.125e-5` at `mesh_scale` 4) lets every edge reach
  its target.
- About 1000–10000× smaller (`3.125e-6` to `3.125e-7` at `mesh_scale` 4) fits
  best, at the cost of some roughness; pair it with a longer polish (1000+
  iterations), which damps the roughness.

If curvature targets aren't being met, the smoothness weight is the first thing
to lower.

#### `smooth_loss_type` (default `"mvs"`)

Which smoothness penalty to use:

- `"mvs"`: the original "minimum variation surface" penalty on how the
  curvature varies along mesh edges.
- `"tensor"`: compares the shape operator (the full second fundamental form)
  between neighboring vertices. It penalizes changes in curvature *direction* as
  well as magnitude, but not uniform curvature.

**How to choose:** `mvs` is the default and what most configs use. `tensor`
works well together with `hessian_smoothness` in the construction method.

### 4.5 The curvature tube

The curvature loss compares, for each edge, the surface's Gaussian curvature at
the mesh vertices within a **tube** around the edge with the edge's target
curvature. These parameters shape that tube.

| Parameter | Default | Unit |
|---|---|---|
| `curvature_tube_radius` | `null` (≈ one mesh diagonal) | fraction of `mesh_scale` |
| `curvature_tube_falloff` | `0.0` (hard edge) | fraction of `mesh_scale` |
| `curvature_tube_falloff_weight` | `false` | |
| `curvature_tube_falloff_ends` | `true` | |

#### `curvature_tube_radius` (default `null`)

The radius of the tube. With `null`, it is just over one mesh diagonal
(`1.01 · √2 · mesh_scale / sides`), so the matched curvature is a strip about
three vertices wide, and it *narrows as you refine the mesh*. The surface often
compensates with opposite curvature right beside this thin strip. A number
instead sets a fixed physical radius as a fraction of `mesh_scale`, giving a
band that doesn't depend on the resolution.

A **list** of radii polishes in stages (only for the `construction` and
`ridges` polishes, and not with `lambda_stretch`): each stage runs the full
`*_polish_maxiter` iterations with its radius, starting from the previous stage's
result. The last radius is the one reported.

**How to choose:** going straight to a wide tube makes the polish stall, but a
staged `[null, 0.025]` (thin tube first, then a band 0.1 mesh units wide at
`mesh_scale` 4) worked well: the curvature in a fixed-width band around each edge
then matches its target closely. Much wider tubes (e.g. 0.0375) fit worse.
Remember the list-of-lists syntax: `"curvature_tube_radius": [[null, 0.025]]`.

#### `curvature_tube_falloff` (default `0.0`)

The width of a ring outside the tube in which the target fades smoothly (with a
cosine) from the edge's curvature to 0. With a hard edge (`0`), the
compensating curvature tends to pile up in a rim just outside the tube, at 2–5×
the target. One value applies to every stage, or give one value per
`curvature_tube_radius` stage.

**How to choose:** use it only in the *last* stage, e.g. `[0, 0.05]` together
with `curvature_tube_radius` `[null, 0.025]`. A falloff in the first stage makes
the polish stall. The trade-off: short, strongly curved edges between tall nodes
lose some of their band curvature.

#### `curvature_tube_falloff_weight` (default `false`)

Whether the vertices in the falloff ring also count for less in the loss (their
weight fades along with the target). With `false`, the ring counts fully, which
enforces the fade-out more strongly. The results with `false` were somewhat
better.

#### `curvature_tube_falloff_ends` (default `true`)

Whether the falloff ring also wraps around the ends of the edge (over its
nodes), or only runs along its sides. Nodes are where edges with *different*
targets meet, so a ring over the node makes them fight. `false` is usually the
better choice.

### 4.6 Route losses

Optional loss terms that try to make the network's routes behave like geodesics
of the surface. Both need a JSON network with routes and traffic, and both are
off by default.

| Parameter | Default | Used by |
|---|---|---|
| `lambda_geodesic` | `0.0` | all methods |
| `geodesic_curve` | `"route"` | when `lambda_geodesic` > 0 |
| `lambda_stretch` | `0.0` | `construction` / `ridges` polish only |
| `stretch_refresh` | `100` | when `lambda_stretch` > 0 |
| `stretch_smoothing` | `0.035` | when `lambda_stretch` > 0 |

#### `lambda_geodesic` and `geodesic_curve`

`lambda_geodesic` weighs the traffic-weighted bending energy ∫κ_g² ds of the
routes on the surface: a curve with zero geodesic curvature is (locally) a
geodesic. `geodesic_curve` chooses which curves:

- `"route"`: a smooth (natural cubic spline) curve through each route's
  waypoints;
- `"link"`: the straight links that the routes use.

**How to choose:** it lowers bending a lot, but in practice it does *not* make
the drawn geodesics follow the routes, because a route can be locally straight
and still not be the shortest path. Prefer geometry (narrow ridges, section
4.8) over this loss.

#### `lambda_stretch`, `stretch_refresh`, `stretch_smoothing`

`lambda_stretch` weighs the route stretch loss, Σ w · (route length / geodesic
length − 1)₊²: it penalizes routes that are longer than the shortest path
between their endpoints, which is exactly the path collation draws. It is only
supported in the polish of `construction` and `ridges` (not `lbfgs`, and not
with a staged tube radius).

- `stretch_refresh` is the number of polish iterations between recomputing the
  competing shortest paths (the polish runs in rounds of this length).
- `stretch_smoothing` is the width of a Gaussian blur, as a fraction of
  `mesh_scale`, applied to the heights before measuring lengths.

**How to choose:** the loss only had an effect at very large weights (about
`1e4`), and then by etching grid-scale grooves; blurring the heights enough to
remove the grooves also removed the gain. As with `lambda_geodesic`, narrow
ridges work better.

### 4.7 Construction method (`hessian_*`)

Only used when `method` is `"construction"`. The construction assigns each edge a
target Hessian (second-derivative matrix) whose determinant is the edge's
curvature, and fits the heights to these targets by least squares. For gentle
slopes, the Gaussian curvature is approximately the Hessian's determinant.

| Parameter | Default |
|---|---|
| `hessian_anisotropy` | `1.0` |
| `hessian_orientation` | `"hill"` |
| `hessian_regularization` | `1e-3` |
| `hessian_tube_radius` | `null` (≈ one mesh diagonal) |
| `hessian_iterations` | `5` |
| `hessian_slope_correction` | `false` |
| `hessian_smoothness` | `0.0` |
| `hessian_polish_maxiter` | `0` (no polish) |
| `hessian_initialization` | `null` (flat) |
| `hessian_boundary` | `"fixed"` |

#### `hessian_anisotropy` (default `1.0`)

The ratio of the curvature *along* an edge to the curvature *across* it, in the
edge's initial target. `1` is isotropic: a round dome, or a symmetric saddle.
Larger values give long ridges or valleys along the edge; smaller values give
features that bend mostly across it. The determinant (and thus the curvature) is
the same either way.

**How to choose:** `1` is a good default, especially since the iterations below
let each target re-adjust its aspect ratio anyway.

#### `hessian_orientation` (default `"hill"`)

With `"hill"`, positive curvature becomes a dome, and negative curvature a
saddle that *rises* along the edge (a mountain pass). `"basin"` flips every
target upside down (bowls, and saddles that dip along the edge). The curvature
is the same; only the look changes.

#### `hessian_regularization` (default `1e-3`)

The weight of a penalty on the Hessian everywhere (relative to the total weight
of all edge targets, which is 1). It keeps the surface flat away from the
network.

**How to choose:** small. Larger values make the surface flatter overall and
fit the targets less well; much smaller values can let the areas between edges
drift. Unlike `hessian_smoothness`, it also penalizes uniform curvature.

#### `hessian_tube_radius` (default `null`)

The radius of the tube around each edge in which the target Hessian is imposed,
in **mesh units** (not a fraction of `mesh_scale`). `null` means the same
one-diagonal width the loss uses by default.

#### `hessian_iterations` (default `5`)

Targets of edges that meet at a node usually can't all be met at once. Each
iteration lets every target rotate and change its aspect ratio (keeping its
determinant, i.e. its curvature) to agree with the current surface, then solves
again.

**How to choose:** `0` (pure targets) recovers only 40–55% of what L-BFGS
achieves on the curvature loss; `5` recovers 76–98%, still in under a second.
More rarely helps much.

#### `hessian_slope_correction` (default `false`)

The fit treats det(Hessian) as the Gaussian curvature, which only holds for
gentle slopes; the true curvature is det(Hessian) / (1 + |∇z|²)². With this
flag, each iteration asks for a larger determinant where the surface is steep,
so steep surfaces still reach their targets. It has no effect with
`hessian_iterations: 0`.

**How to choose:** turn it on for larger `mesh_scale` (about 1 and up), where the
uncorrected fit undershoots badly. It can run away on very steep surfaces (around
`mesh_scale` 2.5–3 with strong reweighting), in which case the run stops with
"slope correction diverged; use a smaller mesh scale".

#### `hessian_smoothness` (default `0.0`)

The weight of the `tensor` smoothness penalty, linearized so it stays a linear
solve. It penalizes changes in curvature (but not uniform curvature).

**How to choose:** about `1e-4 / mesh_scale²` cuts the smoothness loss 2–10×
with no cost to the curvature fit (often improving it). Scale it as
1/`mesh_scale²`.

#### `hessian_polish_maxiter` (default `0`)

After the construction, run this many L-BFGS iterations on the full loss. `0`
means no polish, and `null` polishes until convergence.

**How to choose:** a short polish (about 200 iterations, a few seconds) brings
the result to roughly the level of a full L-BFGS run. Most of what it adds is
smoothness, which the construction itself only partially accounts for.

#### `hessian_initialization` (default `null`)

The surface the construction starts from (and fits offsets to): `null` for flat,
or `"sphere"` for the sphere cap of `initial_radius`. Seeds from
`settings.initialization` are ignored by this method.

#### `hessian_boundary` (default `"fixed"`)

- `"fixed"`: the mesh boundary stays at its initial height. For gentle slopes,
  the total curvature is then pinned by the boundary, so curvature added by one
  edge must be balanced by opposite curvature elsewhere inside the mesh.
- `"free"`: the boundary heights are solved for too, so curvature can be balanced
  through the boundary instead.

**How to choose:** `"free"` gives each edge's feature more freedom and fewer
compensating artifacts between edges; `"fixed"` gives a tidier frame.

### 4.8 Ridges method (`ridge_*`)

Only used when `method` is `"ridges"`. Each node gets a height from the
throughput of its links, and each edge becomes a raised ridge whose crest sags
between its ends (negative curvature, a mountain pass on the ridge) or arches
(positive curvature). The ridges are combined with a smooth maximum, so the
surface stays above z = 0 and the higher ridge wins where they cross. The result
is then smoothed and, normally, polished with L-BFGS.

| Parameter | Default | Unit |
|---|---|---|
| `ridge_width` | `0.3` | fraction of `mesh_scale` (half-width) |
| `ridge_height_scale` | `0.05` | fraction of the half-width |
| `ridge_height_base` | `0.25` | fraction of the full height |
| `ridge_node_height` | `"max"` | `max`, `sum`, `mean` |
| `ridge_height_transform` | `"linear"` | `linear`, `sqrt`, `log` |
| `ridge_crest_height` | `"nodes"` | `nodes`, `edges` |
| `ridge_blend_p` | `8.0` | |
| `ridge_smoothing` | `0.035` | fraction of `mesh_scale` |
| `ridge_polish_maxiter` | `0` | iterations |
| `ridge_height_normalization` | `"batch"` | `batch`, `run` |
| `ridge_height_reference` | `null` | filled in automatically |

> **The polish is essential for this method.** The raw ridge surface fits the
> curvatures poorly (often worse than a flat surface); the ridges are an
> interpretable, throughput-ordered *starting point*, and the L-BFGS polish does
> the fitting while keeping that ordering. Always set `ridge_polish_maxiter`.

#### `ridge_width` (default `0.3`)

The half-width of each ridge (from crest to foot), as a fraction of
`mesh_scale`.

**How to choose:** this is the main control of how well geodesics follow the
network. Wide ridges merge into a plateau, and geodesics cut across it. **Narrow
ridges** (about `0.05`–`0.08`), narrower than the spacing between links, give
geodesics that follow the routes much more closely. Below about `0.05`, the ridge
is only a few cells wide even at 120 sides, and shows grid aliasing. Narrow
ridges need a larger `ridge_height_scale` to keep the same relief.

#### `ridge_height_scale` (default `0.05`)

The height of the highest node, as a fraction of the ridge half-width. This is
roughly the steepest cross slope of a ridge, so it controls how far the surface
is from the gentle-slope regime.

**How to choose:** the tallest node is `ridge_height_scale × ridge_width` of the
mesh width (a bit more where ridges overlap). With the default width of 0.3, the
relief is about 32 × `ridge_height_scale` percent of the mesh width, so tall
surfaces (relief 20–30% of the width) need a height scale of about 0.6–1. Since it is relative
to the half-width, narrowing the ridges by some factor needs the height scale
raised by that factor to keep the same absolute height. For example, the
geodesic-following runs used width `0.08` with height scale `10`–`12` at
`mesh_scale` 4.

#### `ridge_height_base` (default `0.25`)

The height of a node with no throughput, as a fraction of the full height.
Heights run linearly from this (no traffic) to 1 (the reference throughput).

**How to choose:** it keeps idle parts of the network visibly above the plane.
Lower values make throughput differences more dramatic; `0` lets idle nodes sit
on the plane.

#### `ridge_node_height` (default `"max"`)

How a node's throughput is computed from its links' throughputs: the `max`, the
`sum`, or the `mean`.

**How to choose:** `max` makes a node exactly as high as its busiest link, which
gives continuous crests. `sum` emphasizes hubs (nodes with many links).

#### `ridge_height_transform` (default `"linear"`)

Applied to throughputs before normalizing: `linear`, `sqrt`, or `log`
(log(1 + t)).

**How to choose:** `linear` when throughputs are of similar magnitude. `sqrt` or
`log` when a few links carry most of the traffic and would otherwise flatten
everything else to the base height.

#### `ridge_crest_height` (default `"nodes"`)

- `"nodes"`: each ridge's crest runs between the heights of its two end nodes.
- `"edges"`: each ridge's crest sits at the edge's *own* throughput height
  (normalized the same way). A quiet link becomes a low ridge leaving its nodes'
  domes, so a path along it must descend and climb again.

**How to choose:** use `"edges"` when you want geodesics to follow *shifting*
traffic: with `"nodes"`, a quiet link between two busy nodes is as high as a busy
one, and geodesics have no reason to avoid it. `"edges"` with `"sqrt"`, width
`0.08` and height scale 10–12 made geodesics switch paths when traffic
rerouted.

#### `ridge_blend_p` (default `8.0`)

Ridges are combined as the p-norm (a smooth maximum) of their heights. Larger
`p` approaches a hard maximum (sharp creases where ridges meet); smaller `p`
blends them more, but makes overlapping ridges add up into higher ground.

**How to choose:** the default is fine for most purposes.

#### `ridge_smoothing` (default `0.035`)

The width of a final Gaussian blur, as a fraction of `mesh_scale`. It rounds off
the kinks where a sagging crest meets its node's dome, which otherwise become
curvature spikes that grow with resolution and make the polish fail.

**How to choose:** keep it smaller than `ridge_width`, or it erases the ridges.
With narrow ridges and a long polish, `0.05` worked better than `0.035` (with
which the polish stalled); `0.0175` also worked with width `0.08`.

#### `ridge_polish_maxiter` (default `0`)

Number of L-BFGS iterations to polish with (`null` runs to convergence). See the
note above: always polish.

**How to choose:** 200 gives a usable result on small meshes. For 120-side meshes
with a small `lambda_smooth`, 1000–6000 iterations. Staged tube radii multiply
this by the number of stages.

#### `ridge_height_normalization` and `ridge_height_reference`

How node heights are normalized, i.e. which throughput maps to the full height:

- `"batch"` (default): the largest node throughput across the whole batch, so
  heights are comparable from one animation frame to the next. A frame with
  little traffic is genuinely lower.
- `"run"`: each run's own largest throughput, so every frame uses the full
  height range.

`ridge_height_reference` is the resulting throughput value. With `"batch"`, it
is **filled in automatically** before the runs start; you don't normally set it.

### 4.9 Internal parameters

| Parameter | Meaning |
|---|---|
| `index` | A unique counter per run, assigned automatically. Use it in `output_format` for unique directory names; don't set it yourself. |
| `directory_output` | Each run's output directory, computed from the settings (section 3). |
| `initialization_file_path` | Seed file for `lbfgs`, computed from `settings.initialization`. |
| `ridge_height_reference` | Computed for `ridge_height_normalization: "batch"`. |

---

## 5. Collation parameters

These go in `arguments_collation` (and `groups_collation`). They control how
`collation.py` turns a batch's surfaces into an animation; they never change the
surfaces themselves.

| Parameter | Default | Summary |
|---|---|---|
| `postprocess` | `true` | Rescale, center, smooth and trim the heights for display. |
| `postprocess_subtract_initialization` | `true` | Subtract the starting surface first (with postprocessing). |
| `height_scale` | `0.15` | Height range after postprocessing, as a fraction of the mesh width. |
| `bubble_size` | `0.03` | How far around the network the trimmed mesh extends. |
| `vertical_centering` | `"mean"` | How each frame is shifted vertically: `mean`, `min`, or `null`. |
| `exact_geodesics` | `false` | Use exact geodesics instead of the fast approximation. |
| `use_convex_hull` | `false` | Use the network's convex hull as its border. |
| `geodesic_endpoints` | `false` | Draw geodesics between route endpoints only, recomputed per frame. |

#### `postprocess` (default `true`)

With postprocessing, each frame's heights are: reduced by a reference height
(see below), centered, **rescaled so all frames together span `height_scale`
times the mesh width** (one scale shared by all frames), smoothly faded out near
the edge of the bubble, and trimmed to a bubble around the network. Geodesics are
then computed on this processed surface. Each frame also keeps the unprocessed
surface as `original`, so the viewer can show both.

Without postprocessing, the heights are just the computed surface (centered per
`vertical_centering`), and geodesics are computed on the full mesh.

**How to choose:** postprocessing makes very flat or very tall surfaces easy to
look at. But rescaling the heights changes which path is shortest, so **use
`postprocess: false` whenever the geodesics matter** (e.g. checking that they
follow routes). `height_scale: null` has the same effect on the scaling.

#### `postprocess_subtract_initialization` (default `true`)

With postprocessing, subtract the run's starting surface (its `initial` heights,
e.g. the sphere cap for `lbfgs`) so you see only what the method added. For
`construction` and `ridges` the starting surface is usually flat, so this changes
little.

#### `height_scale` (default `0.15`)

With postprocessing, the total height range of the animation as a fraction of the
mesh width. `0.1`–`0.2` gives a readable 3D view.

#### `bubble_size` (default `0.03`)

With postprocessing, the mesh is trimmed to vertices within this distance of the
network (as a fraction of `mesh_scale`). Increase it if isolated edges end up in
disconnected bits of mesh or geodesics run off the trimmed surface.

#### `vertical_centering` (default `"mean"`)

How each frame is shifted vertically, with or without postprocessing: so that its
`"mean"` height is 0, or its `"min"` height is 0, or not at all (`null`), which
keeps real height changes between frames. With postprocessing, the mean or
minimum is taken over the bubble; otherwise over the whole mesh.

**How to choose:** `"min"` suits `ridges` (the plane z = 0 stays at the bottom).
`null` if the absolute heights across frames are meaningful.

#### `exact_geodesics` (default `false`)

Compute geodesics with an exact algorithm (`pygeodesic`) instead of the default
fast approximation (`potpourri3d`). Slower; use it for final figures or when the
approximation looks off.

#### `use_convex_hull` (default `false`)

Use the convex hull of the network's nodes as its border (for the bubble and the
drawn outline), instead of the network's actual shape.

#### `geodesic_endpoints` (default `false`)

- `false`: for each route of the *first* frame, draw a geodesic through that
  route's waypoints, in every frame. The geodesics are pinned to frame 0's
  waypoints.
- `true`: draw one geodesic per source-destination pair (over the routes of all
  frames), between the endpoints only, recomputed on each frame's surface. The
  geodesic can then switch to whichever way that frame's surface makes shortest.

**How to choose:** `true` (with `postprocess: false`) is the honest test of
whether the surface encodes the routing: it shows where geodesics *want* to go.

---

## 6. Recipes and rules of thumb

### A quick look at a new dataset

`method: "construction"`, `sides: 50`, `mesh_scale: 1`, defaults otherwise.
Takes seconds. Add `hessian_polish_maxiter: 200` and `lambda_smooth: 0.005` for a
smoother result.

### Throughput-ordered surfaces whose geodesics follow the routes

The settings from `data/toy/routing_with_volumes/config/traffic_reroute_link_outage_ridges_soft_tube_120_sides.json`:

```json
"method":                       ["ridges"],
"sides":                        [120],
"mesh_scale":                   [4.0],
"lambda_smooth":                [3.125e-07],
"ricci_curvature_reweight":     ["distribution"],
"ricci_curvature_distribution_reweight_scale": [1.0],
"ridge_width":                  [0.08],
"ridge_height_scale":           [12.0],
"ridge_smoothing":              [0.05],
"ridge_crest_height":           ["edges"],
"ridge_height_transform":       ["sqrt"],
"ridge_polish_maxiter":         [6000],
"curvature_tube_radius":        [[null, 0.025]],
"curvature_tube_falloff":       [[0, 0.05]],
"curvature_tube_falloff_ends":  [false]
```

with collation `postprocess: false`, `vertical_centering: "min"`,
`geodesic_endpoints: true`. These runs are expensive (two 6000-iteration
stages at 120 sides); run them on a server.

### An L-BFGS baseline

`method: "lbfgs"`, `initial_radius: 1` or so (not flat), `maxiter` a few
thousand, `settings.initialization: "sequential"` for smooth animations.

### General rules of thumb

- **Change `mesh_scale` and `lambda_smooth` (and `hessian_smoothness`)
  together**, keeping `lambda_smooth × mesh_scale²` in mind.
- **Narrow features need resolution:** keep ridges and tubes several mesh cells
  wide.
- **If curvature targets aren't met**, lower `lambda_smooth` and polish longer
  before changing anything else.
- **If a polish stalls** (stops after very few iterations, or the loss barely
  moves), the surface probably has sharp kinks: raise `ridge_smoothing`, or
  start with a thin tube and widen it in a second stage.
- **Display options go in `arguments_collation`**, so changing them only needs
  `collation.py`, not a new optimization.

---

## 7. Common errors

| Error | Cause and fix |
|---|---|
| `... already exists` | `optimization.py` never overwrites runs. Delete the old `../outputs/...` directory, or change `directory_output`. |
| `Some output directories are duplicated` | Two runs map to the same directory. Add the parameters you're sweeping (or `index`) to `output_format`. |
| `Uneven argument list lengths within group` | Lists in one group (or in the implicit group of everything ungrouped) have different lengths. |
| `Invalid partition` | A parameter is listed in two groups, or a group names a parameter not in `arguments_optimization`. |
| `order_by is not a valid parameter name` | `order_by` must be a parameter in `arguments_optimization`. |
| `Config key "arguments" has been renamed ...` | Old config format; rename the key as the message says. |
| `Unknown collation parameters: ...` | A typo, or an optimization parameter in `arguments_collation`. |
| `The construction method does not support network_trim_radius` | Trimming is for `lbfgs` only. |
| `lambda_stretch is supported only by the polish ...` | Don't combine `lambda_stretch` with `lbfgs`. |
| `Staged curvature_tube_radius needs the construction or ridges polish, without lambda_stretch` | A list of tube radii only works for those polishes, and not with the stretch loss. |
| `curvature_tube_falloff needs one value per curvature_tube_radius stage` | Give a single falloff, or a list as long as the radius list. |
| `The ridges method needs edge throughputs` | Use a JSON network with traffic, or a links CSV with a `throughput` column. |
| `slope correction diverged; use a smaller mesh scale` | `hessian_slope_correction` ran away on a steep surface. Lower `mesh_scale` or the reweighting scale, or turn it off. |
| Any numerical warning (e.g. division by zero) | `optimization.py` turns warnings into errors on purpose, so numerical problems stop the run instead of silently producing a bad surface. |

Always run `python optimization.py <config> --dry-run` first: it checks the
config, the output paths and the collation names without computing anything.
