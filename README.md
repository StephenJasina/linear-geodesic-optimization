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

The optional `backend` argument selects the optimizer implementation. The
`legacy` backend uses the original Python derivative code. The `torch` backend
uses array operations and automatic differentiation and requires PyTorch
(`pip install torch`). The example traffic-two-clusters config selects `torch`.
Both backends use the same objective and output format. The torch backend
uses one CPU thread per optimization process to avoid thread oversubscription.

### Verify and benchmark the optimizer

From `src`, run the parity tests and the repeatable benchmark:

```sh
python -m pytest test/test_torch_optimization.py test/test_optimizer_output_parity.py
python benchmark_optimization.py --sides 20 --maxiter 5 --repeats 3
```

The tests compare legacy and Torch losses and gradients, then run both through
`optimization.py` on the same small GraphML fixture. The network, input, and
initial heights must match exactly; final heights and saved diagnostic losses
must agree within numerical tolerances. The only expected metadata difference
is the `backend` name. The tests cover several mesh sizes and flat, mixed,
positive, and negative curvature targets.

The benchmark warms up each backend, alternates run order, reports median
end-to-end times and their ratio, and fails if final heights differ by more
than its `--atol` threshold. Its default four-node fixture isolates optimizer
runtime from route or curvature preprocessing. Use `--graphml PATH`, `--sides`,
and `--maxiter` to try another GraphML input or mesh size. A five-iteration
benchmark measures a fixed workload; it does not measure time to convergence.
Use `--curvature-scale` and `--curvature-offset` to change every edge's target
before either backend runs, according to `target = scale * original + offset`.
For example, `--sides 30 --curvature-scale 2` tests a larger mesh with twice
the fixture's curvature targets.

To run the webapp, run `npx vite` from the `src/site` directory. This will start the server locally and display a link to the page in the console. The animation can be viewed by dragging and dropping the JSON file produced by the collation process.
