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

To run the webapp, run `npx vite` from the `src/site` directory. This will start the server locally and display a link to the page in the console. The animation can be viewed by dragging and dropping the JSON file produced by the collation process.
