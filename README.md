# Linear Geodesic Optimization
Tool for constructing and viewing a manifold whose geodesic distances are approximately linearly related to some input data. Some (incomplete) documentation can be found in the `doc/` directory.

## Requirements
### System Packages
For building things (in particular, the `MeshUtility` package), you'll need `git`, CMake and Ninja installed. As most of the code is written in Python, a working installation (at least version 3.6) must be installed.

### Python Packages
Most of the packages listed below can be installed via `pip install <package name>`. For those that cannot, additional instructions are included.

You'll need the following to get the optimization routine running:
* `networkx`
* `numpy`
* `POT`
* `python-dcel-mesh`. To install this, clone [this repo](https://github.com/StephenJasina/python-dcel-mesh) and run `pip install .` from its root directory.
* `scikit-learn`
* `scipy`

Additionally helpful packages for viewing the data are
* `adjustText`
* `basemap`. For this one, make sure Python is version at most 3.12
* `potpourri3d`

## Usage
From the `src` directory, run `python optimization.py` to run the optimizer. At the end of the file are some parameters that can be changed to modify the optimization.

To run the webapp, run `npx vite` from the `src/site` directory. This will start the server locally.
