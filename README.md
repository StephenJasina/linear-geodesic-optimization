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
* `scipy`
* `python-dcel-mesh`. To install this, clone [this repo](https://github.com/StephenJasina/python-dcel-mesh) and run `pip install .` from its root directory.
* `MeshUtility`. For this one, follow the [build instructions](https://github.com/zishun/MeshUtility/blob/main/build.md). Note that the `pybind11` submodule in `ext/pybind11` is out of date and currently causes compilation to fail. To update it, just run `cd ext/pybind11` followed by `git checkout master`. You'll also need `cmake` and `ninja` installed.

For the webapp, the following will additionally be required:
* `flask`
* `cvxpy`
* `potpourri3d`

## Usage
From the `src` directory, run `python optimization.py` to run the optimizer. At the end of the file are some parameters that can be changed to modify the optimization.

To run the webapp, run `python site/app.py`. This will start the server locally at `localhost:5000`.
