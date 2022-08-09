# Linear Geodesic Optimization
Finding a manifold whose geodesic distances are approximately linearly related to some input data. For the details, read the markdown files found in `src`. A good ordering is probably:
* `src/main.md`
* `src/linear_optimization/laplacian.md`
* `src/linear_optimization/geodesic.md`
* `src/linear_optimization/smooth.md`
* `src/linear_optimization/linear_regression.md`
* `src/linear_optimization/partial_selection.md`

## Requirements
The code is written entirely in Python. The required packages are `numpy`, `scipy`, and `plotly`, all of which are found in the Python Package Index.

## File Structure
In detail,
* `src/`
  * `main.py`: The main driver. Given a graph $G$ embedded on a mesh (which itself is embedded in $\mathbb{R}^3$) and a mapping from the edges of $G$ to positive reals, it attempts to modify the mesh so that the values associated to the edges are in an approximate linear relation with the lengths of the corresponding geodesics.
  * `linear_ogeodesic_optimization/`
    * `data/`
      * `phony.py`: Some functions for producing fake latency data. This is mostly used for testing purposes.
    * `mesh/`
      * `mesh.py`: An interface describing a mesh.
      * `sphere.py`: A geodesic sphere mesh with icosahedral symmetry.
    * `optimization/`
      * `laplacian.py`: Utilities to compute the discrete Laplace-Beltrami operator on a mesh and its partial derivatives.
      * `geodesic.py`: Utilities to compute the geodesic distances on a mesh and their partial derivatives. The computations here are based off of [Crane et al's heat method](https://www.cs.cmu.edu/~kmcrane/Projects/HeatMethod/). A demonstration of the distance computation is given in `src/heat_method_example.py`.
      * `smooth.py`: Utilities to compute the smoothness of a mesh and its partial derivatives. Currently, this function is not a great implementation, but seems to work "well enough."
      * `linear_regression.py`: Utilities to compute the least squares error in the linear regression setting and its partial derivatives.
      * `partial_selection.py`: Utilities to speed up the computation of the derivatives of the geodesic derivatives.
      * `optimization.py`: A `class` to group the above computations together. This abstracts the computation of the objective function and its gradient.
      * `standard.py`: A collection of standard optimization algorithms. In particular,
        * A function to find a step size satisfying the [(weak) Wolfe conditions](https://en.wikipedia.org/wiki/Wolfe_conditions).
        * An implementation of steepest descent (a.k.a. gradient descent).
        * An implementation of [LBFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS).
    * `plot.py`: Some useful plotting utilities using `plotly`. This includes a function for 3-d animation.
