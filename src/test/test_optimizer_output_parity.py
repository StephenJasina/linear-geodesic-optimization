"""End-to-end output parity between the legacy and Torch optimizers."""

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest


GRAPHML = Path(__file__).parent / "fixtures" / "optimization_parity.graphml"


@pytest.mark.parametrize(
    "sides,lambda_smooth", [(8, 0.0), (8, 0.005), (12, 0.005), (20, 0.005)]
)
@pytest.mark.parametrize(
    "curvature_scale,curvature_offset",
    [(0.0, 0.0), (1.0, 0.0), (0.0, 0.3), (0.0, -0.3)],
    ids=["flat", "mixed", "positive", "negative"],
)
def test_optimized_output_matches_legacy(
    tmp_path, monkeypatch, sides, lambda_smooth, curvature_scale, curvature_offset
):
    # POT needs only its NumPy backend here. Avoid importing an unrelated,
    # potentially incompatible TensorFlow installation during graph loading.
    monkeypatch.setenv("POT_BACKEND_DISABLE_TENSORFLOW", "1")
    pytest.importorskip("torch")
    pytest.importorskip("ot")
    from optimization import optimize

    graph = nx.read_graphml(GRAPHML)
    for _, _, edge in graph.edges(data=True):
        edge["ricciCurvature"] = (
            curvature_scale * edge["ricciCurvature"] + curvature_offset
        )
    graphml = tmp_path / "curvature_goal.graphml"
    nx.write_graphml(graph, graphml)

    outputs = {}
    snapshots = {}
    for backend in ("legacy", "torch"):
        directory = tmp_path / backend
        optimize(
            filename_graphml=graphml,
            initial_radius=80.0,
            sides=sides,
            mesh_scale=0.25,
            coordinates_scale=0.8,
            lambda_curvature=1.0,
            lambda_smooth=lambda_smooth,
            directory_output=directory,
            maxiter=5,
            backend=backend,
        )
        outputs[backend] = json.loads((directory / "output.json").read_text())
        snapshots[backend] = json.loads((directory / "0.json").read_text())

    legacy, fast = outputs["legacy"], outputs["torch"]
    legacy_parameters = {k: v for k, v in legacy["parameters"].items() if k != "backend"}
    fast_parameters = {k: v for k, v in fast["parameters"].items() if k != "backend"}
    assert legacy_parameters == fast_parameters
    assert legacy["network"] == fast["network"]
    assert legacy["routes"] == fast["routes"]
    assert legacy["traffic"] == fast["traffic"]
    assert legacy["initial"] == fast["initial"]
    np.testing.assert_allclose(fast["final"], legacy["final"], rtol=1e-7, atol=1e-7)

    legacy_snapshot, fast_snapshot = snapshots["legacy"], snapshots["torch"]
    assert legacy_snapshot["mesh_parameters"] == fast_snapshot["mesh_parameters"]
    for key in ("L_curvature", "L_smooth"):
        np.testing.assert_allclose(fast_snapshot[key], legacy_snapshot[key], rtol=1e-9, atol=1e-9)
