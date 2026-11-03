"""Checks for collation's vertical centering of frames."""

import json
import pathlib

import numpy as np
import pytest

pytest.importorskip('torch')

import collation
import optimization


TOY_DIRECTORY = (
    pathlib.Path(__file__).parents[2]
    / 'data' / 'toy' / 'routing_with_volumes' / 'traffic' / 'reroute_link_outage'
)
SIDES = 12
SCALE = 0.25


def test_center_heights():
    z = np.array([1., 2., 6.])
    np.testing.assert_allclose(collation.center_heights(z, 'mean'), [-2., -1., 3.])
    np.testing.assert_allclose(collation.center_heights(z, 'min'), [0., 1., 5.])
    np.testing.assert_allclose(collation.center_heights(z, None), z)
    np.testing.assert_allclose(collation.center_heights(z, 'min', np.array([1, 2])), [-1., 0., 4.])
    with pytest.raises(ValueError):
        collation.center_heights(z, 'median')


@pytest.fixture(scope='module')
def batch_directories(tmp_path_factory):
    directory = tmp_path_factory.mktemp('collation')
    directories = []
    for index, name in enumerate(['graph_0.json', 'graph_5.json']):
        directory_output = directory / str(index)
        optimization.optimize(
            filename_json=TOY_DIRECTORY / name, latency_threshold=4,
            clustering_distance=500000, ricci_curvature_reweight='distribution',
            ricci_curvature_distribution_reweight_scale=2., lambda_smooth=0.005,
            sides=SIDES, mesh_scale=SCALE, directory_output=directory_output,
            method='ridges', ridge_height_reference=5.,
        )
        directories.append(directory_output)
    return directory, directories


def collate(batch_directories, **arguments):
    directory, directories = batch_directories
    path = directory / f'animation_{len(list(directory.glob("animation_*")))}.json'
    collation.collate_outputs(directories, [(arguments, path)])
    with open(path) as file:
        return json.load(file)


def finals(batch_directories):
    _, directories = batch_directories
    return [
        np.array(json.load(open(directory / 'output.json'))['final'])
        for directory in directories
    ]


def test_no_centering_keeps_offsets(batch_directories):
    animation = collate(batch_directories, postprocess=False, vertical_centering=None)
    for frame, z in zip(animation['animation'], finals(batch_directories)):
        np.testing.assert_allclose(np.array(frame['height']).ravel(), z / SCALE)


@pytest.mark.parametrize('mode, reduce', [('min', np.min), ('mean', np.mean)])
def test_centering_without_postprocessing(batch_directories, mode, reduce):
    animation = collate(batch_directories, postprocess=False, vertical_centering=mode)
    for frame, z in zip(animation['animation'], finals(batch_directories)):
        np.testing.assert_allclose(np.array(frame['height']).ravel(), (z - reduce(z)) / SCALE, atol=1e-12)


def test_centering_applies_to_originals_with_postprocessing(batch_directories):
    animation = collate(
        batch_directories, postprocess=True, vertical_centering=None,
        postprocess_subtract_initialization=False,
    )
    for frame, z in zip(animation['animation'], finals(batch_directories)):
        np.testing.assert_allclose(np.array(frame['original']['height']).ravel(), z / SCALE)
