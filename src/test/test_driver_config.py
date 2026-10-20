"""Parsing of config files into optimization runs and collation variants."""

import json
import pathlib

import pytest

from linear_geodesic_optimization import driver


def write_config(tmp_path, config):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return pathlib.PurePath(path)


def base_config(**extra):
    return {
        "arguments_optimization": {
            "filename_json": ["a.json", "b.json"],
            "lambda_smooth": [0.0, 0.1],
        },
        "groups_optimization": [["filename_json"], ["lambda_smooth"]],
        "settings": {
            "directory_output": ["toy"],
            "output_format": ["lambda_smooth", "filename_json"],
            "order_by": "filename_json",
        },
        **extra,
    }


def load(tmp_path, config, initialization="sphere"):
    arguments, arguments_collation, settings, defaults = driver.load_config(
        write_config(tmp_path, config)
    )
    output_format = driver.get_output_format(settings, defaults)
    driver.assign_output_directories(
        arguments, settings, output_format, tmp_path / "outputs",
        existence_check="must_not_exist",
    )
    jobs = driver.assign_collation_paths(
        arguments, arguments_collation, settings, defaults, initialization,
        tmp_path / "outputs",
    )
    return arguments, arguments_collation, jobs


def test_without_collation_arguments_uses_one_default_variant(tmp_path):
    arguments, arguments_collation, jobs = load(tmp_path, base_config())

    assert [len(argument_batch) for argument_batch in arguments] == [2, 2]
    assert arguments_collation == [driver.collation_defaults()]
    assert [
        [path.relative_to(tmp_path / "outputs") for _, path in job["variants"]]
        for job in jobs
    ] == [
        [pathlib.Path("toy", "0.0", "a", "animation.json")],
        [pathlib.Path("toy", "0.1", "a", "animation.json")],
    ]
    assert jobs[0]["directories_outputs"] == [
        argument_dict["directory_output"] for argument_dict in arguments[0]
    ]


def test_collation_groups_are_zipped_and_named(tmp_path):
    config = base_config(
        arguments_collation={
            "postprocess": [True, True, False],
            "postprocess_subtract_initialization": [True, False, True],
            "bubble_size": [0.03, 0.05],
        },
        groups_collation=[["postprocess", "postprocess_subtract_initialization"]],
    )
    config["settings"]["output_format_collation"] = [
        ["postprocess", "postprocess_subtract_initialization"], "bubble_size",
    ]
    _, arguments_collation, jobs = load(tmp_path, config)

    assert [
        (variant["postprocess"], variant["postprocess_subtract_initialization"], variant["bubble_size"])
        for variant in arguments_collation
    ] == [
        (True, True, 0.03), (True, True, 0.05),
        (True, False, 0.03), (True, False, 0.05),
        (False, True, 0.03), (False, True, 0.05),
    ]
    assert all(variant["height_scale"] == 0.15 for variant in arguments_collation)
    assert [path.name for _, path in jobs[0]["variants"]] == [
        "animation_True_True_0.03.json", "animation_True_True_0.05.json",
        "animation_True_False_0.03.json", "animation_True_False_0.05.json",
        "animation_False_True_0.03.json", "animation_False_True_0.05.json",
    ]


def test_directory_collation_can_name_by_optimization_parameters(tmp_path):
    config = base_config(arguments_collation={"postprocess": [True, False]})
    config["settings"]["directory_collation"] = ["animations"]
    config["settings"]["output_format_collation"] = ["lambda_smooth", "postprocess"]
    _, _, jobs = load(tmp_path, config)

    assert sorted(
        str(path.relative_to(tmp_path / "outputs"))
        for job in jobs for _, path in job["variants"]
    ) == [
        "animations/animation_0.0_False.json", "animations/animation_0.0_True.json",
        "animations/animation_0.1_False.json", "animations/animation_0.1_True.json",
    ]


def test_seed_runs_are_not_collated(tmp_path):
    arguments, _, jobs = load(tmp_path, base_config(), initialization="first")

    assert jobs[0]["directories_outputs"] == [arguments[0][1]["directory_output"]]
    assert jobs[0]["variants"][0][1].parent == arguments[0][0]["directory_output"]


@pytest.mark.parametrize(
    "settings,message",
    [
        ({}, "duplicated"),  # Two variants, but nothing to tell them apart
        ({"directory_collation": ["animations"], "output_format_collation": ["postprocess"]}, "duplicated"),
        ({"output_format_collation": ["not_a_parameter"]}, "not a valid part"),
    ],
)
def test_bad_collation_paths_raise(tmp_path, settings, message):
    config = base_config(arguments_collation={"postprocess": [True, False]})
    config["settings"].update(settings)
    with pytest.raises(ValueError, match=message):
        load(tmp_path, config)


def test_unknown_collation_parameter_raises(tmp_path):
    with pytest.raises(ValueError, match="Unknown collation parameters: bubble"):
        load(tmp_path, base_config(arguments_collation={"bubble": [0.1]}))


@pytest.mark.parametrize(
    "rename,message",
    [
        ({"arguments_optimization": "arguments"}, 'renamed to "arguments_optimization"'),
        ({"groups_optimization": "groups"}, 'renamed to "groups_optimization"'),
    ],
)
def test_old_keys_raise(tmp_path, rename, message):
    config = base_config()
    for new, old in rename.items():
        config[old] = config.pop(new)
    with pytest.raises(ValueError, match=message):
        load(tmp_path, config)


@pytest.mark.parametrize("key", ["postprocess", "postprocess_subtract_initialization"])
def test_old_postprocess_settings_raise(tmp_path, key):
    config = base_config()
    config["settings"][key] = True
    with pytest.raises(ValueError, match='moved to "arguments_collation"'):
        load(tmp_path, config)
