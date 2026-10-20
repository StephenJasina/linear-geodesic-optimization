import itertools
import os
import pathlib
import typing

import linear_geodesic_optimization.batch as batch


# List of parameters that correspond to extant files. This is used to
# help generate output directory names (by removing extensions when they
# would be unneeded) and determining actual locations of input files (by
# prepending ../data)
parameter_name_filenames = [
    'filename_probes',
    'filename_links',
    'filename_graphml',
    'filename_json',
]

def argument_to_string(
    arguments,
    parameter_name
):
    if parameter_name in parameter_name_filenames:
        return pathlib.PurePath(arguments[parameter_name]).stem
    else:
        return str(arguments[parameter_name])

def base_defaults() -> dict[str, typing.Any]:
    """
    Return a fresh copy of the default arguments shared by the
    optimization and collation config file formats.
    """
    return {
        'filename_probes': None,
        'filename_links': None,
        'filename_graphml': None,
        'filename_json': None,
        'latency_threshold': None,
        'clustering_distance': None,
        'ricci_curvature_alpha': 0.,
        'ricci_curvature_reweight': None,
        'ricci_curvature_distribution_reweight_scale': 0.,
        'lambda_curvature': 1.,
        'lambda_smooth': 0.,
        'smooth_loss_type': 'mvs',
        'initial_radius': None,  # None is an infinite radius (flat)
        'sides': 50,
        'mesh_scale': 1.,
        'coordinates_scale': 0.8,
        'network_trim_radius': None,
        'method': 'construction',  # Or 'lbfgs'
        'backend': 'torch',  # Or 'legacy'
        'maxiter': None,  # Used only when method is 'lbfgs'
        'initialization_file_path': None,  # Ignored when method is 'construction'
        # Parameters used only when method is 'construction'
        'hessian_anisotropy': 1.,
        'hessian_orientation': 'hill',
        'hessian_regularization': 1e-3,
        'hessian_tube_radius': None,
        'hessian_iterations': 5,
        'hessian_slope_correction': False,
        'hessian_smoothness': 0.,
        'hessian_polish_maxiter': 0,
        'hessian_initialization': None,
        'index': None # Additional unique ID
    }

def collation_defaults() -> dict[str, typing.Any]:
    """
    Return a fresh copy of the default collation arguments (the keyword
    options of `collation.render_animation`).
    """
    return {
        'postprocess': True,
        'postprocess_subtract_initialization': True,
        'height_scale': 0.15,
        'bubble_size': 0.03,
        'exact_geodesics': False,
        'use_convex_hull': False,
    }

def resolve_input_file_paths(
    arguments: list[list[dict]],
    settings: dict,
    config_file: pathlib.PurePath,
    data_directory: str | os.PathLike[str] = pathlib.PurePath('..', 'data'),
) -> None:
    """
    Prepend a data directory to every filename-valued parameter (see
    `parameter_name_filenames`) on every argument dict, mutating them in
    place.

    The data directory is derived from `settings['directory_data_type']`
    (one of 'data', 'script', or the default 'config') and, if present,
    `settings['directory_data']`.
    """
    directory_data_type = (
        'config' if 'directory_data_type' not in settings else
        settings['directory_data_type']
    )
    directory_data_parent = (
        data_directory if directory_data_type == 'data' else
        pathlib.PurePath() if directory_data_type == 'script' else
        config_file.parent
    )
    directory_data = directory_data_parent / (pathlib.PurePath(*settings['directory_data']) if 'directory_data' in settings else '')
    for argument_batch in arguments:
        for argument_dict in argument_batch:
            for parameter_name in parameter_name_filenames:
                if argument_dict[parameter_name] is not None:
                    if isinstance(argument_dict[parameter_name], list):
                        to_add = pathlib.PurePath(*argument_dict[parameter_name])
                    else:
                        to_add = argument_dict[parameter_name]
                    argument_dict[parameter_name] = directory_data / to_add

def load_config(
    config_file: pathlib.PurePath,
    data_directory: str | os.PathLike[str] = pathlib.PurePath('..', 'data'),
) -> tuple[list[list[dict]], list[dict], dict, dict]:
    """
    Read a JSON config file and produce its arguments, collation
    arguments, and settings.

    Input file paths (see `parameter_name_filenames`) are resolved
    relative to the appropriate data directory as a side effect. Raise
    if a collation argument isn't a known collation parameter. Returns
    `(arguments, arguments_collation, settings, defaults)`.
    """
    defaults = base_defaults()
    defaults_collation = collation_defaults()
    with open(config_file, 'r') as f:
        arguments, arguments_collation, settings = batch.parse_json(
            f, defaults, default_arguments_collation=defaults_collation
        )
    unknown = sorted({
        name
        for argument_dict in arguments_collation
        for name in argument_dict
    } - set(defaults_collation))
    if unknown:
        raise ValueError(f'Unknown collation parameters: {", ".join(unknown)}')
    resolve_input_file_paths(arguments, settings, config_file, data_directory=data_directory)
    return arguments, arguments_collation, settings, defaults

def validate_output_format(output_format, valid_names) -> None:
    """
    Raise if `output_format` isn't a list of (strings or lists of
    strings), each of which must be in `valid_names`.
    """
    if not isinstance(output_format, list):
        raise TypeError('Output format must be a list of (strings or lists of strings)')
    for output_format_part in output_format:
        if isinstance(output_format_part, str):
            if output_format_part not in valid_names:
                raise ValueError(f'{output_format_part} not a valid part of output format')
        elif isinstance(output_format_part, list):
            for output_format_part_part in output_format_part:
                if isinstance(output_format_part_part, str):
                    if output_format_part_part not in valid_names:
                        raise ValueError(f'{output_format_part_part} not a valid part of output format')
                else:
                    raise TypeError('Output format must be a list of (strings or lists of strings)')
        else:
            raise TypeError('Output format must be a list of (strings or lists of strings)')

def format_output_name(arguments, output_format) -> list[str]:
    """
    Return the name parts described by `output_format` for the given
    arguments. A part that is a list of parameter names becomes their
    values joined with '_'.
    """
    return [
        argument_to_string(arguments, output_format_part) if isinstance(output_format_part, str) else
        '_'.join([argument_to_string(arguments, output_format_part_part) for output_format_part_part in output_format_part])
        for output_format_part in output_format
    ]

def get_output_format(
    settings: dict,
    defaults: dict,
    *,
    validate: bool = False
) -> list:
    """
    Determine the output format used to name output directories.

    If `validate` is true, raise if `output_format` isn't a list of
    (strings or lists of strings), each of which must be a valid
    parameter name.
    """
    output_format = settings['output_format'] if 'output_format' in settings else ['filename_json']
    if validate:
        validate_output_format(output_format, defaults)
    return output_format

def assign_output_directories(
    arguments: list[list[dict]],
    settings: dict,
    output_format: list,
    output_directory: str | os.PathLike[str] = pathlib.PurePath('..', 'outputs'),
    *,
    existence_check: typing.Literal['must_not_exist', 'must_exist'],
) -> None:
    """
    Assign a unique `index` and a `directory_output` to every argument
    dict, mutating them in place. `index` is a running counter across
    all argument batches, so it is unique even when `output_format`
    includes it.

    If `existence_check` is 'must_not_exist', raise if a computed
    directory already exists (for a script about to create fresh
    output). If it is 'must_exist', raise if it doesn't (for a script
    reading already-produced output).
    """
    index = 0
    for argument_batch in arguments:
        for argument_dict in argument_batch:
            argument_dict['index'] = index
            # Also prepend ../outputs (which is the path of the outputs
            # directory relative to the script)
            directory_output = output_directory / pathlib.PurePath(*(
                settings['directory_output'] + format_output_name(argument_dict, output_format)
            ))
            if existence_check == 'must_not_exist':
                if os.path.exists(directory_output):
                    raise ValueError(f'{str(directory_output)} already exists')
            elif existence_check == 'must_exist':
                if not os.path.exists(directory_output):
                    raise ValueError(f'{str(directory_output)} does not exist')
            else:
                raise ValueError(f'Invalid existence_check "{existence_check}"')
            argument_dict['directory_output'] = directory_output
            index += 1

def check_no_duplicate_output_directories(arguments: list[list[dict]]) -> None:
    """
    Raise if any two argument dicts across all batches were assigned the
    same `directory_output`.
    """
    # TODO: This can probably be improved by not constructing a set
    if len(set(str(argument_dict['directory_output']) for argument_batch in arguments for argument_dict in argument_batch)) != sum(len(argument_batch) for argument_batch in arguments):
        raise ValueError('Some output directories are duplicated')

def assign_collation_paths(
    arguments: list[list[dict]],
    arguments_collation: list[dict],
    settings: dict,
    defaults: dict,
    initialization: str,
    output_directory: str | os.PathLike[str] = pathlib.PurePath('..', 'outputs'),
) -> list[dict]:
    """
    Determine the collation jobs: one per argument batch, each writing
    one animation file per collation variant. `assign_output_directories`
    must have been called on `arguments` first.

    Each animation is named 'animation.json' if
    `settings['output_format_collation']` is empty (the default), and
    'animation_<parts>.json' otherwise, where the parts follow the same
    format as `settings['output_format']`. The parts may name collation
    parameters, or optimization parameters (whose values are taken from
    the batch's first run). The files go in the batch's first run
    directory, or in `settings['directory_collation']` (under
    `output_directory`) if it is set.

    Raise if two animations would be written to the same path. Return a
    list of dicts with keys `directories_outputs` (the runs to collate,
    excluding seeds) and `variants` (a list of `(collation arguments,
    animation path)` pairs).
    """
    output_format = settings['output_format_collation'] if 'output_format_collation' in settings else []
    validate_output_format(output_format, set(defaults) | set(collation_defaults()))

    jobs = []
    paths = set()
    for argument_batch in arguments:
        if 'directory_collation' in settings:
            directory = output_directory / pathlib.PurePath(*settings['directory_collation'])
        else:
            directory = argument_batch[0]['directory_output']
        variants = []
        for argument_dict_collation in arguments_collation:
            name_parts = format_output_name(argument_batch[0] | argument_dict_collation, output_format)
            path = directory / ('_'.join(['animation'] + name_parts) + '.json')
            if path in paths:
                raise ValueError(f'Collation path {path} is duplicated')
            paths.add(path)
            variants.append((argument_dict_collation, path))
        jobs.append({
            'directories_outputs': [
                argument_dict['directory_output']
                for argument_dict in (argument_batch[1:] if initialization_excludes_seed(initialization) else argument_batch)
            ],
            'variants': variants,
        })
    return jobs

def get_initialization(settings: dict) -> str:
    """
    Return the initialization strategy ('sphere', 'sequential', or
    'first') specified by settings, defaulting to 'sphere'.
    """
    return settings['initialization'] if 'initialization' in settings else 'sphere'

def initialization_excludes_seed(initialization: str) -> bool:
    """
    Return whether the given initialization strategy uses the first
    element of each argument batch purely as a seed, to be excluded from
    further processing of the batch.
    """
    return initialization == 'first'

def assign_initialization_file_paths(
    arguments: list[list[dict]],
    initialization: str
) -> None:
    """
    Set `initialization_file_path` on every argument dict, mutating them
    in place, according to the given initialization strategy:
    - 'sphere': no argument dict is seeded by another's output.
    - 'sequential': each argument dict (other than the first in its
      batch) is seeded by the previous argument dict's output.
    - 'first': each argument dict (other than the first in its batch) is
      seeded by the first argument dict's output.
    """
    if initialization == 'sphere':
        for argument_batch in arguments:
            for argument_dict in argument_batch:
                argument_dict['initialization_file_path'] = None
    elif initialization == 'sequential':
        for argument_batch in arguments:
            argument_batch[0]['initialization_file_path'] = None
            for argument_dict_previous, argument_dict in itertools.pairwise(argument_batch):
                argument_dict['initialization_file_path'] = argument_dict_previous['directory_output'] / 'output.json'
    elif initialization == 'first':
        for argument_batch in arguments:
            initialization_file_path = argument_batch[0]['directory_output'] / 'output.json'
            for argument_dict in argument_batch[1:]:
                argument_dict['initialization_file_path'] = initialization_file_path
    else:
        raise ValueError(f'Invalid initialization strategy "{initialization}"')

def dispatch_optimization_batches(
    initialization: str,
    arguments: list[list[dict]],
    item_fn: typing.Callable,
    n_cores: typing.Optional[int],
) -> None:
    """
    Run `item_fn` over every argument dict in `arguments`, using
    multiprocessing where possible, in the shape required by the given
    initialization strategy:
    - 'sphere': every argument dict is independent, so all run flat.
    - 'sequential': each batch must run its argument dicts in order (so
      that each one's seed file is written before the next starts), but
      batches are independent of one another.
    - 'first': the first argument dict of each batch must run (and
      write its seed file) before the rest of that batch's argument
      dicts, but otherwise everything is independent.
    """
    if initialization == 'sphere':
        batch.run_multiprocessed(item_fn, itertools.chain(*arguments), n_cores)
    elif initialization == 'sequential':
        batch.run_multiprocessed(
            batch.run_sequential,
            [
                {
                    'f': item_fn,
                    'arguments': argument_batch,
                }
                for argument_batch in arguments
            ],
            n_cores=n_cores
        )
    elif initialization == 'first':
        batch.run_multiprocessed(
            item_fn,
            [
                argument_batch[0]
                for argument_batch in arguments
            ],
            n_cores=n_cores
        )
        batch.run_multiprocessed(
            item_fn,
            itertools.chain(*[
                argument_batch[1:]
                for argument_batch in arguments
            ]),
            n_cores
        )
    else:
        raise ValueError(f'Invalid initialization strategy "{initialization}"')
