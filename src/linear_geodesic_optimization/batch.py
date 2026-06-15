import collections.abc
import concurrent.futures
import functools
import itertools
import json
import os
import pathlib
import traceback
import typing


def group_arguments(
    arguments: dict[str, list[typing.Any]],
    groups: list[list[str]],
    defaults: dict[str, typing.Any] = {},
    pass_arguments: dict[str ,str] = {},
    order_by: typing.Optional[str] = None
) -> list[list[dict[str, typing.Any]]]:
    """
    Convert arguments to actual argument lists.

    In the case that many runs of the same function are to be done with
    only some of the arguments changing, it is useful not to have to
    repeatedly specify the unchanging arguments (for both brevity and
    clarity).

    This function takes in a dictionary `arguments` of arguments
    (mapping parameter names to lists of values).

    To achieve the desired de-duplication of specifications, the
    `groups` parameter describes a partition of the parameter names.
    Each set in the partition refers to parallel lists in `arguments`.
    The function returns all combinations of arguments choosing from
    these sets of parallel lists.

    For example, if `arguments` is
    ```
    {
        'a': [True, False],
        'b': [1, 2, 3],
        'c': ['x', 'y'],
        'd': [0.4],
        'e': [7, 8, 9],
    }
    ```
    and `groups` is
    ```
    [
        ['a', 'c'],
        ['b', 'e'],
        ['d'],
    ]
    ```
    then the function will return
    ```
    [
        [{'a': True,  'b': 1, 'c': 'x', 'd': 0.4, 'e': 7}],
        [{'a': True,  'b': 2, 'c': 'x', 'd': 0.4, 'e': 8}],
        [{'a': True,  'b': 3, 'c': 'x', 'd': 0.4, 'e': 9}],
        [{'a': False, 'b': 1, 'c': 'y', 'd': 0.4, 'e': 7}],
        [{'a': False, 'b': 2, 'c': 'y', 'd': 0.4, 'e': 8}],
        [{'a': False, 'b': 3, 'c': 'y', 'd': 0.4, 'e': 9}],
    ]
    ```

    To make defining `groups` easier, one of the parts of the partition
    may be omitted. Additionally, some arguments may be omitted from
    `arguments` in favor of using a default value found in `defaults`.

    The returned arguments are partitioned according to `order_by`,
    which is an parameter name. This parameter name specifies a (unique)
    argument group. The partitions are then the sets of argument
    settings with all arguments constant *except* those in the specified
    argument group. The elements within the partitions are ordered in
    the same way as in the argument groups. Continuing the above
    example, if `order_by` is set to `'c'`, then the return would then
    be
    ```
    [
        [
            {'a': True,  'b': 1, 'c': 'x', 'd': 0.4, 'e': 7},
            {'a': False, 'b': 1, 'c': 'y', 'd': 0.4, 'e': 7},
        ],
        [
            {'a': True,  'b': 2, 'c': 'x', 'd': 0.4, 'e': 8},
            {'a': False, 'b': 2, 'c': 'y', 'd': 0.4, 'e': 8},
        ],
        [
            {'a': True,  'b': 3, 'c': 'x', 'd': 0.4, 'e': 9},
            {'a': False, 'b': 3, 'c': 'y', 'd': 0.4, 'e': 9},
        ],
    ]
    ```
    If `order_by` is `None`, then the returned partitions are all
    singletons.

    Finally, if certain arguments need to be passed from one list to the
    next (e.g., for a sequential set of runs of a function where the
    output of one run is used as input for the subsequent run),
    `pass_arguments` can be used to define source-destination pairs for
    these parameter names. The initial passed arguments will be `None`.
    """
    # Check validity of groups
    unseen_parameter_names = set(arguments.keys())
    order_by_index = None
    for index, group in enumerate(groups):
        if len(group) == 0:
            raise ValueError('Invalid partition')
        for parameter_name in group:
            if parameter_name not in unseen_parameter_names:
                raise ValueError('Invalid partition')
            if parameter_name == order_by:
                order_by_index = index
            unseen_parameter_names.remove(parameter_name)
    if len(unseen_parameter_names) != 0:
        # Allow for one element of the partition to be elided
        groups = groups + [list(unseen_parameter_names)]
        if order_by in unseen_parameter_names:
            order_by_index = len(groups) - 1
    # Check that order_by is valid
    if order_by is not None and order_by_index is None:
        raise ValueError('order_by is not a valid parameter name')
    for group in groups:
        length = len(arguments[group[0]])
        for element in group[1:]:
            if len(arguments[element]) != length:
                raise ValueError('Uneven argument list lengths within group')
    # Check validity of sequential argument passing
    for parameter_source, parameter_destination in pass_arguments.items():
        if parameter_source not in arguments.keys():
            raise ValueError(f'Parameter {parameter_source} cannot be passed on')
        if parameter_destination in arguments.keys():
            raise ValueError(f'Parameter destination {parameter_destination} already occupied')

    # Re-sort the groups
    length_batch = 1
    if order_by_index is not None:
        groups = groups[:order_by_index] + groups[order_by_index+1:] + [groups[order_by_index]]
        length_batch = len(arguments[groups[-1][0]])

    # Construct partial argument lists. These are what the lists of
    # arguments would be if we restrict our attention to individual
    # groups
    argument_sublists = [
        [
            {
                parameter_name: argument_value
                for parameter_name, argument_value in zip(group, argument_tuple)
            } | {
                parameter_name_destination: argument_destination_value
                for parameter_name_source, argument_destination_value  in zip(group, argument_tuple_previous)
                if parameter_name_source in pass_arguments
                for parameter_name_destination in (pass_arguments[parameter_name_source],)
            }
            for argument_tuples in (list(zip(*[
                arguments[element]
                for element in group
            ])),) # List of tuples of arguments within group
            for argument_tuple, argument_tuple_previous in zip(
                argument_tuples,
                itertools.chain([tuple(None for _ in range(len(group)))], argument_tuples)
            )
        ]
        for group in groups
    ]

    # Combine the partial argument lists
    return [
        list(batch)
        for batch in itertools.batched([
            functools.reduce(lambda x, y: x | y, split_argument_list, defaults)
            for split_argument_list in itertools.product(*argument_sublists)
        ], length_batch)
    ]

def parse_json(
    f: str | bytes | os.PathLike,
    default_arguments: dict[str, typing.Any]={},
    default_settings: dict[str, typing.Any]={}
) -> tuple[list[dict[str, typing.Any]], dict[str, typing.Any]]:
    """
    Convert a JSON file into a list of argument lists and other settings.

    As input, take a path to a file. Also, take default arguments (that can be
    grouped together) and default settings (that are separate from the
    arguments and their groups).

    Return a tuple of the arguments and settings after parsing the file.
    """
    json_blob = json.load(f)
    settings = default_settings | json_blob['settings'] if 'settings' in json_blob else {}
    arguments = group_arguments(
        json_blob['arguments'],
        json_blob['groups'] if 'groups' in json_blob else [],
        default_arguments,
        pass_arguments=settings['pass_arguments'] if 'pass_arguments' in settings else {},
        order_by=settings['order_by'] if 'order_by' in settings else None
    )
    return arguments, settings

def run_multiprocessed(
    f: collections.abc.Callable,
    arguments: typing.Iterable[dict[str, typing.Any]],
    n_cores: typing.Optional[int]=None
) -> None:
    """
    Run a function with many sets of arguments.

    `f` is run in parallel multiple times, once for each element of
    `arguments`. As arguments, each element `arguments` is used as `f`'s
    kwargs.
    """
    if n_cores is None:
        n_cores = max(len(os.sched_getaffinity(0)) - 2, 1)

    with concurrent.futures.ProcessPoolExecutor(n_cores) as executor:
        futures = []
        for argument_list in arguments:
            future = executor.submit(f, **argument_list)
            futures.append(future)
        futures = concurrent.futures.wait(futures, return_when='FIRST_EXCEPTION')
        for future in futures.done:
            try:
                future.result()
            except Exception as exc:
                print(traceback.format_exc())
    return [future.result() for future in futures.done]

def run_sequential(
    f: collections.abc.Callable,
    arguments: typing.Iterable[dict[str, typing.Any]],
) -> None:
    """
    Run a function with many sets of arguments.

    `f` is run in parallel multiple times, once for each element of
    `arguments`. As arguments, each element `arguments` is used as `f`'s
    kwargs.
    """
    returns = []
    for argument_list in arguments:
        returns.append(f(**argument_list))
    return returns
