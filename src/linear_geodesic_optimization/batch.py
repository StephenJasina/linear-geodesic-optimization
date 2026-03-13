import concurrent.futures
import functools
import itertools
import json
import os
import pathlib
import traceback
import typing


def group_parameters(
    settings: typing.Dict[str, typing.List[typing.Any]],
    groups: typing.List[typing.List[str]],
    defaults: typing.Dict[str, typing.Any]={}
):
    """
    Convert settings to actual parameter lists.

    In the case that many runs of the same function are to be done with
    only some of the parameters changing, it is useful not to have to
    repeatedly specify the unchanging parameters (for both brevity and
    clarity).

    This function takes in a dictionary of `settings` of parameters
    (mapping parameter names to lists of values).

    To achieve the desired de-duplication of specifications, the
    `groups` parameter describes a partition of the parameter names.
    Each set in the partition refers to parallel lists in `settings`.
    The function returns all combinations of parameter settings choosing
    from these sets of parallel lists.

    For example, if `settings` is
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
        {'a': True, 'b': 1, 'c': 'x', 'd': 0.4, 'e': 7},
        {'a': True, 'b': 2, 'c': 'x', 'd': 0.4, 'e': 8},
        {'a': True, 'b': 3, 'c': 'x', 'd': 0.4, 'e': 9},
        {'a': False, 'b': 1, 'c': 'y', 'd': 0.4, 'e': 7},
        {'a': False, 'b': 2, 'c': 'y', 'd': 0.4, 'e': 8},
        {'a': False, 'b': 3, 'c': 'y', 'd': 0.4, 'e': 9},
    ]
    ```

    To make defining `groups` easier, one of the parts of the partition
    may be omitted. Additionally, some parameters may be omitted from
    `settings` in favor of using a default value found in `defaults`.
    """
    # Check validity of groups
    unseen_parameter_names = set(settings.keys())
    for group in groups:
        if len(group) == 0:
            raise ValueError('Invalid partition')
        for parameter_name in group:
            if parameter_name not in unseen_parameter_names:
                raise ValueError('Invalid partition')
            unseen_parameter_names.remove(parameter_name)
    if len(unseen_parameter_names) != 0:
        groups = groups + [list(unseen_parameter_names)]
    for group in groups:
        length = len(settings[group[0]])
        for element in group[1:]:
            if len(settings[element]) != length:
                raise ValueError('Uneven parameter list lengths within group')

    # Construct partial parameter lists
    parameter_sublists = [
        [
            {
                parameter_name: parameter_value
                for parameter_name, parameter_value in zip(group, parameter_tuple)
            }
            for parameter_tuple in zip(*[
                settings[element]
                for element in group
            ])
        ]
        for group in groups
    ]

    # Combine the partial parameter lists
    return [
        functools.reduce(lambda x, y: x | y, split_parameter_list, defaults)
        for split_parameter_list in itertools.product(*parameter_sublists)
    ]


def parse_json(f, defaults={}):
    """
    Convert a JSON file into a list of parameter lists.
    """
    json_blob = json.load(f)
    return group_parameters(
        json_blob['settings'],
        json_blob['groups'] if 'groups' in json_blob else [],
        defaults
    )

def run_multiprocessed(f, parameters, n_cores=None):
    if n_cores is None:
        n_cores = max(len(os.sched_getaffinity(0)) - 2, 1)

    with concurrent.futures.ProcessPoolExecutor(n_cores) as executor:
        futures = []
        for parameter_list in parameters:
            future = executor.submit(f, **parameter_list)
            futures.append(future)
        futures = concurrent.futures.wait(futures, return_when = 'FIRST_EXCEPTION')
        for future in futures.done:
            try:
                future.result()
            except Exception as exc:
                print(traceback.format_exc())
