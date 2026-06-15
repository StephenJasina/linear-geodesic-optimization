import sys

sys.path.insert(0, '.')
from linear_geodesic_optimization import batch

arguments = {
    'alpha': [True, False],
    'beta': [1, 2, 3],
    'gamma': ['x', 'y'],
    'delta': [0.4],
    'epsilon': [7, 8, 9],
}
groups = [
    ['alpha', 'gamma'],
    ['beta', 'epsilon'],
    ['delta'],
]

for index, argument_batch in enumerate(batch.group_arguments(arguments, groups, {'f': 'default'}, order_by='zeta')):
    print(f'Batch {index}:')
    for argument_list in argument_batch:
        print(f'\t{argument_list}')
