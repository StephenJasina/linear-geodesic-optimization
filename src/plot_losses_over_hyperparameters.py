import os
import pickle
import sys

from matplotlib import pyplot as plt


plt.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
})

max_iterations = 1000

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python3 <directory name>')
        sys.exit(0)

    directory = sys.argv[1]

    lambda_geodesics = []
    L_geodesics = []
    L_smooths = []
    L_curvatures = []

    for subdirectory in os.scandir(directory):
        if not subdirectory.is_dir():
            continue

        with open(os.path.join(subdirectory, 'parameters'), 'rb') as f:
            parameters = pickle.load(f)
            lambda_geodesics.append(parameters['lambda_geodesic'])

        iteration = min(
            max_iterations,
            max(int(name) for name in os.listdir(subdirectory) if name.isdigit())
        )
        with open(os.path.join(subdirectory, str(iteration)), 'rb') as f:
            iteration_data = pickle.load(f)
            L_geodesics.append(iteration_data['L_geodesic'])
            L_curvatures.append(iteration_data['L_curvature'])
            L_smooths.append(iteration_data['L_smooth'])

    x, y = zip(*sorted(zip(lambda_geodesics, L_curvatures)))
    plt.plot(x, y, 'b-', label='Curvature Loss')
    x, y = zip(*sorted(zip(lambda_geodesics, L_geodesics)))
    plt.plot(x, y, 'r-', label='Geodesic Loss')
    plt.title(f'{max_iterations} Iterations')
    plt.xlabel('$\lambda_{\mathrm{geodesic}}$')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join('stuff', f'loss_{max_iterations}.png'), dpi=500)
