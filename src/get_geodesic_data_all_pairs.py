import argparse
import csv

from linear_geodesic_optimization import data
from linear_geodesic_optimization.optimization.geodesic import Computer as Geodesic

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-directory', '-i', type=str, required=True,
                        dest='input_directory', metavar='<filename>',
                        help='Input file')
    parser.add_argument('--output-file', '-o', type=str, required=True,
                        dest='output_filename', metavar='<filename>',
                        help='Output file')
    args = parser.parse_args()
    input_directory = args.input_directory
    output_filename = args.output_filename

    mesh = data.get_mesh_output(input_directory, postprocessed=False)
    n = mesh.get_topology().n_vertices()

    with open(output_filename, 'w') as output_file:
        csv_writer = csv.writer(output_file)
        for u in range(n):
            for v in range(n):
                print((u, v))
                geodesic = Geodesic(mesh, u, v)
                geodesic.forward()
                csv_writer.writerow([u, v, geodesic.distance])
