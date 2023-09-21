import os

from matplotlib import pyplot as plt
import networkx as nx

from linear_geodesic_optimization.plot import get_network_plot

cutoff = 7
data_file_name = os.path.join('graph_Europe_clustered', f'graph{cutoff}.graphml')
data_file_path = os.path.join('..', 'data', data_file_name)
network_plot = get_network_plot(nx.graphml.read_graphml(data_file_path))
network_plot.savefig('network.png', dpi=1000, bbox_inches='tight')
plt.show()
