import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import os
import shutil
from collections import Counter
import networkx as nx
import math
import pyvis
import matplotlib

random.seed(21)

number_of_nodes = 10

socres = [random.randint(0, number_of_nodes*2) for i in range(number_of_nodes)]
socres = np.array(socres)
graph = nx.barabasi_albert_graph(number_of_nodes, 2)

# assign scores to nodes based on degree
degree = nx.degree(graph)
degree = np.array([val for (node, val) in degree])

ind_degree = np.argsort(degree)[::-1]
ind_socres = np.argsort(socres)[::-1]


for n_x, i_x in zip(ind_degree, ind_socres):
    graph.nodes[n_x]['score'] = socres[i_x]


def draw_network_html(network, scores):
    net = pyvis.network.Network(notebook=True, cdn_resources='remote',
                                height="1000px", width="100%", bgcolor="#FDFEFF", font_color="#000c1f")
    net.force_atlas_2based()
    max_size = 30
    min_size = 15
    centrality = nx.degree_centrality(network)
    min_centrality = min(centrality.values())
    max_centrality = max(centrality.values())
    chosen_weights = np.array([network.nodes[node]['score']
                               for node in network.nodes])
    best_weight = max(chosen_weights)
    worst_weight = min(chosen_weights)
    for i, node in enumerate(network.nodes):
        node_color_based_on_weight = matplotlib.colors.to_hex(plt.cm.Blues(
            (network.nodes[node]['score'] - worst_weight) / (best_weight - worst_weight)))

        net.add_node(i, label=f"{network.nodes[node]['score']}", size=min_size + (max_size - min_size) * (
            centrality[node] - min_centrality) / (max_centrality - min_centrality), color=node_color_based_on_weight, font_size=25)

    for edge in network.edges:
        net.add_edge(edge[0], edge[1])

    net.show('./network.html')


draw_network_html(graph, socres)
