from enum import Enum
import networkx as nx
import random
import numpy as np
import math


class NetworkModelType(Enum):
    BARABASI_ALBERT = 1
    WATTS_STROGATZ = 2
    ERDOS_RENYI = 3
    COMPLETE = 4
    RANDOM = 5


class NetworkModel:
    def __init__(self, network_model_type, population_size):
        self.network_model_type = network_model_type
        self.population_size = population_size
        self.network = None
        self._generate_network()

    def _generate_network(self):
        if self.network_model_type == NetworkModelType.BARABASI_ALBERT:
            self.network = self._generate_barabasi_albert_network()
        elif self.network_model_type == NetworkModelType.WATTS_STROGATZ:
            self.network = self._generate_watts_strogatz_network()
        elif self.network_model_type == NetworkModelType.ERDOS_RENYI:
            self.network = self._generate_erdos_renyi_network()
        elif self.network_model_type == NetworkModelType.COMPLETE:
            self.network = self._generate_complete_network()
        elif self.network_model_type == NetworkModelType.RANDOM:
            self.network = self._generate_random_network()

    def _generate_barabasi_albert_network(self):
        self.m0 = 4
        self.m = 4
        self.A = 1
        self.alpha = 1

        G = nx.Graph()

        # Fully connected initial graph
        G.add_nodes_from(range(self.m0))
        for i in range(self.m0):
            for j in range(i + 1, self.m0):
                G.add_edge(i, j)

        # Add new nodes
        for i in range(self.m0, self.population_size):
            new_node_index = i
            k = [self.A + math.pow(G.degree(node), self.alpha)
                 for node in G.nodes]
            p = [k_i / sum(k) for k_i in k]
            selected_nodes = np.random.choice(
                G.nodes, self.m, p=p, replace=False)
            G.add_node(new_node_index)
            for node in selected_nodes:
                # if not self loop
                if node != new_node_index:
                    G.add_edge(new_node_index, node)

        return G

    def _generate_watts_strogatz_network(self):
        # the higher is rewiring factor the faster the convergance but lower SR

        return nx.watts_strogatz_graph(self.population_size, 6, 0.2)

    def _generate_erdos_renyi_network(self):
        return nx.erdos_renyi_graph(self.population_size, 0.038)

    def _generate_complete_network(self):
        return nx.complete_graph(self.population_size)

    def _generate_random_network(self):
        return nx.random_regular_graph(random.randint(1, 2), self.population_size)
