import collections
from utils.math import *
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import time
import math
import random
from scipy.stats import linregress
from matplotlib import animation


class DrawManager:
    def __init__(self, is_maximization):
        self.is_maximization = is_maximization

    def _cosine_similarity(self, a, b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        similarity = dot_product / (norm_a * norm_b)
        return similarity

    def draw_selection_pressure(self, history, name="selection_pressure"):
        populations = history.get_populations()
        fitnesses = history.get_fitnesses()
        selection_pressure = []

        for i in range(len(populations)):
            best_fitness = np.max(
                fitnesses[i]) if history._is_maximization else np.min(fitnesses[i])
            count = 0
            for fitness in fitnesses[i]:
                # within a certain precision
                if abs(fitness - best_fitness) < 0.001:
                    count += 1
            selection_pressure.append(count / len(fitnesses[i]))

        plt.plot(selection_pressure)
        plt.title('Selection Pressure')
        plt.ylabel('Selection Pressure')
        plt.xlabel('Generation')

        current_time = time.strftime("%d_%H_%M")
        filename = f"./logs/metrics/selection_pressure/{name}_{current_time}.png"
        plt.savefig(filename)
        plt.close()

    def draw_avg_phenotype_diversity(self, history, name="avg_phenotype_diversity"):
        populations = history.get_populations()
        fitnesses = history.get_fitnesses()
        fitness_diversity = []
        for i in range(len(populations)):
            best_fitness = np.max(
                fitnesses[i]) if history._is_maximization else np.min(fitnesses[i])
            worst_fitness = np.min(
                fitnesses[i]) if history._is_maximization else np.max(fitnesses[i])

            fitness_diversity_value = abs(best_fitness - worst_fitness)
            fitness_diversity_value_normalized = fitness_diversity_value / \
                (abs(best_fitness) + 0.0001)
            fitness_diversity.append(fitness_diversity_value_normalized)

        # normalize
        fitness_diversity = [x / max(fitness_diversity)
                             for x in fitness_diversity]

        plt.plot(fitness_diversity)
        plt.title('Phenotype Diversity')
        plt.ylabel('Phenotype Diversity')
        plt.xlabel('Generation')

        current_time = time.strftime("%d_%H_%M")
        filename = f"./logs/metrics/phenotype_diversity/{name}_{current_time}.png"
        plt.savefig(filename)
        plt.close()

    def draw_avg_genotype_diversity(self, history, name="avg_genotype_diversity"):
        populations = history.get_populations()
        diversity = []
        for i in range(len(populations)):
            current_population = populations[i]
            num_samples_to_test = len(current_population) * 0.2
            similarities = []
            for _ in range(int(num_samples_to_test)):
                individuals = random.sample(current_population, 2)
                individual1 = individuals[0]
                individual2 = individuals[1]

                similarities.append(
                    self._cosine_similarity(individual1, individual2))

            diversity.append(np.mean(similarities))

        plt.plot(diversity)
        plt.title('Diversity')
        plt.ylabel('Diversity')
        plt.xlabel('Generation')

        current_time = time.strftime("%d_%H_%M")
        filename = f"./logs/metrics/genotype_diversity/{name}_{current_time}.png"
        plt.savefig(filename)
        plt.close()

    def draw_avg_fitness(self, history, name="avg_diversity", optimum=None):
        fig, axs = plt.subplots(1, 2)
        fig.suptitle('Fitness Average & Best Fitness')
        fig.set_size_inches(16, 8)

        axs[0].plot(history._avg_fitnesses)
        axs[0].set_title('Fitness Average')
        axs[0].set_ylabel('Fitness Average')
        axs[0].set_xlabel('Generation')
        axs[0].set_xscale('log')

        axs[1].plot(history._best_fitnesses)
        axs[1].set_title('Best Fitness')
        axs[1].set_ylabel('Best Fitness')
        axs[1].set_xlabel('Generation')
        axs[1].set_xscale('log')
        # add optimum as horizontal line if given
        # if optimum != None:
        #     axs[0].axhline(y=optimum, color='r', linestyle='-')
        #     axs[1].axhline(y=optimum, color='r', linestyle='-')

        fig.tight_layout()
        current_time = time.strftime("%d_%H_%M")
        filename = f"./logs/metrics/fitness/{name}_{current_time}.png"
        plt.savefig(filename)
        plt.close()

    def draw_number_of_nodes_to_remove(self, history, name="number_of_nodes_to_remove"):
        number_of_nodes_to_remove_each_gen = history.get_number_of_nodes_to_remove()

        if len(number_of_nodes_to_remove_each_gen) == 0:
            return

        plt.plot(number_of_nodes_to_remove_each_gen)
        plt.title('Number of Nodes to Remove')
        plt.ylabel('Number of Nodes to Remove')
        plt.xlabel('Generation')

        current_time = time.strftime("%d_%H_%M")
        filename = f"./logs/metrics/number_of_nodes_to_remove/{name}_{current_time}.png"
        plt.savefig(filename)
        plt.close()

    def draw_degree_exponent(self, history, name="degree_exponent"):
        networks = history.get_networks()

        if len(networks) == 0:
            return

        degree_exponents = []
        for network in networks:
            degree_sequence = sorted(
                [d for n, d in network.degree()], reverse=True)
            degree_count = collections.Counter(degree_sequence)
            deg, cnt = zip(*degree_count.items())
            degree_exponents.append(
                -linregress(np.log(deg), np.log(cnt)).slope + 1)

        plt.plot(degree_exponents)
        plt.title('Degree Exponent')
        plt.ylabel('Degree Exponent')
        plt.xlabel('Generation')

        current_time = time.strftime("%d_%H_%M")
        filename = f"./logs/metrics/degree_exponent/{name}_{current_time}.png"
        plt.savefig(filename)
        plt.close()

    def draw_avg_clustering_coefficient(self, history, name="avg_clustering_coefficient"):
        networks = history.get_networks()

        if len(networks) == 0:
            return

        avg_clustering_coefficients = []
        for network in networks:
            if nx.is_connected(network) == False:
                largest_cc = max(nx.connected_components(
                    network), key=len)
                largest_cc_network = network.subgraph(largest_cc)

                avg_clustering_coefficients.append(
                    nx.average_clustering(largest_cc_network))
            else:
                avg_clustering_coefficients.append(
                    nx.average_clustering(network))

        plt.plot(avg_clustering_coefficients)
        plt.title('Average Clustering Coefficient')
        plt.ylabel('Average Clustering Coefficient')
        plt.xlabel('Generation')

        current_time = time.strftime("%d_%H_%M")
        filename = f"./logs/metrics/clustering_coefficient/{name}_{current_time}.png"
        plt.savefig(filename)
        plt.close()

    def draw_avg_path_length(self, history, name="avg_path_length"):
        networks = history.get_networks()

        if len(networks) == 0:
            return

        avg_path_lengths = []
        for network in networks:
            # check if network is connected
            if nx.is_connected(network) == False:
                largest_cc = max(nx.connected_components(
                    network), key=len)
                largest_cc_network = network.subgraph(largest_cc)

                avg_path_lengths.append(
                    nx.average_clustering(largest_cc_network))
            else:
                avg_path_lengths.append(
                    nx.average_shortest_path_length(network))

        plt.plot(avg_path_lengths)
        plt.title('Average Path Length')
        plt.ylabel('Average Path Length')
        plt.xlabel('Generation')

        current_time = time.strftime("%d_%H_%M")
        filename = f"./logs/metrics/path_length/{name}_{current_time}.png"
        plt.savefig(filename)
        plt.close()

    def draw_avg_degree(self, history, name="avg_degree"):
        networks = history.get_networks()

        if len(networks) == 0:
            return

        avg_degrees = []
        for network in networks:
            avg_degrees.append(np.mean([network.degree(node)
                                        for node in network.nodes]))

        plt.plot(avg_degrees)
        plt.title('Average Degree')
        plt.ylabel('Average Degree')
        plt.xlabel('Generation')

        current_time = time.strftime("%d_%H_%M")
        filename = f"./logs/metrics/degree/{name}_{current_time}.png"
        plt.savefig(filename)
        plt.close()

    def draw_network(self, network):
        node_fitness = {
            node: network.nodes[node]['node'].fitness for node in network.nodes
        }

        node_fitness_sorted = sorted(node_fitness.items(
        ), key=lambda x: x[1], reverse=self.is_maximization)

        top_five = [node for node, fitness in node_fitness_sorted[:5]]
        degree_sorted = sorted(
            network.degree, key=lambda x: x[1], reverse=True)
        best_degrees = [degree for _, degree in degree_sorted[:5]]
        nodes_with_best_degree = [
            node for node in network.nodes if network.degree(node) in best_degrees]

        colors = []
        for node in network.nodes:
            if node in top_five and node in nodes_with_best_degree:
                colors.append('#FDE725')
            elif node in top_five:
                colors.append('#B5DE2B')
            elif node in nodes_with_best_degree:
                colors.append('#1F9E89')
            else:
                colors.append('#3E4989')

        def get_fitness_label(node):
            return f"{round(network.nodes[node]['node'].fitness)}" if len(str(int(network.nodes[node]['node'].fitness))) >= 3 else f"{round(network.nodes[node]['node'].fitness, 2)}"
        labels = {
            node: get_fitness_label(node) + " | " + str(round(network.nodes[node]['node_score'], 3)) for node in network.nodes
        }

        plt.figure(figsize=(16, 12))
        ax = plt.gca()
        ax.set_title('Population Network')

        pos = nx.spring_layout(network, k=1.2)
        nx.draw(network, node_color=colors,
                pos=pos,
                node_size=700,
                with_labels=True, labels=labels, font_size=6, ax=ax)
        _ = ax.axis('off')
        # add best individual as text
        best_individual = network.nodes[top_five[0]]['node'].individual
        best_individual_text = f"Best Individual: {best_individual}"
        best_fitness = network.nodes[top_five[0]]['node'].fitness
        best_fitness_text = f"Best Fitness: {best_fitness}"
        plt.text(0.5, 0.05, best_fitness_text, horizontalalignment='center',
                 verticalalignment='bottom', transform=ax.transAxes, fontsize=8)
        plt.text(0.5, 0, best_individual_text, horizontalalignment='center',
                 verticalalignment='bottom', transform=ax.transAxes, fontsize=8)
        plt.show()

    def draw_animated_network(self, history):
        networks = history.get_networks()

        fig, ax = plt.subplots()

        pos = nx.spring_layout(networks[0], k=1.4)

        def update(i):
            plt.cla()
            network = networks[i]
            node_fitness = {
                node: network.nodes[node]['node'].fitness for node in network.nodes
            }

            node_fitness_sorted = sorted(node_fitness.items(
            ), key=lambda x: x[1], reverse=self.is_maximization)

            firts_chunk_size = 2
            second_chunk_size = 10
            third_chunk_size = 15
            first_best_nodes = [node for node,
                                fitness in node_fitness_sorted[:firts_chunk_size]]

            second_best_nodes = [
                node for node, fitness in node_fitness_sorted[firts_chunk_size:firts_chunk_size+second_chunk_size]]

            third_best_nodes = [
                node for node, fitness in node_fitness_sorted[firts_chunk_size+second_chunk_size:firts_chunk_size+second_chunk_size+third_chunk_size]]

            colors = []
            for node in network.nodes:
                if node in first_best_nodes:
                    colors.append('#6ECE58')
                elif node in second_best_nodes:
                    colors.append('#35B779')
                elif node in third_best_nodes:
                    colors.append('#1F9E89')
                else:
                    colors.append('#3E4989')

            def get_fitness_label(node):
                return f"{round(network.nodes[node]['node'].fitness)}" if len(str(int(network.nodes[node]['node'].fitness))) >= 3 else f"{round(network.nodes[node]['node'].fitness, 2)}"
            labels = {
                node: str(round(network.nodes[node]['node_score'], 3)) for node in network.nodes
            }

            # nx.draw(network, node_color=colors,
            #         pos=pos,
            #         node_size=500,
            #         with_labels=False, labels=labels, font_size=6, ax=ax)

            nx.draw_networkx_nodes(network, pos=pos, node_color=colors,
                                   node_size=500, ax=ax)
            nx.draw_networkx_labels(
                network, pos=pos, labels=labels, font_size=6, ax=ax)
            nx.draw_networkx_edges(network, pos=pos, ax=ax, alpha=0.1)
            ax.set_title(f"Generation {i}")

        num_frames = len(networks)
        ani = animation.FuncAnimation(fig, update, frames=num_frames,
                                      interval=100, repeat=True)

        plt.show()

    def temprature_of_node(self, G: nx.Graph, node):
        """
        Temprature of a node i, is the sum of 1/degree of neighbor, over all its neighbors

        Returns
        -------
        float : temprature of node i
        """

        temp = 0
        for neighbor in G.neighbors(node):
            temp += 1/G.degree(neighbor)
        return temp

    def heat_heteroginity_network(self, G: nx.Graph,  plt_show=True):
        """
        Draw the heat heterogenity network of a graph
        """
        for node in G.nodes:
            G.nodes[node]['temprature'] = self.temprature_of_node(G, node)

        max_temp = max([G.nodes[node]['temprature'] for node in G.nodes])
        min_temp = min([G.nodes[node]['temprature'] for node in G.nodes])

        for node in G.nodes:
            G.nodes[node]['temprature'] = (
                G.nodes[node]['temprature'] - min_temp) / (max_temp-min_temp)

        max_temp_color = (1, 0, 0)
        min_temp_color = (0, 0, 1)

        node_color = []
        for node in G.nodes:
            if abs(G.nodes[node]['temprature'] - 1) < 1e-5:
                node_color.append(max_temp_color)
            elif abs(G.nodes[node]['temprature'] - 0) < 1e-5:
                node_color.append(min_temp_color)
            else:
                normalized_temp = G.nodes[node]['temprature']

                node_color.append(interopolate_color(
                    max_temp_color, min_temp_color, normalized_temp))

        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        ax.set_title('Population Network')
        nx.draw(G, node_color=node_color, node_size=500, with_labels=True, labels={
            node: round(G.nodes[node]['temprature'], 1) for node in G.nodes
        }, ax=ax)
        _ = ax.axis('off')
        if plt_show:
            plt.show()

    def heat_heteroginity_histogram(self, G: nx.Graph,  plt_show=True):
        """
        Draw the heat heterogenity histogram of a graph
        """

        heat_sequence = sorted([round(self.temprature_of_node(G, node), 1)
                                for node in G.nodes], reverse=True)

        plt.hist(heat_sequence, bins=20, edgecolor='black', alpha=0.7)

        # You can customize labels and titles if needed
        plt.xlabel('heat heterogenity')
        plt.ylabel('Frequency')
        plt.title(f'Heat Heterogenity Histogram')

        plt.grid(True)
        if plt_show:
            plt.show()

    def clustering_of_node(self, G: nx.Graph, node):
        """
        Clustering coefficient of a node i = 2 * Li / ki(ki-1)

        where Li is the number of edges between neighbors of node i
        and ki is the degree of node i

        Returns
        -------
        float : clustering coefficient of node i, range [0, 1]
        """

        neighbors = list(G.neighbors(node))
        edges = 0
        for i in range(len(neighbors)):
            for j in range(i+1, len(neighbors)):
                if G.has_edge(neighbors[i], neighbors[j]):
                    edges += 1
        if len(neighbors) <= 1:
            return 0
        return 2*edges/(len(neighbors)*(len(neighbors)-1))

    def clustering_coefficient_histogram(self, G: nx.Graph, plt_show=True):
        """
        Draw the clustering coefficient histogram of a graph
        """

        clustering_sequence = sorted([round(self.clustering_of_node(G, node), 2)
                                      for node in G.nodes], reverse=True)

        plt.hist(clustering_sequence, bins=20, edgecolor='black', alpha=0.7)

        # You can customize labels and titles if needed
        plt.xlabel('clustering coefficient')
        plt.ylabel('Frequency')
        plt.title(f'Clustering Coefficient Histogram')

        plt.grid(True)
        if plt_show:
            plt.show()

    # Detect communities using Girvan-Newman algorithm
    def community_network(self, G: nx.Graph):
        communities_generator = nx.community.girvan_newman(G)
        next(communities_generator)
        next(communities_generator)
        next(communities_generator)
        top_level_communities = next(communities_generator)
        community_colors = {}
        for i, community in enumerate(top_level_communities):
            color = plt.cm.tab20(i % 20)
            for node in community:
                community_colors[node] = color

        # Generate a list of colors for each node
        node_colors = [community_colors[node] for node in G.nodes]

        # Create a layout for the graph (e.g., spring layout)
        pos = nx.spring_layout(G)
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        ax.set_title(f'Communities')
        # Draw the graph with specified node colors
        nx.draw(G, pos, node_color=node_colors,
                with_labels=True, node_size=300, font_size=10)
        _ = ax.axis('off')

        # Show the plot
        plt.show()

    def centrality_network(self, G: nx.Graph, func=nx.degree_centrality, plt_show=True):
        """
        Draw the eigenvector centrality network of a graph
        """

        centrality = func(G)
        max_centrality = max([centrality[node] for node in G.nodes])
        min_centrality = min([centrality[node] for node in G.nodes])

        max_centrality_color = (1, 0, 0)
        min_centrality_color = (0, 0, 1)

        node_color = []
        for node in G.nodes:
            if abs(centrality[node] - max_centrality) < 1e-5:
                node_color.append(max_centrality_color)
            elif abs(centrality[node] - min_centrality) < 1e-5:
                node_color.append(min_centrality_color)
            else:
                normalized_centrality = (centrality[node] - min_centrality) / \
                    (max_centrality-min_centrality)

                node_color.append(interopolate_color(
                    max_centrality_color, min_centrality_color, normalized_centrality))
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        ax.set_title(f'Network with emphasis on {func.__name__}')
        nx.draw(G, node_color=node_color, node_size=500, with_labels=True, labels={
                node: round(centrality[node], 2) for node in G.nodes}, ax=ax)
        _ = ax.axis('off')
        if plt_show:
            plt.show()

    def centrality_histogram(self, G: nx.Graph, func=nx.degree_centrality, plt_show=True, save=False, name=""):
        """
        Draw the eigenvector centrality histogram of a graph
        """
        centrality_sequence = sorted([round(func(G)[node], 3)
                                      for node in G.nodes], reverse=True)

        plt.hist(centrality_sequence, bins=20, edgecolor='black', alpha=0.7)

        # You can customize labels and titles if needed
        plt.xlabel(f'{func.__name__} Centrality Value')
        plt.ylabel('Frequency')
        plt.title(f'{func.__name__} Centrality Distribution Histogram')

        plt.grid(True)
        if plt_show:
            plt.show()

        if save:
            if os.path.exists("./logs/metrics") == False:
                os.mkdir("./logs/metrics")

            current_time = time.strftime("%d_%H_%M")
            filename = f"./logs/metrics/{name}_{current_time}.png"
            plt.savefig(filename)

        # plot loglog
        plt.loglog(centrality_sequence, 'o')
        plt.ylabel(f'{func.__name__} Centrality Value')
        plt.xlabel('Node')
        plt.title(f'{func.__name__} Centrality Distribution LogLog Plot')
        plt.grid(True)
        if plt_show:
            plt.show()
