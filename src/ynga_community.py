from genetic_algorithm import GeneticAlgorithm, GAHistory
import networkx as nx
import random
import math
import numpy as np
from enum import Enum
from crossover import CrossOverType
from mutation import MutationType
from selection import SelectionType
import matplotlib.pyplot as plt
from networked_assignment import NetworkAssignment, NetworkAssignmentType, Node
from networked_model import NetworkModel, NetworkModelType
from networked_selection import NetworkSelection, InitialSelectionType, SecondarySelectionType
from tqdm import tqdm
from utils.drawable import DrawManager
import infomap


class YNGA(GeneticAlgorithm, NetworkAssignment, NetworkModel, NetworkSelection):
    def __init__(self, number_of_genes: int, domain, number_of_generations: int, population_size: int, number_elites: int,
                 probability_crossover: float, probability_mutation: float, decimal_precision: int, create_random_individual, fitness_function,
                 is_maximization: bool, verbose: bool = True, random_seed: int = None):

        super().__init__(number_of_genes, domain, number_of_generations, population_size, number_elites, probability_crossover, probability_mutation, decimal_precision,
                         create_random_individual, fitness_function, is_maximization, verbose, random_seed)

        # NetworkModel.__init__(
        #     self, NetworkModelType.BARABASI_ALBERT, population_size)
        # NetworkAssignment.__init__(
        #     self, NetworkAssignmentType.DEGREE_CENTRALITY_FITNESS_ASSIGNMENT)
        # NetworkSelection.__init__(self, initial_selection_type=InitialSelectionType.TOURNAMENT,
        #                           secondary_selection_type=SecondarySelectionType.RANDOM_NEIGHBOR)

        # Adds the node object to each node, i.e. self.network.nodes[index]['node']
        # self._assign_individuals_to_nodes()
        self.draw_manager = DrawManager(self.is_maximization)

    # Override the default selection method of GA
    def select_parents(self):
        return NetworkSelection.select_parents(self)

    def find_assignment(self, child1, child2, parent1, parent2):
        parent1_choice = random.choice([child1, child2])
        parent2_choice = child2 if parent1_choice == child1 else child1

        return {
            parent1: parent1_choice,
            parent2: parent2_choice,
        }

    def is_better(self, child1, child2):
        f1 = self._evaluate(child1)
        f2 = self._evaluate(child2)

        if self.is_maximization:
            return f1 >= f2

        return f1 <= f2

    def _similarity(self, individual1, individual2):
        # normalize the individuals
        epsilon = 1e-10
        individual1 = [gene / (max(individual1) + epsilon)
                       for gene in individual1]
        individual2 = [gene / (max(individual2) +
                       epsilon) for gene in individual2]

        return math.sqrt(sum([(gene1 - gene2)**2 for gene1, gene2 in zip(individual1, individual2)]))

    def create_network(self, population):
        # create similarity matrix between individuals
        similarity_matrix = np.zeros(
            (len(population), len(population)))
        for i in range(len(population)):
            for j in range(len(population)):
                similarity_matrix[i][j] = self._similarity(
                    population[i], population[j])

        similarity_matrix = similarity_matrix / np.max(similarity_matrix)

        fitnesses = self._evaluate_population(population)

        network = nx.Graph()
        for i in range(len(population)):
            network.add_node(i, node=Node(
                node_index=i, individual=population[i], fitness=fitnesses[i]))

        number_of_nodes_based_on_sim = 10
        number_of_nodes_based_on_fitness = 2
        # for each node, get the number_of_nodes_based_on_sim most similar nodes and from those get the number_of_nodes_based_on_fitness most fit nodes and add them as neighbors
        for i in range(len(population)):
            # get the number_of_nodes_based_on_sim most similar nodes
            most_similar_nodes = np.argsort(
                similarity_matrix[i])[-number_of_nodes_based_on_sim:]
            # get the number_of_nodes_based_on_fitness most fit nodes
            most_fit_nodes = np.argsort(
                fitnesses[most_similar_nodes])[-number_of_nodes_based_on_fitness:]
            for node in most_fit_nodes:
                if node != i:
                    network.add_edge(i, node)

        # threshold = 0.7
        # for i in range(self.popsize):
        #     for j in range(self.popsize):
        #         if i != j and similarity_matrix[i][j] > threshold:
        #             network.add_edge(i, j)

        return network

    def run(self):
        """
        E NGA algorithm is:
        1. Create a random population
        2. Create a Barabasi-Albert network
        3. Assign each individual to a node in the network based on fitness/degree of individual/node
        4. For each generation:
            4.1. Create a copy of the network (new network)
            4.1. For population size/2 times:
                4.1.1. Select two parents from the OLD network (initial selection + secondary selection) returns 2 assignments object (each contain individual index and node index and fitness)
                4.1.2. Crossover the parents
                4.1.3. Mutate the children
                4.1.4 Pick an Assignment between parents and children
                4.1.5 If child fitness "better" than parents fitness child replaces parent in NEW Network, if it is not add a new Node with the childern as individual
            4.2. Remove Nodes from the network using (RemovalMethod: Roullete, Tournemant, Least Degree, Lest fitness, Least Closenes) to make sure that the size of the network remains the same across generations
            4.3  If there exists nodes with no degree, re add m edges to them randomly
            4.3. Replace the OLD network with the NEW network
        5. Return the best individual from the network
        """

        history = GAHistory(is_maximization=self.is_maximization)

        for generation in tqdm(range(self.n_generations)) if self.verbose else range(self.n_generations):

            # Create a copy of the network
            self.network = self.create_network(self.population)
            new_network = self.network.copy()
            history._add_network(new_network.copy())

            fitnesses = self._evaluate_population(
                [self.network.nodes[node]['node'].individual for node in self.network.nodes])

            if sum(fitnesses) == 0:
                break

            ind = np.argmax(
                fitnesses) if self.is_maximization else np.argmin(fitnesses)
            best_individual = self.network.nodes[ind]['node'].individual
            best_fitness = fitnesses[ind]
            if generation % 1 == 0:
                print(
                    f"{generation} | {best_individual} | {best_fitness} ")

            node_to_int = {node: i for i,
                           node in enumerate(self.network.nodes)}
            info = infomap.Infomap(silent=True)

            for u, v in self.network.edges:
                info.addLink(node_to_int[u], node_to_int[v])

            info.run()
            communities = {}

            for node in info.iterTree():
                if node.isLeaf():
                    if node.moduleIndex() not in communities:
                        communities[node.moduleIndex()] = []

                    communities[node.moduleIndex()].append(
                        node.physicalId)

            for i, nodes in enumerate(communities.values()):
                for node in nodes:
                    self.network.nodes[node]['community'] = i

            def get_local_importance(node, community, network, communities):
                neighbors = list(network.neighbors(node))
                neighbors_in_community = [
                    neighbor for neighbor in neighbors if network.nodes[neighbor]['community'] == community]
                return len(neighbors_in_community) / len(communities[community])

            # local importance = number of neighbors in the same community / number of nodes in the community
            local_importance = {}
            for node in self.network.nodes:
                local_importance[node] = {}
                for community in communities:
                    local_importance[node][community] = get_local_importance(
                        node, community, self.network, communities)

            deno = {}
            for node in self.network.nodes:
                deno[node] = sum([local_importance[node][j]
                                  for j in communities])

            # global importance = sum over each community l()
            importance_concentration = {}
            for node in self.network.nodes:
                cs = []
                for l in communities:
                    cs.append((local_importance[node][l] / deno[node]) ** 2)
                importance_concentration[node] = math.sqrt(sum(cs))

            representativity = {}
            for node in self.network.nodes:
                representativity[node] = local_importance[node][self.network.nodes[node]
                                                                ['community']] * importance_concentration[node]
            print(
                f"Number of nodes in each community: {[len(communities[community]) for community in communities]}")
            # from each community compute its fitness
            community_fitness = {}
            for community in communities:
                community_fitness[community] = 0
                for node in communities[community]:
                    community_fitness[community] += self.network.nodes[node]['node'].fitness
                community_fitness[community] /= len(communities[community])

            print(
                f"Fitness of each community: {[community_fitness[community] for community in communities]}")

            take_nodes = self.popsize // 2

            # normalize thee fitness of each community to be between 0 and 1 such that lowest fitness is 1 and highest fitness is 0
            normalized_community_fitness = {}
            if len(community_fitness) == 1:
                normalized_community_fitness[0] = 1
            else:
                for community in communities:
                    normalized_community_fitness[community] = (
                        community_fitness[community] - min(community_fitness.values())) / (max(community_fitness.values()) - min(community_fitness.values()))

                # make sure least fit community has normalized fitness of 1 and most fit community has normalized fitness of 0
                normalized_community_fitness = {
                    community: (1 - normalized_community_fitness[community]) + 0.00001 for community in communities}

                normalized_community_fitness = {community: normalized_community_fitness[community] /
                                                sum(normalized_community_fitness.values()) for community in communities}

            print(
                f"Normalized fitness of each community: {[normalized_community_fitness[community] for community in communities]} = {sum(normalized_community_fitness.values())}")

            selected_nodes_per_community = np.random.multinomial(
                take_nodes, list(normalized_community_fitness.values()))

            selected_nodes_per_community = {
                community: selected_nodes_per_community[i] for i, community in enumerate(communities)}

            print(
                f"select_nodes_per_community: {selected_nodes_per_community} = {sum(selected_nodes_per_community.values())}")

            # fix the select_nodes_per_community if there are less nodes in a community than required, distribute the excess nodes equally to all communities
            while any([selected_nodes_per_community[community] > len(communities[community]) for community in communities]):
                for community in communities:
                    if selected_nodes_per_community[community] > len(communities[community]):
                        # Calculate the excess nodes that need to be distributed equally among other communities
                        excess_nodes = selected_nodes_per_community[community] - len(
                            communities[community])
                        other_communities = [
                            c for c in communities if c != community]
                        nodes_to_distribute = excess_nodes // len(
                            other_communities)

                        added_till_now = 0

                        # Distribute the excess nodes equally among other communities
                        for other_community in other_communities:
                            selected_nodes_per_community[other_community] += nodes_to_distribute
                            added_till_now += nodes_to_distribute

                        if added_till_now < excess_nodes:
                            # Distribute the remaining excess nodes to the first few communities
                            for i in range(excess_nodes - added_till_now):
                                # only add if that community did not already get the maximum number of nodes
                                if selected_nodes_per_community[other_communities[i]] < len(communities[other_communities[i]]):
                                    selected_nodes_per_community[other_communities[i]
                                                                 ] += 1

                        # Set the selected nodes for the current community to the maximum available nodes
                        selected_nodes_per_community[community] = len(
                            communities[community])

            print(
                f"selected_nodes_per_community: {selected_nodes_per_community} = {sum(selected_nodes_per_community.values())}")

            chosen_nodes = []
            for community in communities:
                representativity_in_community = {
                    node: representativity[node] for node in communities[community]}
                sorted_representativity = sorted(
                    representativity_in_community.items(), key=lambda x: x[1], reverse=True)
                nodes = [node for node, _ in sorted_representativity]
                nodes = nodes[:selected_nodes_per_community[community]]

                chosen_nodes.extend(nodes)

            chosen_individuals = [
                (node, self.network.nodes[node]['node'].individual) for node in chosen_nodes]

            print(
                f"chosen_individuals = {len(chosen_individuals)}")

            # use strict mating on the chosen nodes
            new_population = []

            for parent1_index, parent1 in chosen_individuals:

                tournament_size = 2
                # tournmant on neighbors of parent1
                neighbors = list(self.network.neighbors(parent1_index))

                if len(neighbors) >= tournament_size:
                    tournament = random.sample(
                        neighbors, tournament_size)
                    sorted_tournament = sorted(
                        tournament, key=lambda x: self.network.nodes[x]['node'].fitness)

                    parent2_index = sorted_tournament[0]

                parent2 = self.network.nodes[parent2_index]['node'].individual
                child1, child2 = self.crossover(
                    parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                new_population.append(child1)
                new_population.append(child2)

            pick_from = new_population + [ind for _, ind in chosen_individuals]
            print(f"Size of pick_from: {len(pick_from)}")

            if len(pick_from) > self.popsize:
                print(f"removing {len(pick_from) - self.popsize} individuals")
                fitnesses = self._evaluate_population(pick_from)
                print(f"fitnesses: {len(fitnesses)}")
                ind_fitnesses = np.argsort(fitnesses)
                ind_fitnesses = ind_fitnesses[:self.popsize]
                pick_from = [pick_from[i] for i in ind_fitnesses]

            print(f"Size of pick_from: {len(pick_from)}")
            self.population = pick_from

            # if generation % 10 == 0:
            #     largest_communities = sorted(
            #         communities.values(), key=lambda x: len(x), reverse=True)
            #     colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink',
            #               'brown', 'black', 'gray', 'cyan', 'magenta', 'lime', 'olive', 'teal']
            #     node_colors = {}
            #     for i, nodes in enumerate(largest_communities):
            #         for node in nodes:
            #             node_colors[node] = colors[i % len(colors)]
            #     default_node_color = 'black'
            #     for node in self.network.nodes:
            #         if node not in node_colors:
            #             node_colors[node] = default_node_color

            #     # top 10 representativity have highest size
            #     node_sizes = {}
            #     for node in self.network.nodes:
            #         if node in representativity:
            #             node_sizes[node] = representativity[node] * 1000
            #         else:
            #             node_sizes[node] = 100

            #     nx.draw(self.network, pos=nx.kamada_kawai_layout(
            #         self.network), node_color=[node_colors[node] for node in self.network.nodes], node_size=[node_sizes[node] for node in self.network.nodes])
            #     plt.show()

            history._add_population_fitness(fitnesses, self.population)

        history.stop()
        print(history)
        return history
