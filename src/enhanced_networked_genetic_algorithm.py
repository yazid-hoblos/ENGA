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


class SAEnhancedNetworkGeneticAlgorithm(GeneticAlgorithm, NetworkAssignment, NetworkModel, NetworkSelection):
    def __init__(self, number_of_genes: int, domain, number_of_generations: int, population_size: int, number_elites: int,
                 probability_crossover: float, probability_mutation: float, decimal_precision: int, create_random_individual, fitness_function,
                 is_maximization: bool, verbose: bool = True, random_seed: int = None):

        super().__init__(number_of_genes, domain, number_of_generations, population_size, number_elites, probability_crossover, probability_mutation, decimal_precision,
                         create_random_individual, fitness_function, is_maximization, verbose, random_seed)

        NetworkModel.__init__(
            self, NetworkModelType.BARABASI_ALBERT, population_size)
        NetworkAssignment.__init__(
            self, NetworkAssignmentType.DEGREE_CENTRALITY_FITNESS_ASSIGNMENT)
        NetworkSelection.__init__(self, initial_selection_type=InitialSelectionType.TOURNAMENT,
                                  secondary_selection_type=SecondarySelectionType.RANDOM_NEIGHBOR)

        # Adds the node object to each node, i.e. self.network.nodes[index]['node']
        self._assign_individuals_to_nodes()

        self.draw_manager = DrawManager(self.is_maximization)
        # self.draw_manager.draw_network(self.network)

    # Override the default selection method of GA
    def select_parents(self):
        return NetworkSelection.select_parents(self)

    def find_assignment(self, child1, child2, parent1, parent2):

        # Associate the higher fitness with the higher degree parent

        higher_degree = parent1 if self.network.degree(
            parent1.node_index) >= self.network.degree(parent2.node_index) else parent2
        lower_degree = parent1 if higher_degree == parent2 else parent2

        higher_fitness = child1 if self.is_better(child1, child2) else child2
        lower_fitness = child1 if higher_fitness == child2 else child2

        return {
            higher_degree: higher_fitness,
            lower_degree: lower_fitness,
        }

    def is_better(self, child1, child2):
        f1 = self._evaluate(child1)
        f2 = self._evaluate(child2)

        if self.is_maximization:
            return f1 >= f2

        return f1 <= f2

    # if child better then parent replace it if not add it as new node (should have edges with parents?)
    # mutation of edges + normal mutation + get value from neighbor
    # Network Restructuring (addition or deletion or rewiring) (maybe if plateau)
    # Community fitness selection
    # use tournement for ealy convergence

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
                4.1.5 If child fitness "better" than parents fitness child replaces parent in NEW Network, otherwise replace it with probability related to temprature (exploration/exploitation)
            4.2. Remove Nodes from the network using (RemovalMethod: Roullete, Tournemant, Least Degree, Least Closenes) to make sure that the size of the network remains the same across generations
            4.3. Replace the OLD network with the NEW network
        5. Return the best individual
        """

        history = GAHistory(is_maximization=self.is_maximization)

        temparture = 1.5
        min_temp = 0.0001
        step = 0.0015

        for generation in tqdm(range(self.n_generations)) if self.verbose else range(self.n_generations):

            # Create a copy of the network
            new_network = self.network.copy()

            history._add_network(new_network)

            fitnesses = self._evaluate_population(
                [self.network.nodes[node]['node'].individual for node in self.network.nodes])

            history._add_population_fitness(
                fitnesses, [self.network.nodes[node]['node'].individual for node in self.network.nodes])

            if sum(fitnesses) == 0:
                break

            ind = np.argsort(fitnesses)

            if not self.is_maximization:
                ind = ind[::-1]

            # Elites, array of indicies of parents (nodes) that should not be replaced
            elites = [self.network.nodes[i]['node'].individual for i in ind[-self.n_elites:]
                      ] if self.n_elites > 0 else []

            for parent1 in self.network.nodes:
                # Select two parents from the OLD network (initial selection + secondary selection)

                parent1 = self.network.nodes[parent1]['node']
                parent2 = NetworkSelection._secondary_node_selection(
                    self, parent1)

                # Crossover the parents
                child1, child2 = self.crossover(
                    parent1.individual, parent2.individual)

                # Mutate the children
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                better_child = child1 if self.is_better(
                    child1, child2) else child2

                better_child_fitness = self._evaluate(better_child)

                if ((better_child_fitness >= parent1.fitness if self.is_maximization else better_child_fitness <= parent1.fitness)):
                    new_network.nodes[parent1.node_index]['node'] = Node(
                        node_index=parent1.node_index, individual=better_child, fitness=better_child_fitness)

                else:
                    x = better_child_fitness-parent1.fitness
                    probability = math.exp(
                        -(1/(2+100*math.exp(-0.1*x))) / max(temparture, min_temp))
                    if random.random() < probability:
                        if not parent1.individual in elites:
                            new_network.nodes[parent1.node_index]['node'] = Node(
                                node_index=parent1.node_index, individual=better_child, fitness=better_child_fitness)

            # Replace the OLD network with the NEW network
            self.network = new_network
            temparture -= step

        history.stop()
        # print(history)
        # self.draw_manager.draw_avg_fitness(history)
        return history


class EnhancedNetworkGeneticAlgorithm(GeneticAlgorithm, NetworkAssignment, NetworkModel, NetworkSelection):
    def __init__(self, number_of_genes: int, domain, number_of_generations: int, population_size: int, number_elites: int,
                 probability_crossover: float, probability_mutation: float, decimal_precision: int, create_random_individual, fitness_function,
                 is_maximization: bool, verbose: bool = True, random_seed: int = None):

        super().__init__(number_of_genes, domain, number_of_generations, population_size, number_elites, probability_crossover, probability_mutation, decimal_precision,
                         create_random_individual, fitness_function, is_maximization, verbose, random_seed)

        NetworkModel.__init__(
            self, NetworkModelType.BARABASI_ALBERT, population_size)
        NetworkAssignment.__init__(
            self, NetworkAssignmentType.DEGREE_CENTRALITY_FITNESS_ASSIGNMENT)
        NetworkSelection.__init__(self, initial_selection_type=InitialSelectionType.TOURNAMENT,
                                  secondary_selection_type=SecondarySelectionType.RANDOM_NEIGHBOR)

        # Adds the node object to each node, i.e. self.network.nodes[index]['node']
        self._degree_centrality_fitness_assignment(0, self.n_generations)
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

            if generation % (self.n_generations-1) == 0:
                self.draw_manager.draw_network(self.network)

            # Create a copy of the network
            new_network = self.network.copy()
            history._add_network(new_network.copy())

            fitnesses = self._evaluate_population(
                [self.network.nodes[node]['node'].individual for node in self.network.nodes])

            if sum(fitnesses) == 0:
                break

            # Strict mating (each parent is selected once)
            new_population = []
            for i, parent1 in enumerate(self.network.nodes):
                parent1 = self.network.nodes[parent1]['node']
                parent2 = NetworkSelection._secondary_node_selection(
                    self, parent1)

                # Crossover the parents
                child1, child2 = self.crossover(
                    parent1.individual, parent2.individual)

                # Mutate the children
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                better_child = child1 if self.is_better(
                    child1, child2) else child2

                if (self.is_better(better_child, parent1.individual)):
                    new_population.append(better_child)

                else:
                    new_population.append(parent1.individual)

            self.population = new_population
            self._degree_centrality_fitness_assignment(
                generation + 1, self.n_generations)

            fitnesses = self._evaluate_population(self.population)
            history._add_population_fitness(fitnesses, self.population)

        history.stop()
        self.draw_manager.draw_animated_network(history)
        print(history)
        return history


class OnlySelectionENGA(GeneticAlgorithm, NetworkAssignment, NetworkModel, NetworkSelection):
    def __init__(self, number_of_genes: int, domain, number_of_generations: int, population_size: int, number_elites: int,
                 probability_crossover: float, probability_mutation: float, decimal_precision: int, create_random_individual, fitness_function,
                 is_maximization: bool, verbose: bool = True, random_seed: int = None):

        super().__init__(number_of_genes, domain, number_of_generations, population_size, number_elites, probability_crossover, probability_mutation, decimal_precision,
                         create_random_individual, fitness_function, is_maximization, verbose, random_seed)

        NetworkModel.__init__(
            self, NetworkModelType.BARABASI_ALBERT, population_size)
        NetworkAssignment.__init__(
            self, NetworkAssignmentType.DEGREE_CENTRALITY_FITNESS_ASSIGNMENT)
        NetworkSelection.__init__(self, initial_selection_type=InitialSelectionType.TOURNAMENT,
                                  secondary_selection_type=SecondarySelectionType.RANDOM_NEIGHBOR)

        # Adds the node object to each node, i.e. self.network.nodes[index]['node']
        self._degree_centrality_fitness_assignment(0, self.n_generations)
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
            new_network = self.network.copy()

            fitnesses = self._evaluate_population(
                [self.network.nodes[node]['node'].individual for node in self.network.nodes])

            if sum(fitnesses) == 0:
                break

            # if generation % 500 == 0:
            # ind = np.argsort(fitnesses)

            # if not self.is_maximization:
            #     ind = ind[::-1]

            # best = self.network.nodes[ind[-1]]['node'].individual
            # best_fitness = self._evaluate(best)
            #     print(f"\n{generation} | {ind[-1]} | {best} | {best_fitness}")

            # Strict mating (each parent is selected once)
            new_population = []
            for i, parent1 in enumerate(self.network.nodes):
                parent1 = self.network.nodes[parent1]['node']
                parent2 = NetworkSelection._secondary_node_selection(
                    self, parent1)

            # for _ in range(self.population_size // 2):
            #     parent1, parent2 = self.select_parents()

                # Crossover the parents
                child1, child2 = parent1.individual, parent2.individual

                better_child = child1 if self.is_better(
                    child1, child2) else child2

                # Replace if better (IDEALIZED SELECTION)
                # Add child as new node if no better

                if (self.is_better(better_child, parent1.individual)):
                    # new_network.nodes[parent1.node_index]['node'] = Node(
                    #     node_index=parent1.node_index, individual=better_child, fitness=self._evaluate(better_child))
                    new_population.append(better_child)

                else:
                    new_population.append(parent1.individual)
                    # add the child as a new node to the network and adds m edges to the new node, using the preferential attachment mechanism
                    # new_node_index = new_network.number_of_nodes()
                    # k = [self.A + math.pow(new_network.degree(node), self.alpha)
                    #      for node in new_network.nodes]
                    # p = [k_i / sum(k) for k_i in k]
                    # selected_nodes = np.random.choice(
                    #     new_network.nodes, self.m, p=p, replace=False)
                    # new_network.add_node(new_node_index, node=Node(
                    #     node_index=new_node_index, individual=better_child, fitness=self._evaluate(better_child)))
                    # for node in selected_nodes:
                    #     # if not self loop
                    #     if node != new_node_index:
                    #         new_network.add_edge(new_node_index, node)

            # self.draw_manager.draw_network(self.network)
            self.population = new_population
            self._degree_centrality_fitness_assignment(
                generation, self.n_generations)

            # self.draw_manager.draw_network(self.network)

            # Perform selection on the network to remove nodes so that the size of the network remains the same across generations
            # number_of_nodes_to_remove = new_network.number_of_nodes() - self.population_size

            # history._add_number_of_nodes_to_remove(number_of_nodes_to_remove)

            # nodes = list(new_network.nodes)
            # nodes_fitness = [new_network.nodes[node]
            #                  ['node'].fitness for node in nodes]
            # nodes_fitness = np.array(nodes_fitness)
            # nodes_fitness_sorted = np.argsort(nodes_fitness) if self.is_maximization else np.argsort(
            #     nodes_fitness)[::-1]

            # nodes_to_remove = nodes_fitness_sorted[:number_of_nodes_to_remove] if number_of_nodes_to_remove > 0 else [
            # ]

            # # create a new network without the nodes to remove, copy the edges from the old network to the new network
            # edge_look_up_dict = {}
            # new_network_with_removed_nodes = nx.Graph()
            # for node in new_network.nodes:
            #     if not node in nodes_to_remove:
            #         node_index = new_network_with_removed_nodes.number_of_nodes()
            #         new_node_object = Node(
            #             node_index=node_index, individual=new_network.nodes[node][
            #                 'node'].individual, fitness=new_network.nodes[node]['node'].fitness
            #         )
            #         new_network_with_removed_nodes.add_node(
            #             node_index, node=new_node_object)

            #         edge_look_up_dict[node] = node_index

            # for edge in new_network.edges:
            #     if not edge[0] in nodes_to_remove and not edge[1] in nodes_to_remove:
            #         new_network_with_removed_nodes.add_edge(
            #             edge_look_up_dict[edge[0]], edge_look_up_dict[edge[1]])

            # # delete new_network
            # del new_network

            # # if a node does not have any edges, add m edges to the nodes using the non linear preferential attachment mechanism
            # for node in new_network_with_removed_nodes.nodes:
            #     if new_network_with_removed_nodes.degree(node) == 0:
            #         k = [self.A + math.pow(new_network_with_removed_nodes.degree(node), self.alpha)
            #              for node in new_network_with_removed_nodes.nodes]
            #         p = [k_i / sum(k) for k_i in k]
            #         selected_nodes = np.random.choice(
            #             new_network_with_removed_nodes.nodes, self.m, p=p, replace=False)
            #         for selected_node in selected_nodes:
            #             # if not self loop
            #             if node != selected_node:
            #                 new_network_with_removed_nodes.add_edge(
            #                     node, selected_node)

            # self.network = new_network_with_removed_nodes

            # if generation % 100 == 0:
            #     self.draw_manager.draw_network(self.network)
            #     self.draw_manager.community_network(self.network)
            # self.draw_manager.draw_number_of_nodes_to_remove(history)
            # self.draw_manager.centrality_histogram(self.network)
            #     self.draw_manager.centrality_network(
            #         self.network, func=nx.closeness_centrality)
            #     self.draw_manager.heat_heteroginity_network(self.network)
            #     self.draw_manager.heat_heteroginity_histogram(self.network)
            #     self.draw_manager.clustering_coefficient_histogram(
            #         self.network)

            fitnesses = self._evaluate_population(self.population)

            history._add_population_fitness(fitnesses, self.population)

        history.stop()
        print(history)
        return history
