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


class NetworkGeneticAlgorithm(GeneticAlgorithm, NetworkAssignment, NetworkModel, NetworkSelection):
    def __init__(self, number_of_genes: int, domain, number_of_generations: int, population_size: int, number_elites: int,
                 probability_crossover: float, probability_mutation: float, decimal_precision: int, create_random_individual, fitness_function,
                 is_maximization: bool, verbose: bool = True, random_seed: int = None):

        super().__init__(number_of_genes, domain, number_of_generations, population_size, number_elites, probability_crossover, probability_mutation, decimal_precision,
                         create_random_individual, fitness_function, is_maximization, verbose, random_seed)

        NetworkModel.__init__(
            self, NetworkModelType.BARABASI_ALBERT, population_size)
        NetworkAssignment.__init__(
            self, NetworkAssignmentType.RANDOM_ASSIGNMENT)
        NetworkSelection.__init__(self, initial_selection_type=InitialSelectionType.ROULETTE,
                                  secondary_selection_type=SecondarySelectionType.RANDOM_NEIGHBOR)

        # Adds the node object to each node, i.e. self.network.nodes[index]['node']
        self._assign_individuals_to_nodes()

        self.draw_manager = DrawManager(self.is_maximization)
        # self.draw_manager.draw_network(self.network)

    # Override the default selection method of GA
    def select_parents(self):
        return NetworkSelection.select_parents(self)

    def get_who_is_better(self, child1, child2):
        return (child1 if self._evaluate(child1) >= self._evaluate(child2) else child2) if self.is_maximization else ((child1 if self._evaluate(child1) <= self._evaluate(child2) else child2))

    def run(self):
        """
        NGA algorithm of the article networked evolutionary algorithms
        """

        history = GAHistory(is_maximization=self.is_maximization)

        for generation in tqdm(range(self.n_generations)) if self.verbose else range(self.n_generations):

            # Create a copy of the network
            new_network = self.network.copy()
            history._add_network(new_network)

            fitnesses = self._evaluate_population(
                [self.network.nodes[node]['node'].individual for node in self.network.nodes])

            if sum(fitnesses) == 0:
                break

            history._add_population_fitness(
                fitnesses, [self.network.nodes[node]['node'].individual for node in self.network.nodes])

            ind = np.argsort(fitnesses)

            if not self.is_maximization:
                ind = ind[::-1]

            # Elites, array of indicies of parents (nodes) that should not be replaced
            elites = [i for i in ind[-self.n_elites:]
                      ] if self.n_elites > 0 else []

            offsprings = []

            for node in self.network.nodes:
                parent1 = self.network.nodes[node]['node']
                parent2 = NetworkSelection._secondary_node_selection(
                    self, parent1)
                # Crossover the parents
                child1, child2 = self.crossover(
                    parent1.individual, parent2.individual)

                # Mutate the children
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                chosen = self.get_who_is_better(child1, child2)

                offsprings.append(chosen)

            combined_fitness = list(fitnesses)
            combined_fitness.extend([self._evaluate(individual)
                                    for individual in offsprings])

            seperator_indx = len(offsprings) - 1
            selected_offsprings_indicies = []
            for node in new_network.nodes:
                new_network.nodes[node]['selected'] = False

            # mark the elites as selected
            for elite in elites:
                new_network.nodes[elite]['selected'] = True

            i = self.n_elites
            while i < self.population_size:
                selected = self._roulette_selection(combined_fitness, 1)[0]
                if selected > seperator_indx and selected not in selected_offsprings_indicies:
                    i += 1
                    selected_offsprings_indicies.append(selected)

                if selected <= seperator_indx:
                    node = new_network.nodes[selected]
                    if node['selected'] == False:
                        i += 1
                        new_network.nodes[selected]['selected'] = True

            selected_offsprings = [offsprings[i - len(offsprings) - 1]
                                   for i in selected_offsprings_indicies]

            for node in new_network.nodes:
                # If it was not selected by roullette => remove all its edges
                if new_network.nodes[node]['selected'] == False:
                    neighbors = new_network.neighbors(node)

                    # this hack, because neighbors is a dict key iterator, thus we cannot remove edge if we loop over it
                    neighbors2 = [n for n in neighbors]
                    for neighbor in neighbors2:
                        new_network.remove_edge(node, neighbor)

            for node in new_network.nodes:
                # If it was not selected by roullette => replace it with an offspring to avoid removing a node
                if new_network.nodes[node]['selected'] == False:
                    index = random.randint(
                        0, len(selected_offsprings) - 1)
                    random_offspring = selected_offsprings.pop(index)

                    degrees = []
                    selected_parents = []

                    for node2 in new_network.nodes:
                        if new_network.nodes[node2]['selected'] == True:
                            degrees.append(new_network.degree(node2))
                            selected_parents.append(node2)

                    new_network.nodes[node]['node'] = Node(
                        node_index=node, individual=random_offspring, fitness=self._evaluate(random_offspring))
                    new_network.nodes[node]['selected'] = True

                    m = 4
                    total_degrees = sum(degrees)
                    nodes_with_prob = [(node, degree / total_degrees)
                                       for node, degree in zip(selected_parents, degrees)]

                    # Initialize an empty list to store the choices
                    choices = []

                    # Perform preferential attachment to select m nodes without duplicates
                    for _ in range(m):
                        selected_node = random.choices(nodes_with_prob)[0][0]
                        while selected_node in choices or selected_node == node:
                            selected_node = random.choices(
                                nodes_with_prob)[0][0]
                        new_network.add_edge(node, selected_node)

            for node in new_network.nodes:
                if new_network.degree(node) == 0:
                    degrees = []
                    selected_parents = []

                    for node2 in new_network.nodes:
                        if new_network.degree(node2) != 0:
                            degrees.append(new_network.degree(node2))
                            selected_parents.append(node2)

                    m = 4
                    total_degrees = sum(degrees)
                    nodes_with_prob = [(node, degree / total_degrees)
                                       for node, degree in zip(selected_parents, degrees)]

                    # Initialize an empty list to store the choices
                    choices = []

                    # Perform preferential attachment to select m nodes without duplicates
                    for _ in range(m):
                        selected_node = random.choices(nodes_with_prob)[0][0]
                        while selected_node in choices or selected_node == node:
                            selected_node = random.choices(
                                nodes_with_prob)[0][0]
                        new_network.add_edge(node, selected_node)

            # Replace the OLD network with the NEW network
            self.network = new_network

        history.stop()
        print(history)
        # self.draw_manager.draw_avg_fitness(history)
        return history


class OnlySelectionNGA(GeneticAlgorithm, NetworkAssignment, NetworkModel, NetworkSelection):
    def __init__(self, number_of_genes: int, domain, number_of_generations: int, population_size: int, number_elites: int,
                 probability_crossover: float, probability_mutation: float, decimal_precision: int, create_random_individual, fitness_function,
                 is_maximization: bool, verbose: bool = True, random_seed: int = None):

        super().__init__(number_of_genes, domain, number_of_generations, population_size, number_elites, probability_crossover, probability_mutation, decimal_precision,
                         create_random_individual, fitness_function, is_maximization, verbose, random_seed)

        NetworkModel.__init__(
            self, NetworkModelType.BARABASI_ALBERT, population_size)
        NetworkAssignment.__init__(
            self, NetworkAssignmentType.RANDOM_ASSIGNMENT)
        NetworkSelection.__init__(self, initial_selection_type=InitialSelectionType.ROULETTE,
                                  secondary_selection_type=SecondarySelectionType.RANDOM_NEIGHBOR)

        # Adds the node object to each node, i.e. self.network.nodes[index]['node']
        self._assign_individuals_to_nodes()

        self.draw_manager = DrawManager(self.is_maximization)
        # self.draw_manager.draw_network(self.network)

    # Override the default selection method of GA
    def select_parents(self):
        return NetworkSelection.select_parents(self)

    def get_who_is_better(self, child1, child2):
        return (child1 if self._evaluate(child1) >= self._evaluate(child2) else child2) if self.is_maximization else ((child1 if self._evaluate(child1) <= self._evaluate(child2) else child2))

    def run(self):
        """
        NGA algorithm of the article networked evolutionary algorithms
        """

        history = GAHistory(is_maximization=self.is_maximization)

        for generation in tqdm(range(self.n_generations)) if self.verbose else range(self.n_generations):

            # Create a copy of the network
            new_network = self.network.copy()
            history._add_network(new_network)

            fitnesses = self._evaluate_population(
                [self.network.nodes[node]['node'].individual for node in self.network.nodes])

            if sum(fitnesses) == 0:
                break

            history._add_population_fitness(
                fitnesses, [self.network.nodes[node]['node'].individual for node in self.network.nodes])

            ind = np.argsort(fitnesses)

            if not self.is_maximization:
                ind = ind[::-1]

            # Elites, array of indicies of parents (nodes) that should not be replaced
            elites = [i if self.n_elites > 0 else []
                      for i in ind[-self.n_elites:]]

            offsprings = []

            for node in self.network.nodes:
                parent1 = self.network.nodes[node]['node']
                parent2 = NetworkSelection._secondary_node_selection(
                    self, parent1)

                child1, child2 = parent1.individual, parent2.individual

                chosen = self.get_who_is_better(child1, child2)

                offsprings.append(chosen)

            combined_fitness = list(fitnesses)
            combined_fitness.extend([self._evaluate(individual)
                                    for individual in offsprings])

            seperator_indx = len(offsprings) - 1
            selected_offsprings_indicies = []
            for node in new_network.nodes:
                new_network.nodes[node]['selected'] = False

            i = 0
            while i < self.population_size:
                # select the first population_size indicies from the sorted fitnesses
                selected = self._roulette_selection(combined_fitness, 1)[0]

                while selected in elites:
                    selected = self._roulette_selection(combined_fitness, 1)[0]

                if selected > seperator_indx and selected not in selected_offsprings_indicies:
                    i += 1
                    selected_offsprings_indicies.append(selected)

                if selected <= seperator_indx:
                    node = new_network.nodes[selected]
                    if node['selected'] == False:
                        i += 1
                        new_network.nodes[selected]['selected'] = True

            selected_offsprings = [offsprings[i - len(offsprings) - 1]
                                   for i in selected_offsprings_indicies]

            for node in new_network.nodes:
                # If it was not selected by roullette => remove all its edges
                if new_network.nodes[node]['selected'] == False:
                    neighbors = new_network.neighbors(node)

                    # this hack, because neighbors is a dict key iterator, thus we cannot remove edge if we loop over it
                    neighbors2 = [n for n in neighbors]
                    for neighbor in neighbors2:
                        new_network.remove_edge(node, neighbor)

            for node in new_network.nodes:
                # If it was not selected by roullette => replace it with an offspring to avoid removing a node
                if new_network.nodes[node]['selected'] == False:
                    index = random.randint(
                        0, len(selected_offsprings) - 1)
                    random_offspring = selected_offsprings.pop(index)

                    degrees = []
                    selected_parents = []

                    for node2 in new_network.nodes:
                        if new_network.nodes[node2]['selected'] == True:
                            degrees.append(new_network.degree(node2))
                            selected_parents.append(node2)

                    new_network.nodes[node]['node'] = Node(
                        node_index=node, individual=random_offspring, fitness=self._evaluate(random_offspring))
                    new_network.nodes[node]['selected'] = True

                    m = 4
                    total_degrees = sum(degrees)
                    nodes_with_prob = [(node, degree / total_degrees)
                                       for node, degree in zip(selected_parents, degrees)]

                    # Initialize an empty list to store the choices
                    choices = []

                    # Perform preferential attachment to select m nodes without duplicates
                    for _ in range(m):
                        selected_node = random.choices(nodes_with_prob)[0][0]
                        while selected_node in choices or selected_node == node:
                            selected_node = random.choices(
                                nodes_with_prob)[0][0]
                        new_network.add_edge(node, selected_node)

            for node in new_network.nodes:
                if new_network.degree(node) == 0:
                    degrees = []
                    selected_parents = []

                    for node2 in new_network.nodes:
                        if new_network.degree(node2) != 0:
                            degrees.append(new_network.degree(node2))
                            selected_parents.append(node2)

                    m = 4
                    total_degrees = sum(degrees)
                    nodes_with_prob = [(node, degree / total_degrees)
                                       for node, degree in zip(selected_parents, degrees)]

                    # Initialize an empty list to store the choices
                    choices = []

                    # Perform preferential attachment to select m nodes without duplicates
                    for _ in range(m):
                        selected_node = random.choices(nodes_with_prob)[0][0]
                        while selected_node in choices or selected_node == node:
                            selected_node = random.choices(
                                nodes_with_prob)[0][0]
                        new_network.add_edge(node, selected_node)

            # Replace the OLD network with the NEW network
            self.network = new_network

        history.stop()
        print(history)
        return history
