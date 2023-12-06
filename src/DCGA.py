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


class DCGA(GeneticAlgorithm, NetworkAssignment, NetworkModel, NetworkSelection):
    def __init__(self, number_of_genes: int, domain, number_of_generations: int, population_size: int, number_elites: int,
                 probability_crossover: float, probability_mutation: float, decimal_precision: int, create_random_individual, fitness_function,
                 is_maximization: bool, verbose: bool = True, random_seed: int = None):

        super().__init__(number_of_genes, domain, number_of_generations, population_size, number_elites, probability_crossover, probability_mutation, decimal_precision,
                         create_random_individual, fitness_function, is_maximization, verbose, random_seed)

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

            # use strict mating on the chosen nodes
            new_population = []
            for parent1_index in self.network.nodes:
                parent1 = self.network.nodes[parent1_index]['node'].individual
                parent2 = random.choice(
                    list(self.network.neighbors(parent1_index)))
                parent2 = self.network.nodes[parent2]['node'].individual

                child1, child2 = self.crossover(
                    parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                new_population.append(child1)
                new_population.append(child2)

            self.population = new_population
            history._add_population_fitness(fitnesses, self.population)

        history.stop()
        print(history)
        return history
