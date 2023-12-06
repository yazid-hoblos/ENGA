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
