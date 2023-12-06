import numpy as np
from enum import Enum
from typing import Callable, Tuple
import random
from selection import Selection, SelectionType
from crossover import CrossOver, CrossOverType
from mutation import Mutation, MutationType
from tqdm import tqdm
import sys
import math
import time


class GAHistory:
    def __init__(self, is_maximization: bool):
        self._start_time = time.time()
        self._number_of_generations = 0
        self._is_maximization = is_maximization
        # used to plot best fitnesses with time
        # and average fitnesses with time
        self._fitnesses = []
        self._avg_fitnesses = []
        self._populations = []
        self._best_fitnesses = []
        self._networks = []
        self._number_of_nodes_to_remove = []
        self.extra = {}

    def stop(self):
        self._end_time = time.time()
        self._total_time = self._end_time - self._start_time

    def _add_population_fitness(self, fitnesses, population):
        self._number_of_generations += 1
        self._fitnesses.append(fitnesses)
        self._avg_fitnesses.append(np.mean(fitnesses))
        self._populations.append(population)
        self._best_fitnesses.append(
            max(fitnesses) if self._is_maximization else min(fitnesses))

    def _add_network(self, network):
        self._networks.append(network)

    def _add_number_of_nodes_to_remove(self, number_of_nodes_to_remove):
        self._number_of_nodes_to_remove.append(number_of_nodes_to_remove)

    def get_number_of_nodes_to_remove(self):
        return self._number_of_nodes_to_remove

    def get_total_time(self):
        return f"{self._total_time:.2f} seconds"

    def get_number_of_generations(self):
        return self._number_of_generations

    def get_fitnesses(self):
        return self._fitnesses

    def get_populations(self):
        return self._populations

    def get_best_fitnesses(self):
        return self._best_fitnesses

    def get_networks(self):
        return self._networks

    def __str__(self):
        avg_fitnesses = np.mean(self._best_fitnesses)
        best_fitness = np.max(
            self._best_fitnesses) if self._is_maximization else np.min(self._best_fitnesses)

        return f"\n\n---------------------------------------\n\n" + \
            f"Number of generations: {self._number_of_generations}\n" + \
            f"Total time: {self.get_total_time()}\n" + \
            f"Average fitness: {avg_fitnesses}\n" + \
            f"Best fitness: {best_fitness}\n" + \
            f"Best individual: {self._populations[-1][np.argmax(self._fitnesses[-1]) if self._is_maximization else np.argmin(self._fitnesses[-1])]}" + \
            f"\n\n---------------------------------------\n\n"


class GeneticAlgorithm(Selection, CrossOver, Mutation):
    def __init__(self, number_of_genes: int, domain, number_of_generations: int, population_size: int, number_elites: int,
                 probability_crossover: float, probability_mutation: float, decimal_precision: int, create_random_individual, fitness_function,
                 is_maximization: bool, verbose: bool = True, random_seed: int = None):
        """
        A constructor for the genetic algorithm class

        Parameters
        ----------
        number_of_genes : int
            The number of genes in each individual

        domain: 
            The domain of each gene

        number_of_generations : int
            The number of generations to run the algorithm for

        population_size : int
            The size of the population

        number_elites : int
            The number of elites to keep in the population

        probability_crossover : float
            The probability of crossover

        probability_mutation : float
            The probability of mutation

        crossover_type : CrossOverType
            The type of crossover to use

        mutation_type : MutationType
            The type of mutation to use

        selection_type : SelectionType
            The type of selection to use

        create_random_individual : Callable[[], Tuple[int]]
            A function that creates a random individual

        fitness_function : Callable[[Tuple[int]], float]
            A function that evaluates the fitness of an individual

        is_maximization : bool
            Whether the fitness function is a maximization or minimization function

        verbose : bool
            Whether to print the progress of the algorithm

        random_seed : int
            The random seed to use

        Returns
        -------
        None
        """

        Selection.__init__(self, SelectionType.ROULETTE,
                           number_of_parents=2, is_maximization=is_maximization)
        CrossOver.__init__(
            self, prob_crossover=probability_crossover, crossover_type=CrossOverType.ONE_POINT)
        Mutation.__init__(
            self, prob_mutation=probability_mutation, mutation_type=MutationType.FLIP, domain=domain, decimal_precision=decimal_precision)

        if random_seed is not None:
            random.seed(random_seed)

        self.n_genes = number_of_genes
        self.domain = domain
        self.decimal_precision = decimal_precision
        self.n_generations = number_of_generations
        self.popsize = population_size
        self.n_elites = number_elites
        self.create_random_individual = create_random_individual
        self.fitness_function = fitness_function
        self.is_maximization = is_maximization
        self.verbose = verbose

        self.population = [self.create_random_individual(self.n_genes, self.domain, self.decimal_precision)
                           for i in range(self.popsize)]

    def _evaluate(self, individual) -> float:
        """
        Evaluate the population

        Parameters
        ----------
        individual
            The individual to evaluate

        Returns
        -------
        float
            The fitness of the individual
        """
        return self.fitness_function(individual)

    def _evaluate_population(self, population) -> np.ndarray:
        """
        Evaluate the population

        Parameters
        ----------
        population
            The population to evaluate

        Returns
        -------
        float
            The fitness of the population
        """
        return np.array([self._evaluate(individual)
                         for individual in population])

    def run(self):
        """
        Run the genetic algorithm

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        if self.verbose:
            print("Running genetic algorithm...")

        history = GAHistory(is_maximization=self.is_maximization)

        number_of_generation_best_ind_fitness_stayed_same = 0
        best_individual_fitness = float('inf')
        best_individual_fitness_threshold = 0.0001

        for generation in tqdm(range(self.n_generations)) if self.verbose else range(self.n_generations):
            # Evaluate the population
            fitnesses = self._evaluate_population(self.population)
            history._add_population_fitness(fitnesses, self.population)

            ind = np.argsort(fitnesses)
            if not self.is_maximization:
                ind = ind[::-1]

            if sum(fitnesses) == 0:
                break

            # if number_of_generation_best_ind_fitness_stayed_same >= max(200, 0.5 * self.n_generations):
            #     break

            new_population = [self.population[i]
                              for i in ind[-self.n_elites:]] if self.n_elites > 0 else []

            while len(new_population) < self.popsize:
                # Select two parents
                parent1, parent2 = self.select_parents(fitnesses)

                # Crossover
                child1, child2 = self.crossover(
                    self.population[parent1], self.population[parent2])

                # Mutate
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                new_population.append(child1)
                new_population.append(child2)

            self.population = new_population

        history.stop()
        print(history)

        return history
