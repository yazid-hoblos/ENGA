from typing import List
from genetic_algorithm import GAHistory, GeneticAlgorithm
from selection import SelectionType
from mutation import MutationType
from crossover import CrossOverType
import prettytable
import math
import random
import time
import os
from enum import Enum
from utils.drawable import DrawManager


class FunctionCharacteristic(Enum):
    UNIMODAL = 1
    MULTIMODAL = 2
    SEPARETABLE = 3
    NON_SEPARETABLE = 4
    SYMMETRIC = 5
    NON_SYMMETRIC = 6
    PERMUTATION_INVARIENT = 7
    PERMUTATION_VARIANT = 8
    NP_HARD = 9


class GABenchmarkFunction:
    def __init__(self, name, lower_bound, upper_bound, optimal_value, number_of_genes, create_random_individual, fitness_function, is_maximization: bool, decimal_precision: int, function_characteristics: List[FunctionCharacteristic] = []):
        self.name = name
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.optimal_value = optimal_value
        self.number_of_genes = number_of_genes
        self.create_random_individual = create_random_individual
        self.fitness_function = fitness_function
        self.is_maximization = is_maximization
        self.decimal_precision = decimal_precision
        self.function_characteristics = function_characteristics


class Benchmark:
    def __init__(self, genetic_algorithm: GeneticAlgorithm, number_of_runs: int, benchmark_functions: List[GABenchmarkFunction], number_of_generations: int, population_size: int, number_elites: int, probability_crossover: float, probability_mutation: float, verbose: bool = False, save_metrics: bool = False, comment: str = ""):
        self.genetic_algorithm = genetic_algorithm
        self.number_of_runs = number_of_runs
        self.benchmark_functions = benchmark_functions
        self.number_of_generations = number_of_generations
        self.population_size = population_size
        self.number_elites = number_elites
        self.probability_crossover = probability_crossover
        self.probability_mutation = probability_mutation
        self.verbose = verbose
        self.comment = comment
        self.save_metrics = save_metrics
        self.table = None

    def run(self):
        self.table = prettytable.PrettyTable()
        self.table.field_names = ["Function Name", "Min or Max", "Optimal Value", "Best Fitness",  "Avg Error", "Std Error",  "Avg Generations",
                                  "Avg Best Fitness", "Std Best Fitness", "Avg Total Time"]
        best_fitnesses = {benchmark_function.name: []
                          for benchmark_function in self.benchmark_functions}

        for benchmark_function in self.benchmark_functions:
            self.draw_manager = DrawManager(benchmark_function.is_maximization)
            list_history = []
            for _ in range(self.number_of_runs):
                genetic_algorithm_instance = self.genetic_algorithm(number_of_genes=benchmark_function.number_of_genes, domain=[
                                                                    benchmark_function.lower_bound, benchmark_function.upper_bound], number_of_generations=self.number_of_generations, population_size=self.population_size, number_elites=self.number_elites, probability_crossover=self.probability_crossover, probability_mutation=self.probability_mutation, decimal_precision=benchmark_function.decimal_precision, create_random_individual=benchmark_function.create_random_individual, fitness_function=benchmark_function.fitness_function, is_maximization=benchmark_function.is_maximization, verbose=self.verbose, random_seed=937162211)
                history = genetic_algorithm_instance.run()
                list_history.append(history)
                if self.save_metrics:
                    # Network Specific Metrics i.e. the functions itself will check if the history have the networks data
                    self.draw_manager.draw_avg_path_length(
                        history, name=f"avg_path_length_{self.genetic_algorithm.__name__}_{benchmark_function.name}_run_{_}")
                    self.draw_manager.draw_avg_clustering_coefficient(
                        history, name=f"avg_clustering_coefficient_{self.genetic_algorithm.__name__}_{benchmark_function.name}_run_{_}")
                    self.draw_manager.draw_avg_degree(
                        history, name=f"avg_degree_{self.genetic_algorithm.__name__}_{benchmark_function.name}_run_{_}")
                    self.draw_manager.draw_degree_exponent(
                        history, name=f"degree_exponent_{self.genetic_algorithm.__name__}_{benchmark_function.name}_run_{_}")
                    self.draw_manager.draw_number_of_nodes_to_remove(
                        history, name=f"number_of_nodes_to_remove_{self.genetic_algorithm.__name__}_{benchmark_function.name}_run_{_}")

                    self.draw_manager.draw_selection_pressure(
                        history, name=f"selection_pressure_{self.genetic_algorithm.__name__}_{benchmark_function.name}_run_{_}")
                    self.draw_manager.draw_avg_phenotype_diversity(
                        history, name=f"avg_phenotype_diversity_{self.genetic_algorithm.__name__}_{benchmark_function.name}_run_{_}")
                    self.draw_manager.draw_avg_genotype_diversity(
                        history, name=f"avg_genotype_diversity_{self.genetic_algorithm.__name__}_{benchmark_function.name}_run_{_}")
                    self.draw_manager.draw_avg_fitness(
                        history, name=f"avg_fitness_{self.genetic_algorithm.__name__}_{benchmark_function.name}_run_{_}", optimum=benchmark_function.optimal_value)

            metrics = self.get_metrics(benchmark_function, list_history)
            best_fitnesses[benchmark_function.name].append(
                metrics[7])
            self.table.add_row(metrics)
            self.display_results()

        if self.verbose:
            self.save_results()

        return best_fitnesses

    def get_metrics(self, func, his):
        func_name = func.name
        min_or_max = "Max" if func.is_maximization else "Min"
        optimal_value = func.optimal_value
        avg_generations = sum([history._number_of_generations
                               for history in his])/self.number_of_runs
        avg_best_fitness = sum([min(history._best_fitnesses)
                                for history in his])/self.number_of_runs
        std_best_fitness = math.sqrt(sum([(history._best_fitnesses[-1]-avg_best_fitness)**2
                                          for history in his])/self.number_of_runs)
        avg_total_time = sum([history._total_time
                              for history in his])/self.number_of_runs
        best_fitness = max([history._best_fitnesses
                            for history in his]) if func.is_maximization else min([min(history._best_fitnesses) for history in his])
        print(best_fitness)
        avg_error = sum([abs(history._best_fitnesses[-1] - optimal_value)
                         for history in his])/self.number_of_runs
        std_error = math.sqrt(sum([(abs(history._best_fitnesses[-1] - optimal_value)-avg_error)**2
                                   for history in his])/self.number_of_runs)

        return [func_name, min_or_max, optimal_value, best_fitness,  avg_error, std_error,  avg_generations, avg_best_fitness, std_best_fitness, f"{avg_total_time:.2f} seconds"]

    def display_results(self):
        str_to_print = f"Benchmark for {self.genetic_algorithm.__name__}\n"
        str_to_print += f"Number of runs: {self.number_of_runs}\n"
        str_to_print += f"Number of generations: {self.number_of_generations}\n"
        str_to_print += f"Population size: {self.population_size}\n"
        str_to_print += f"Number of elites: {self.number_elites}\n"
        str_to_print += f"Probability of crossover: {self.probability_crossover * 100}%\n"
        str_to_print += f"Probability of mutation: {self.probability_mutation * 100}%\n"
        if self.comment != "":
            str_to_print += f"Comment: {self.comment}\n"

        str_to_print += str(self.table)

        print(str_to_print)

    def save_results(self):
        str_to_print = f"Benchmark for {self.genetic_algorithm.__name__}\n"
        str_to_print += f"Number of runs: {self.number_of_runs}\n"
        str_to_print += f"Number of generations: {self.number_of_generations}\n"
        str_to_print += f"Population size: {self.population_size}\n"
        str_to_print += f"Number of elites: {self.number_elites}\n"
        str_to_print += f"Probability of crossover: {self.probability_crossover * 100}%\n"
        str_to_print += f"Probability of mutation: {self.probability_mutation * 100}%\n"
        if self.comment != "":
            str_to_print += f"Comment: {self.comment}\n"

        str_to_print += str(self.table)

        print(str_to_print)

        if os.path.exists("./logs/results") == False:
            os.mkdir("./logs/results")

        # formatted D - H:M
        current_time = time.strftime("%d_%H_%M")
        filename = "./logs/results/benchmark_" + \
            str(self.genetic_algorithm.__name__) + \
            "_" + str(current_time) + ".txt"

        with open(filename, "w") as f:
            f.write(str_to_print)
            f.write("\n")
