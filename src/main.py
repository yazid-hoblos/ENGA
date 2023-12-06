import random
from selection import SelectionType
from mutation import MutationType
from crossover import CrossOverType
from genetic_algorithm import GeneticAlgorithm, GAHistory
from enhanced_networked_genetic_algorithm import EnhancedNetworkGeneticAlgorithm
from networked_genetic_algorithm import NetworkGeneticAlgorithm
from benchmark import Benchmark, GABenchmarkFunction, FunctionCharacteristic
from benchmark_functions import get_functions_by_name, get_functions_with_characteristics, get_all_functions
import sys
import math
from CRGA import CRGA
from DCGA import DCGA
import matplotlib.pyplot as plt

sys.path.append('.')


def main():
    benchmark_functions = get_all_functions()

    # Enhanced Networked Genetic Algorithm
    enhanced_nga_benchmark = Benchmark(genetic_algorithm=EnhancedNetworkGeneticAlgorithm, number_of_runs=1, benchmark_functions=benchmark_functions,
                                       number_of_generations=2000, population_size=125, number_elites=0, probability_crossover=0.95, probability_mutation=0.1, verbose=True, save_metrics=True)

    # Networked Genetic Algorithm
    nga_benchmark = Benchmark(genetic_algorithm=NetworkGeneticAlgorithm, number_of_runs=1, benchmark_functions=benchmark_functions,
                              number_of_generations=2000, population_size=125, number_elites=0, probability_crossover=0.95, probability_mutation=0.1, verbose=True, save_metrics=True)

    # Genetic Algorithm
    ga_benchmark = Benchmark(genetic_algorithm=GeneticAlgorithm, number_of_runs=1, benchmark_functions=benchmark_functions,
                             number_of_generations=2000, population_size=125, number_elites=10, probability_crossover=0.95, probability_mutation=0.1, verbose=True, save_metrics=True)

    # Communities Representatives Genetic Algorithm
    crga_benchmark = Benchmark(genetic_algorithm=CRGA, number_of_runs=1, benchmark_functions=benchmark_functions,
                               number_of_generations=2000, population_size=124, number_elites=0, probability_crossover=0.95, probability_mutation=0.1, verbose=True, save_metrics=True)

    # Dynamic Communities Genetic Algorithm
    dcga_benchmark = Benchmark(genetic_algorithm=DCGA, number_of_runs=1, benchmark_functions=benchmark_functions,
                               number_of_generations=2000, population_size=125, number_elites=0, probability_crossover=0.95, probability_mutation=0.1, verbose=True, save_metrics=True)

    # enhanced_nga_benchmark.run()
    # nga_benchmark.run()
    # ga_benchmark.run()

    # Warning: crga_benchmark.run() and dcga_benchmark.run() will take a long time to run

    # crga_benchmark.run()
    # dcga_benchmark.run()


if __name__ == '__main__':
    main()
