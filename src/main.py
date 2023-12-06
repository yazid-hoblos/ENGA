import random
from selection import SelectionType
from mutation import MutationType
from crossover import CrossOverType
from genetic_algorithm import GeneticAlgorithm, GAHistory
from enhanced_networked_genetic_algorithm import EnhancedNetworkGeneticAlgorithm
from networked_genetic_algorithm import NetworkGeneticAlgorithm
from benchmark import Benchmark, GABenchmarkFunction, FunctionCharacteristic
from benchmark_functions import get_functions_by_name, get_functions_with_characteristics
import sys
import math
from CRGA import CRGA
import matplotlib.pyplot as plt

sys.path.append('.')


def main():
    benchmark_functions = get_functions_by_name(['f3'])
    benchmark_function = benchmark_functions[0]

    enga = EnhancedNetworkGeneticAlgorithm(number_of_genes=benchmark_function.number_of_genes, domain=[benchmark_function.lower_bound, benchmark_function.upper_bound], fitness_function=benchmark_function.fitness_function,
                                           is_maximization=benchmark_function.is_maximization, population_size=125, number_elites=0, probability_crossover=0.95, probability_mutation=0.2, verbose=True, save_metrics=False, comment="official run mutation 0.2")

    nga = NetworkGeneticAlgorithm(number_of_genes=benchmark_function.number_of_genes, domain=[benchmark_function.lower_bound, benchmark_function.upper_bound], fitness_function=benchmark_function.fitness_function,
                                  is_maximization=benchmark_function.is_maximization, population_size=125, number_elites=0, probability_crossover=0.95, probability_mutation=0.2, verbose=True, save_metrics=False, comment="official run mutation 0.2")

    ga = GeneticAlgorithm(number_of_genes=benchmark_function.number_of_genes, domain=[benchmark_function.lower_bound, benchmark_function.upper_bound], fitness_function=benchmark_function.fitness_function,
                          is_maximization=benchmark_function.is_maximization, population_size=125, number_elites=10, probability_crossover=0.95, probability_mutation=0.2, verbose=True, save_metrics=False, comment="official run mutation 0.2")

    history_enga = enga.run()
    history_nga = nga.run()
    history_ga = ga.run()

    # plot using different line styles, the history._best_fitnesses for each algorithm on same plot
    plt.plot(history_enga._best_fitnesses, label='ENGA', linestyle='dashed')
    plt.plot(history_nga._best_fitnesses, label='NGA', linestyle='dotted')
    plt.plot(history_ga._best_fitnesses, label='GA', linestyle='solid')
    plt.legend()
    plt.show()

    enhanced_nga_benchmark = Benchmark(genetic_algorithm=EnhancedNetworkGeneticAlgorithm, number_of_runs=1, benchmark_functions=benchmark_functions,
                                       number_of_generations=2000, population_size=125, number_elites=0, probability_crossover=0.95, probability_mutation=0.1, verbose=True, save_metrics=True, comment="official run mutation 0.2")

    nga_benchmark = Benchmark(genetic_algorithm=NetworkGeneticAlgorithm, number_of_runs=1, benchmark_functions=benchmark_functions,
                              number_of_generations=2000, population_size=125, number_elites=0, probability_crossover=0.95, probability_mutation=0.1, verbose=True, save_metrics=True, comment="official run mutation 0.2")

    ga_benchmark = Benchmark(genetic_algorithm=GeneticAlgorithm, number_of_runs=1, benchmark_functions=benchmark_functions,
                             number_of_generations=2000, population_size=125, number_elites=10, probability_crossover=0.95, probability_mutation=0.1, verbose=True, save_metrics=True, comment="official run mutation 0.2")

    crga_benchmark = Benchmark(genetic_algorithm=CRGA, number_of_runs=1, benchmark_functions=benchmark_functions,
                               number_of_generations=2000, population_size=124, number_elites=0, probability_crossover=0.95, probability_mutation=0.1, verbose=True, save_metrics=True, comment="official run mutation 0.2")

    # enhanced_nga_benchmark.run()
    # nga_benchmark.run()
    # ga_benchmark.run()
    # crga_benchmark.run()


if __name__ == '__main__':
    main()
