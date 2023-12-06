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
from ynga_community import YNGA
import matplotlib.pyplot as plt

sys.path.append('.')


def main():
    benchmark_functions = get_functions_by_name([
        'f2',
    ])

    enhanced_nga_benchmark = Benchmark(genetic_algorithm=EnhancedNetworkGeneticAlgorithm, number_of_runs=10, benchmark_functions=benchmark_functions,
                                       number_of_generations=2000, population_size=125, number_elites=0, probability_crossover=0.95, probability_mutation=0.2, verbose=True, save_metrics=False, comment="official run mutation 0.2")

    nga_benchmark = Benchmark(genetic_algorithm=NetworkGeneticAlgorithm, number_of_runs=1, benchmark_functions=benchmark_functions,
                              number_of_generations=2000, population_size=125, number_elites=0, probability_crossover=0.95, probability_mutation=0.1, verbose=True, save_metrics=False, comment="official run mutation 0.2")

    ga_benchmark = Benchmark(genetic_algorithm=GeneticAlgorithm, number_of_runs=50, benchmark_functions=benchmark_functions,
                             number_of_generations=2000, population_size=125, number_elites=10, probability_crossover=0.95, probability_mutation=0.2, verbose=True, save_metrics=False, comment="official run mutation 0.2")

    ynga_benchmark = Benchmark(genetic_algorithm=YNGA, number_of_runs=1, benchmark_functions=benchmark_functions,
                               number_of_generations=2000, population_size=124, number_elites=20, probability_crossover=1, probability_mutation=0.25, verbose=True, save_metrics=False,)
    # runs = {}
    # start_mutation = 0.1
    # end_mutation = 0.8
    # step_mutation = 0.1
    # current_mutation = start_mutation
    # while current_mutation <= end_mutation:
    #     benchmark = Benchmark(genetic_algorithm=GeneticAlgorithmCopy, number_of_runs=2, benchmark_functions=benchmark_functions,
    #                           number_of_generations=2000, population_size=125, number_elites=10, probability_crossover=1, probability_mutation=current_mutation, verbose=True, save_metrics=False, comment="async")
    #     runs[current_mutation] = benchmark.run()
    #     current_mutation += step_mutation

    # # plot for each function the progression of best fitness for each mutation rate
    # for function in benchmark_functions:
    #     plt.figure()
    #     plt.title(f"Function {function.name}")
    #     plt.xlabel("Elites")
    #     plt.ylabel("Best fitness")
    #     plt.grid()
    #     plt.xlim(start_mutation, end_mutation)
    #     x = []
    #     y = []
    #     for mutation_rate, run in runs.items():
    #         x.append(mutation_rate)
    #         y.append(run[function.name])
    #     plt.plot(x, y, label=function.name)
    #     plt.legend()
    #     plt.show()

    # enhanced_nga_benchmark.run()
    nga_benchmark.run()
    # ga_benchmark.run()
    # ynga_benchmark.run()


if __name__ == '__main__':
    main()
