import random
from selection import SelectionType
from mutation import MutationType
from crossover import CrossOverType
from genetic_algorithm_mod import GeneticAlgorithmModified
from genetic_algorithm import GeneticAlgorithm, GAHistory
from enhanced_networked_genetic_algorithm import SAEnhancedNetworkGeneticAlgorithm, EnhancedNetworkGeneticAlgorithm, OnlySelectionENGA
from networked_genetic_algorithm import NetworkGeneticAlgorithm, OnlySelectionNGA
from benchmark import Benchmark, GABenchmarkFunction, FunctionCharacteristic
from benchmark_functions import get_functions_by_name, get_cec_2017_functions, get_functions_with_characteristics, get_all_paper_functions, final_chosen_functions
from genetic_algorithm_copy import GeneticAlgorithmCopy
import sys
import math
from ga_random import GA_RANDOM
from ynga_community import YNGA
from ynga2 import YNGA2
from ynga3 import YNGA3
from genetic_algorithm_strict_selection import GeneticAlgorithmStrictSelection
import matplotlib.pyplot as plt

sys.path.append('.')


def main():
    benchmark_functions = get_functions_by_name([
        'f2', 'f4', 'f5', 'f6', 'f7', 'dixonprice', 'discus', 'schwefel'
    ])

    enhanced_nga_benchmark = Benchmark(genetic_algorithm=EnhancedNetworkGeneticAlgorithm, number_of_runs=10, benchmark_functions=benchmark_functions,
                                       number_of_generations=2000, population_size=125, number_elites=0, probability_crossover=0.95, probability_mutation=0.2, verbose=True, save_metrics=False, comment="official run mutation 0.2")

    only_selection_enga_benchmark = Benchmark(genetic_algorithm=OnlySelectionENGA, number_of_runs=1, benchmark_functions=benchmark_functions, number_of_generations=2000,
                                              population_size=125, number_elites=0, probability_crossover=0.95, probability_mutation=0.2, verbose=True, save_metrics=True, comment="s")

    saenhanced_nga_benchmark = Benchmark(genetic_algorithm=SAEnhancedNetworkGeneticAlgorithm, number_of_runs=10, benchmark_functions=benchmark_functions,
                                         number_of_generations=2000, population_size=125, number_elites=10, probability_crossover=1, probability_mutation=0.5, verbose=True, save_metrics=False, comment="strict mating")

    nga_benchmark = Benchmark(genetic_algorithm=NetworkGeneticAlgorithm, number_of_runs=10, benchmark_functions=benchmark_functions,
                              number_of_generations=2000, population_size=125, number_elites=0, probability_crossover=0.95, probability_mutation=0.1, verbose=True, save_metrics=False, comment="official run mutation 0.2")

    ga_random_benchmark = Benchmark(genetic_algorithm=GA_RANDOM, number_of_runs=1, benchmark_functions=benchmark_functions,
                                    number_of_generations=2000, population_size=125, number_elites=0, probability_crossover=0.95, probability_mutation=0.1, verbose=True, save_metrics=False, comment="GA RANDOM")

    only_selection_nga_benchmark = Benchmark(genetic_algorithm=OnlySelectionNGA, number_of_runs=1, benchmark_functions=benchmark_functions,
                                             number_of_generations=2000, population_size=125, number_elites=0, probability_crossover=0.95, probability_mutation=0.1, verbose=True, save_metrics=False,)

    ga_mod_benchmark = Benchmark(genetic_algorithm=GeneticAlgorithmModified, number_of_runs=1, benchmark_functions=benchmark_functions,
                                 number_of_generations=2000, population_size=125, number_elites=0, probability_crossover=1, probability_mutation=0.4, verbose=True, save_metrics=False,)

    ga_benchmark = Benchmark(genetic_algorithm=GeneticAlgorithm, number_of_runs=50, benchmark_functions=benchmark_functions,
                             number_of_generations=2000, population_size=125, number_elites=10, probability_crossover=0.95, probability_mutation=0.2, verbose=True, save_metrics=False, comment="official run mutation 0.2")

    ga_copy_benchmark = Benchmark(genetic_algorithm=GeneticAlgorithmCopy, number_of_runs=5, benchmark_functions=benchmark_functions,
                                  number_of_generations=2000, population_size=125, number_elites=20, probability_crossover=0.95, probability_mutation=0.2, verbose=True, save_metrics=False,)

    ga_strict_benchmark = Benchmark(genetic_algorithm=GeneticAlgorithmStrictSelection, number_of_runs=1, benchmark_functions=benchmark_functions,
                                    number_of_generations=2000, population_size=125, number_elites=20, probability_crossover=1, probability_mutation=0.1, verbose=False, save_metrics=False,)

    ynga_benchmark = Benchmark(genetic_algorithm=YNGA, number_of_runs=1, benchmark_functions=benchmark_functions,
                               number_of_generations=2000, population_size=124, number_elites=20, probability_crossover=1, probability_mutation=0.25, verbose=True, save_metrics=False,)

    ynga2_benchmark = Benchmark(genetic_algorithm=YNGA2, number_of_runs=3, benchmark_functions=benchmark_functions,
                                number_of_generations=2000, population_size=125, number_elites=20, probability_crossover=1, probability_mutation=0.1, verbose=True, save_metrics=False,)
    ynga3_benchmark = Benchmark(genetic_algorithm=YNGA3, number_of_runs=3, benchmark_functions=benchmark_functions,
                                number_of_generations=2000, population_size=124, number_elites=20, probability_crossover=1, probability_mutation=0.2, verbose=True, save_metrics=False,)

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

    enhanced_nga_benchmark.run()
    # only_selection_enga_benchmark.run()
    # saenhanced_nga_benchmark.run()
    # nga_benchmark.run()
    # ga_random_benchmark.run()
    # only_selection_nga_benchmark.run()
    # ga_mod_benchmark.run()
    # ga_benchmark.run()
    # ga_copy_benchmark.run()
    # ga_strict_benchmark.run()
    # ynga2_benchmark.run()
    # ynga3_benchmark.run()


if __name__ == '__main__':
    main()
