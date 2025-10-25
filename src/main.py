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
import argparse
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

    # Simple CLI: allow selecting which benchmark(s) to run.
    parser = argparse.ArgumentParser(description='Run GA benchmarks')
    parser.add_argument('--run', nargs='+', choices=['enhanced', 'networked', 'ga', 'crga', 'dcga', 'all'],
                        help='Which benchmark(s) to run')
    args = parser.parse_args()

    if not args.run:
        print('No benchmarks requested.')
        print('Use --run with one or more of: enhanced, networked, ga, crga, dcga, all')
        print('Example: python3 main.py --run ga')
        return

    to_run = set(args.run)
    if 'all' in to_run:
        to_run = {'enhanced', 'networked', 'ga', 'crga', 'dcga'}

    mapping = {
        'enhanced': enhanced_nga_benchmark,
        'networked': nga_benchmark,
        'ga': ga_benchmark,
        'crga': crga_benchmark,
        'dcga': dcga_benchmark,
    }

    for name in to_run:
        bench = mapping.get(name)
        if bench is None:
            print(f"Unknown benchmark: {name}")
            continue
        print(f"Running {name} benchmark...")
        bench.run()
        print(f"Finished {name} benchmark")

if __name__ == '__main__':
    main()
