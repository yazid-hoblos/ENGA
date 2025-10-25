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
    # Simple CLI: allow selecting which benchmark(s) to run and whether to save metrics.
    parser = argparse.ArgumentParser(description='Run GA benchmarks')
    parser.add_argument('--run', nargs='+', choices=['enhanced', 'networked', 'ga', 'crga', 'dcga', 'all'],
                        help='Which benchmark(s) to run')
    parser.add_argument('--save-metrics', action='store_true', help='Save metrics to the logs/metrics directory')

    # Optional parameters (when omitted, per-benchmark defaults are preserved)
    parser.add_argument('--generations', type=int, help='Number of generations to run (overrides default per-benchmark)')
    parser.add_argument('--population-size', type=int, help='Population size for all benchmarks')
    parser.add_argument('--number-elites', type=int, help='Number of elites to keep (overrides per-benchmark default)')
    parser.add_argument('--crossover-prob', type=float, help='Crossover probability (0.0-1.0)')
    parser.add_argument('--mutation-prob', type=float, help='Mutation probability (0.0-1.0)')
    parser.add_argument('--nruns', type=int, help='Number of independent runs to perform')

    # Verbosity option: default is verbose (True). Use --no-verbose to disable.
    parser.add_argument('--no-verbose', dest='verbose', action='store_false', help='Disable verbose output')
    parser.set_defaults(verbose=True)

    args = parser.parse_args()

    # Select whether to save metrics based on CLI flag
    save_metrics_flag = bool(args.save_metrics)

    # Helper: use provided CLI value if not None, otherwise keep per-benchmark default later
    cli_generations = args.generations
    cli_population = args.population_size
    cli_elites = args.number_elites
    cli_crossover = args.crossover_prob
    cli_mutation = args.mutation_prob
    cli_runs = args.nruns if args.nruns is not None else None
    verbose_flag = bool(args.verbose)

    benchmark_functions = get_all_functions()

    # Enhanced Networked Genetic Algorithm
    enhanced_nga_benchmark = Benchmark(
        genetic_algorithm=EnhancedNetworkGeneticAlgorithm,
        number_of_runs=(cli_runs if cli_runs is not None else 1),
        benchmark_functions=benchmark_functions,
        number_of_generations=(cli_generations if cli_generations is not None else 2000),
        population_size=(cli_population if cli_population is not None else 125),
        number_elites=(cli_elites if cli_elites is not None else 0),
        probability_crossover=(cli_crossover if cli_crossover is not None else 0.95),
        probability_mutation=(cli_mutation if cli_mutation is not None else 0.1),
        verbose=verbose_flag,
        save_metrics=save_metrics_flag,
    )

    # Networked Genetic Algorithm
    nga_benchmark = Benchmark(
        genetic_algorithm=NetworkGeneticAlgorithm,
        number_of_runs=(cli_runs if cli_runs is not None else 1),
        benchmark_functions=benchmark_functions,
        number_of_generations=(cli_generations if cli_generations is not None else 2000),
        population_size=(cli_population if cli_population is not None else 125),
        number_elites=(cli_elites if cli_elites is not None else 0),
        probability_crossover=(cli_crossover if cli_crossover is not None else 0.95),
        probability_mutation=(cli_mutation if cli_mutation is not None else 0.1),
        verbose=verbose_flag,
        save_metrics=save_metrics_flag,
    )

    # Genetic Algorithm
    ga_benchmark = Benchmark(
        genetic_algorithm=GeneticAlgorithm,
        number_of_runs=(cli_runs if cli_runs is not None else 1),
        benchmark_functions=benchmark_functions,
        number_of_generations=(cli_generations if cli_generations is not None else 2000),
        population_size=(cli_population if cli_population is not None else 125),
        number_elites=(cli_elites if cli_elites is not None else 10),
        probability_crossover=(cli_crossover if cli_crossover is not None else 0.95),
        probability_mutation=(cli_mutation if cli_mutation is not None else 0.1),
        verbose=verbose_flag,
        save_metrics=save_metrics_flag,
    )

    # Communities Representatives Genetic Algorithm
    crga_benchmark = Benchmark(
        genetic_algorithm=CRGA,
        number_of_runs=(cli_runs if cli_runs is not None else 1),
        benchmark_functions=benchmark_functions,
        number_of_generations=(cli_generations if cli_generations is not None else 2000),
        population_size=(cli_population if cli_population is not None else 124),
        number_elites=(cli_elites if cli_elites is not None else 0),
        probability_crossover=(cli_crossover if cli_crossover is not None else 0.95),
        probability_mutation=(cli_mutation if cli_mutation is not None else 0.1),
        verbose=verbose_flag,
        save_metrics=save_metrics_flag,
    )

    # Dynamic Communities Genetic Algorithm
    dcga_benchmark = Benchmark(
        genetic_algorithm=DCGA,
        number_of_runs=(cli_runs if cli_runs is not None else 1),
        benchmark_functions=benchmark_functions,
        number_of_generations=(cli_generations if cli_generations is not None else 2000),
        population_size=(cli_population if cli_population is not None else 125),
        number_elites=(cli_elites if cli_elites is not None else 0),
        probability_crossover=(cli_crossover if cli_crossover is not None else 0.95),
        probability_mutation=(cli_mutation if cli_mutation is not None else 0.1),
        verbose=verbose_flag,
        save_metrics=save_metrics_flag,
    )

    # enhanced_nga_benchmark.run()
    # nga_benchmark.run()
    # ga_benchmark.run()

    # Warning: crga_benchmark.run() and dcga_benchmark.run() will take a long time to run

    # crga_benchmark.run()
    # dcga_benchmark.run()

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
