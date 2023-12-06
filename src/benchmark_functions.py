from benchmark import GABenchmarkFunction, FunctionCharacteristic
from typing import List
import random
import numpy as np
import math
import os
from tsp.tsp import tsp
from cec2017.cec2017 import get_cec2017

# Ackely - acceptance : 0.01
# multimodal
f1: GABenchmarkFunction = GABenchmarkFunction(
    name="f1",
    create_random_individual=lambda nb_genes, domain, decimal_precision: [
        round(random.uniform(domain[0], domain[1]), decimal_precision) for _ in range(nb_genes)],
    fitness_function=lambda individual: 20 - 20 *
    math.exp(-0.2*math.sqrt((1/30) *
                            sum([individual[i]**2 for i in range(len(individual))]))) + math.e - math.exp((1/30) * sum(
                                [math.cos(2*math.pi*individual[i]) for i in range(len(individual))])),
    decimal_precision=0,
    is_maximization=False, lower_bound=-32, upper_bound=32, number_of_genes=30, optimal_value=0,
    function_characteristics=[FunctionCharacteristic.MULTIMODAL, FunctionCharacteristic.NON_SEPARETABLE, FunctionCharacteristic.NON_SYMMETRIC, FunctionCharacteristic.PERMUTATION_INVARIENT])

# De Jong - acceptance : 0.01
# unimodal
f2: GABenchmarkFunction = GABenchmarkFunction(
    name="f2",
    create_random_individual=lambda nb_genes, domain, decimal_precision: [
        round(random.uniform(domain[0], domain[1]), decimal_precision) for _ in range(nb_genes)],
    fitness_function=lambda individual: sum(
        [individual[i]**2 for i in range(len(individual))]),
    decimal_precision=0,
    is_maximization=False, lower_bound=-100, upper_bound=100, number_of_genes=30, optimal_value=0,
    function_characteristics=[FunctionCharacteristic.UNIMODAL, FunctionCharacteristic.SEPARETABLE, FunctionCharacteristic.SYMMETRIC, FunctionCharacteristic.PERMUTATION_INVARIENT])

# Unintresting function - acceptance : 0.01
# unimodal
f3: GABenchmarkFunction = GABenchmarkFunction(
    name="f3",
    create_random_individual=lambda nb_genes, domain, decimal_precision: [
        round(random.uniform(domain[0], domain[1]), decimal_precision) for _ in range(nb_genes)],
    fitness_function=lambda individual: sum(
        [abs(individual[i]) for i in range(len(individual))]) + np.prod([abs(individual[i]) for i in range(len(individual))]),
    decimal_precision=0,
    is_maximization=False, lower_bound=-10, upper_bound=10, number_of_genes=30, optimal_value=0,
    function_characteristics=[FunctionCharacteristic.UNIMODAL, FunctionCharacteristic.SEPARETABLE, FunctionCharacteristic.SYMMETRIC, FunctionCharacteristic.PERMUTATION_INVARIENT])

# acceptance : 0.01
# unimodal
f4: GABenchmarkFunction = GABenchmarkFunction(
    name="f4",
    create_random_individual=lambda nb_genes, domain, decimal_precision: [
        round(random.uniform(domain[0], domain[1]), decimal_precision) for _ in range(nb_genes)],
    fitness_function=lambda individual: sum(
        [i * math.pow(individual[i], 4) for i in range(len(individual))]) + random.uniform(0, 1),
    decimal_precision=2,
    is_maximization=False, lower_bound=-1.28, upper_bound=1.28, number_of_genes=30, optimal_value=0,
    function_characteristics=[FunctionCharacteristic.UNIMODAL, FunctionCharacteristic.SEPARETABLE, FunctionCharacteristic.SYMMETRIC, FunctionCharacteristic.PERMUTATION_INVARIENT])

# rastring acceptance : 50
# multimodal
f5: GABenchmarkFunction = GABenchmarkFunction(
    name="f5",
    create_random_individual=lambda nb_genes, domain, decimal_precision: [
        round(random.uniform(domain[0], domain[1]), decimal_precision) for _ in range(nb_genes)],
    fitness_function=lambda individual: sum(
        [individual[i] ** 2 - 10*math.cos(2*math.pi*individual[i]) + 10 for i in range(len(individual))]),
    decimal_precision=2,
    is_maximization=False, lower_bound=-5.12, upper_bound=5.12, number_of_genes=30, optimal_value=0,
    function_characteristics=[FunctionCharacteristic.MULTIMODAL, FunctionCharacteristic.NON_SEPARETABLE, FunctionCharacteristic.SYMMETRIC, FunctionCharacteristic.PERMUTATION_INVARIENT])

# griewank acceptance : 0.01
# multimodal
f6: GABenchmarkFunction = GABenchmarkFunction(
    name="f6",
    create_random_individual=lambda nb_genes, domain, decimal_precision: [
        round(random.uniform(domain[0], domain[1]), decimal_precision) for _ in range(nb_genes)],
    fitness_function=lambda individual: sum([(individual[i]**2)/4000 for i in range(len(individual))]) - np.prod(
        [math.cos(individual[i]/math.sqrt(i+1)) for i in range(len(individual))]) + 1,
    decimal_precision=0,
    is_maximization=False, lower_bound=-600, upper_bound=600, number_of_genes=30, optimal_value=0,
    function_characteristics=[FunctionCharacteristic.MULTIMODAL, FunctionCharacteristic.NON_SEPARETABLE, FunctionCharacteristic.SYMMETRIC, FunctionCharacteristic.PERMUTATION_INVARIENT])

# rosenbrock acceptance : 100
# unimodal
f7: GABenchmarkFunction = GABenchmarkFunction(
    name="f7",
    create_random_individual=lambda nb_genes, domain, decimal_precision: [
        round(random.uniform(domain[0], domain[1]), decimal_precision) for _ in range(nb_genes)],
    fitness_function=lambda individual: sum(
        [100*(individual[i+1] - individual[i]**2)**2 + (individual[i] - 1)**2 for i in range(len(individual)-1)]),
    decimal_precision=0,
    is_maximization=False, lower_bound=-32, upper_bound=32, number_of_genes=30, optimal_value=0,
    function_characteristics=[FunctionCharacteristic.MULTIMODAL, FunctionCharacteristic.NON_SEPARETABLE, FunctionCharacteristic.NON_SYMMETRIC, FunctionCharacteristic.PERMUTATION_VARIANT])

eggholder: GABenchmarkFunction = GABenchmarkFunction(
    name="eggholder",
    create_random_individual=lambda nb_genes, domain, decimal_precision: [
        round(random.uniform(domain[0], domain[1]), decimal_precision) for _ in range(nb_genes)],
    fitness_function=lambda individual: -(individual[1] + 47) * math.sin(math.sqrt(abs(individual[0]/2 + (
        individual[1] + 47)))) - individual[0] * math.sin(math.sqrt(abs(individual[0] - (individual[1] + 47)))),
    decimal_precision=0,
    is_maximization=False, lower_bound=-512, upper_bound=512, number_of_genes=2, optimal_value=-959.6407,
    function_characteristics=[FunctionCharacteristic.MULTIMODAL, FunctionCharacteristic.NON_SEPARETABLE, FunctionCharacteristic.NON_SYMMETRIC, FunctionCharacteristic.PERMUTATION_VARIANT])

dixonprice: GABenchmarkFunction = GABenchmarkFunction(
    name="dixonprice",
    create_random_individual=lambda nb_genes, domain, decimal_precision: [
        round(random.uniform(domain[0], domain[1]), decimal_precision) for _ in range(nb_genes)],
    fitness_function=lambda individual: (individual[0] - 1)**2 + sum([(i+1) * (
        2 * individual[i]**2 - individual[i-1])**2 for i in range(1, len(individual))]),
    decimal_precision=0,
    is_maximization=False, lower_bound=-10, upper_bound=10, number_of_genes=30, optimal_value=0,

    function_characteristics=[FunctionCharacteristic.UNIMODAL, FunctionCharacteristic.NON_SEPARETABLE, FunctionCharacteristic.NON_SYMMETRIC, FunctionCharacteristic.PERMUTATION_VARIANT])

bent_cigar: GABenchmarkFunction = GABenchmarkFunction(
    name="bent_cigar",
    create_random_individual=lambda nb_genes, domain, decimal_precision: [
        round(random.uniform(domain[0], domain[1]), decimal_precision) for _ in range(nb_genes)],
    fitness_function=lambda individual: individual[0]**2 + 10**6 * sum(
        [individual[i]**2 for i in range(1, len(individual))]),
    decimal_precision=0,
    is_maximization=False, lower_bound=-100, upper_bound=100, number_of_genes=30, optimal_value=0,
    function_characteristics=[FunctionCharacteristic.UNIMODAL, FunctionCharacteristic.NON_SEPARETABLE, FunctionCharacteristic.NON_SYMMETRIC, FunctionCharacteristic.PERMUTATION_VARIANT])


discus: GABenchmarkFunction = GABenchmarkFunction(
    name="discus",
    create_random_individual=lambda nb_genes, domain, decimal_precision: [
        round(random.uniform(domain[0], domain[1]), decimal_precision) for _ in range(nb_genes)],
    fitness_function=lambda individual: (10**6)*individual[0]**2 + sum(
        [individual[i]**2 for i in range(1, len(individual))]),
    decimal_precision=0,
    is_maximization=False, lower_bound=-100, upper_bound=100, number_of_genes=30, optimal_value=0,
    function_characteristics=[FunctionCharacteristic.UNIMODAL, FunctionCharacteristic.NON_SEPARETABLE, FunctionCharacteristic.NON_SYMMETRIC, FunctionCharacteristic.PERMUTATION_VARIANT])

schwefel: GABenchmarkFunction = GABenchmarkFunction(
    name="schwefel",
    create_random_individual=lambda nb_genes, domain, decimal_precision: [
        round(random.uniform(domain[0], domain[1]), decimal_precision) for _ in range(nb_genes)],
    fitness_function=lambda individual: 418.9829 * len(individual) - sum(
        [individual[i] * math.sin(math.sqrt(abs(individual[i]))) for i in range(len(individual))]),
    decimal_precision=4,
    is_maximization=False, lower_bound=-500, upper_bound=500, number_of_genes=30, optimal_value=0,
    function_characteristics=[FunctionCharacteristic.MULTIMODAL, FunctionCharacteristic.NON_SEPARETABLE, FunctionCharacteristic.NON_SYMMETRIC, FunctionCharacteristic.PERMUTATION_VARIANT])


target_wave_params = [1.0, 5.0, 1.5, 4.8, 2.0, 4.9]
theta = 2 * math.pi / 100


def wave(t, a1, w1, a2, w2, a3, w3):
    return a1 * math.sin(w1 * t) + a2 * math.sin(w2 * t) + a3 * math.sin(w3 * t)


#  Precompute the target wave to save time
target_wave = [wave(t, *target_wave_params) for t in range(100)]

frequency_modulated_sound_waves: GABenchmarkFunction = GABenchmarkFunction(
    name="fm_sound_waves",
    create_random_individual=lambda nb_genes, domain, decimal_precision: [
        round(random.uniform(domain[0], domain[1]), decimal_precision) for _ in range(nb_genes)],
    fitness_function=lambda individual: sum(
        [(wave(t, *individual) - target_wave[t])**2 for t in range(100)]),
    decimal_precision=1,
    is_maximization=False, lower_bound=-6.4, upper_bound=6.4, number_of_genes=6, optimal_value=0,
    function_characteristics=[FunctionCharacteristic.MULTIMODAL, FunctionCharacteristic.NON_SEPARETABLE, FunctionCharacteristic.NON_SYMMETRIC, FunctionCharacteristic.PERMUTATION_VARIANT])


a = [1.0, 1.2, 3.0, 3.2]
A = [[3, 10, 30],
     [0.1, 10, 35],
     [3, 10, 30],
     [0.1, 10, 35]]

P = [[0.3689, 0.1170, 0.2673],
     [0.4699, 0.4387, 0.7470],
     [0.1091, 0.8732, 0.5547],
     [0.0381, 0.5743, 0.8828]]


HARTMANN: GABenchmarkFunction = GABenchmarkFunction(
    name="hartmann3",
    create_random_individual=lambda nb_genes, domain, decimal_precision: [
        round(random.uniform(domain[0], domain[1]), decimal_precision) for _ in range(nb_genes)],
    fitness_function=lambda individual: -
    sum([a[i] * math.exp(-sum([A[i][j] * (individual[j] - P[i][j])
        ** 2 for j in range(3)])) for i in range(4)]),
    decimal_precision=6,
    is_maximization=False, lower_bound=0, upper_bound=1, number_of_genes=3, optimal_value=-3.32237,
    function_characteristics=[FunctionCharacteristic.MULTIMODAL, FunctionCharacteristic.NON_SEPARETABLE, FunctionCharacteristic.NON_SYMMETRIC, FunctionCharacteristic.PERMUTATION_VARIANT])


def get_all_paper_functions():
    functions: List[GABenchmarkFunction] = []

    functions.append(f1)
    functions.append(f2)
    functions.append(f3)
    functions.append(f4)
    functions.append(f5)
    functions.append(f6)
    functions.append(f7)

    return functions


def get_all_functions():
    functions: List[GABenchmarkFunction] = []

    functions.append(f1)
    functions.append(f2)
    functions.append(f3)
    functions.append(f4)
    functions.append(f5)
    functions.append(f6)
    functions.append(f7)
    functions.append(tsp)
    functions.append(frequency_modulated_sound_waves)
    functions.append(eggholder)
    functions.append(dixonprice)
    functions.append(HARTMANN)
    functions.append(bent_cigar)
    functions.append(discus)
    functions.append(schwefel)

    return functions


def final_chosen_functions():
    functions: List[GABenchmarkFunction] = []

    functions.append(f2)
    functions.append(f4)
    functions.append(f5)
    functions.append(f6)
    functions.append(f7)
    functions.append(dixonprice)
    functions.append(discus)
    functions.append(schwefel)

    return functions


def get_functions_with_characteristics(characteristics: List[FunctionCharacteristic]):
    functions: List[GABenchmarkFunction] = []
    for function in get_all_functions():
        if all(characteristic in function.function_characteristics for characteristic in characteristics):
            functions.append(function)

    if len(functions) == 0:
        raise Exception("No function with the given characteristics")

    return functions


def get_random_function():
    return random.choice(get_all_functions())


def get_functions_by_name(names: List[str]):
    functions: List[GABenchmarkFunction] = []
    for function in get_all_functions():
        if function.name in names:
            functions.append(function)

    if len(functions) == 0:
        raise Exception("No function with the given names")

    return functions


def get_cec_2017_functions():
    functions: List[GABenchmarkFunction] = []

    for func_name, func in get_cec2017().items():
        functions.append(GABenchmarkFunction(
            name=func_name,
            create_random_individual=lambda nb_genes, domain, decimal_precision: [
                round(random.uniform(domain[0], domain[1]), decimal_precision) for _ in range(nb_genes)],
            fitness_function=lambda individual: func(
                np.array([individual]))[0],
            decimal_precision=0,
            is_maximization=False, lower_bound=-100, upper_bound=100, number_of_genes=30, optimal_value=0,
            function_characteristics=[]))

    return functions
