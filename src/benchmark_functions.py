from benchmark import GABenchmarkFunction, FunctionCharacteristic
from typing import List
import random
import numpy as np
import math
import os

# De Jong - acceptance : 0.01
# unimodal
f1: GABenchmarkFunction = GABenchmarkFunction(
    name="f1",
    create_random_individual=lambda nb_genes, domain, decimal_precision: [
        round(random.uniform(domain[0], domain[1]), decimal_precision) for _ in range(nb_genes)],
    fitness_function=lambda individual: sum(
        [individual[i]**2 for i in range(len(individual))]),
    decimal_precision=0,
    is_maximization=False, lower_bound=-100, upper_bound=100, number_of_genes=30, optimal_value=0,
    function_characteristics=[FunctionCharacteristic.UNIMODAL, FunctionCharacteristic.SEPARETABLE, FunctionCharacteristic.SYMMETRIC, FunctionCharacteristic.PERMUTATION_INVARIENT])

# acceptance : 0.01
# unimodal
f2: GABenchmarkFunction = GABenchmarkFunction(
    name="f2",
    create_random_individual=lambda nb_genes, domain, decimal_precision: [
        round(random.uniform(domain[0], domain[1]), decimal_precision) for _ in range(nb_genes)],
    fitness_function=lambda individual: sum(
        [i * math.pow(individual[i], 4) for i in range(len(individual))]) + random.uniform(0, 1),
    decimal_precision=2,
    is_maximization=False, lower_bound=-1.28, upper_bound=1.28, number_of_genes=30, optimal_value=0,
    function_characteristics=[FunctionCharacteristic.UNIMODAL, FunctionCharacteristic.SEPARETABLE, FunctionCharacteristic.SYMMETRIC, FunctionCharacteristic.PERMUTATION_INVARIENT])

# rastring acceptance : 50
# multimodal
f3: GABenchmarkFunction = GABenchmarkFunction(
    name="f3",
    create_random_individual=lambda nb_genes, domain, decimal_precision: [
        round(random.uniform(domain[0], domain[1]), decimal_precision) for _ in range(nb_genes)],
    fitness_function=lambda individual: sum(
        [individual[i] ** 2 - 10*math.cos(2*math.pi*individual[i]) + 10 for i in range(len(individual))]),
    decimal_precision=2,
    is_maximization=False, lower_bound=-5.12, upper_bound=5.12, number_of_genes=30, optimal_value=0,
    function_characteristics=[FunctionCharacteristic.MULTIMODAL, FunctionCharacteristic.NON_SEPARETABLE, FunctionCharacteristic.SYMMETRIC, FunctionCharacteristic.PERMUTATION_INVARIENT])

# griewank acceptance : 0.01
# multimodal
f4: GABenchmarkFunction = GABenchmarkFunction(
    name="f4",
    create_random_individual=lambda nb_genes, domain, decimal_precision: [
        round(random.uniform(domain[0], domain[1]), decimal_precision) for _ in range(nb_genes)],
    fitness_function=lambda individual: sum([(individual[i]**2)/4000 for i in range(len(individual))]) - np.prod(
        [math.cos(individual[i]/math.sqrt(i+1)) for i in range(len(individual))]) + 1,
    decimal_precision=0,
    is_maximization=False, lower_bound=-600, upper_bound=600, number_of_genes=30, optimal_value=0,
    function_characteristics=[FunctionCharacteristic.MULTIMODAL, FunctionCharacteristic.NON_SEPARETABLE, FunctionCharacteristic.SYMMETRIC, FunctionCharacteristic.PERMUTATION_INVARIENT])

# rosenbrock acceptance : 100
# unimodal
f5: GABenchmarkFunction = GABenchmarkFunction(
    name="f5",
    create_random_individual=lambda nb_genes, domain, decimal_precision: [
        round(random.uniform(domain[0], domain[1]), decimal_precision) for _ in range(nb_genes)],
    fitness_function=lambda individual: sum(
        [100*(individual[i+1] - individual[i]**2)**2 + (individual[i] - 1)**2 for i in range(len(individual)-1)]),
    decimal_precision=0,
    is_maximization=False, lower_bound=-32, upper_bound=32, number_of_genes=30, optimal_value=0,
    function_characteristics=[FunctionCharacteristic.MULTIMODAL, FunctionCharacteristic.NON_SEPARETABLE, FunctionCharacteristic.NON_SYMMETRIC, FunctionCharacteristic.PERMUTATION_VARIANT])

f6: GABenchmarkFunction = GABenchmarkFunction(
    name="f6",
    create_random_individual=lambda nb_genes, domain, decimal_precision: [
        round(random.uniform(domain[0], domain[1]), decimal_precision) for _ in range(nb_genes)],
    fitness_function=lambda individual: (individual[0] - 1)**2 + sum([(i+1) * (
        2 * individual[i]**2 - individual[i-1])**2 for i in range(1, len(individual))]),
    decimal_precision=0,
    is_maximization=False, lower_bound=-10, upper_bound=10, number_of_genes=30, optimal_value=0,

    function_characteristics=[FunctionCharacteristic.UNIMODAL, FunctionCharacteristic.NON_SEPARETABLE, FunctionCharacteristic.NON_SYMMETRIC, FunctionCharacteristic.PERMUTATION_VARIANT])

f7: GABenchmarkFunction = GABenchmarkFunction(
    name="f7",
    create_random_individual=lambda nb_genes, domain, decimal_precision: [
        round(random.uniform(domain[0], domain[1]), decimal_precision) for _ in range(nb_genes)],
    fitness_function=lambda individual: (10**6)*individual[0]**2 + sum(
        [individual[i]**2 for i in range(1, len(individual))]),
    decimal_precision=0,
    is_maximization=False, lower_bound=-100, upper_bound=100, number_of_genes=30, optimal_value=0,
    function_characteristics=[FunctionCharacteristic.UNIMODAL, FunctionCharacteristic.NON_SEPARETABLE, FunctionCharacteristic.NON_SYMMETRIC, FunctionCharacteristic.PERMUTATION_VARIANT])

f8: GABenchmarkFunction = GABenchmarkFunction(
    name="f8",
    create_random_individual=lambda nb_genes, domain, decimal_precision: [
        round(random.uniform(domain[0], domain[1]), decimal_precision) for _ in range(nb_genes)],
    fitness_function=lambda individual: 418.9829 * len(individual) - sum(
        [individual[i] * math.sin(math.sqrt(abs(individual[i]))) for i in range(len(individual))]),
    decimal_precision=4,
    is_maximization=False, lower_bound=-500, upper_bound=500, number_of_genes=30, optimal_value=0,
    function_characteristics=[FunctionCharacteristic.MULTIMODAL, FunctionCharacteristic.NON_SEPARETABLE, FunctionCharacteristic.NON_SYMMETRIC, FunctionCharacteristic.PERMUTATION_VARIANT])


def get_all_functions():
    functions: List[GABenchmarkFunction] = []

    functions.append(f1)
    functions.append(f2)
    functions.append(f3)
    functions.append(f4)
    functions.append(f5)
    functions.append(f6)
    functions.append(f7)
    functions.append(f8)

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
