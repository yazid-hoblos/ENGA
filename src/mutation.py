from enum import Enum
import numpy as np
import random


class MutationType(Enum):
    FLIP = 1
    SWAP = 2


class Mutation:
    def __init__(self, prob_mutation: float, mutation_type: MutationType, domain, decimal_precision: int):
        self.prob_mutation = prob_mutation
        self.domain = domain
        self.decimal_precision = decimal_precision
        if isinstance(mutation_type, MutationType):
            self.mutation_type = mutation_type
        else:
            raise ValueError("Unknown mutation type")

    def mutate(self, individual):
        """
        Mutate an individual

        Parameters
        ----------
        individual
            The individual to mutate

        Returns
        -------
        The mutated individual
        """

        if random.random() < self.prob_mutation:
            if self.mutation_type == MutationType.FLIP:
                return self._flip_mutation(individual)
            elif self.mutation_type == MutationType.SWAP:
                return self._swap_mutation(individual)
            else:
                raise ValueError("Unknown mutation type")
        else:
            return individual

    def _flip_mutation(self, individual):
        index = random.randint(0, len(individual) - 1)
        individual[index] = round(random.uniform(
            self.domain[0], self.domain[1]), self.decimal_precision)
        return individual

    def _swap_mutation(self, individual):
        index1, index2 = random.sample(range(len(individual)), 2)
        individual[index1], individual[index2] = individual[index2], individual[index1]
        return individual
