from enum import Enum
import numpy as np
import random


class CrossOverType(Enum):
    ONE_POINT = 1
    TWO_POINT = 2
    UNIFORM = 3

class CrossOver:
    def __init__(self, prob_crossover: float, crossover_type: CrossOverType):
        self.prob_crossover = prob_crossover
        if isinstance(crossover_type, CrossOverType):
            self.crossover_type = crossover_type
        else:
            raise ValueError("Unknown crossover type")

    def crossover(self, parent1, parent2):
        """
        Crossover two parents

        Parameters
        ----------
        parent1
            The first parent

        parent2
            The second parent

        Returns
        -------
        The two children
        """

        if random.random() < self.prob_crossover:
            if self.crossover_type == CrossOverType.ONE_POINT:
                return self._one_point_crossover(parent1, parent2)
            elif self.crossover_type == CrossOverType.TWO_POINT:
                return self._two_point_crossover(parent1, parent2)
            elif self.crossover_type == CrossOverType.UNIFORM:
                return self._uniform_crossover(parent1, parent2)
            else:
                raise ValueError("Unknown crossover type")
        else:
            return parent1, parent2

    def _one_point_crossover(self, parent1, parent2):
        index = random.randint(0, len(parent1))
        child1 = parent1[:index] + parent2[index:]
        child2 = parent2[:index] + parent1[index:]
        return child1, child2

    def _two_point_crossover(self, parent1, parent2):
        index1 = random.randint(0, len(parent1))
        index2 = random.randint(index1, len(parent1))
        child1 = parent1[:index1] + parent2[index1:index2] + parent1[index2:]
        child2 = parent2[:index1] + parent1[index1:index2] + parent2[index2:]
        return child1, child2

    def _uniform_crossover(self, parent1, parent2):
        child1 = []
        child2 = []
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])
        return child1, child2
