from enum import Enum
import numpy as np
import random


class SelectionType(Enum):
    ROULETTE = 1
    TOURNAMENT = 2
    RANK = 3
    RANDOM = 4


class Selection:
    def __init__(self, selection_type: SelectionType, number_of_parents: int, is_maximization: bool):
        self.number_of_parents = number_of_parents
        self.is_maximization = is_maximization
        if isinstance(selection_type, SelectionType):
            self.selection_type = selection_type
        else:
            raise ValueError("Unknown selection type")

        if self.selection_type == SelectionType.TOURNAMENT:
            self.tounement_size = int(input("Enter the tournament size: "))
            if self.tounement_size < 1:
                raise ValueError(
                    "The tournament size must be greater than or equal to 1")

    def select_parents(self, fitnesses: np.ndarray):
        """
        Select two parents from the population

        Parameters
        ----------
        None

        Returns
        -------
        The indices of the selected parents
        """

        if self.selection_type == SelectionType.ROULETTE:
            return self._roulette_selection(fitnesses, self.number_of_parents)
        elif self.selection_type == SelectionType.TOURNAMENT:
            return self._tournament_selection(fitnesses, self.number_of_parents, self.tounement_size)
        elif self.selection_type == SelectionType.RANK:
            return self._rank_selection(fitnesses, self.number_of_parents)
        elif self.selection_type == SelectionType.RANDOM:
            return self._random_selection(fitnesses, self.number_of_parents)
        else:
            raise ValueError("Unknown selection type")

    def _roulette_selection(self, fitnesses: np.ndarray, number_of_parents: int):
        """
        Roulette selection, select two individuals from the population based on their fitnesses, the higher the fitness the higher the probability of being selected

        Parameters
        ----------
        fitnesses : np.ndarray
            The fitnesses of the population

        number_of_parents : int
            The number of parents to select

        Returns
        -------
        The indices of the selected individuals
        """

        if isinstance(fitnesses, list):
            fitnesses = np.array(fitnesses)

        if len(fitnesses) == 1:
            return [0]

        fitness_sum = np.sum(fitnesses)

        if fitness_sum == 0:
            return np.random.choice(range(len(fitnesses)), size=number_of_parents)

        if any(fitnesses < 0):
            probs = abs(fitnesses / fitness_sum)
        else:
            probs = fitnesses / fitness_sum

        if not self.is_maximization:
            probs = (1 - probs)

            if any(probs < 0):
                probs = abs(probs)

            probs = probs / np.sum(probs)

        return np.random.choice(range(len(fitnesses)), size=number_of_parents, p=probs, replace=False)

    def _tournament_selection(self, fitnesses: np.ndarray, number_of_parents: int, K_tournament: int):
        """
        Tournament selection, select two individuals from the population based on their fitnesses, the higher the fitness the higher the probability of being selected

        Parameters
        ----------
        fitnesses : np.ndarray
            The fitnesses of the population

        Returns
        -------
        The indices of the selected individuals
        """

        if isinstance(fitnesses, list):
            fitnesses = np.array(fitnesses)

        fitness_sorted = list(np.argsort(fitnesses))
        parents_indices = []

        for parent_num in range(number_of_parents):
            # Generate random indices for the candiadate solutions.
            rand_indices = np.random.randint(
                low=0.0, high=len(fitnesses), size=K_tournament)

            # Find the rank of the candidate solutions. The lower the rank, the better the solution.
            rand_indices_rank = [fitness_sorted.index(
                rand_idx) for rand_idx in rand_indices]
            # Select the solution with the lowest rank as a parent.
            selected_parent_idx = rand_indices_rank.index(
                min(rand_indices_rank))

            # Append the index of the selected parent.
            parents_indices.append(rand_indices[selected_parent_idx])
            # Insert the selected parent.

        return parents_indices

    def _rank_selection(self, fitnesses: np.ndarray, number_of_parents: int):
        """
        Rank selection, select two individuals from the population based on their fitnesses, the higher the fitness the higher the probability of being selected

        Parameters
        ----------
        fitnesses : np.ndarray
            The fitnesses of the population

        Returns
        -------
        The indices of the selected individuals
        """
        if isinstance(fitnesses, list):
            fitnesses = np.array(fitnesses)

        fitness_sorted = np.argsort(fitnesses)
        rank = np.arange(1, len(fitnesses) + 1)
        probs = rank / rank.sum()
        probs_start, probs_end = self._wheel_cumulative_probs(probs=probs.copy(),
                                                              number_of_parents=number_of_parents)
        parents_indices = []
        for parent_num in range(number_of_parents):
            rand_prob = np.random.rand()
            for idx in range(probs.shape[0]):
                if (rand_prob >= probs_start[idx] and rand_prob < probs_end[idx]):
                    # The variable idx has the rank of solution but not its index in the population.
                    # Return the correct index of the solution.
                    mapped_idx = fitness_sorted[idx]
                    parents_indices.append(mapped_idx)
                    break

        return parents_indices

    def _random_selection(self, fitnesses: np.ndarray, number_of_parents: int):
        """
        Random selection, select two individuals from the population randomly

        Parameters
        ----------
        fitnesses : np.ndarray
            The fitnesses of the population

        number_of_parents : int
            The number of parents to select
        Returns
        -------
        The indices of the selected individuals
        """
        if isinstance(fitnesses, list):
            fitnesses = np.array(fitnesses)

        return random.choices(range(len(fitnesses)), k=number_of_parents)

    def _wheel_cumulative_probs(self, probs, number_of_parents):
        """
        A helper function to calculate the wheel probabilities for these 2 methods:
            1) roulette_wheel_selection
            2) rank_selection
        It accepts a single 1D array representing the probabilities of selecting each solution.
        It returns 2 1D arrays:
            1) probs_start has the start of each range.
            2) probs_start has the end of each range.
        """

        # An array holding the start values of the ranges of probabilities.
        probs_start = np.zeros(probs.shape, dtype=float)
        # An array holding the end values of the ranges of probabilities.
        probs_end = np.zeros(probs.shape, dtype=float)

        curr = 0.0

        # Calculating the probabilities of the solutions to form a roulette wheel.
        for _ in range(probs.shape[0]):
            min_probs_idx = np.where(probs == np.min(probs))[0][0]
            probs_start[min_probs_idx] = curr
            curr = curr + probs[min_probs_idx]
            probs_end[min_probs_idx] = curr
            # Replace 99999999999 by float('inf')
            # probs[min_probs_idx] = 99999999999
            probs[min_probs_idx] = float('inf')

        return probs_start, probs_end,
