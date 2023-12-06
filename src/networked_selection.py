from enum import Enum
import networkx as nx
import random
import numpy as np

"""
Selection is to select 2 nodes, ways to do that
1 - Select a node in the network by:
    - Random node
    - Roulette on nodes
    - Tournament on nodes
    - Centrality
        - Degree centrality
        - Closeness centrality
        - Eigenvector centrality

2 - After selecting the node, select the other node by:
    - Random neighbor of the node
    - Roulette on neighbor of the node
    - Tournament on neighbor of the node
    - A random node in the community of the node
    - Tournament on nodes in the community of the node
    - Roulette on nodes in the community of the node
    - A random node from a different community than the node
    - Tournament on nodes from different communities than the node
    - Roulette on nodes from different communities than the node
"""


class InitialSelectionType(Enum):
    RANDOM = 1
    ROULETTE = 2
    TOURNAMENT = 3
    DEGREE_CENTRALITY = 4
    CLOSENESS_CENTRALITY = 5
    EIGENVECTOR_CENTRALITY = 6


class SecondarySelectionType(Enum):
    RANDOM_NEIGHBOR = 1
    ROULETTE_NEIGHBOR = 2
    TOURNAMENT_NEIGHBOR = 3
    RANDOM_SAME_COMMUNITY = 4
    TOURNAMENT_SAME_COMMUNITY = 5
    ROULETTE_SAME_COMMUNITY = 6
    RANDOM_DIFFERENT_COMMUNITY = 7
    TOURNAMENT_DIFFERENT_COMMUNITY = 8
    ROULETTE_DIFFERENT_COMMUNITY = 9


class NetworkSelection:
    def __init__(self, initial_selection_type: InitialSelectionType, secondary_selection_type: SecondarySelectionType):
        self.initial_selection_type = initial_selection_type
        if self.initial_selection_type == InitialSelectionType.TOURNAMENT:
            self.tourn_size = 4  # int(input("Enter TOURNAMENT size: "))

        self.secondary_selection_type = secondary_selection_type

        self.global_mating_probability = 0

    def select_parents(self):
        initial_assignment = self._initial_node_selection()
        secondary_assignment = self._secondary_node_selection(
            initial_assignment)

        return initial_assignment, secondary_assignment

    def _initial_node_selection(self):
        if self.initial_selection_type == InitialSelectionType.RANDOM:
            return self._random_node_selection()
        elif self.initial_selection_type == InitialSelectionType.ROULETTE:
            return self._roulette_node_selection()
        elif self.initial_selection_type == InitialSelectionType.TOURNAMENT:
            return self._tournament_node_selection()
        elif self.initial_selection_type == InitialSelectionType.DEGREE_CENTRALITY:
            return self._degree_centrality_node_selection()
        elif self.initial_selection_type == InitialSelectionType.CLOSENESS_CENTRALITY:
            return self._closeness_centrality_node_selection()
        elif self.initial_selection_type == InitialSelectionType.EIGENVECTOR_CENTRALITY:
            return self._eigenvector_centrality_node_selection()

    def _secondary_node_selection(self, assignment):
        # Global mating -> select a random node in the network instead of a neighbor
        if random.random() < self.global_mating_probability:
            return self._random_node_selection()

        if self.secondary_selection_type == SecondarySelectionType.RANDOM_NEIGHBOR:
            return self._random_neighbor_selection(assignment)
        elif self.secondary_selection_type == SecondarySelectionType.ROULETTE_NEIGHBOR:
            return self._roulette_neighbor_selection(assignment)
        elif self.secondary_selection_type == SecondarySelectionType.TOURNAMENT_NEIGHBOR:
            return self._tournament_neighbor_selection(assignment)
        elif self.secondary_selection_type == SecondarySelectionType.RANDOM_SAME_COMMUNITY:
            return self._random_same_community_selection(assignment)
        elif self.secondary_selection_type == SecondarySelectionType.TOURNAMENT_SAME_COMMUNITY:
            return self._tournament_same_community_selection(assignment)
        elif self.secondary_selection_type == SecondarySelectionType.ROULETTE_SAME_COMMUNITY:
            return self._roulette_same_community_selection(assignment)
        elif self.secondary_selection_type == SecondarySelectionType.RANDOM_DIFFERENT_COMMUNITY:
            return self._random_different_community_selection(assignment)
        elif self.secondary_selection_type == SecondarySelectionType.TOURNAMENT_DIFFERENT_COMMUNITY:
            return self._tournament_different_community_selection(assignment)
        elif self.secondary_selection_type == SecondarySelectionType.ROULETTE_DIFFERENT_COMMUNITY:
            return self._roulette_different_community_selection(assignment)

    def _random_node_selection(self):
        """
        Select a random node in the network
        """

        return self.network.nodes[random.choice(list(self.network.nodes))]['node']

    def _roulette_node_selection(self):
        """
        Select a node in the network by roulette on the fitness of the nodes
        """
        fitnesses = [self.network.nodes[node]
                     ['node'].fitness for node in self.network.nodes]

        node_index = self._roulette_selection(fitnesses, 1)[0]
        return self.network.nodes[node_index]['node']

    def _tournament_node_selection(self):
        random_nodes = np.random.choice(
            list(self.network.nodes), size=self.tourn_size, replace=False)

        fitnesses = [self.network.nodes[node]
                     ['node'].fitness for node in random_nodes]
        ind = np.argsort(fitnesses)
        if self.is_maximization:
            ind = ind[::-1]

        return self.network.nodes[random_nodes[ind[0]]]['node']

    def _degree_centrality_node_selection(self):
        raise NotImplementedError

    def _closeness_centrality_node_selection(self):
        raise NotImplementedError

    def _eigenvector_centrality_node_selection(self):
        raise NotImplementedError

    def _random_neighbor_selection(self, assignment):
        neighbors = list(self.network.neighbors(assignment.node_index))
        if len(neighbors) == 0:
            return assignment
        return self.network.nodes[random.choice(neighbors)]['node']

    def _roulette_neighbor_selection(self, assignment):
        neighbors = list(self.network.neighbors(assignment.node_index))
        if len(neighbors) == 0:
            return assignment

        fitnesses = [self.network.nodes[node]
                     ['node'].fitness for node in neighbors]

        if all(fitnesses[i] == 0 for i in range(len(fitnesses))):
            node_index = random.choice(neighbors)
            return self.network.nodes[node_index]['node']

        node_index = self._roulette_selection(fitnesses, 1)[0]
        return self.network.nodes[node_index]['node']

    def _tournament_neighbor_selection(self, assignment):
        neighbors = list(self.network.neighbors(assignment.node_index))
        if len(neighbors) == 0:
            return assignment
        k = 2
        random_nodes = random.choices(neighbors, k=k)
        fitnesses = [self.network.nodes[node]
                     ['node'].fitness for node in random_nodes]
        ind = np.argsort(fitnesses)
        if self.is_maximization:
            ind = ind[::-1]

        return self.network.nodes[random_nodes[ind[0]]]['node']

    def _random_same_community_selection(self, assignment):
        raise NotImplementedError

    def _tournament_same_community_selection(self, assignment):
        raise NotImplementedError

    def _roulette_same_community_selection(self, assignment):
        raise NotImplementedError

    def _random_different_community_selection(self, assignment):
        raise NotImplementedError

    def _tournament_different_community_selection(self, assignment):
        raise NotImplementedError

    def _roulette_different_community_selection(self, assignment):
        raise NotImplementedError
