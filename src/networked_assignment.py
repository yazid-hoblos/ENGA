from enum import Enum
import networkx as nx
from typing import List
import random
import numpy as np


class NetworkAssignmentType(Enum):
    RANDOM_ASSIGNMENT = 1
    # Split as communities and assign fitnesses such that each community has a high fitness node/hub
    COMMUNITY_FITNESS_ASSIGNMENT = 2
    DEGREE_CENTRALITY_FITNESS_ASSIGNMENT = 3
    DEGREE_CENTRALITY_FITNESS_COMMUNITY_ASSIGNMENT = 4
    EIGENVECTOR_CENTRALITY_FITNESS_ASSIGNMENT = 5
    CLOSEST_CENTRALITY_FITNESS_ASSIGNMENT = 6


class Node:
    def __init__(self, node_index, individual, fitness):
        self.node_index = node_index
        self.individual = individual
        self.fitness = fitness


class NetworkAssignment:
    def __init__(self, networked_assignment_type: NetworkAssignmentType):
        self.networked_assignment_type = networked_assignment_type

    def _assign_individuals_to_nodes(self):
        if self.networked_assignment_type == NetworkAssignmentType.RANDOM_ASSIGNMENT:
            self._random_assignment()
        elif self.networked_assignment_type == NetworkAssignmentType.COMMUNITY_FITNESS_ASSIGNMENT:
            self._community_fitness_assignment()
        elif self.networked_assignment_type == NetworkAssignmentType.DEGREE_CENTRALITY_FITNESS_ASSIGNMENT:
            self._degree_centrality_fitness_assignment()
        elif self.networked_assignment_type == NetworkAssignmentType.DEGREE_CENTRALITY_FITNESS_COMMUNITY_ASSIGNMENT:
            self._degree_centrality_fitness_community_assignment()
        elif self.networked_assignment_type == NetworkAssignmentType.EIGENVECTOR_CENTRALITY_FITNESS_ASSIGNMENT:
            self._eigenvector_centrality_fitness_assignment()
        elif self.networked_assignment_type == NetworkAssignmentType.CLOSEST_CENTRALITY_FITNESS_ASSIGNMENT:
            self._closest_centrality_fitness_assignment()

    def _random_assignment(self):
        nodes_indexes = list(self.network.nodes)
        individuals_indexes = list(range(len(self.population)))

        random.shuffle(nodes_indexes)
        random.shuffle(individuals_indexes)
        for n_x, i_x in zip(nodes_indexes, individuals_indexes):
            node = Node(
                n_x, self.population[i_x], self._evaluate(self.population[i_x]))

            self.network.nodes[n_x]['node'] = node

    def _community_fitness_assignment(self):
        raise NotImplementedError

    def _degree_centrality_fitness_assignment(self, generation, max_generations):
        population = np.array(self.population.copy())

        # Calculate the pairwise Euclidean distances
        pairwise_distances = np.linalg.norm(
            population[:, np.newaxis, :] - population, axis=2)

        # Set diagonal elements to a large value to exclude them from calculations
        np.fill_diagonal(pairwise_distances, 0)

        similarity_for_each_individual = np.mean(pairwise_distances, axis=1)

        # high similarity means high distance, mean this individual is not similar to other individuals (good)
        normalized_similarity = (
            similarity_for_each_individual - np.min(similarity_for_each_individual)) / (np.max(similarity_for_each_individual) - np.min(similarity_for_each_individual) + 1e-6)

        fitnesses = list(self._evaluate_population(self.population))
        fitnesses = np.array(fitnesses)
        normalized_fitnesses = (fitnesses - np.min(fitnesses)) / \
            (np.max(fitnesses) - np.min(fitnesses) + 1e-10)

        # high normalized_fitnesses means low fitness, mean this individual is good
        normalized_fitnesses = 1 - normalized_fitnesses

        # at max generation, we want fitness weight to be 0.7 and similarity weight to be 0.3
        # at min generation i.e. gen = 0, we want fitness weight to be 0.3 and similarity weight to be 0.7

        start = 0.3
        end = 0.9

        # linear decay
        fitness_weight = start + (end - start) * \
            ((generation/max_generations))

        similarity_weight = 1 - fitness_weight

        # Calculate the combined score using a weighted sum
        node_score = fitness_weight * normalized_fitnesses + \
            similarity_weight * normalized_similarity

        degrees = [self.network.degree(node) for node in self.network.nodes]

        node_score = np.array(node_score)
        degrees = np.array(degrees)

        node_score_argsort = np.argsort(node_score)[::-1]
        degrees_argsort = np.argsort(degrees)[::-1]

        for n_x, i_x in zip(degrees_argsort, node_score_argsort):
            fitness = fitnesses[i_x]
            individual = self.population[i_x]
            node = Node(
                n_x, individual, fitness)

            # if generation % 100 == 0:
            #     print(
            #         f"Assigning: degree: {degrees[n_x]}, node_score: {node_score[i_x]}, fitness: {fitness}, normalized_similarity: {normalized_similarity[i_x]}")

            self.network.nodes[n_x]['node'] = node
            self.network.nodes[n_x]['degree'] = degrees[n_x]
            self.network.nodes[n_x]['node_score'] = node_score[i_x]

        # if generation % 100 == 0:
        #     print("---------------")

    def _degree_centrality_fitness_community_assignment(self):
        raise NotImplementedError

    def _eigenvector_centrality_fitness_assignment(self):
        fitnesses = list(self._evaluate_population(self.population))
        eigenvector_centralities = list(
            nx.eigenvector_centrality(self.network).values())

        fitnesses = np.array(fitnesses)
        eigenvector_centralities = np.array(eigenvector_centralities)

        fitnesses_argsort = np.argsort(fitnesses)
        eigenvector_centralities_argsort = np.argsort(
            eigenvector_centralities)[::-1]

        if self.is_maximization:
            fitnesses_argsort = fitnesses_argsort[::-1]

        for n_x, i_x in zip(eigenvector_centralities_argsort, fitnesses_argsort):
            node = Node(
                n_x, self.population[i_x], self._evaluate(self.population[i_x]))

            self.network.nodes[n_x]['node'] = node

    def _closest_centrality_fitness_assignment(self):
        raise NotImplementedError
