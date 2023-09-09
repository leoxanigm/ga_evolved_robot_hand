import numpy as np
from copy import deepcopy
import random
import math

from constants import GeneDesc, Limits


class Cross:
    '''
    Crosses two two specimen parents.
    It takes phalanx genomes and corresponding convolution matrices from
    two parents and returns a child genome encoding of fingers and brain.
    It performs a Biased Uniform Crossing where fitter phalanges are
    taken from each parent.
    '''

    @staticmethod
    def __get_genome_mapping():
        # To Do
        # Check if genome mapping works for crossing
        # map fitness to genome encoding
        # take either from parent a or b by which ever fitness is greater
        # use np boolean indexing
        pass
    @staticmethod
    def cross_genomes(parent_1: np.ndarray, parent_2: np.ndarray):
        assert parent_1.shape == parent_2.shape

        child = deepcopy(parent_1)

        for i in range(len(child)):  # loop through fingers
            if random.random() < 0.5:  # Replace the whole finger
                child[i] = parent_2[i]
            else:
                for j in range(len(child[i])):
                    if random.random() < 0.5:  # Replace a phalanx
                        child[i][j] = parent_2[i][j]

        child = CrossFingers.__move_non_zero_to_top(child)

        return child

    @staticmethod
    def __move_non_zero_to_top(child):
        '''Move zero values to the end to avoid unwanted finger structure
        For example: It restructures the array [1, 1, 0, 1] to [1, 1, 1, 0]'''

        child_modified = deepcopy(child)

        # Rearrange fingers by sorting based on zero values
        # This works by checking if all the values in the finger array are zero,
        # and if there are, moves them to the bottom
        child_modified = np.array(sorted(child, key=lambda x: np.all(x == 0)))

        # Do the same for the phalanges in each finger
        for i in range(len(child_modified)):
            child_modified[i] = np.array(
                sorted(child_modified[i], key=lambda x: np.all(x == 0))
            )

        return child_modified


class MutateFingers:
    '''Randomly adds a small value (positive or negative) to the genetic encoding
    of fingers to give them new physical characters'''

    @staticmethod
    def mutate(child):
        mutated_child = deepcopy(child)

        for i in range(len(child)):  # Fingers
            for j in range(len(child[i])):  # Phalanges
                if random.random() < 0.3:
                    # Calculate a small value from each of x, y, z dimensions'abs
                    # small limits
                    m_v_x = Limits.DIM_X_LOWER / 10
                    m_v_y = Limits.DIM_Y_LOWER / 10
                    m_v_z = Limits.DIM_Z_LOWER / 10

                    # Add/subtract a random amount based on the above values
                    child[i][j][GeneDesc.DIM_X] += np.random.uniform(-m_v_x, m_v_x)
                    child[i][j][GeneDesc.DIM_Y] += np.random.uniform(-m_v_y, m_v_y)
                    child[i][j][GeneDesc.DIM_Z] += np.random.uniform(-m_v_z, m_v_z)

        return mutated_child


class MutateBrain:
    '''Randomly adds a small value (positive or negative) to the genetic encoding
    of weights and biases to give them new values'''

    @staticmethod
    def mutate(child):
        assert isinstance(child, list)
        assert isinstance(child[0], np.ndarray)

        mutated_child = deepcopy(child)

        # Loop through each weight and bias np array and randomly chose a value
        # from each of the parents
        for i in range(len(mutated_child)):
            # We will use the same logic used to generate random weight and
            # bias values to generate a random value to augment the values
            features = len(mutated_child[i])
            m_v = np.sqrt(1 / features)

            for j in range(len(mutated_child[i])):
                if isinstance(mutated_child[i][j], np.ndarray):
                    for k in range(len(mutated_child[i][j])):
                        if random.random() < 0.5:
                            mutated_child[i][j][k] += np.random.uniform(-m_v, m_v)
                else:
                    if random.random() < 0.5:
                        mutated_child[i][j] += np.random.uniform(-m_v, m_v)

        return mutated_child


class CrossBrain:
    '''Crosses weights and biases of two parents to give a child with a
    new set of weights and biases'''

    @staticmethod
    def cross_genomes(parent_1: list[np.ndarray], parent_2: list[np.ndarray]):
        # Check there are equal number of weights and biases for each parent
        for i in range(len(parent_1)):
            assert parent_1[i].shape == parent_2[i].shape

        child = deepcopy(parent_1)

        # Loop through each weight and bias np array and randomly chose a value
        # from each of the parents
        for i in range(len(child)):
            for j in range(len(child[i])):
                if random.random() < 0.5:
                    # Both parents have equal probability of passing a genome information
                    child[i][j] = parent_2[i][j]

        return child
