import numpy as np
from copy import deepcopy
import random
import math


class CrossFingers:
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


class CrossBrain:
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
                if random.random() < 0.5:  # Replace the whole finger
                    child[i][j] = parent_2[i][j]

        return child
