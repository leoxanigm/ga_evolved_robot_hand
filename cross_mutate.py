import numpy as np
from copy import deepcopy
import random
import math

from specimen import Phalanx

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
    def __get_genome_mask(
        fitness_map_1: np.ndarray, fitness_map_2: np.ndarray
    ) -> np.ndarray[np.bool_]:
        '''Returns a boolean map array by comparing two fitness maps.
        Array value will be true if fitness phalanx fitness in fitness_map_1
        is greater than a corresponding fitness index in fitness_map_2'''

        assert isinstance(fitness_map_1, np.ndarray) and isinstance(
            fitness_map_2, np.ndarray
        )
        assert fitness_map_1.shape == fitness_map_2.shape

        return fitness_map_1 > fitness_map_2

    @staticmethod
    def __shorten_child(
        parent_1: np.ndarray, parent_2: np.ndarray, child_genome: np.ndarray
    ) -> np.ndarray:
        '''This makes sure number of child phalanges is between the number
        of phalanges of its parent. This is to avoid longer and longer fingers
        with every crossing'''

        try:
            # Get random target phalanx count between the two parents
            target_num = np.random.randint(
                *sorted(
                    (np.sum(parent_1[:, :, 0] != 0), np.sum(parent_2[:, :, 0] != 0))
                )
            )
        except ValueError:
            # Same number of parents
            return child_genome

        # Start from the end phalanges and remove them to get to
        # the target count
        # For now this will diverge to fingers with equal length
        # ToDo: introduce some randomness
        p_i = -1  # phalanx index
        while np.sum(child_genome[:, :, 0] != 0) > target_num:
            if np.all(child_genome[:, p_i] == 0):
                p_i -= 1
            child_genome[:, p_i] = 0

        return child_genome

    @staticmethod
    def cross_genomes(
        f_g_parent_1: np.ndarray,
        b_g_parent_1: np.ndarray,
        fit_parent_1: np.ndarray,
        f_g_parent_2: np.ndarray,
        b_g_parent_2: np.ndarray,
        fit_parent_2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        '''
        Crosses fingers and brain genome of two parents

        Args:
            f_g_parent_1: parent 1 finger genome
            b_g_parent_1: parent 1 brain genome
            fit_parent_1: parent 1 phalanx fitness map
            f_g_parent_2: parent 2 finger genome
            b_g_parent_2: parent 2 brain genome
            fit_parent_2: parent 2 phalanx fitness map
        '''

        assert all(isinstance(arr, np.ndarray) for arr in locals().values())

        # Initialize children genome with zeros
        f_g_child = np.zeros(f_g_parent_1.shape)  # fingers genome
        b_g_child = np.zeros(b_g_parent_1.shape)  # brain genome

        fitness_mask = Cross.__get_genome_mask(fit_parent_1, fit_parent_2)

        # Cross fingers genome
        # First take fittest phalanges from parent 1
        f_g_child[fitness_mask] = f_g_parent_1[fitness_mask]
        b_g_child[fitness_mask] = b_g_parent_1[fitness_mask]
        # Next take fittest phalanges from parent 2
        f_g_child[np.invert(fitness_mask)] = f_g_parent_2[np.invert(fitness_mask)]
        b_g_child[np.invert(fitness_mask)] = b_g_parent_2[np.invert(fitness_mask)]

        f_g_child = Cross.__shorten_child(f_g_parent_1, f_g_parent_2, f_g_child)

        return f_g_child, b_g_child


class Mutate:
    '''Randomly adds a small value (positive or negative) to the genetic encodings'''

    @staticmethod
    def mutate(
        finger_genome: np.ndarray,
        brain_genome: np.ndarray,
        f_amount: float,
        b_amount: float,
        f_rate: float = np.random.uniform(),
        b_rate: float = np.random.uniform(),
    ):
        '''
        Mutates supplied fingers and brain genomes by specified amount and given rate
        
        Args:
            finger_genome: fingers genome encoding
            brain_genome: brain genome encoding
            f_amount: amount to mutate fingers genome by
            b_amount: amount to mutate brain genome by
            f_rate: rate to mutate fingers genome by
            b_rate: rate to mutate brain genome by
        '''

        assert isinstance(finger_genome, np.ndarray) and isinstance(
            brain_genome, np.ndarray
        )

        # Mutate fingers genome
        # Source: https://numpy.org/doc/stable/reference/arrays.nditer.html#modifying-array-values
        with np.nditer(finger_genome, op_flags=['readwrite']) as it:
            for encoding in it:
                if np.random.uniform() < f_rate and encoding != 0:
                    # For finger genome, the values shouldn't be out of the bounds 0.01 and 1
                    if encoding + f_amount < 0.01 or encoding + f_amount > 1:
                        continue
                    encoding += f_amount

        # Mutate brain genome
        with np.nditer(brain_genome, op_flags=['readwrite']) as it:
            for encoding in it:
                if np.random.uniform() < b_rate and encoding != 0:
                    encoding += b_amount

        return finger_genome, brain_genome
