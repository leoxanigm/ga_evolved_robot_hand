import os
import numpy as np
import pickle

import constants as c


class FingersGenome:
    @staticmethod
    def random_genome() -> np.ndarray:
        '''
        Returns random fingers genome matrix based on maximum number of
        fingers, maximum number of phalanges and length of gene description
        '''

        max_fingers, max_phalanges = c.MAX_NUM_FINGERS, c.MAX_NUM_PHALANGES
        gene_len = len(c.GeneDesc)

        genome_matrix = np.random.uniform(
            low=0.01, high=1, size=(max_fingers, max_phalanges, gene_len)
        )

        # Generate random number of fingers, done by setting array values to
        # zero after a random index
        fingers = np.random.randint(2, max_fingers)
        genome_matrix[fingers:] = 0

        # Generate random number of phalanges for each finger
        for f in range(max_fingers):
            p = np.random.randint(2, max_phalanges)
            genome_matrix[f, p:] = 0

        return genome_matrix


class BrainGenome:
    @staticmethod
    def random_genome() -> np.ndarray:
        '''
        Genetic coding for the brain.
        The shape of the matrix is decided by the maximum number of fingers,
        maximum number of phalanges a specimen would have and the length
        of gene description.
        '''

        max_fingers, max_phalanges = c.MAX_NUM_FINGERS, c.MAX_NUM_PHALANGES
        gene_len = len(c.GeneDesc)

        shape = max_fingers, max_phalanges, gene_len
        num_inputs = c.NUMBER_OF_INPUTS

        # Take number of finger and phalanges. Size of weight array equals
        # number of inputs. For example: for a brain genome with shape (7, 10, 9),
        # and for 3 inputs, the convolution matrix will have a shape (7, 10, 7, 10, 3)
        size = (*shape[:2], *shape[:2], num_inputs)

        return np.random.uniform(low=-0.5, high=0.5, size=size)


def save_genome(genome: np.ndarray, file_path: str) -> None:
    '''Saves genome encoding to disk'''

    assert isinstance(genome, np.ndarray)

    full_path = os.path.join(os.getcwd(), file_path)

    try:
        with open(full_path, 'wb') as f:
            pickle.dump(genome, f)
    except FileNotFoundError:
        raise FileNotFoundError('Make sure the folder exists before loading the genome')


def load_genome(file_path: str) -> np.ndarray:
    '''Loads saved genome encoding from disk'''

    full_path = os.path.join(os.getcwd(), file_path)

    if not os.path.exists(full_path):
        raise FileNotFoundError('Could not load genome from the specified path')

    with open(full_path, 'rb') as f:
        genome = pickle.load(f)

    return genome


# ToDo
# Look into encoding the genome so it looks like below
# It gives an output of signed binary
# Maybe use something like np.sign(np.sum(g, axis=(2, 3, 4)))

# a = array([[[0, 1],
#          [1, 1]],

#         [[0, 0],
#          [1, 1]]]),
# g = array([[[[[-1, -1],
#            [ 0, -1]],

#           [[ 1,  0],
#            [ 0, -1]]],


#          [[[ 1,  0],
#            [ 0, -1]],

#           [[-1,  0],
#            [ 0, -1]]]],


#         [[[[-1,  0],
#            [ 0, -1]],

#           [[ 1, -1],
#            [ 1, -1]]],


#          [[[ 1,  1],
#            [ 1,  0]],

#           [[ 1, -1],
#            [ 0,  1]]]]]))
