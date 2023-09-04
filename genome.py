import numpy as np
from torch.nn.parameter import Parameter
from types import GeneratorType


class FingersGenome:
    @staticmethod
    def genome(gene_description, rows=7, columns=10):
        '''
        Returns random fingers genome matrix

        Args:
            gene_description (Enum): attributes for a typical phalanx. Defined
                in the constants.py file
            rows: maximum number of fingers
            columns: maximum number of phalanges
        '''

        genome_matrix = FingersGenome.__set_default_matrix(
            len(gene_description), rows, columns
        )
        genome_matrix = FingersGenome.__generate_rand_fingers(genome_matrix)

        return genome_matrix

    @staticmethod
    def __set_default_matrix(gene_len: int, max_fingers: int, max_phalanges: int):
        '''
        Returns a default matrix of zeros.
        Rows are list of fingers
        Columns describe each finger with list of phalanges

        Args:
            max_fingers: maximum number of fingers
            max_phalanges: maximum number of phalanges in each finger
            gene_len: length of genome description for each phalanx
        '''

        genome_matrix = np.zeros(
            (max_fingers, max_phalanges, gene_len), dtype=np.float64
        )
        return genome_matrix

    @staticmethod
    def __generate_rand_fingers(genome_matrix: np.ndarray):
        '''
        Generates random fingers by setting random array of floats

        Args:
            genome_matrix: a genome matrix of zeros with
                shape (fingers, phalanges, gene desc length)
        '''

        # random number of fingers
        no_of_fingers = np.random.randint(2, len(genome_matrix))

        for i in range(no_of_fingers):
            # random number of phalanx
            no_of_phalanx = np.random.randint(2, len(genome_matrix[i]))

            for j in range(no_of_phalanx):
                genome_matrix[i][j] = np.random.uniform(
                    0.01, 1, len(genome_matrix[i][j])
                )

        return genome_matrix


class BrainGenome:
    @staticmethod
    def genome(layers=[(5, 8), (8, 1)]):
        '''
        Genetic coding for the brain.
        This represents the parameters (weights and biases) of the hidden
        neurons for the neural network control mechanism.

        The brain calculates the target rotation angle for each phalanx.
        The neural network will get the inputs:
            Distance of a phalanx from the target object:
                Calculated using PyBullet's getClosestPoints().
            Rotation axis:
                The rotational axis of a phalanx in the shape [x, y, z]
            Distance of the center of mass of a phalanx from the palm:

        The input will be tensor of length five. The first index is the distance
        from the target object, indices 1, 2, and 3 are a the axis of rotation
        (For example, an input of 1,0,0 means the phalanx rotates around x axis),
        and the last index is the distance of center of mass of the phalanx from
        the palm.

        The outputs will a rotation angle for each phalanx.

        Args:
            layers: A list containing the number of neurons for the hidden layers.
                Default is for a neural network with 1 hidden layer of 8
                neurons.
        '''

        return BrainGenome.__generate_default_genome_array(layers)

    @staticmethod
    def __generate_default_genome_array(layers):
        '''
        Returns a list of numpy arrays with shapes defined by layers input,
        The numpy arrays represent weights and biases.
        For the default configuration, it returns a random float list of
        numpy arrays with shape: [(5, 8), (8,), (8, 1), (1,)]
        '''
        assert type(layers) == list

        genome_array = []

        for i, j in layers:
            # The random values will be initialized based on in-features and
            # out-features to follow how weights and biases are initialized
            # for PyTorch Linear layer models.
            # Source: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
            rand_range = np.sqrt(1 / i)
            random_weights = np.random.uniform(-rand_range, rand_range, (i, j))
            random_weights = random_weights.astype(np.float32)
            genome_array.append(random_weights)  # weights

            random_biases = np.random.uniform(-rand_range, rand_range, (j,))
            random_biases = random_biases.astype(np.float32)
            genome_array.append(random_biases)  # biases

        return genome_array
