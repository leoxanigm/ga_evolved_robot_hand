import numpy as np
from torch.nn.parameter import Parameter
from types import GeneratorType


class FingersGenome:
    '''
    Genetic coding for the fingers.
    Genome is a 7 column by 10 row matrix. Each cell contains genotype info
    for a phalanx. If there is an array of random floats, there exists a phalanx
    if there is a 0, there is no data. Hence, no phenotype will be generated.

    Args:
        gene_description (Enum): attributes for a typical phalanx. Defined
            in the constants.py file
        rows: maximum number of fingers
        columns: maximum number of phalanges
    '''

    def __init__(self, gene_description, rows=7, columns=10):
        self.gene_len = len(gene_description)
        self.max_fingers = rows
        self.max_phalanges = columns

        self.genome_matrix = None

        # Initialize genome
        self.__set_default_matrix()
        self.__generate_rand_fingers()

    @property
    def genome(self):
        '''Returns random fingers genome matrix'''

        return self.genome_matrix

    def __set_default_matrix(self):
        '''
        Returns a default matrix of zeros.
        Rows are list of fingers
        Columns describe each finger with list of phalanges
        '''

        self.genome_matrix = np.zeros(
            (self.max_fingers, self.max_phalanges, self.gene_len), dtype=np.float64
        )

    def __generate_rand_fingers(self):
        '''
        Generates random fingers by setting random array of floats
        '''

        assert self.genome_matrix is not None

        # random number of fingers
        no_of_fingers = np.random.randint(2, len(self.genome_matrix))

        for i in range(no_of_fingers):
            # random number of phalanx
            no_of_phalanx = np.random.randint(2, len(self.genome_matrix[i]))

            for j in range(no_of_phalanx):
                self.genome_matrix[i][j] = np.random.uniform(0.01, 1, self.gene_len)


class BrainGenome:
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

    def __init__(self, layers=[(5, 8), (8, 1)]):
        self.layers = layers
        self.genome_array = None

        # Initialize genome
        self.__generate_default_genome_array()

    @property
    def genome(self):
        '''Returns random brain genome array'''

        return self.genome_array

    def get_genome_shape(self):
        '''Returns layer shape used to initialize genome'''
        return self.layers

    def set_genome(self, model_parameters):
        '''Sets genome from model parameters
        Args:
            model_parameters (generator[torch.nn.parameter.Parameter])
        '''
        genome_array = []

        isinstance(model_parameters, GeneratorType)

        # We have to first convert PyTorch tensor to numpy
        for parameter in model_parameters:
            genome_array.append(parameter.data.numpy())

        self.genome_array = genome_array

    def save_genome(self):
        pass

    def load_genome(self):
        pass

    def __generate_default_genome_array(self):
        '''
        Returns a list of numpy arrays with shapes defined by layers input,
        The numpy arrays represent weights and biases.
        For the default configuration, it returns a random float list of
        numpy arrays with shape: [(5, 8), (8,), (8, 1), (1,)]
        '''
        assert type(self.layers) == list

        genome_array = []

        for i, j in self.layers:
            # The random values will be initialized based on in-features and
            # out-features to follow how weights and biases are initialized
            # for PyTorch Linear layer models.
            # Source: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
            rand_range = np.sqrt(1 / i)
            genome_array.append(
                np.random.uniform(-rand_range, rand_range, (i, j))
            )  # weights
            genome_array.append(
                np.random.uniform(-rand_range, rand_range, (j,))
            )  # biases

        self.genome_array = genome_array
