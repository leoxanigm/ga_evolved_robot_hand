import numpy as np


class FingersGenome:
    '''
    Genetic coding for the fingers.
    Genome is a 7 column by 10 row matrix. Each cell contains genotype info
    for a phalanx. If there is an array of random floats, there exists a phalanx
    if there is a 0, there is no data. Hence, no phenotype will be generated.
    '''

    def __init__(self, gene_description, rows=7, columns=10):
        self.gene_len = len(gene_description)
        self.max_fingers = rows
        self.max_phalanges = columns

        self.genome_matrix = None

    def get_genome(self):
        # initialize genome
        self.__set_default_matrix()
        self.__generate_rand_fingers()

        return self.genome_matrix

    def __set_default_matrix(self):
        '''
        Returns a default matrix of zeros.
        Rows are list of fingers
        Columns describe each finger with list of phalanges
        '''

        print(f'Gene_len {self.gene_len}')

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
    This represents the parameters for the neural network control mechanism.

    The neural network will have the inputs:
        Target object bounding box:
            Calculated using PyBullet's getAABB function.
            Returns [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        Object class (type of target object):
            numbers from 1 (For the scope of this project, manually assigned)
        Action:
            pickup / drop (0 / 1)
        Genome matrix:
            Gives the brain information about fingers' anatomy it is supposed to control.
            The genome matrix has the shape (7, 10, 9)

    The outputs will be rotation angles for each phalanx. The number of output is 70, as
    there can be a maximum of 70 phalanges in the 7 by 10 genome matrix.
    '''

    def __init__(self, no_of_inputs=638, layers=[52, 52, 52], no_of_outputs=70):
        '''
        Params:
            no_of_inputs: The number of input neurons. Default is 638 for (2, 3) shape
                            target object bounding box, (1, ) shape object class, (1, )
                            shape action and (7, 10, 9) shape genome matrix
            layers: A list containing the number of neurons for the hidden layers. Default
                    is for a neural network with 3 hidden layers of 52 neurons each.
            no_of_outputs: rotation angles for each phalanx. Default is 70
        '''
        self.no_of_inputs = no_of_inputs
        self.layers = layers
        self.no_of_outputs = no_of_outputs

        self.genome_array = None

    def get_genome(self):
        # initialize genome
        self.__generate_default_genome_array()

        return self.genome_array

    def __generate_default_genome_array(self):
        '''
        Returns a list of numpy arrays with shapes defined by inputs, hidden layers and outputs.
        The numpy arrays represent weights and biases.
        '''
        assert type(self.layers) == list

        list_template = self.layers + [self.no_of_outputs]
        genome_array = []

        for i in range(len(list_template)):
            if i == 0:
                # First np array should be weights with shape of
                # (number of neurons of first hidden layer, number of inputs)
                np_rand_weights = np.random.rand(list_template[i], self.no_of_inputs)
                np_rand_biases = np.random.rand(list_template[i])
                genome_array.append(np_rand_weights)
                genome_array.append(np_rand_biases)
            else:
                np_rand_weights = np.random.rand(list_template[i], list_template[i - 1])
                np_rand_biases = np.random.rand(list_template[i])
                genome_array.append(np_rand_weights)
                genome_array.append(np_rand_biases)

        self.genome_array = genome_array
