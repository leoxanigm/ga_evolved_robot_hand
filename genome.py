import numpy as np


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
    def genome(genome_matrix, num_inputs=3) -> np.ndarray:
        '''
        Genetic coding for the brain.
        The shape of the matrix is decided by the shape of genome_matrix.

        The brain decides the rotation direction of a phalanx: either +ve,
        -ve or no rotation. It takes a tuple of binary inputs for each phalanx
        that have structure (distance [1/0], collision with target [1/0]) and
        (collision with obstacle [1/0]). It multiples the inputs for all
        phalanges with a convolution matrix of a phalanx and outputs rotation
        direction for the said phalanx.

        Args:
            genome_matrix: genome encoding for the fingers
            num_inputs: number of inputs
        '''
        assert isinstance(genome_matrix, np.ndarray)
        shape = genome_matrix.shape
        return BrainGenome.__generate_convolution_matrix(shape, num_inputs)

    @staticmethod
    def __generate_convolution_matrix(shape: tuple, num_inputs: int) -> np.ndarray:
        '''
        Returns a convolution matrix with random weights with shape
        the same as input.
        '''

        # Take number of finger and phalanges. Size of weight array equals
        # number of inputs. For example: for a brain genome with shape (7, 10, 9),
        # and for 3 inputs, the convolution matrix will have a shape (7, 10, 7, 10, 3)
        size = (*shape[:2], *shape[:2], num_inputs)
        convolution_matrix = np.random.uniform(low=-0.5, high=0.5, size=size)

        return convolution_matrix
