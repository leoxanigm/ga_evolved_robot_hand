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

        max_fingers, max_phalanges = rows, columns
        
        # A random matrix with num of max fingers and phalanges
        gene_len = len(gene_description)
        genome_matrix = np.random.uniform(
            low=0.01, high=1, size=(max_fingers, max_phalanges, gene_len)
        )

        # Random number of fingers between 2 and max_fingers
        fingers = np.random.randint(2, max_fingers)
        genome_matrix[fingers:] = 0

        # Random number of phalanges for each finger
        for f in range(max_fingers):
            p = np.random.randint(2, max_phalanges)
            genome_matrix[f, p:] = 0

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

        # Take number of finger and phalanges. Size of weight array equals
        # number of inputs. For example: for a brain genome with shape (7, 10, 9),
        # and for 3 inputs, the convolution matrix will have a shape (7, 10, 7, 10, 3)
        size = (*shape[:2], *shape[:2], num_inputs)

        return np.random.uniform(low=-0.5, high=0.5, size=size)
