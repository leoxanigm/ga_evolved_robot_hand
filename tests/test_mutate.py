import unittest
import numpy as np
from copy import deepcopy

from constants import GeneDesc
from genome import FingersGenome, BrainGenome
from cross_mutate import Mutate


class TestMutate(unittest.TestCase):
    def test_mutate(self):
        f_genome = FingersGenome.genome(GeneDesc)
        b_genome = BrainGenome.genome(f_genome)

        f_genome_c = deepcopy(f_genome)
        b_genome_c = deepcopy(b_genome)

        mut_f_genome, mut_b_genome = Mutate.mutate(
            f_genome,
            b_genome,
            np.random.uniform(-0.05, 0.05),
            np.random.uniform(-0.1, 0.1),
        )

        assert all(
            (np.any(mut_f_genome != f_genome_c), np.any(mut_b_genome != b_genome_c))
        )


if __name__ == '__main__':
    unittest.main()
