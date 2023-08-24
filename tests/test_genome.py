import unittest
import hypothesis.strategies as st
from hypothesis import given
import numpy as np

from genome import FingersGenome, BrainGenome
from constants import GeneDesc


class FingersGenomeTest(unittest.TestCase):
    @given(fingers=st.integers(3, 10), phalanges=st.integers(3, 20))
    def test_random_genome_matrix(self, fingers, phalanges):
        fingers_genome = FingersGenome(GeneDesc, rows=fingers, columns=phalanges).genome

        assert isinstance(fingers_genome, np.ndarray)
        assert fingers_genome.shape == (fingers, phalanges, len(GeneDesc))
        assert np.all((fingers_genome >= 0) & (fingers_genome < 1))


class BrainGenomeTest(unittest.TestCase):
    def test_random_genome(self):
        brain_genome = BrainGenome(layers=[(5, 8), (8, 1)]).genome

        assert isinstance(brain_genome, list)
        assert isinstance(brain_genome[0], np.ndarray)
        assert brain_genome[0].shape == (5, 8)
        assert brain_genome[1].shape == (8, )
        assert brain_genome[2].shape == (8, 1)
        assert brain_genome[3].shape == (1, )


if __name__ == '__main__':
    unittest.main()
