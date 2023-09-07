import unittest
import hypothesis.strategies as st
from hypothesis import given
import numpy as np

from genome import FingersGenome, BrainGenome
from constants import GeneDesc


class FingersGenomeTest(unittest.TestCase):
    @given(fingers=st.integers(3, 10), phalanges=st.integers(3, 20))
    def test_random_genome_matrix(self, fingers, phalanges):
        fingers_genome = FingersGenome.genome(GeneDesc, rows=fingers, columns=phalanges)

        assert isinstance(fingers_genome, np.ndarray)
        assert fingers_genome.shape == (fingers, phalanges, len(GeneDesc))
        assert np.all((fingers_genome >= 0) & (fingers_genome < 1))


class BrainGenomeTest(unittest.TestCase):
    @given(
        fingers=st.integers(3, 10),
        phalanges=st.integers(3, 20),
        num_inputs=st.integers(3, 7),
    )
    def test_convolution_matrix(self, fingers, phalanges, num_inputs):
        fingers_genome = FingersGenome.genome(GeneDesc, rows=fingers, columns=phalanges)
        brain_genome = BrainGenome.genome(fingers_genome, num_inputs)

        assert isinstance(brain_genome, np.ndarray)
        assert brain_genome.shape == (fingers, phalanges, fingers, phalanges, num_inputs)


if __name__ == '__main__':
    unittest.main()
