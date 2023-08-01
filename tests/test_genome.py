import unittest
import hypothesis.strategies as st
from hypothesis import given
import numpy as np

from genome import FingersGenome, BrainGenome
from constants import GeneDesc


class FingersGenomeTest(unittest.TestCase):
    @given(values=st.floats(min_value=0.0, max_value=1.0))
    def test_random_genome_matrix(self, values):
        finger_genome = FingersGenome(GeneDesc).get_genome()
        assert isinstance(finger_genome, np.ndarray)
        assert finger_genome.shape == (7, 10, 9)
        assert np.all((finger_genome >= 0) & (finger_genome < 1))


if __name__ == '__main__':
    unittest.main()

class BrainGenomeTest(unittest.TestCase):
    @given(
        no_of_inputs=st.integers(min_value=1, max_value=1000),
        layers=st.lists(st.integers(min_value=10, max_value=60), min_size=1),
        no_of_outputs=st.integers(min_value=20, max_value=100)
    )
    def test_random_genome(self, no_of_inputs, layers, no_of_outputs):
        brain_genome = BrainGenome(no_of_inputs, layers, no_of_outputs).get_genome()

        # For example, a NN with 3 hidden layers, there will be a total of 4 weight and
        # 4 bias numpy arrays
        assert len(brain_genome) == (len(layers) + 1) * 2

        for i, np_genome in enumerate(brain_genome):
            assert isinstance(np_genome, np.ndarray)

            if i == 0:
                # First np array should be weights with shape of
                # (number of neurons of first hidden layer, number of inputs)
                assert np_genome.shape == (layers[0], no_of_inputs)

            if i == len(brain_genome) - 1:
                # Last np array should be biases with shape of
                # (number of outputs, )
                assert np_genome.shape == (no_of_outputs, )

            assert np.all((np_genome >= 0) & (np_genome <= 1))