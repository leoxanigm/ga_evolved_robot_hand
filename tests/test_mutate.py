import unittest
import hypothesis.strategies as st
from hypothesis import given, settings
import numpy as np

from constants import GeneDesc
from genome import FingersGenome, BrainGenome
from cross_mutate import CrossFingers, CrossBrain, MutateBrain, MutateFingers


class TestMutateBrain(unittest.TestCase):
    def test_mutate(self):
        genome = BrainGenome().genome
        mutated = MutateBrain.mutate(genome)

        different_from_original = False
        for i in range(len(mutated)):
            if np.any(mutated[i] != genome[i]):
                different_from_original = True

        assert different_from_original


class TestMutateFingers(unittest.TestCase):
    def test_mutate(self):
        genome = FingersGenome(GeneDesc).genome
        mutated = MutateFingers.mutate(genome)

        assert not np.all(mutated == genome)


if __name__ == '__main__':
    unittest.main()
