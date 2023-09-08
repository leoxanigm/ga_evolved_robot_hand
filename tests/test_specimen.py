import unittest
import hypothesis.strategies as st
from hypothesis import given, settings
import numpy as np
import os

from constants import GeneDesc, TRAINING_DIR
from genome import FingersGenome, BrainGenome
from specimen import Specimen
from helpers.misc_helpers import clear_training_dir


class TestSpecimen(unittest.TestCase):
    def tearDown(self):
        clear_training_dir()

    def test_random_specimen_genome(self):
        clear_training_dir()

        specimen = Specimen()

        assert isinstance(specimen.fingers_genome, np.ndarray)
        assert isinstance(specimen.brain_genome, np.ndarray)

    @given(fingers=st.integers(3, 10), phalanges=st.integers(3, 20))
    def test_set_genomes(self, fingers, phalanges):
        fingers_genome = FingersGenome.genome(GeneDesc, rows=fingers, columns=phalanges)
        brain_genome = BrainGenome.genome(fingers_genome)

        specimen = Specimen(
            fingers_genome=fingers_genome,
            brain_genome=brain_genome,
        )

        assert np.all(specimen.fingers_genome == fingers_genome)
        assert np.all(specimen.brain_genome == brain_genome)


if __name__ == '__main__':
    unittest.main()
