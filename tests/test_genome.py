import os
import unittest
import hypothesis.strategies as st
from hypothesis import given
import numpy as np

from genome import FingersGenome, BrainGenome, save_genome, load_genome
import constants as c
from helpers.misc_helpers import clear_dir


class TestFingersGenome(unittest.TestCase):
    def test_random_genome_matrix(self):
        fingers_genome = FingersGenome.random_genome()

        assert isinstance(fingers_genome, np.ndarray)
        assert fingers_genome.shape == (
            c.MAX_NUM_FINGERS,
            c.MAX_NUM_PHALANGES,
            len(c.GeneDesc),
        )
        # Check there are random number of fingers
        assert not np.all(fingers_genome > 0)


class TestBrainGenome(unittest.TestCase):
    def test_convolution_matrix(self):
        brain_genome = BrainGenome.random_genome()

        assert isinstance(brain_genome, np.ndarray)
        assert brain_genome.shape == (
            c.MAX_NUM_FINGERS,
            c.MAX_NUM_PHALANGES,
            c.MAX_NUM_FINGERS,
            c.MAX_NUM_PHALANGES,
            c.NUMBER_OF_INPUTS,
        )


class TestSaveLoadGenome(unittest.TestCase):
    def tearDown(self):
        clear_dir(c.FIT_DIR)

    def test_save_genome(self):
        fingers_genome = FingersGenome.random_genome()
        brain_genome = BrainGenome.random_genome()

        save_genome(fingers_genome, c.FIT_DIR + '/f_g.pk')
        save_genome(brain_genome, c.FIT_DIR + '/b_g.pk')

        assert os.path.exists(c.FIT_DIR + '/f_g.pk')
        assert os.path.exists(c.FIT_DIR + '/b_g.pk')

    def test_load_genome(self):
        fingers_genome = FingersGenome.random_genome()
        brain_genome = BrainGenome.random_genome()

        save_genome(fingers_genome, c.FIT_DIR + '/f_g_2.pk')
        save_genome(brain_genome, c.FIT_DIR + '/b_g_2.pk')

        load_fingers_genome = load_genome(c.FIT_DIR + '/f_g_2.pk')
        load_brain_genome = load_genome(c.FIT_DIR + '/b_g_2.pk')

        assert np.all(fingers_genome == load_fingers_genome)
        assert np.all(brain_genome == load_brain_genome)


if __name__ == '__main__':
    unittest.main()
