import unittest
import numpy as np
import os
import time
import pybullet as p

import constants as c
from genome import FingersGenome, BrainGenome
from phenome import FingersPhenome, BrainPhenome
from specimen import Specimen
from simulation import Simulation

from helpers.misc_helpers import clear_dir


class TestSpecimen(unittest.TestCase):
    def setUp(self):
        self.fingers_genome = FingersGenome.random_genome()
        self.fingers_phenome = FingersPhenome.genome_to_phenome(self.fingers_genome)
        self.brain_genome = BrainGenome.random_genome()
        self.specimen = Specimen()

    def test_fitness_exists(self):
        '''Check fitness is calculated for each phalanx'''

        with Simulation() as sim:
            sim.run_specimen(self.specimen)

        for phalanx in self.specimen.phalanges:
            assert sum(phalanx.get_performance()) > 0


if __name__ == '__main__':
    unittest.main()
