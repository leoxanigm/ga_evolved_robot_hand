import unittest
import numpy as np
import os
import time
import pybullet as p
from uuid import uuid4

import constants as c
from genome import FingersGenome, BrainGenome
from phenome import FingersPhenome, BrainPhenome
from specimen import Specimen

from helpers.misc_helpers import clear_dir


class TestSpecimen(unittest.TestCase):
    def setUp(self):
        self.specimen = Specimen()

    def tearDown(self):
        clear_dir(c.FIT_DIR)

    def test_save_specimen(self):
        self.specimen.save_specimen()

        output_f_g = os.path.join(c.FIT_DIR, f'{self.specimen.id}_fingers.pk')
        output_b_g = os.path.join(c.FIT_DIR, f'{self.specimen.id}_brain.pk')

        assert os.path.exists(output_f_g)
        assert os.path.exists(output_b_g)


if __name__ == '__main__':
    unittest.main()
