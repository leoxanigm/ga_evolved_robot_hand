import unittest
import hypothesis.strategies as st
from hypothesis import given
import numpy as np

from genome import FingersGenome, BrainGenome
from phenome import FingersPhenome, BrainPhenome
import constants as c
from constants import GeneDesc, Limits



class TestFingersPhenome(unittest.TestCase):
    fingers_genome = None
    fingers_genome = None
    palm_dim = 0.25  # robot palm size

    def setUp(self):
        self.fingers_genome = FingersGenome.random_genome()
        self.fingers_phenome = FingersPhenome.genome_to_phenome(self.fingers_genome)

    def test_set_link_dimensions(self):
        '''Test link dimensions are in the range of defined constants.'''

        non_zero_map = self.fingers_phenome != 0

        assert np.all(
            self.fingers_phenome[:, :, GeneDesc.RADIUS][
                non_zero_map[:, :, GeneDesc.RADIUS]
            ]
            >= Limits.RADIUS_LOWER
        ) and np.all(
            self.fingers_phenome[:, :, GeneDesc.RADIUS][
                non_zero_map[:, :, GeneDesc.RADIUS]
            ]
            <= Limits.RADIUS_UPPER
        )
        assert np.all(
            self.fingers_phenome[:, :, GeneDesc.LENGTH][
                non_zero_map[:, :, GeneDesc.LENGTH]
            ]
            >= Limits.LENGTH_LOWER
        ) and np.all(
            self.fingers_phenome[:, :, GeneDesc.LENGTH][
                non_zero_map[:, :, GeneDesc.LENGTH]
            ]
            <= Limits.LENGTH_UPPER
        )

    def test_set_joint_origin(self):
        '''
        Test no two joint origins overlap
        '''

        joint_origins = self.fingers_phenome[
            :, 0, GeneDesc.JOINT_ORIGIN_X : GeneDesc.JOINT_ORIGIN_Z
        ]
        tracked = []
        for x, y in joint_origins:
            if not x == y == 0:
                assert (x, y) not in tracked
                tracked.append((x, y))


class TestBrainPhenome(unittest.TestCase):
    def test_brain_trajectories(self):
        fingers_genome = FingersGenome.random_genome()
        brain_genome = BrainGenome.random_genome()

        inputs = np.random.randint(0, 2, size=(*brain_genome.shape[:2], c.NUMBER_OF_INPUTS))
        outputs = BrainPhenome.trajectories(brain_genome, inputs)

        assert outputs.shape == (*brain_genome.shape[:2],)

        # All zero inputs must return no rotation (0)
        inputs = np.zeros((*brain_genome.shape[:2], c.NUMBER_OF_INPUTS))
        outputs = BrainPhenome.trajectories(brain_genome, inputs)

        assert np.all(outputs == 0)


if __name__ == '__main__':
    unittest.main()
