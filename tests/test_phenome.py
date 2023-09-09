import unittest
import hypothesis.strategies as st
from hypothesis import given
import numpy as np

from genome import FingersGenome, BrainGenome
from phenome import FingersPhenome, BrainPhenome
from constants import GeneDesc, Limits


class TestFingersPhenome(unittest.TestCase):
    fingers_genome = None
    fingers_genome = None
    palm_dim = 0.25  # robot palm size

    def setUp(self):
        self.fingers_genome = FingersGenome.genome(GeneDesc)
        self.fingers_phenome = FingersPhenome(self.fingers_genome).phenome

    def test_set_list_dimensions(self):
        '''Test link dimensions are in the range of defined constants.'''
        for i in range(len(self.fingers_phenome)):
            if np.all(self.fingers_phenome[i] == 0):
                break

            for j in range(len(self.fingers_phenome[i])):
                if np.all(self.fingers_phenome[i][j] == 0):
                    break

                phalanx_x = self.fingers_phenome[i][j][GeneDesc.DIM_X]
                phalanx_y = self.fingers_phenome[i][j][GeneDesc.DIM_Y]
                phalanx_z = self.fingers_phenome[i][j][GeneDesc.DIM_Z]
                assert Limits.DIM_X_LOWER <= phalanx_x <= Limits.DIM_X_UPPER
                assert Limits.DIM_Y_LOWER <= phalanx_y <= Limits.DIM_Y_UPPER
                assert Limits.DIM_Z_LOWER <= phalanx_z <= Limits.DIM_Z_UPPER

    def test_set_joint_origin(self):
        '''
        Test joint origin is at edge of palm for posterior phalanges.
        Test joint origin at phalanx edge for other phalanges.
        '''
        for i in range(len(self.fingers_phenome)):
            if np.all(self.fingers_phenome[i] == 0):
                break

            for j in range(len(self.fingers_phenome[i])):
                if np.all(self.fingers_phenome[i][j] == 0):
                    break
                
                phalanx_x = self.fingers_phenome[i][j][GeneDesc.JOINT_ORIGIN_X]
                phalanx_y = self.fingers_phenome[i][j][GeneDesc.JOINT_ORIGIN_Y]
                phalanx_z = self.fingers_phenome[i][j][GeneDesc.JOINT_ORIGIN_Z]

                if j == 0:
                    assert -self.palm_dim <= phalanx_x <= self.palm_dim
                    assert -self.palm_dim <= phalanx_y <= self.palm_dim
                else:
                    assert phalanx_x == 0
                    assert phalanx_y == 0

                # Joint attachment at z axis should be at edge. We just check if it is
                # in the link length limit.
                # assert Limits.DIM_Z_LOWER <= phalanx_z <= Limits.DIM_Z_UPPER

class TestBrainPhenome(unittest.TestCase):
    @given(
        fingers=st.integers(3, 10),
        phalanges=st.integers(3, 20),
        num_inputs=st.integers(3, 7),
    )
    def test_brain_trajectories(self, fingers, phalanges, num_inputs):
        fingers_genome = FingersGenome.genome(GeneDesc, rows=fingers, columns=phalanges)
        brain_genome = BrainGenome.genome(fingers_genome, num_inputs)
        brain_phenome = BrainPhenome(brain_genome)

        inputs = np.random.randint(0, 2, size=(*brain_genome.shape[:2], num_inputs))

        outputs = brain_phenome.trajectories(inputs)

        assert outputs.shape == (*brain_genome.shape[:2], )

        # All zero inputs must return no rotation (0)
        inputs = np.zeros((*brain_genome.shape[:2], num_inputs))
        outputs = brain_phenome.trajectories(inputs)

        assert np.all(outputs == 0)


if __name__ == '__main__':
    unittest.main()
