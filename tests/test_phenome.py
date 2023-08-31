import unittest
import numpy as np

from genome import FingersGenome, BrainGenome
from phenome import FingersPhenome, BrainPhenome
from constants import GeneDesc, Limits


class TestFingersPhenome(unittest.TestCase):
    fingers_genome = None
    fingers_genome = None
    palm_dim = 0.25  # robot palm size

    def setUp(self):
        self.fingers_genome = FingersGenome(GeneDesc).genome
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
                assert Limits.DIM_Z_LOWER <= phalanx_z <= Limits.DIM_Z_UPPER

class TestBrainPhenome(unittest.TestCase):
    def test_model_to_genome(self):
        brain_genome = BrainGenome().genome
        brain_phenome = BrainPhenome(brain_genome)
        model_layers = []
        
        for layer in brain_phenome.model_layers.parameters():
            model_layers.append(np.array(layer.data))

        for i in range(len(model_layers)):
            model_layer = model_layers[i]
            if i % 2 == 0:
                model_layer = model_layer.T

            assert model_layer.shape == brain_genome[i].shape
            assert np.all(model_layer == brain_genome[i])

    def test_genome_update_after_learn(self):
        brain_genome = BrainGenome().genome
        brain_phenome = BrainPhenome(brain_genome)

        output = brain_phenome.move([0.13, 0, 1, 0, 0.185])
        target = -np.arctan(0.13/0.185)
        brain_phenome.learn([target])

        new_brain_genome = brain_phenome.genome

        assert len(new_brain_genome) == len(brain_genome)

        for i in range(len(new_brain_genome)):
            assert new_brain_genome[i].shape == brain_genome[i].shape
            assert np.any(new_brain_genome[i] != brain_genome[i])


if __name__ == '__main__':
    unittest.main()
