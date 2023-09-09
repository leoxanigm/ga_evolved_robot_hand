import unittest
import hypothesis.strategies as st
from hypothesis import given, settings
import numpy as np
import os
import time

from constants import GeneDesc, TRAINING_DIR
from genome import FingersGenome, BrainGenome
from phenome import FingersPhenome, BrainPhenome
from specimen import Specimen
from simulation import Simulation
from fitness_fun import FitnessFunction

from helpers.misc_helpers import clear_training_dir


class BrainPhenomeTest(BrainPhenome):
    '''A test brain that overwrites move method to output custom
    values for tests'''

    def __init__(self, *args, **kwargs):
        super(BrainPhenomeTest, self).__init__(*args, **kwargs)

    def trajectories(self, *args, target_object: int, iteration: int):
        output = np.zeros((7, 10))
        
        if target_object == 2 and iteration == 1: # cube
            output[:4, 0] = 1
        if target_object == 3 and iteration == 1: # sphere
            output[:4, 1] = 1
        if target_object == 3 and iteration in [2, 3, 4]:  # sphere
            output[:4, 2] = 1
        if target_object == 4 and iteration == 1:  # cylinder
            output[0, 0] = 1
            output[0, 1] = -1
            output[1:3, 1] = 1
            output[1, 2] = 1
        if target_object == 4 and iteration == 1:  # cylinder
            output[1, 2] = 1
        return output


class FingerPhenomeTest(FingersPhenome):
    '''A test finger phenome that gives a manually generated
    fit specimen'''

    def __init__(self, *args, **kwargs):
        super(FingerPhenomeTest, self).__init__(*args, **kwargs)

    @property
    def genome(self):
        self.genome_matrix = np.zeros((7, 10, 9))
        # Joint attachment for four fingers
        self.genome_matrix[0, 0, GeneDesc.JOINT_ORIGIN_X] = -0.125
        self.genome_matrix[1, 0, GeneDesc.JOINT_ORIGIN_X] = 0.125
        self.genome_matrix[2, 0, GeneDesc.JOINT_ORIGIN_Y] = 0.125
        self.genome_matrix[3, 0, GeneDesc.JOINT_ORIGIN_Y] = -0.125
        self.genome_matrix[:4, :3, GeneDesc.JOINT_ORIGIN_Z] = 0.1
        # Dimensions
        self.genome_matrix[:4, :3, GeneDesc.DIM_X] = 0.012
        self.genome_matrix[:4, :3, GeneDesc.DIM_Y] = 0.022
        self.genome_matrix[:4, :3, GeneDesc.DIM_Z] = 0.1
        # Rotation axis
        self.genome_matrix[0, :3, GeneDesc.JOINT_AXIS_Y] = 1
        self.genome_matrix[1, :3, GeneDesc.JOINT_AXIS_Y] = -1
        self.genome_matrix[2, :3, GeneDesc.JOINT_AXIS_X] = 1
        self.genome_matrix[3, :3, GeneDesc.JOINT_AXIS_X] = -1
        return self.genome_matrix


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

    def test_fit_specimen(self):
        '''Test a manually created fit specimen satisfies the
        fitness requirements'''

        start = time.time()

        fingers_genome = FingersGenome.genome(GeneDesc)
        brain_genome = BrainGenome.genome(fingers_genome)
        finger_phenome = FingerPhenomeTest(fingers_genome)
        brain_phenome = BrainPhenomeTest(brain_genome)
        specimen = Specimen()
        specimen.fingers = finger_phenome.genome
        specimen.brain = brain_phenome
        specimen.write_training_urdf() # New urdf for our phenome
        simulation = Simulation(conn_method='GUI')
        # simulation = Simulation()
        simulation.run_specimen(specimen)
        print(specimen.fitness)

if __name__ == '__main__':
    unittest.main()
