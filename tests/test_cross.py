import unittest
import hypothesis.strategies as st
from hypothesis import given, settings
import numpy as np
from copy import deepcopy

from constants import GeneDesc
from genome import FingersGenome, BrainGenome
from specimen import Specimen
from simulation import Simulation
from fitness_fun import FitnessFunction
from cross_mutate import CrossMutate, Mutate


class TestCrossMutate(unittest.TestCase):
    specimen_1 = None
    specimen_2 = None
    child = None

    @classmethod
    def setUpClass(cls):
        super(TestCrossMutate, cls).setUpClass()

        cls.specimen_1 = Specimen()
        cls.specimen_2 = Specimen()

        with Simulation() as simulation:
            simulation.run_specimen(cls.specimen_1)
            cls.fit_parent_1 = FitnessFunction.get_fitness_map(
                cls.specimen_1.phalanges, cls.specimen_1.fingers.shape
            )
        with Simulation() as simulation:
            simulation.run_specimen(cls.specimen_2)
            cls.fit_parent_2 = FitnessFunction.get_fitness_map(
                cls.specimen_2.phalanges, cls.specimen_2.fingers.shape
            )

        cls.child = CrossMutate.cross_mutate_genomes(
            cls.specimen_1.fingers_genome,
            cls.specimen_1.brain_genome,
            cls.fit_parent_1,
            cls.specimen_2.fingers_genome,
            cls.specimen_2.brain_genome,
            cls.fit_parent_2,
        )

    def test_child_shapes(self):
        assert (
            self.child[0].shape
            == self.specimen_1.fingers_genome.shape
            == self.specimen_2.fingers_genome.shape
        )
        assert (
            self.child[1].shape
            == self.specimen_1.brain_genome.shape
            == self.specimen_2.brain_genome.shape
        )

    def test_child_different_from_parents(self):
        assert all(
            (
                np.any(self.child[0] != self.specimen_1.fingers_genome),
                np.any(self.child[0] != self.specimen_2.fingers_genome),
            )
        )
        assert all(
            (
                np.any(self.child[1] != self.specimen_1.brain_genome),
                np.any(self.child[1] != self.specimen_2.brain_genome),
            )
        )

    def test_no_unwanted_finger_structure(self):
        '''
        Test the child contains no fingers or phalanges after empty spots
        For example: [1, 1, 0, 1] this is unwanted finger structure
        which could be caused by finger crossing which will later break
        our code. Hence the test makes sure no arrays like the example are
        created because of crossing between two parents with different
        amount of fingers.
        '''

        non_zero_index = np.nonzero(self.child[0] != 0)
        curr_i = -1
        index_sum = curr_i
        for f, p, _ in zip(*non_zero_index):
            if f != curr_i:
                assert f == (curr_i + 1)
                curr_i = f
                index_sum = curr_i
            else:
                assert (f + p) <= (index_sum + 1)
                if (f + p) == (index_sum + 1):
                    index_sum += f + p

    def test_child_cross(self):
        '''Test child took genes based on fitness mask'''

        # No mutation
        self.child = CrossMutate.cross_mutate_genomes(
            self.specimen_1.fingers_genome,
            self.specimen_1.brain_genome,
            self.fit_parent_1,
            self.specimen_2.fingers_genome,
            self.specimen_2.brain_genome,
            self.fit_parent_2,
            0,
        )

        mask = self.fit_parent_1 > self.fit_parent_2
        inv_mask = np.invert(mask)

        assert np.all(self.child[0][mask] == self.specimen_1.fingers_genome[mask])
        assert np.all(self.child[1][mask] == self.specimen_1.brain_genome[mask])
        assert np.all(
            self.child[0][inv_mask] == self.specimen_2.fingers_genome[inv_mask]
        )
        assert np.all(self.child[1][inv_mask] == self.specimen_2.brain_genome[inv_mask])


if __name__ == '__main__':
    unittest.main()
