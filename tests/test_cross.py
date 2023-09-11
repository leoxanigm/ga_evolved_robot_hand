import unittest
import hypothesis.strategies as st
from hypothesis import given, settings
import numpy as np

from constants import GeneDesc
from genome import FingersGenome, BrainGenome
from specimen import Specimen
from simulation import Simulation
from fitness_fun import FitnessFunction
from cross_mutate import Cross


class TestCross(unittest.TestCase):
    specimen_1 = None
    specimen_2 = None
    child = None

    @classmethod
    def setUpClass(cls):
        super(TestCross, cls).setUpClass()

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

        cls.child = Cross.cross_genomes(
            cls.specimen_1.fingers,
            cls.specimen_1.brain.genome,
            cls.fit_parent_1,
            cls.specimen_2.fingers,
            cls.specimen_2.brain.genome,
            cls.fit_parent_2,
        )

    def test_child_shapes(self):
        assert (
            self.child[0].shape
            == self.specimen_1.fingers.shape
            == self.specimen_2.fingers.shape
        )
        assert (
            self.child[1].shape
            == self.specimen_1.brain.genome.shape
            == self.specimen_2.brain.genome.shape
        )

    def test_child_different_from_parents(self):
        assert all(
            (
                np.any(self.child[0] != self.specimen_1.fingers),
                np.any(self.child[0] != self.specimen_2.fingers),
            )
        )
        assert all(
            (
                np.any(self.child[1] != self.specimen_1.brain.genome),
                np.any(self.child[1] != self.specimen_2.brain.genome),
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

    # def test_child_len_between_parents(self):
    #     '''Test number of phalanges in child are not fewer or greater
    #     than number of phalanges in the parents.'''

    #     num_pha_par_1 = np.sum(self.specimen_1.fingers != 0)
    #     num_pha_par_2 = np.sum(self.specimen_2.fingers != 0)
    #     num_pha_par_child = np.sum(self.child[0] != 0)

    #     assert all(
    #         (
    #             num_pha_par_child >= min(num_pha_par_1, num_pha_par_2),
    #             num_pha_par_child <= max(num_pha_par_1, num_pha_par_2),
    #         )
    #     )


if __name__ == '__main__':
    unittest.main()
