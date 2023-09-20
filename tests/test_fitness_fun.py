import unittest
import hypothesis.strategies as st
from hypothesis import given, settings
import numpy as np

from specimen import Specimen, Phalanx
from fitness_fun import FitnessFunction


def init_phalanges() -> list[Phalanx]:
    phalanges = []
    l_i = 0
    for i in range(7):
        for j in range(10):
            phalanx = Phalanx(i, j, l_i)
            phalanges.append(phalanx)
            l_i += 1
    return phalanges


def populate_performance(phalanges, dis, t_coll, o_coll):
    for _ in range(5):  # 5 iterations
        for phalanx in phalanges:
            phalanx.set_performance(dis, t_coll, o_coll)
    return phalanges


class TestSpecimen(unittest.TestCase):
    def test_no_distance(self):
        '''Ideal fit specimen.
        All phalanges contact with target, and move it'''
        phalanges = init_phalanges()
        phalanges = populate_performance(phalanges, 0, 1, 0)
        fitness = FitnessFunction.get_total_fitness(phalanges, 3)
        assert fitness == 1

    def test_all_distance(self):
        '''Somewhat performing specimen.
        All phalanges don't contact with target, and never move it.
        They also don't collide with obstacle'''
        phalanges = init_phalanges()
        phalanges = populate_performance(phalanges, 100, 0, 0)
        fitness = FitnessFunction.get_total_fitness(phalanges, 0)
        assert fitness < 0.1

    def test_all_distance_all_obs_collision(self):
        '''Least fittest specimen.
        All phalanges don't contact with target, and never move it.
        They also all collide with obstacle'''
        phalanges = init_phalanges()
        phalanges = populate_performance(phalanges, 100, 0, 1)
        fitness = FitnessFunction.get_total_fitness(phalanges, 0)
        assert fitness < 0.1

    def test_fitness_map_fit(self):
        '''Test fitness map of an ideal specimen'''
        phalanges = init_phalanges()
        phalanges = populate_performance(phalanges, 0, 1, 0)
        fitness_map = FitnessFunction.get_fitness_map(phalanges, (7, 10, 9))
        assert np.all(fitness_map == 1)

    def test_fitness_map_least_fit(self):
        '''Test fitness map of an ideal specimen'''
        phalanges = init_phalanges()
        phalanges = populate_performance(phalanges, 100, 0, 1)
        fitness_map = FitnessFunction.get_fitness_map(phalanges, (7, 10, 9))
        assert np.all(fitness_map < 0.2) # picking performance is not accounted

if __name__ == '__main__':
    unittest.main()
