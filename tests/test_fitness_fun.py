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
    for _ in range(5): # 5 iterations
        for phalanx in phalanges:
            # No distance, all collision with target, no collision with
            phalanx.set_performance(dis, t_coll, o_coll)
    return phalanges

class TestSpecimen(unittest.TestCase):
    def test_no_distance(self):
        '''total distance should be 0, all phalanges contact with target'''
        phalanges = init_phalanges()
        phalanges = populate_performance(phalanges, 0, 1, 0)
        distance, t_collision, o_collision = FitnessFunction.get_norm_performance(phalanges)
        assert distance == 0
        assert t_collision == 1
        assert o_collision == 0

    def test_all_distance(self):
        '''total distance should be 1, no phalanges contact with target'''
        phalanges = init_phalanges()
        phalanges = populate_performance(phalanges, 1, 0, 0)
        distance, t_collision, o_collision = FitnessFunction.get_norm_performance(phalanges)
        assert distance == 1
        assert t_collision == 0
        assert o_collision == 0

    def test_all_collision(self):
        '''total distance should be 0, no phalanges contact with target
        all phalanges contact with obstacle'''
        phalanges = init_phalanges()
        phalanges = populate_performance(phalanges, 0, 0, 1)
        distance, t_collision, o_collision = FitnessFunction.get_norm_performance(phalanges)
        assert distance == 0
        assert t_collision == 0
        assert o_collision == 1
        
        


if __name__ == '__main__':
    unittest.main()