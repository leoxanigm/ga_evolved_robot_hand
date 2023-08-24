import unittest
import hypothesis.strategies as st
from hypothesis import given, settings
import numpy as np

from constants import GeneDesc
from genome import FingersGenome, BrainGenome
from cross_mutate import CrossFingers, CrossBrain


class TestCrossFingers(unittest.TestCase):
    @given(fingers=st.integers(3, 10), phalanges=st.integers(3, 20))
    def test_random_genome_matrix(self, fingers, phalanges):
        '''Test the child of two genomes of a given shape has the same
        shape as its parents'''

        genome_1 = FingersGenome(GeneDesc, rows=fingers, columns=phalanges).genome
        genome_2 = FingersGenome(GeneDesc, rows=fingers, columns=phalanges).genome

        child = CrossFingers.cross_genomes(genome_1, genome_2)

        assert isinstance(child, np.ndarray)
        assert child.shape == (fingers, phalanges, len(GeneDesc))
        assert np.all((child >= 0) & (child < 1))

    @given(fingers=st.integers(3, 100), phalanges=st.integers(3, 100))
    # @settings(max_examples=10000)
    def test_unwanted_finger_structure(self, fingers, phalanges):
        '''
        Test the child contains no fingers after empty spots
        For example: [1, 1, 0, 1] this is unwanted finger structure
        caused by finger crossing which will later break our code.
        Hence the test makes sure no arrays like the example are
        created because of crossing between two parents with different
        amount of fingers.
        '''

        genome_1 = FingersGenome(GeneDesc, rows=fingers, columns=phalanges).genome
        genome_2 = FingersGenome(GeneDesc, rows=fingers, columns=phalanges).genome

        # Purposefully make genome_2 have only two fingers so that the crossing
        # will likely yield the unwanted finger structure
        genome_2[2:] = np.zeros((phalanges, len(GeneDesc)))

        child = CrossFingers.cross_genomes(genome_1, genome_2)

        found_empty = False
        for finger in child:
            if np.all(finger == 0):
                found_empty = True
            try:
                assert not (found_empty and np.any(finger != 0))
            except:
                print(child)

    @given(fingers=st.integers(3, 100), phalanges=st.integers(3, 100))
    # @settings(max_examples=10000)
    def test_unwanted_phalanx_structure(self, fingers, phalanges):
        '''
        Test each finger of the child contains no phalanges after
        empty spots For example: [[1], [1], [0], [1]] this is unwanted
        phalanx structure in an arbitrary finger caused by phalanx crossing
        which will later break our code. Hence the test makes sure no phalanx
        like the example are created because of crossing between two
        parents with different amount of phalanges.
        '''

        genome_1 = FingersGenome(GeneDesc, rows=fingers, columns=phalanges).genome
        genome_2 = FingersGenome(GeneDesc, rows=fingers, columns=phalanges).genome

        child = CrossFingers.cross_genomes(genome_1, genome_2)

        for finger in child:
            found_empty = False
            for phalanx in finger:
                if np.all(phalanx == 0):
                    found_empty = True
                assert not (found_empty and np.any(phalanx != 0))

class TestCrossBrain:
    genome_1 = None
    genome_2 = None
    child = None

    def _setup_test_child_genome_array(self, inputs, hidden, outputs):
        self.inputs = inputs
        self.hidden = hidden
        self.outputs = outputs
        self.genome_1 = BrainGenome(layers=[(inputs, hidden),(hidden, outputs)]).genome
        self.genome_2 = BrainGenome(layers=[(inputs, hidden),(hidden, outputs)]).genome

        self.child = CrossBrain.cross_genomes(self.genome_1, self.genome_2)


    @given(inputs=st.integers(5, 10), hidden=st.integers(11, 50), outputs=st.integers(1, 4))
    def test_random_genome_array(self, inputs, hidden, outputs):
        '''Test the child of two genomes of a given shape has the same
        shape as its parents'''

        self._setup_test_child_genome_array(inputs, hidden, outputs)

        assert isinstance(self.child, list)
        assert isinstance(self.child[0], np.ndarray)
        assert self.child[0].shape == (inputs, hidden)
        assert self.child[1].shape == (hidden, )
        assert self.child[2].shape == (hidden, outputs)
        assert self.child[3].shape == (outputs, )

    @given(inputs=st.integers(5, 10), hidden=st.integers(11, 50), outputs=st.integers(1, 4))
    def test_cross_genome_array(self, inputs, hidden, outputs):
        '''Test the child of two genomes of a given shape is different from 
        its parents'''

        self._setup_test_child_genome_array(inputs, hidden, outputs)

        different_from_parent_1 = False
        for i in range(len(self.child)):
            for j in range(len(self.child[i])):
                if np.any(self.child[i][j] != self.genome_1[i][j]):
                    different_from_parent_1 = True

        different_from_parent_2 = False
        for i in range(len(self.child)):
            for j in range(len(self.child[i])):
                if np.any(self.child[i][j] != self.genome_1[i][j]):
                    different_from_parent_2 = True

        assert different_from_parent_1 and different_from_parent_2

        


if __name__ == '__main__':
    unittest.main()