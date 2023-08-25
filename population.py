from specimen import Specimen
from constants import GeneDesc
from cross_mutate import CrossFingers, CrossBrain, MutateFingers, MutateBrain


class Population:
    def __init__(self, pop_size):
        self._specimen = [Specimen(GeneDesc, 'robot_hand.urdf') for _ in range(pop_size)]
        self.fits: list[Specimen] = None # Top 30% fittest specimen

    def get_fits(self):
        pass

    def cross(self):
        assert self.fits is not None
        assert len(self.fits) % 2 == 0

        children = []
        for i in range(0, len(self.fits), 2):
            parent_1 = self.fits[i].fingers


    def mutate(self):
        pass

    def repopulate(self):
        pass

    @property
    def specimen(self):
        return self._specimen

    @specimen.setter
    def specimen(self, new_specimen):
        self._specimen = new_specimen
    

    