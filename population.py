from specimen import Specimen
from constants import GeneDesc


class Population:
    def __init__(self, pop_size):
        self._specimen = [Specimen(GeneDesc, 'robot_hand.urdf') for _ in range(pop_size)]

    @property
    def specimen(self):
        return self._specimen

    @specimen.setter
    def specimen(self, new_specimen):
        self._specimen = new_specimen
    

    