from specimen import Specimen
from constants import GeneDesc


class Population:
    def __init__(self, pop_size):
        self.specimen = [Specimen(GeneDesc, 'robot_hand.urdf') for _ in range(pop_size)]