import os
import uuid
import numpy as np

from genome import FingersGenome, BrainGenome
from phenome import FingersPhenome, BrainPhenome
from generate_urdf import GenerateURDF
from constants import GeneDesc


class Specimen:
    '''
    Defines the specimen and makes it ready for simulation.

    Args:
        gene_description (Enum): GeneDesc enum class
        robot_hand: URDF file containing the main robot hand the
            fingers will attach to
    '''

    id = None
    fingers = None
    brain = None
    finger_positions = None
    specimen_URDF = None

    def __init__(self, gene_description, robot_hand):
        self.gene_desc = gene_description
        self.robot_hand = robot_hand

        # Assign a unique id
        self.id = uuid.uuid4()

        # Initialize specimen
        self.__init_fingers()
        self.__init_brain()

    def __init_fingers(self):
        # Initialize fingers genome
        fingers_genome = FingersGenome(self.gene_desc).get_genome()

        assert os.path.exists(self.robot_hand)
        self.fingers = FingersPhenome(
            fingers_genome, self.gene_desc, self.robot_hand
        ).get_genome()

        # Write fingers URDF definition file
        output_file = f'intraining_specimen/r_{self.id}.urdf'

        generate_urdf = GenerateURDF(self.fingers)
        urdf_written = generate_urdf.generate_robot_fingers(
            self.robot_hand, output_file
        )
        if urdf_written:
            self.specimen_URDF = output_file
        else:
            raise Exception('Can not initialize robot fingers')

    def __init_brain(self):
        # Initialize specimen brain
        brain_genome = BrainGenome().get_genome()
        self.brain = BrainPhenome(self.fingers, brain_genome)

    def train_brain(self):
        self.brain.train([[0.1, 0.06, 1.2], [0.2, 0.16, 1.7]], 1, 1, self.move_fingers)

    def move_fingers(self, target_angles):
        '''
        Moves the fingers to the specified target angles.

        Args:
            target_angles (list): target rotation angles determined by the brain

        Returns:
            the fingers' distance from the target object after the angles have
            been applied
        '''
        my_arr = np.random.rand(20) * 10
        return my_arr.tolist()

    def calc_fitness(self):
        pass
