import os
import uuid
import numpy as np

from genome import FingersGenome, BrainGenome
from phenome import FingersPhenome, BrainPhenome
from generate_urdf import GenerateURDF
from constants import GeneDesc


class Specimen:
    '''
    Defines a specimen and makes it ready for simulation.

    Args:
        gene_description (Enum): GeneDesc enum class
        robot_hand: URDF file containing the main robot hand the
            fingers will attach to
    '''

    id = None
    fingers = None
    brain = None
    specimen_URDF = None
    fitness = 0
    distances = []  # distance of each phalanx in the fingers from the target object

    def __init__(self, gene_description, robot_hand):
        # Phalanx gene description Enum
        self.gene_desc = gene_description
        # URDF robot hand file. This contains the palm the fingers will be attached to
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
        brain_genome = BrainGenome()
        self.brain = BrainPhenome(brain_genome)

    def move_fingers(self, distances):
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
        assert len(self.distances) == self.num_of_phalanges

        # Calculate mean-square fitness from distance of the phalanges
        # from the target object
        self.fitness = np.mean(np.square(self.distances))

    @property
    def num_of_phalanges(self):
        finger_count = 0
        for finger in self.fingers:  # loop through fingers
            if np.all(finger == 0):
                # No need to continue looping are the rest of array elements will be None
                break

            for phalanx in finger:  # loop through phalanges
                if np.all(phalanx == 0):
                    break

                finger_count += 0
