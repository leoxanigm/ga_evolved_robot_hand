import os
import uuid
import numpy as np
from types import FunctionType
import time
import pybullet as p
from operator import add

from genome import FingersGenome, BrainGenome
from phenome import FingersPhenome, BrainPhenome
from generate_urdf import GenerateURDF
from constants import GeneDesc
from helpers.pybullet_helpers import apply_rotation


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
    # Fitness for each phalange
    # The first n elements are distances of fingers from the target object
    # The last element of distance of the palm from the target object
    fitness_array = None
    fitness_total = None  # The total fitness for the specimen

    prev_arm_angles = []  # Keeps track of previously applied angles for the arm
    prev_finger_angles = []  # Keeps track of previously applied angles for fingers

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

    def move_fingers(
        self,
        action: str,
        body_id: int,
        get_distances: FunctionType,
        p_id: int,
    ):
        '''
        Moves the fingers to the specified target angles.

        Args:
            target_angles (list): target rotation angles determined by the brain

        Returns:
            the fingers' distance from the target object after the angles have
            been applied
        '''
        # Center of mass distance of each phalanx from the palm
        center_of_mass = 0

        distances = get_distances()

        # Length of the target angles list should be the same as
        # the number of finger links
        target_angles = [(0, index) for d, index in distances]

        link_index = 0  # Keep track of link indices
        for finger in self.fingers:
            if np.all(finger == 0):
                # No need to continue looping as the rest of array elements will be zero
                break
            for i, phalanx in enumerate(finger):
                if np.all(phalanx == 0):
                    break

                # Moving a parent phalanx will also move the child.
                # So we have to get distances for each iteration
                distances = get_distances()

                if action == 'ready_to_pick' or action == 'drop':
                    # Spread out the fingers
                    if i == 0:
                        target_angle = np.pi / 4
                    else:
                        target_angle = 0
                else:
                    # Model calculates the target angles

                    # Get distance for this phalanx
                    # distances[i] is a tuple, (distance, index) and we only want
                    # the first element
                    distance = distances[link_index][0]

                    # Axes of rotation. Either 0 or 1
                    x_axis = phalanx[GeneDesc.JOINT_AXIS_X]
                    y_axis = phalanx[GeneDesc.JOINT_AXIS_Y]
                    z_axis = phalanx[GeneDesc.JOINT_AXIS_Z]

                    # Center of mass is a the mid-point
                    center_of_mass += phalanx[GeneDesc.DIM_Z] / 2

                    target_angle = self.brain.move(
                        [distance, x_axis, y_axis, z_axis, center_of_mass]
                    )

                    target_angle = target_angle.tolist()[0]

                    # Move center of mass to the bottom (from the center of the phalanx)
                    # as a reference point for the next phalanx
                    center_of_mass += phalanx[GeneDesc.DIM_Z] / 2

                apply_rotation(body_id, distances[link_index][1], target_angle, p_id)

                link_index += 1

    def move_arm(
        self,
        action: str,
        body_id: int,
        p_id: int,
    ):
        '''Moves the robot hand's elbow and base depending on the action specified
        Args:
            action (str):
                ready_to_pick - goes to object pick up position
                pick - picks up the object
                move_to_drop - rotates to object drop position
                drop - goes to drop position
                reset - rotates back to original position
        Returns:
            rotation angle (tuple[float]): [base_rotation, elbow_rotation]
        '''
        action_dict = {
            'ready_to_pick': [0, np.pi / 4],
            'pick': [0, np.pi / 2],
            'move_to_drop': [np.pi, np.pi / 4],
            'drop': [np.pi, np.pi / 2],
            'reset': [0, np.pi / 4],
        }

        assert action in action_dict

        if len(self.prev_arm_angles) == 0:
            self.prev_arm_angles = [0, 0]

        # Rotate the arm by the desired angle
        apply_rotation(body_id, [0, 1], action_dict[action], p_id, self.prev_arm_angles)

        # Keep track of applied angles
        # This is later used for smooth movement animation
        self.prev_arm_angles = action_dict[action]

    def calc_fitness(self, get_distances: FunctionType):
        '''Calculates fitness of the specimen for each target object'''
        if self.fitness_array is None:
            self.fitness_array = [dis for dis, _ in get_distances('fingers')]
            self.fitness_array.append(get_distances('palm')[0][0])
        else:
            # If fitness_array is not empty, that means we have a record of
            # fitnesses for the first target object. In this case we add the
            # distances for the next target object to the original distances.
            new_fitness_array = [dis for dis, _ in get_distances('fingers')]
            new_fitness_array.append(get_distances('palm')[0][0])

            # Add the new fitness to the previous records
            # Source: https://www.javatpoint.com/how-to-add-two-lists-in-python
            self.fitness_array = list(map(add, self.fitness_array, new_fitness_array))

        # If the distance of the target object from the palm is small,
        # the arm has picked up the object, making the specimen more fit.
        # Hence we square the distance of the palm from the object to give
        # it more importance.
        self.fitness_array[-1] = self.fitness_array[-1] ** 2

        # The final goal is to get the distances as close as possible
        # to zero. That is, we don't want neither positive or negative
        # distances. So, we use mean-square of the fitnesses to calculate
        # the total fitness value
        self.fitness_total = np.mean(np.square(self.fitness_array))

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