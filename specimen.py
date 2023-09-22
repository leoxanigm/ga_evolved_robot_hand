import os
import uuid
import numpy as np
from types import FunctionType
import time
import pybullet as p
from operator import add
from collections import namedtuple

from genome import FingersGenome, BrainGenome, save_genome
from phenome import FingersPhenome, BrainPhenome
from generate_urdf import GenerateURDF

import constants as c
from helpers.pybullet_helpers import apply_rotation


class Phalanx:
    '''
    A state that tracks attributes and performance for each phalanx in a specimen.
    The attributes are:
        finger_index: the index of finger the phalanx is located in a genome encoding
        phalanx_index: the index in a finger
        link_index: the link index in a simulation
        inputs: distance from target, collision with target and collision with obstacle
        output: the aggregated rotation angle to be applied
        total_distance: accumulated distance from targets in a simulation
        target_collision: accumulated target collision in a simulation
        obstacle_collision: accumulated obstacle collision in a simulation
    '''

    def __init__(self, finger_index, phalanx_index, link_index, inputs=[], output=0):
        self.finger_index = finger_index
        self.phalanx_index = phalanx_index
        self.link_index = link_index
        self.inputs = inputs
        self.output = output

        # The distance of the phalanx from the target object for each iteration,
        # later used to calculate fitness
        self._total_distance = 0
        self._target_collision = 0
        self._obstacle_collision = 0

        # Keep track of iterations. Later used to normalize the recorded values
        self.iterations = 0

        # Fitness of the finger the current phalanx is located
        self.finger_fitness = 0
        # Fitness of the current phalanx
        self.phalanx_fitness = 0

    def set_performance(self, distance, t_collision, o_collision):
        '''
        Increments the distance of the phalanx from the target object,
        collision with target and collision for obstacle for each iteration
        '''
        self._total_distance += distance
        self._target_collision += t_collision
        self._obstacle_collision += o_collision

        self.iterations += 1

    def get_performance(self):
        '''
        Returns normalized values for
        (total_distance, total_target_collisions, total_obstacle_collisions)
        less distance, more performing
        less obstacle collision, more performing
        '''
        return (
            self._total_distance / self.iterations,
            self._target_collision / self.iterations,
            self._obstacle_collision / self.iterations,
        )


class Specimen:
    '''
    Defines a specimen and makes it ready for simulation.
    Args:
        fingers_genome: already saved fingers genome [optional]
        brain_genome: already saved brain genome [optional]
    '''

    id = None
    fingers = None

    _fitness = 0  # Total fitness of the specimen

    def __init__(
        self,
        fingers_genome: np.ndarray = None,
        brain_genome: list = None,
    ):
        # Assign a unique id
        # Use first 8 characters to avoid long file names
        self.id = str(uuid.uuid4())[:8]
        self._fingers_genome = None
        self._brain_genome = None

        self._phalanges: list[Phalanx] = []

        # Increment angle for each iteration of fingers' movement
        self.angle_increment = np.pi / 32

        # Keeps track of previously applied angles for the arm, helps in smooth animation
        self.prev_arm_angles = []

        if fingers_genome is not None and brain_genome is not None:
            # Initialize specimen from provided genomes
            self.__init_state(fingers_genome=fingers_genome, brain_genome=brain_genome)
        else:  # New specimen
            # Initialize new specimen
            self.__init_state()

    def __init_state(
        self,
        fingers_genome=None,
        brain_genome=None,
        generation_id: str = None,
        specimen_id: str = None,
    ):
        if fingers_genome is not None and brain_genome is not None:
            assert isinstance(fingers_genome, np.ndarray)
            assert isinstance(brain_genome, np.ndarray)

            self._fingers_genome = fingers_genome
            self._brain_genome = brain_genome
        else:
            # Initialize random fingers genome
            self._fingers_genome = FingersGenome.random_genome()
            # Initialize random brain genome
            self._brain_genome = BrainGenome.random_genome()

        self.fingers = FingersPhenome.genome_to_phenome(self._fingers_genome)

    def load_state(self, genome_link_indices: list[tuple]):
        '''Loads genome indices (fingers and phalanges) and link indices to
        the phalanges state'''

        assert isinstance(genome_link_indices, list)
        assert len(genome_link_indices) > 0

        # Populate state for phalanges
        for (finger_index, phalanx_index), link_index in genome_link_indices:
            self._phalanges.append(Phalanx(finger_index, phalanx_index, link_index))

    def save_specimen(self):
        # Save fingers genome
        save_genome(
            self._fingers_genome, os.path.join(c.FIT_DIR, f'{self.id}_fingers.pk')
        )
        # Save brain genome
        save_genome(self._brain_genome, os.path.join(c.FIT_DIR, f'{self.id}_brain.pk'))

        return self.id

    def move_fingers(
        self,
        action: str,
        body_id: int,
        check_distances: FunctionType,
        check_collisions: FunctionType,
        p_id: int,
    ):
        '''
        Moves the fingers based on inputs retrieved from callback functions.
        The inputs are: distance from the target object, contact with target
        object and contact with table.

        Args:
            action (str): the current state of the arm
            body_id (int): the specimen body id in the PyBullet simulation
            check_distances (function): callback function to get distance of
                each phalanx from the target object
            check_collisions (function): callback function to get collision
                of a phalanx - target object and phalanx - table. It returns
                a tuple with boolean values 0/1 for (target, table)
            p_id (int): PyBullet connected server's simulation id
        '''

        p.stepSimulation(physicsClientId=p_id)

        if action == 'ready_to_pick' or action == 'drop':
            for phalanx in self._phalanges:
                if phalanx.phalanx_index == 0:
                    # Spread out the fingers
                    phalanx.output = -np.pi / 4
                else:
                    phalanx.output = 0
        elif action == 'before_pick':
            # Move the phalanges in a straight position
            for phalanx in self._phalanges:
                phalanx.output = 0
        elif action == 'pick':
            time.sleep(0.1)
            # Get link indices for distance and collistions, for input
            distances = check_distances()
            target_collisions = check_collisions('target')
            obstacle_collisions = check_collisions('obstacle')

            for phalanx in self._phalanges:
                link_index = phalanx.link_index
                distance = target_collision = obstacle_collision = 0
                if link_index in distances:
                    distance = 1
                if link_index in target_collisions:
                    target_collision = 1
                if link_index in obstacle_collisions:
                    obstacle_collision = 1

                # Set inputs for brain
                phalanx.inputs = [distance, target_collision, obstacle_collision]

            # Get shapes to construct input np array
            genome_shape = self.fingers_genome.shape[:2]
            input_shape = len(self._phalanges[0].inputs)

            # Initialize the input array with zeros
            inputs = np.zeros((*genome_shape, input_shape))

            # Populate the inputs
            for phalanx in self._phalanges:
                f_i = phalanx.finger_index
                p_i = phalanx.phalanx_index

                inputs[f_i][p_i] = np.array(phalanx.inputs)

            # Get trajectories
            outputs = BrainPhenome.trajectories(self._brain_genome, inputs)

            # Get link indices for distance and collistions, to get performance
            distances = check_distances()
            target_collisions = check_collisions('target')
            obstacle_collisions = check_collisions('obstacle')

            # Populate output
            for phalanx in self._phalanges:
                f_i = phalanx.finger_index
                p_i = phalanx.phalanx_index

                phalanx.output += outputs[f_i][p_i] * self.angle_increment

                link_index = phalanx.link_index
                distance = target_collision = obstacle_collision = 0
                if link_index in distances:
                    distance = 1
                if link_index in target_collisions:
                    target_collision = 1
                if link_index in obstacle_collisions:
                    obstacle_collision = 1

                # Record phalanx performance
                phalanx.set_performance(distance, target_collision, obstacle_collision)

        link_indices = [p.link_index for p in self._phalanges]
        target_angles = [p.output for p in self._phalanges]

        apply_rotation(body_id, link_indices, target_angles, p_id)

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
                return_to_pick - returns to object pick up position
        Returns:
            rotation angle (tuple[float]): [base_rotation, elbow_rotation]
        '''
        action_dict = {
            'ready_to_pick': [0, 0],
            'pick': [0, np.pi / 4],
            'move_to_drop': [np.pi, 0],
            'drop': [np.pi, np.pi / 4],
            'return_to_pick': [0, 0],
        }

        assert action in action_dict

        if len(self.prev_arm_angles) == 0:
            self.prev_arm_angles = [0, 0]

        if action == 'return_to_pick':
            # Don't animate the return motion in direct connection to save time
            conn_info = p.getConnectionInfo(physicsClientId=p_id)
            if conn_info['connectionMethod'] == p.DIRECT:  #
                apply_rotation(body_id, [0, 1], action_dict[action], p_id)
        else:
            apply_rotation(
                body_id, [0, 1], action_dict[action], p_id, self.prev_arm_angles
            )

        # Keep track of applied angles
        # This is later used for smooth movement animation
        self.prev_arm_angles = action_dict[action]

    @property
    def fitness(self):
        return self._fitness

    @fitness.setter
    def fitness(self, fitness: float):
        self._fitness = fitness

    @property
    def phalanges(self):
        return self._phalanges

    @property
    def fingers_genome(self):
        return self._fingers_genome

    @property
    def brain_genome(self):
        return self._brain_genome

    def __repr__(self):
        return f'id - {self.id}, fitness - {self.fitness}'
