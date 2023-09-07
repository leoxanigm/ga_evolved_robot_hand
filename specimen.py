import os
import uuid
import numpy as np
from types import FunctionType
import time
import pybullet as p
from operator import add
from collections import namedtuple

from genome import FingersGenome, BrainGenome
from phenome import FingersPhenome, BrainPhenome
from generate_urdf import GenerateURDF
from constants import GeneDesc, ROBOT_HAND, TRAINING_DIR
from helpers.pybullet_helpers import apply_rotation


class Phalanx:
    def __init__(self, finger_index, phalanx_index, link_index, inputs=[], output=0):
        self.finger_index = finger_index
        self.phalanx_index = phalanx_index
        self.link_index = link_index
        self.inputs = inputs
        self.output = output


class Specimen:
    '''
    Defines a specimen and makes it ready for simulation.

    Args:
        gene_description (Enum): GeneDesc enum class
        robot_hand (str): URDF file containing the main robot hand the
            fingers will attach to
        generation_id (str): Generation ID for previously run evaluated specimen
        specimen_id (str): Specimen ID for previously run evaluated specimen
    '''

    id = None
    fingers = None
    brain = None
    specimen_URDF = None
    # Distance for each phalange
    # The first n elements are distances of fingers from the target object
    # The last element of distance of the palm from the target object
    distance_array = None
    distance_total = None  # The accumulated distance for the specimen

    _fitness = 0

    prev_arm_angles = []  # Keeps track of previously applied angles for the arm
    prev_finger_angles = []  # Keeps track of previously applied angles for fingers

    angle_increment = np.pi / 16  # How much angle each phalanx moves at a time

    def __init__(
        self,
        gene_description=GeneDesc,
        robot_hand: str = ROBOT_HAND,
        fingers_genome: np.ndarray = None,
        brain_genome: list = None,
        generation_id: str = None,
        specimen_id: str = None,
    ):
        # Phalanx gene description Enum
        self.gene_desc = gene_description
        # URDF robot hand file. This contains the palm the fingers will be attached to
        self.robot_hand = robot_hand

        self._fingers_genome = None
        self._brain_genome = None

        self.phalanges: list[Phalanx] = []

        # Assign a unique id
        # Use first 8 characters to avoid long file names
        self.id = str(uuid.uuid4())[:8]

        if generation_id and specimen_id:  # Specimen from saved data
            # Initialize saved specimen
            self.__init_state(generation_id=generation_id, specimen_id=specimen_id)
        elif fingers_genome is not None and brain_genome is not None:
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
        if generation_id and specimen_id:
            # Load fingers from saved pickle dump
            self.fingers_phenome = FingersPhenome()
            self.fingers = self.fingers_phenome.load_genome(
                f'fit_specimen/genome_encodings/{generation_id}_fingers_{specimen_id}.pickle'
            )

            self.specimen_URDF = f'{generation_id}_{specimen_id}.urdf'

            # Load brain from saved pickle dump
            self.brain = BrainPhenome()
            self.brain.load_genome(
                f'fit_specimen/genome_encodings/{generation_id}_brain_{specimen_id}.pickle'
            )

            return

        if fingers_genome is not None and brain_genome is not None:
            self.fingers_phenome = FingersPhenome(fingers_genome)
            self.fingers = self.fingers_phenome.phenome

            self.brain = BrainPhenome(brain_genome)

        else:
            # Initialize random fingers genome
            fingers_genome = FingersGenome.genome(self.gene_desc)

            self.fingers_phenome = FingersPhenome(fingers_genome)
            self.fingers = self.fingers_phenome.phenome

            # Initialize random brain genome
            brain_genome = BrainGenome.genome(fingers_genome)

        # Write fingers URDF definition file
        training_folder_path = TRAINING_DIR
        output_file = f'{self.id}.urdf'

        assert os.path.exists(self.robot_hand)

        generate_urdf = GenerateURDF(self.fingers)
        urdf_written = generate_urdf.generate_robot_fingers(
            self.robot_hand, training_folder_path + output_file
        )
        if urdf_written:
            self.specimen_URDF = output_file
        else:
            raise Exception('Can not initialize robot fingers')

        self._fingers_genome = self.fingers_phenome.genome

        self.brain = BrainPhenome(brain_genome)
        self._brain_genome = self.brain.genome

    def load_state(self, genome_link_indices: list[tuple]):
        '''Loads genome indices (fingers and phalanges) and link indices to
        the phalanges state'''

        assert isinstance(genome_link_indices, list)
        assert len(genome_link_indices) > 0

        # Populate state for phalanges
        for (finger_index, phalanx_index), link_index in genome_link_indices:
            self.phalanges.append(Phalanx(finger_index, phalanx_index, link_index))

    def move_fingers(
        self,
        action: str,
        body_id: int,
        get_distances: FunctionType,
        get_collisions: FunctionType,
        p_id: int,
        iteration: int = 0,
    ):
        '''
        Moves the fingers based on inputs retrieved from callback functions.
        The inputs are: distance from the target object, contact with target
        object and contact with table.

        Args:
            action (str): the current state of the arm
            body_id (int): the specimen body id in the PyBullet simulation
            get_distances (function): callback function to get distance of
                each phalanx from the target object
            get_collisions (function): callback function to get collision
                of a phalanx - target object and phalanx - table. It returns
                a tuple with boolean values 0/1 for (target, table)
            iteration (int): the current iteration in the movement steps
            p_id (int): PyBullet connected server's simulation id
        '''

        p.stepSimulation(physicsClientId=p_id)
        if action == 'ready_to_pick' or action == 'drop':
            for phalanx in self.phalanges:
                if phalanx.phalanx_index == 0:
                    # Spread out the fingers
                    phalanx.output = np.pi / 4
                else:
                    phalanx.output = 0
        elif action == 'pick' and iteration == 0:
            # For the first iteration of the grabbing steps, move the
            # phalanges in a straight position
            for phalanx in self.phalanges:
                phalanx.output = 0
        else:
            for phalanx in self.phalanges:
                link_index = phalanx.link_index
                distances = get_distances(link_index)
                collisions = get_collisions(link_index)
                phalanx.inputs = [distances, *collisions]

            time.sleep(30)

            # Get shapes to construct input np array
            genome_shape = self.fingers_genome.shape[:2]
            input_shape = len(self.phalanges[0].inputs)

            # Initialize the input array with zeros
            inputs = np.zeros((*genome_shape, input_shape))

            # Populate the inputs
            for phalanx in self.phalanges:
                f_i = phalanx.finger_index
                p_i = phalanx.phalanx_index

                inputs[f_i][p_i] = np.array(phalanx.inputs)

            # print('===============')
            # print(f'inputs {inputs}')

            # Get trajectories
            outputs = self.brain.trajectories(inputs)

            # print(f'outputs {outputs}')
            # print('===============')

            # Populate output
            for phalanx in self.phalanges:
                f_i = phalanx.finger_index
                p_i = phalanx.phalanx_index

                phalanx.output += (outputs[f_i][p_i] * self.angle_increment)

        link_indices = [p.link_index for p in self.phalanges]
        target_angles = [p.output for p in self.phalanges]


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
                reset - rotates back to original position
        Returns:
            rotation angle (tuple[float]): [base_rotation, elbow_rotation]
        '''
        action_dict = {
            'ready_to_pick': [0, np.pi / 4],
            'pick': [0, np.pi / 2],
            'move_to_drop': [np.pi, np.pi / 4],
            'drop': [np.pi, np.pi / 2],
        }

        assert action in action_dict

        if len(self.prev_arm_angles) == 0:
            self.prev_arm_angles = [0, 0]

        apply_rotation(body_id, [0, 1], action_dict[action], p_id, self.prev_arm_angles)

        # Keep track of applied angles
        # This is later used for smooth movement animation
        self.prev_arm_angles = action_dict[action]

    def calc_fitness(self, get_distances: FunctionType, link_type: str = 'all'):
        '''Calculates fitness of the specimen for each target object
        Args:
            get_distances (function): callback to get distances from simulation
            link_type (str): wether to get distances of just fingers or for
                both fingers and the palm. This is to calculate the fitness
                of the specimen when fingers are picking up the object in
                addition to check if the specimen actually moved the object
        '''
        if self.distance_array is None:
            if link_type == 'fingers':
                self.distance_array = [dis for dis, _ in get_distances('fingers')]
                self.distance_array.append(0)
            else:
                self.distance_array = [0] * len(get_distances('fingers'))
                self.distance_array.append(get_distances('palm')[0][0])

        else:
            # If distance_array is not empty, that means we have a record of
            # fitnesses for the first target object. In this case we add the
            # distances for the next target object to the original distances.

            if link_type == 'fingers':
                new_distance_array = [dis for dis, _ in get_distances('fingers')]
                new_distance_array.append(0)
            else:
                new_distance_array = [0] * (len(self.distance_array) - 1)
                new_distance_array.append(get_distances('palm')[0][0])

            # Add the new fitness to the previous records
            # Source: https://www.javatpoint.com/how-to-add-two-lists-in-python
            self.distance_array = list(
                map(add, self.distance_array, new_distance_array)
            )

        # If the distance of the target object from the palm is small,
        # the arm has picked up the object, making the specimen more fit.
        # Hence we square the distance of the palm from the object to give
        # it more importance.
        self.distance_array[-1] = self.distance_array[-1] ** 2

        # The final goal is to get the distances as close as possible
        # to zero. That is, we don't want neither positive or negative
        # distances. So, we use mean-square of the fitnesses to calculate
        # the total fitness value
        self.distance_total = np.mean(np.square(self.distance_array))

        # Use the minimum of the accumulated distance or 50 as exorbitantly
        # large distance would not add that much difference to the fitness
        self.distance_total = min(self.distance_total, 50)

        self._fitness = 50 - self.distance_total

    def save_specimen(self, generation_id: str):
        '''
        Save specimen to disk. Copies the specimen URDF file to
        fit_specimen/urdf_files folder, saves fingers and brain genome
        to fit_specimen/genome_encodings folder.

        Args:
            generation_id (str): unique id of the generation the specimen was run
        '''

        training_urdf = f'intraining_specimen/{self.specimen_URDF}'
        target_urdf = f'fit_specimen/urdf_files/{generation_id}_{self.specimen_URDF}'

        assert os.path.exists(training_urdf)

        # Move the URDF file
        # Source: https://www.learndatasci.com/solutions/python-move-file/
        os.replace(training_urdf, target_urdf)

        # Save fingers genome matrix
        self.fingers_phenome.save_genome(
            f'fit_specimen/genome_encodings/{generation_id}_fingers_{self.id}.pickle'
        )

        # Save brain genome
        self.brain.save_genome(
            f'fit_specimen/genome_encodings/{generation_id}_brain_{self.id}.pickle'
        )

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

    @property
    def fitness(self):
        return self._fitness

    @property
    def fingers_genome(self):
        return self._fingers_genome

    @fingers_genome.setter
    def fingers_genome(self, fingers_genome):
        assert isinstance(fingers_genome, np.ndarray)

        self._fingers_genome = fingers_genome
        self.__init_fingers()

    @property
    def brain_genome(self):
        return self._brain_genome

    @brain_genome.setter
    def brain_genome(self, brain_genome):
        assert isinstance(brain_genome, list)
        assert isinstance(brain_genome[0], np.ndarray)

        self._brain_genome = brain_genome
        self.__init_brain()
