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
        '''Returns normalized values for
        (total_distance, total_target_collisions, total_obstacle_collisions)'''
        return (
            self._total_distance / self.iterations,
            self._target_collision / self.iterations,
            self._obstacle_collision / self.iterations,
        )


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

    _fitness = 0  # Total fitness of the specimen

    prev_arm_angles = []  # Keeps track of previously applied angles for the arm

    angle_increment = np.pi / 16  # How much angle each phalanx moves at a time

    def __init__(
        self,
        fingers_genome: np.ndarray = None,
        brain_genome: list = None,
        generation_id: str = None,
        specimen_id: str = None,
        gene_description=GeneDesc,
        robot_hand: str = ROBOT_HAND,
    ):
        # Phalanx gene description Enum
        self.gene_desc = gene_description
        # URDF robot hand file. This contains the palm the fingers will be attached to
        self.robot_hand = robot_hand

        self._fingers_genome = None
        self._brain_genome = None

        self._phalanges: list[Phalanx] = []

        # Increment angle for each iteration of fingers' movement
        self.angle_increment = np.pi / 16

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
            self._fingers_genome = self.fingers_phenome.genome

            self.specimen_URDF = f'{generation_id}_{specimen_id}.urdf'

            # Load brain from saved pickle dump
            self.brain = BrainPhenome()
            self.brain.load_genome(
                f'fit_specimen/genome_encodings/{generation_id}_brain_{specimen_id}.pickle'
            )
            self._brain_genome = self.brain.genome

            return

        if fingers_genome is not None and brain_genome is not None:
            self.fingers_phenome = FingersPhenome(fingers_genome)
            self.fingers = self.fingers_phenome.phenome
        else:
            # Initialize random fingers genome
            fingers_genome = FingersGenome.genome(self.gene_desc)

            self.fingers_phenome = FingersPhenome(fingers_genome)
            self.fingers = self.fingers_phenome.phenome

            # Initialize random brain genome
            brain_genome = BrainGenome.genome(fingers_genome)

        self._fingers_genome = self.fingers_phenome.genome

        self.brain = BrainPhenome(brain_genome)
        self._brain_genome = self.brain.genome

        self.write_training_urdf()

    def write_training_urdf(self):
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

    def load_state(self, genome_link_indices: list[tuple]):
        '''Loads genome indices (fingers and phalanges) and link indices to
        the phalanges state'''

        assert isinstance(genome_link_indices, list)
        assert len(genome_link_indices) > 0

        # Populate state for phalanges
        for (finger_index, phalanx_index), link_index in genome_link_indices:
            self._phalanges.append(Phalanx(finger_index, phalanx_index, link_index))

    def move_fingers(
        self,
        action: str,
        body_id: int,
        get_distances: FunctionType,
        get_collisions: FunctionType,
        p_id: int,
        target_object: int = 0,
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
            for phalanx in self._phalanges:
                if phalanx.phalanx_index == 0:
                    # Spread out the fingers
                    phalanx.output = np.pi / 4
                    print('#######################################')
                    print(self.fingers[phalanx.finger_index, phalanx.phalanx_index])
                    print('#######################################')
                else:
                    phalanx.output = 0
        elif action == 'pick' and iteration == 0:
            # For the first iteration of the grabbing steps, move the
            # phalanges in a straight position
            for phalanx in self._phalanges:
                phalanx.output = 0
        else:
            for phalanx in self._phalanges:
                link_index = phalanx.link_index
                distance = get_distances(link_index)
                collisions = get_collisions(link_index)

                # Get collisions
                target_collision = collisions[0]
                obstacle_collision = collisions[1]

                # Record phalanx performance
                phalanx.set_performance(distance, target_collision, obstacle_collision)

                # Set inputs for brain
                # We just care if there is a distance not the quantity
                distance = 0 if distance == 0 else 1
                phalanx.inputs = [distance, target_collision, obstacle_collision]

            # time.sleep(30)

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
            outputs = self.brain.trajectories(
                inputs, target_object=target_object, iteration=iteration
            )

            # Populate output
            for phalanx in self._phalanges:
                f_i = phalanx.finger_index
                p_i = phalanx.phalanx_index

                phalanx.output += outputs[f_i][p_i] * self.angle_increment

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
            'ready_to_pick': [0, np.pi / 4],
            'pick': [0, np.pi / 2],
            'move_to_drop': [np.pi, np.pi / 4],
            'drop': [np.pi, np.pi / 2],
            'return_to_pick': [0, np.pi / 4],
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

    # def calc_fitness(self, moved_object_ids: list[int], target_box_id: int, p_id: int):
    #     '''Calculates fitness of the specimen
    #     Args:
    #         moved_object_ids: list of successfully moved object ids
    #         target_box_id: target dop box id
    #         p_id: connected physics client id
    #     '''

    #     grabbing_performance = .get_grabbing_performance(self._phalanges)
    #     picking_performance = 0
    #     for id in moved_object_ids:
    #         picking_performance += .get_picking_performance(
    #             id, target_box_id, p_id
    #         )

    #     self._fitness = .get_total_fitness(
    #         grabbing_performance, picking_performance
    #     )

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
