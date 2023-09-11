import os
import pybullet as p
import numpy as np
from multiprocessing import Pool
import time

from typing import Literal

from helpers.pybullet_helpers import (
    get_distance_of_bodies,
    apply_rotation,
    check_collisions,
    get_genome_link_indices,
    check_in_target_box,
)

from population import Population
from specimen import Specimen, Phalanx
from fitness_fun import FitnessFunction

object_URDF = ['cube.urdf', 'sphere.urdf', 'cylinder.urdf']


class Simulation:
    def __init__(self, conn_method='DIRECT', sim_id=0):
        self.conn_method = conn_method

        if conn_method == 'DIRECT':
            self.p_id = p.connect(p.DIRECT)
        elif conn_method == 'GUI':
            self.p_id = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.p_id)
            # p.resetDebugVisualizerCamera(
            #     5, 0, 200, [0, -3, -0.5], physicsClientId=self.p_id
            # )
            p.resetDebugVisualizerCamera(
                3.26,
                -74.8,
                197.2,
                (-1.58, -0.46, -0.13),
                physicsClientId=self.p_id,
            )

        p.resetSimulation(physicsClientId=self.p_id)

        self.sim_id = sim_id

        # The PyBullet ID of the specimen in the simulation
        self.robot = None

        # Serves as a spot where target object can be placed and
        # as an obstacle to avoid for the fingers
        self.table = None
        # Target contains where the objects will be dropped
        self.target_box = None

        # List of target object ids
        self.target_objects = []

        # Target objects dropped in box
        self.targets_in_box = []

        self.__initialize_pybullet()
        self.__initialize_bodies()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        # Disconnect physics server when simulation is done
        # Source: https://stackoverflow.com/questions/865115/how-do-i-correctly-clean-up-a-python-object
        p.disconnect(self.p_id)

    def __initialize_pybullet(self):
        # Remove everything from the current world
        p.resetSimulation(physicsClientId=self.p_id)

        # Disable file caching incase we load different URDFs with same name
        p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=self.p_id)

        # Set downward gravity
        p.setGravity(0, 0, -10, physicsClientId=self.p_id)

    def __initialize_bodies(self):
        # Create plane shape
        plane_shape = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=self.p_id)
        floor = p.createMultiBody(plane_shape, plane_shape, physicsClientId=self.p_id)

        # Load table to place target objects on
        self.table = p.loadURDF(
            'objects/table.urdf', useFixedBase=1, physicsClientId=self.p_id
        )
        p.resetBasePositionAndOrientation(
            self.table, [0.8, -2, 0], [0, 0, 0, 1], physicsClientId=self.p_id
        )
        # Get table top position
        table_top_z = p.getAABB(
            self.table,
            p.getNumJoints(self.table, physicsClientId=self.p_id) - 1,
            physicsClientId=self.p_id,
        )[1][2]

        # Load target objects and separate them by a certain distance
        for pos, urdf_file in enumerate(object_URDF):
            body = p.loadURDF(f'objects/{urdf_file}', physicsClientId=self.p_id)

            # For the object to sit perfectly on the table, we get the object's
            # base z coordinate and add the table top position
            body_top_z = p.getAABB(body, physicsClientId=self.p_id)[1][2]
            body_z_pos = table_top_z + body_top_z

            p.resetBasePositionAndOrientation(
                body, [0.8, -pos, body_z_pos], [0, 0, 0, 1], physicsClientId=self.p_id
            )
            self.target_objects.append(body)

        # Load target box
        self.target_box = p.loadURDF('objects/assets/target_box.urdf', useFixedBase=1)
        p.resetBasePositionAndOrientation(self.target_box, [-1, 0, 0], [1, 0, 0, 1])

    def __load_next_target_object(self):
        '''Removes the current target object from the target_objects array
        and moves the next one to the pick up location'''

        assert len(self.target_objects) > 0

        in_target_box = check_in_target_box(
            [self.target_objects[0]], self.target_box, self.p_id
        )

        # If the target is not moved to the drop location, the arm was
        # not successful and we need to remove the object to clear the
        # place for the next target.
        if sum(in_target_box) == 0:
            p.removeBody(self.target_objects[0], physicsClientId=self.p_id)
        else:
            self.targets_in_box.append(self.target_objects[0])

        self.target_objects.pop(0)

        if len(self.target_objects) < 1:
            # Finished iterating the target objects
            return

        next_target = self.target_objects[0]

        # Get the current position and orientation of the next target as lists
        # getBasePositionAndOrientation() returns a tuple of tuples. Conversion
        # to list is required so as to modify next_pos's first value
        next_pos, next_orientation = map(
            list,
            p.getBasePositionAndOrientation(next_target, physicsClientId=self.p_id),
        )

        next_pos[1] = 0  # Move to target position in y direction

        p.resetBasePositionAndOrientation(
            next_target,
            next_pos,
            next_orientation,
            physicsClientId=self.p_id,
        )

    def __get_distances(self, link_index) -> float:
        '''
        Returns a distance for a specified link in the robot hand.

        Args:
            link_index (int): link index in the simulation
        '''

        return get_distance_of_bodies(
            self.robot, self.target_objects[0], link_index, self.p_id
        )

    def __get_collisions(self, link_index) -> tuple[Literal[0, 1], Literal[0, 1]]:
        '''
        Checks for a contact between a phalanx - target object and
        a phalanx - table

        Args:
            link_index (int): link indices in the simulation

        Returns:
            A tuple of binary values for target and table collision
            respectively.
        '''
        return check_collisions(
            self.robot, self.target_objects[0], self.table, link_index, self.p_id
        )

    def __calc_fitness(self, phalanges: list[Phalanx]):
        picking_performance = FitnessFunction.get_picking_performance(
            self.targets_in_box, self.target_box, self.p_id
        )
        return FitnessFunction.get_total_fitness(phalanges, picking_performance)

    def run_specimen(self, specimen: Specimen, in_training=True):
        # Load specimen
        if in_training:
            urdf_file = f'intraining_specimen/{specimen.specimen_URDF}'
            self.robot = p.loadURDF(
                urdf_file, useFixedBase=1, physicsClientId=self.p_id
            )
        else:
            urdf_file = f'fit_specimen/urdf_files/{specimen.specimen_URDF}'
            self.robot = p.loadURDF(
                urdf_file, useFixedBase=1, physicsClientId=self.p_id
            )

        # Populate phalanx state with link index
        specimen.load_state(get_genome_link_indices(self.robot, self.p_id))

        # Iterate over all target objects
        while len(self.target_objects) > 0:
            # Move robot arm to ready to pick location
            specimen.move_arm('ready_to_pick', self.robot, self.p_id)

            # Spread out the fingers to ready to pick position
            specimen.move_fingers(
                'ready_to_pick',
                self.robot,
                self.__get_distances,
                self.__get_collisions,
                self.p_id,
            )

            # Move robot arm to pick up location
            specimen.move_arm('pick', self.robot, self.p_id)

            # Move fingers to vertical position
            specimen.move_fingers(
                'before_pick',
                self.robot,
                self.__get_distances,
                self.__get_collisions,
                self.p_id,
            )

            # The grabbing motion is performed by some steps
            for _ in range(5):
                # Pick up the target object
                specimen.move_fingers(
                    'pick',
                    self.robot,
                    self.__get_distances,
                    self.__get_collisions,
                    self.p_id,
                )

            # Move robot arm back to ready to pick location
            specimen.move_arm('ready_to_pick', self.robot, self.p_id)

            # Move robot arm to drop off location
            specimen.move_arm('drop', self.robot, self.p_id)

            # Drop object
            specimen.move_fingers(
                'ready_to_pick',
                self.robot,
                self.__get_distances,
                self.__get_collisions,
                self.p_id,
            )

            self.__load_next_target_object()

            # Move robot arm to drop off location
            specimen.move_arm('drop', self.robot, self.p_id)

            specimen.move_arm('return_to_pick', self.robot, self.p_id)

        # Set specimen fitness
        specimen.fitness = self.__calc_fitness(specimen.phalanges)

        # Keep simulation running is connected via GUI
        # while self.conn_method == 'GUI':
        #     p.stepSimulation(physicsClientId=self.p_id)


class ThreadedSim:
    '''
    Run the multiple simulation in multiple threads

    Args:
        pool_size (int): how many simulations to run at a time

    Source:
        M. King. (n.d.). CM3020 Artificial Intelligence, Creatures 1:
        Automatic design using genetic algorithms. Coursera.
        https://www.coursera.org/learn/uol-cm3020-artificial-intelligence/home/week/6
    '''

    def __init__(self, pool_size=os.cpu_count()):
        self.pool_size = pool_size

    @staticmethod
    def static_run_specimen(specimen: Specimen):
        simulation = Simulation()
        simulation.run_specimen(specimen)

        return specimen

    def run_population(self, population: Population):
        '''
        Runs simulation for all specimen in a population
        in multi-threaded environment

        Args:
            population (Population): A population of specimen
        '''

        pool_args = []
        start_ind = 0

        len_specimen = len(population.specimen)

        # Add a list of arguments to run the specimen for
        # the number of pool_size
        while start_ind < len_specimen:
            # Arguments for one run of a pool
            curr_pool_args = []

            for i in range(start_ind, start_ind + self.pool_size):
                if i >= len_specimen:  # reached the end
                    break

                curr_pool_args.append([population.specimen[i]])

            pool_args.append(curr_pool_args)

            start_ind = start_ind + self.pool_size

        new_specimen = []

        for pool_argset in pool_args:
            with Pool(self.pool_size) as pool:
                # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool.starmap
                evaluated_specimen = pool.starmap(
                    ThreadedSim.static_run_specimen, pool_argset
                )

                new_specimen.extend(evaluated_specimen)

        # update the population
        population.specimen = new_specimen
