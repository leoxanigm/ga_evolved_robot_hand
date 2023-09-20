import os
import pybullet as p
import numpy as np
from multiprocessing import Process, Manager, current_process, Semaphore
import time

from typing import Literal

from helpers.pybullet_helpers import (
    check_distances,
    check_collisions,
    apply_rotation,
    get_genome_link_indices,
    check_in_target_box,
)
from helpers.init_sim import init_sim
from helpers.misc_helpers import clear_dir

from population import Population
from specimen import Specimen, Phalanx
from fitness_fun import FitnessFunction
import constants as c
from generate_urdf import GenerateURDF

object_URDF = ['cube.urdf', 'sphere.urdf', 'cylinder.urdf']


class Simulation:
    def __init__(self, conn_method='DIRECT', training=True):
        self.conn_method = conn_method

        # The PyBullet ID of the specimen in the simulation
        self.robot = None
        if training:
            # Initialize training simulation
            self.p_id, self.table, self.target_objects, self.target_box = init_sim(
                conn_method, 'training'
            )
        else:
            # Initialize testing simulation
            self.p_id, self.table, self.target_objects, self.target_box = init_sim(
                conn_method, 'testing'
            )

        # Target objects dropped in box
        self.targets_in_box = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        # Disconnect physics server when simulation is done
        # Source: https://stackoverflow.com/questions/865115/how-do-i-correctly-clean-up-a-python-object
        p.disconnect(self.p_id)
        # Clear training directory
        clear_dir()

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

        # Move target object to palm location
        next_pos = [2.82, 1.45, 2]

        p.resetBasePositionAndOrientation(
            next_target,
            next_pos,
            [0, 0, 0, 1],
            physicsClientId=self.p_id,
        )

    def __check_distances(self) -> list[int]:
        '''
        Returns a distance for a specified link in the robot hand.
        Args:
            link_index (int): link index in the simulation
        '''
        return check_distances(self.robot, self.target_objects[0], self.p_id)

    def __check_collisions(self, with_body: str) -> list[int]:
        '''
        Checks for a contact between a phalanx with specified body
        Args:
            with_body: target object or obstacle
        Returns:
            A tuple of binary values for target and table collision
            respectively.
        '''
        if with_body == 'target':
            return check_collisions(self.robot, self.target_objects[0], self.p_id)
        elif with_body == 'obstacle':
            return check_collisions(self.robot, self.table, self.p_id)

    def __calc_fitness(self, phalanges: list[Phalanx]):
        picking_performance = FitnessFunction.get_picking_performance(
            self.targets_in_box, self.target_box, self.p_id
        )
        return FitnessFunction.get_total_fitness(phalanges, picking_performance)

    def run_specimen(self, specimen: Specimen):
        # Load specimen urdf
        output_file = os.path.join(c.TRAINING_DIR, f'{specimen.id}.urdf')
        GenerateURDF.generate_robot_fingers(specimen.fingers, output_file)
        self.robot = p.loadURDF(output_file, useFixedBase=1, physicsClientId=self.p_id)

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
                self.__check_distances,
                self.__check_collisions,
                self.p_id,
            )

            # Move robot arm to pick up location
            specimen.move_arm('pick', self.robot, self.p_id)

            # Move fingers to vertical position
            specimen.move_fingers(
                'before_pick',
                self.robot,
                self.__check_distances,
                self.__check_collisions,
                self.p_id,
            )

            # The grabbing motion is performed by some steps
            for _ in range(5):
                # Pick up the target object
                specimen.move_fingers(
                    'pick',
                    self.robot,
                    self.__check_distances,
                    self.__check_collisions,
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
                self.__check_distances,
                self.__check_collisions,
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
        # Number of cores to use
        self.pool_size = pool_size

    @staticmethod
    def run_specimen(specimen: Specimen):
        with Simulation() as simulation:
            simulation.run_specimen(specimen)

        return specimen

    def run_population(self, population: Population):
        '''
        Runs simulation for all specimen in a population
        in multi-threaded environment
        Source: https://docs.python.org/3/library/multiprocessing.html

        Args:
            population (Population): A population of specimen
        '''

        def add_specimen(
            specimen: Specimen, specimen_list: list[Specimen], sema: Semaphore
        ):
            '''Function to be passed to a process. It Evaluates a specimen
            and adds it to a result list'''

            specimen = ThreadedSim.run_specimen(specimen)
            specimen_list.append(specimen)

            # Add 1 to sema so that processes can continue running
            sema.release()

        specimens = population.specimen

        evaluated_specimen = []
        processes = []

        # Using Semaphore to only run specified amount of threads
        # Useful when we want to do other tasks while evaluation is running
        # Source: https://stackoverflow.com/questions/20886565/using-multiprocessing-process-with-a-maximum-number-of-simultaneous-processes
        sema = Semaphore(self.pool_size)

        with Manager() as manager:
            specimen_list = manager.list()

            for i, specimen in enumerate(specimens):
                # Reserve process (add 1) for each one started
                sema.acquire()

                process = Process(
                    target=add_specimen,
                    args=(specimen, specimen_list, sema),
                    name=f'{i}',
                )
                process.start()
                processes.append(process)

            # We are going to run a specimen for a maximum of 20 seconds
            end_time = time.time() + 20

            while time.time() < end_time and processes:
                for process in processes.copy():
                    process.join(timeout=0.1)  # Periodically join to check status
                    if not process.is_alive():
                        # We don't need to track the process if it's finished
                        processes.remove(process)

            # Terminate any still-running processes after 20 seconds
            for process in processes:
                if process.is_alive():
                    # If evaluation for a specimen runs more than 20 seconds,
                    # discard the specimen by setting its fitness to 0.
                    specimens[int(process.name)].fitness = 0
                    evaluated_specimen.append(specimens[int(process.name)])
                    process.terminate()

            evaluated_specimen.extend(list(specimen_list))

        # update the population
        population.specimen = evaluated_specimen
