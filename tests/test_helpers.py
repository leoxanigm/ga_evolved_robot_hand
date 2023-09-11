import os
import time
import unittest
import hypothesis.strategies as st
from hypothesis import given
import numpy as np
import pybullet as p

from genome import FingersGenome, BrainGenome
from phenome import FingersPhenome, BrainPhenome
from specimen import Specimen
from simulation import Simulation
from tests.test_specimen import FingerPhenomeTest, BrainPhenomeTest
from constants import GeneDesc

from helpers.pybullet_helpers import get_distance_of_bodies, check_collisions, check_in_target_box

target_urdf = ['cube.urdf', 'sphere.urdf', 'cylinder.urdf']
target_obj_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'objects')

def init_pybullet(robot_file):
    # Comments describing the steps here can be found in specimen.py file
    global robot
    global table
    global target_objects
    global target_pos

    target_objects = []
    try:
        p.disconnect()
    except:
        pass
    p.connect(p.DIRECT)
    # p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setPhysicsEngineParameter(enableFileCaching=0)
    plane_shape = p.createCollisionShape(p.GEOM_PLANE)
    plane = p.createMultiBody(plane_shape, plane_shape)
    p.setGravity(0, 0, -10)
    table = p.loadURDF(target_obj_dir + '/table.urdf', useFixedBase=1)
    p.resetBasePositionAndOrientation(table, [0.8, -2, 0], [0, 0, 0, 1])
    table_top_z = p.getAABB(table, p.getNumJoints(table) - 1)[1][2]
    for pos, urdf_file in enumerate(target_urdf):
        body = p.loadURDF(target_obj_dir + f'/{urdf_file}')
        body_top_z = p.getAABB(body)[1][2]
        body_z_pos = table_top_z + body_top_z
        if pos == 0:
            target_pos = [0.8, -pos, body_z_pos]
            p.resetBasePositionAndOrientation(body, target_pos, [0, 0, 0, 1])
        else:
            p.resetBasePositionAndOrientation(
                body, [0.8, -pos, body_z_pos], [0, 0, 0, 1]
            )
        target_objects.append(body)
    robot = p.loadURDF(robot_file, useFixedBase=1)
    for i in range(4, p.getNumJoints(robot)):
        if i in [4, 7, 10, 13]:
            p.setJointMotorControl2(robot, i, p.POSITION_CONTROL, -np.pi / 8)
        else:
            p.setJointMotorControl2(robot, i, p.POSITION_CONTROL, 0)
    step_simulation()
    p.setJointMotorControlArray(robot, [0, 1], p.POSITION_CONTROL, [0, np.pi / 2])
    step_simulation()
    for i in range(4, p.getNumJoints(robot)):
        p.setJointMotorControl2(robot, i, p.POSITION_CONTROL, 0)
    step_simulation()


def step_simulation(time_step=240):
    for _ in range(time_step):
        p.stepSimulation()


class TestHelpers(unittest.TestCase):
    def test_positive_distance(self):
        distances = []

        init_pybullet('tests/urdf_files/fit.urdf')  # manually created fit robot

        for i in range(4, p.getNumJoints(robot)):
            d = get_distance_of_bodies(robot, target_objects[0], i, 0)
            distances.append(d)

        assert min(distances) > 0

    def test_target_collision(self):
        '''Test collision with target and no collision with obstacle'''

        # p.disconnect()
        init_pybullet('tests/urdf_files/fit.urdf')  # manually created fit robot
        collisions = []

        # Move only one finger
        p.setJointMotorControl2(robot, 4, p.POSITION_CONTROL, np.pi / 8)
        step_simulation()

        for i in range(4, p.getNumJoints(robot)):
            c = check_collisions(robot, target_objects[0], table, i, 0)
            collisions.append(c)

        # Collisions have structure (target collision, obstacle collision)
        # zip them to their own set of tuples for easier checking
        collisions = list(zip(*collisions))

        # Collision with target, only one finger has moved
        assert min(collisions[0]) == 0
        assert max(collisions[0]) == 1
        # No collision with table
        assert max(collisions[1]) == 0

    def test_obstacle_collision(self):
        '''Test no collision with target and collision with obstacle'''

        # p.disconnect()
        init_pybullet('tests/urdf_files/coll_table.urdf')  # manually created fit robot
        collisions = []

        # Move only one finger
        p.setJointMotorControl2(robot, 4, p.POSITION_CONTROL, np.pi / 8)
        step_simulation()

        for i in range(4, p.getNumJoints(robot)):
            c = check_collisions(robot, target_objects[0], table, i, 0)
            collisions.append(c)

        # Collisions have structure (target collision, obstacle collision)
        # zip them to their own set of tuples for easier checking
        collisions = list(zip(*collisions))

        # Collision with table
        assert max(collisions[1]) == 1

    def test_in_target(self):
        '''
        Test checking target objects are in target box works.
        This runs a fit specimen that puts all the target object in the box.
        Then checks the helper correctly calculates that all target are in box.
        '''
        fingers_genome = FingersGenome.genome(GeneDesc)
        brain_genome = BrainGenome.genome(fingers_genome)
        finger_phenome = FingerPhenomeTest(fingers_genome)
        brain_phenome = BrainPhenomeTest(brain_genome)
        specimen = Specimen()
        specimen.fingers = finger_phenome.genome
        specimen.brain = brain_phenome
        specimen.write_training_urdf() # New urdf for our phenome

        with Simulation('GUI') as simulation:
            simulation.run_specimen(specimen)
            target_box = simulation.target_box
            moved_objects = simulation.targets_in_box
            in_box = check_in_target_box(moved_objects, target_box, simulation.p_id)

        assert sum(in_box) == 3

    def test_not_in_target(self):
        '''
        Test checking target objects are not in target box works.
        Runs random specimen.
        '''

        specimen = Specimen()
        with Simulation() as simulation:
            simulation.run_specimen(specimen)
            target_box = simulation.target_box
            moved_objects = simulation.targets_in_box
            in_box = check_in_target_box(moved_objects, target_box, simulation.p_id)

        assert sum(in_box) == 0


if __name__ == '__main__':
    unittest.main()
