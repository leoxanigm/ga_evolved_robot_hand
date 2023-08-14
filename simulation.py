import pybullet as p
import numpy as np
import time

from helpers.pybullet_helpers import get_distance_of_bodies, apply_rotation

from population import Population
from specimen import Specimen

object_URDF = ['cube.urdf', 'sphere.urdf', 'cylinder.urdf']


class Simulation:
    def __init__(self, conn_method='DIRECT', sim_id=0):
        self.conn_method = conn_method

        if conn_method == 'DIRECT':
            self.p_id = p.connect(p.DIRECT)
        elif conn_method == 'GUI':
            self.p_id = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.p_id)
        self.sim_id = sim_id

        # The PyBullet ID of the specimen in the simulation
        self.robot = None

        # List of target object ids
        self.target_objects = []

        self.__initialize_pybullet()
        self.__initialize_bodies()

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
        table = p.loadURDF(
            'objects/table.urdf', useFixedBase=1, physicsClientId=self.p_id
        )
        p.resetBasePositionAndOrientation(
            table, [0.8, -2, 0], [0, 0, 0, 1], physicsClientId=self.p_id
        )
        # Get table top position
        table_top_z = p.getAABB(
            table,
            p.getNumJoints(table, physicsClientId=self.p_id) - 1,
            physicsClientId=self.p_id,
        )[1][2]

        # Load target objects and separate them by a certain distance
        for pos, urdf_file in enumerate(object_URDF):
            body = p.loadURDF(f'objects/{urdf_file}', physicsClientId=self.p_id)
            body_top_z = p.getAABB(body, physicsClientId=self.p_id)[1][2]
            body_z_pos = table_top_z + body_top_z
            p.resetBasePositionAndOrientation(
                body, [0.8, -pos, body_z_pos], [0, 0, 0, -1], physicsClientId=self.p_id
            )
            self.target_objects.append(body)

    def get_distances(self, link_type='fingers') -> list[tuple[float, int]]:
        '''Returns a list of distances and their corresponding link index
        for phalanx links or palm from the target object.
        Args:
            link_type (str): either fingers or palm
        '''

        return get_distance_of_bodies(
            self.robot, self.target_objects[0], link_type, self.p_id
        )

    def set_rotation_angles(self, joint_index: int, target_angle: int):
        '''Sets rotation angle for a phalanx link
        Args:
            joint_index (int): target joint index
            target_angle (int): target position for the joint
        '''

        p.setJointMotorControl2(
            bodyUniqueId=self.robot,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_angle,
            physicsClientId=self.p_id,
        )

    def run_specimen(self, specimen: Specimen, iterations=2400):
        # Load specimen
        self.robot = p.loadURDF(
            specimen.specimen_URDF, useFixedBase=1, physicsClientId=self.p_id
        )

        # p.resetDebugVisualizerCamera(2, 50,-25, [0, 0, 1])

        # Move robot arm to ready to pick location
        specimen.move_arm('ready_to_pick', self.robot, self.p_id)

        # Spread out the fingers to ready to pick position
        specimen.move_fingers(
            'ready_to_pick', self.robot, self.get_distances, self.p_id
        )

        specimen.calc_fitness(self.get_distances)
        print(specimen.fitness_total)

        # Move robot arm to pick up location
        specimen.move_arm('pick', self.robot, self.p_id)

        # Pick up the target object
        specimen.move_fingers('pick', self.robot, self.get_distances, self.p_id)

        # Move robot arm back to ready to pick location
        specimen.move_arm('ready_to_pick', self.robot, self.p_id)

        specimen.calc_fitness(self.get_distances)
        print(specimen.fitness_total)

        while self.conn_method == 'GUI':
            p.stepSimulation(physicsClientId=self.p_id)