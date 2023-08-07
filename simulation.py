import pybullet as p
import numpy as np

from helpers.pybullet_helpers import get_distance_of_bodies, apply_rotation

from population import Population
from specimen import Specimen


class Simulation:
    def __init__(self, sim_id=0, conn_method='DIRECT'):
        if conn_method == 'DIRECT':
            self.physicsClientId = p.connect(p.DIRECT)
        elif conn_method == 'GUI':
            self.physicsClientId = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self.sim_id = sim_id

    def run_specimen(self, specimen: Specimen, iterations=2400):
        p_id = self.physicsClientId

        # Remove everything from the current world
        p.resetSimulation(physicsClientId=p_id)

        # Disable file caching incase we load different URDFs with same name
        p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=p_id)

        # Set downward gravity
        p.setGravity(0, 0, -10, physicsClientId=p_id)

        # Create plane shape
        plane_shape = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=p_id)
        floor = p.createMultiBody(plane_shape, plane_shape, physicsClientId=p_id)

        specimen_id = p.loadURDF(
            specimen.specimen_URDF, useFixedBase=1, physicsClientId=p_id
        )

        cube_id = p.loadURDF('objects/cube.urdf')

        count = 0
        while True:
            p.stepSimulation(physicsClientId=p_id)

            if count % 480 == 0:
                specimen.move_fingers(distances)

            try:
                pos, _ = p.getBasePositionAndOrientation(c_id, physicsClientId=p_id)

                creature.update_position(pos)
                # print(creature.get_distance_traveled())
            except Exception as err:
                print(type(err), err)
                continue
            if time.time() - tick > 10:
                break

            count += 1
