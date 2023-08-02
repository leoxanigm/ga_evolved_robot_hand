import pybullet as p
import pybullet_data as pd
import numpy as np

from specimen import Specimen
from constants import GeneDesc

p.connect(p.DIRECT)

# Config
p.setPhysicsEngineParameter(enableFileCaching=0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.resetDebugVisualizerCamera(5, 0, 200, [0, -3, -0.5])

# Create plane collision shape
plane_shape = p.createCollisionShape(p.GEOM_PLANE)

# Create plane shape
plane = p.createMultiBody(plane_shape, plane_shape)

# Set downward gravity
p.setGravity(0, 0, -10)

specimen = Specimen(GeneDesc, 'robot_hand.urdf')

r = p.loadURDF(specimen.specimen_URDF, useFixedBase=1)

startOrientation = p.getQuaternionFromEuler([0, 0, 0])

# specimen.train_brain()
# print(
#     specimen.brain.move_object(
#         [[0.1, 0.06, 1.2], [0.2, 0.16, 1.7]], 1, 1, specimen.fingers
#     )
# )

# load table object
table = p.loadURDF('objects/table.urdf', useFixedBase=1)
p.resetBasePositionAndOrientation(table, [0.8, -2, 0], startOrientation)

# load cube object
cube = p.loadURDF('objects/cube.urdf')
p.resetBasePositionAndOrientation(cube, [0.8, 0, 0.7], startOrientation)

contact_points = p.getClosestPoints(r, cube, 1000)
contact_points = contact_points[5:] # We do not need distance for parts of the base robot arm

distances = [point[8] for point in contact_points]
print(len(distances))
print(p.getNumJoints(r))