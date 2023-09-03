import pybullet as p
import numpy as np
import time
import sys
import math

from helpers.pybullet_helpers import get_distance_of_bodies

if len(sys.argv) == 2 and sys.argv[1] == 'DIRECT':
    p.connect(p.DIRECT)
else:
    p.connect(p.GUI)

# Config
p.setPhysicsEngineParameter(enableFileCaching=0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)

# Create plane collision shape
plane_shape = p.createCollisionShape(p.GEOM_PLANE)

# Create plane shape
plane = p.createMultiBody(plane_shape, plane_shape)

p.resetDebugVisualizerCamera(
    1.003999948501587,
    179.199951171875,
    -13.799997329711914,
    (0.7531526684761047, -0.4100551903247833, 0.6280001997947693),
)

# Set downward gravity
p.setGravity(0, 0, -10)


def distance(c):
    d_1 = c[4][8]
    d_2 = c[5][8]
    d_3 = c[6][8]
    return d_1, d_2, d_3


def calculate_distance(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    distance = math.sqrt(dx**2 + dy**2 + dz**2)
    return distance

r = p.loadURDF('intraining_specimen/5fb86384.urdf', useFixedBase=1)
c = p.loadURDF('objects/cube.urdf', useFixedBase=1)

p.setJointMotorControl2(r, 0, p.POSITION_CONTROL, 0)
p.setJointMotorControl2(r, 1, p.POSITION_CONTROL, np.pi / 2)

for i in range(4, p.getNumJoints(r)):
    p.setJointMotorControl2(r, i, p.POSITION_CONTROL, 0)

p.setJointMotorControl2(r, 4, p.POSITION_CONTROL, np.pi)

for _ in range(2400):
    p.stepSimulation()

p.resetBasePositionAndOrientation(c, [0.8, 0, 0.8], [0, 0, 0, 1])

cube_loc = p.getBasePositionAndOrientation(c)[0]

length = 0
link_length = 0.062  # to be taken from gene encoding

print(get_distance_of_bodies(r, c, 'fingers'))
print('=========================')
print('----------------------------')

new_dis = []

for i in range(4, p.getNumJoints(r)):
    length += link_length
    link_loc = list(p.getLinkState(r, i)[0])  # link location at link origin
    # link_loc[2] -= length  # get location of the link's end
    target_loc = p.rayTest(link_loc, cube_loc)  # cast ray from current link to cube
    target_loc = target_loc[0][3]  # 3'rd index is where the intersection loc is
    dis = calculate_distance(target_loc, link_loc)
    angle = math.atan(dis / length)  # target angle

    p.setJointMotorControl2(r, i, p.POSITION_CONTROL, targetPosition=-angle)

    for _ in range(2400):
        p.stepSimulation()

    link_loc = list(p.getLinkState(r, i)[0])  # link location at link origin
    # link_loc[2] -= length  # get location of the link's end
    target_loc = p.rayTest(link_loc, cube_loc)  # cast ray from current link to cube
    target_loc = target_loc[0][2]  # 3'rd index is where the intersection loc is
    dis = calculate_distance(target_loc, link_loc)
    new_dis.append(target_loc)
    

print(get_distance_of_bodies(r, c, 'fingers'))
print('=========================')
print(new_dis)

while True:
    p.stepSimulation()

