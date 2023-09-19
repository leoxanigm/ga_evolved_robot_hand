from enum import IntEnum

MAX_NUM_FINGERS = 7
MAX_NUM_PHALANGES = 10


class GeneDesc(IntEnum):
    DIM_X = 0
    DIM_Y = 1
    DIM_Z = 2
    JOINT_ORIGIN_X = 3
    JOINT_ORIGIN_Y = 4
    JOINT_ORIGIN_Z = 5
    JOINT_AXIS_X = 6
    JOINT_AXIS_Y = 7
    JOINT_AXIS_Z = 8


class Limits:
    DIM_X_LOWER = DIM_Y_LOWER = 0.01  # 1cm
    DIM_X_UPPER = DIM_Y_UPPER = 0.05  # 5cm
    DIM_Z_LOWER = 0.05  # 1cm
    DIM_Z_UPPER = 0.1  # 20cm


# The PyBullet body index of the first finger, excluding the robot arm
FINGER_START_INDEX = 4

# The inputs are:
#   distance of phalanx from target object (1/0)
#   collision with target object (1/0)
#   collision with obstacle (1/0)
NUMBER_OF_INPUTS = 3

# Object URDF files
TRAINING_DIR = 'intraining_specimen/'
ROBOT_HAND = 'assets/robot_arm/robot_arm.urdf'
CUBE = 'assets/objects/cube.urdf'
SPHERE = 'assets/objects/sphere.urdf'
CYLINDER = 'assets/objects/cylinder.urdf'
CONE = 'assets/objects/cone.urdf'
TABLE = 'assets/obstacle/table.urdf'
TARGET_BOX = 'assets/target_box/target_box.urdf'

# Phalanx STL file
PHALANX = 'assets/robot_arm/phalnax.stl'