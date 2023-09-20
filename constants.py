from enum import IntEnum, auto

MAX_NUM_FINGERS = 7
MAX_NUM_PHALANGES = 5


class GeneDesc(IntEnum):
    RADIUS = 0
    LENGTH = auto()
    JOINT_ORIGIN_X = auto()
    JOINT_ORIGIN_Y = auto()
    JOINT_ORIGIN_Z = auto()
    JOINT_AXIS_X = auto()
    JOINT_AXIS_Y = auto()
    JOINT_AXIS_Z = auto()

class Limits:
    RADIUS_LOWER = 0.05  # 5cm
    RADIUS_UPPER = 0.1  # 10cm
    LENGTH_LOWER = 0.2  # 10cm
    LENGTH_UPPER = 0.5  # 50cm


# The PyBullet body index of the first finger, excluding the robot arm
FINGER_START_INDEX = 3

# The inputs are:
#   distance of phalanx from target object (1/0)
#   collision with target object (1/0)
#   collision with obstacle (1/0)
NUMBER_OF_INPUTS = 3

TRAINING_DIR = 'intraining_specimen/'
FIT_DIR = 'fit_specimen/'

# Object URDF files
ROBOT_ARM = 'assets/robot_arm/robot_arm.urdf'

CUBE = 'assets/objects/cube.urdf'
SPHERE = 'assets/objects/sphere.urdf'
CYLINDER = 'assets/objects/cylinder.urdf'
CONE = 'assets/objects/cone.urdf'
BOTTLE = 'assets/objects/bottle.urdf'
CUP = 'assets/objects/cup.urdf'

TABLE = 'assets/obstacle/table.urdf'
TARGET_BOX = 'assets/target_box/target_box.urdf'

# Phalanx STL file
PHALANX = 'assets/robot_arm/phalnax.stl'