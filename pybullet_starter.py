import pybullet as p
import numpy as np
import time
import sys
import math

from helpers.pybullet_helpers import get_distance_of_bodies
from helpers.debug_helpers import draw_debug_boundary_box, draw_debug_sphere

if len(sys.argv) == 2 and sys.argv[1] == 'DIRECT':
    p.connect(p.DIRECT)
else:
    p.connect(p.GUI)

# Config
p.setPhysicsEngineParameter(enableFileCaching=0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

# Create plane collision shape
plane_shape = p.createCollisionShape(p.GEOM_PLANE)

# Create plane shape
plane = p.createMultiBody(plane_shape, plane_shape)

# Set downward gravity
p.setGravity(0, 0, -10)