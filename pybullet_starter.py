import pybullet as p
import pybullet_data as pd
import numpy as np
import time

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