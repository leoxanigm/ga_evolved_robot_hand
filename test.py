import pybullet as p
import numpy as np
import helpers.pybullet_helpers as p_h
import helpers.math_functions as m_f

r = p.loadURDF(
    'intraining_specimen/r_9be70e16-e36e-4124-8bfa-6fae9efd751d.urdf', useFixedBase=1
)
c = p.loadURDF('objects/cube.urdf')
distances = p_h.get_distance_of_bodies(r, c)
targets = np.random.uniform(
    low=(np.pi / 2) - 0.01, high=(np.pi / 2) + 0.01, size=len(distances)
).tolist()
joint_indices = [i for i in range(4, p.getNumJoints(r))]

# print(len(distances), joint_indices)
count = 0
while True:
    count += 1
    p.stepSimulation()

    if count % 2400 == 0:
        for i, d in enumerate(distances):
            sign = d / np.abs(d)
            max_angle = sign * np.pi / 1000
            target_dis = m_f.normalize(d, 0, max_angle, 0, d)
            targets[i] = target_dis

        distances = p_h.get_distance_of_bodies(r, c)

        print(distances)

        p_h.apply_rotation(r, joint_indices, targets)
