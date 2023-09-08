import pybullet as p

def draw_debug_boundary_box(aabb, color=[1, 0, 0]):
    '''
    Draws debug lines around a boundary box
    Source: https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/getAABB.py
    '''

    aabb_min = aabb[0]
    aabb_max = aabb[1]

    f = [aabb_min[0], aabb_min[1], aabb_min[2]]
    t = [aabb_max[0], aabb_min[1], aabb_min[2]]
    p.addUserDebugLine(f, t, color)

    f = [aabb_min[0], aabb_min[1], aabb_min[2]]
    t = [aabb_min[0], aabb_max[1], aabb_min[2]]
    p.addUserDebugLine(f, t, color)

    f = [aabb_min[0], aabb_min[1], aabb_min[2]]
    t = [aabb_min[0], aabb_min[1], aabb_max[2]]
    p.addUserDebugLine(f, t, color)

    f = [aabb_min[0], aabb_min[1], aabb_max[2]]
    t = [aabb_min[0], aabb_max[1], aabb_max[2]]
    p.addUserDebugLine(f, t, color)

    f = [aabb_min[0], aabb_min[1], aabb_max[2]]
    t = [aabb_max[0], aabb_min[1], aabb_max[2]]
    p.addUserDebugLine(f, t, color)

    f = [aabb_max[0], aabb_min[1], aabb_min[2]]
    t = [aabb_max[0], aabb_min[1], aabb_max[2]]
    p.addUserDebugLine(f, t, color)

    f = [aabb_max[0], aabb_min[1], aabb_min[2]]
    t = [aabb_max[0], aabb_max[1], aabb_min[2]]
    p.addUserDebugLine(f, t, color)

    f = [aabb_max[0], aabb_max[1], aabb_min[2]]
    t = [aabb_min[0], aabb_max[1], aabb_min[2]]
    p.addUserDebugLine(f, t, color)

    f = [aabb_min[0], aabb_max[1], aabb_min[2]]
    t = [aabb_min[0], aabb_max[1], aabb_max[2]]
    p.addUserDebugLine(f, t, color)

    f = [aabb_max[0], aabb_max[1], aabb_max[2]]
    t = [aabb_min[0], aabb_max[1], aabb_max[2]]
    p.addUserDebugLine(f, t, color)

    f = [aabb_max[0], aabb_max[1], aabb_max[2]]
    t = [aabb_max[0], aabb_min[1], aabb_max[2]]
    p.addUserDebugLine(f, t, color)

    f = [aabb_max[0], aabb_max[1], aabb_max[2]]
    t = [aabb_max[0], aabb_max[1], aabb_min[2]]
    p.addUserDebugLine(f, t, color)

def draw_debug_sphere(pos):
    '''Draws a small red sphere at specified position'''
    s_1 = p.loadURDF('objects/sphere_small.urdf', useFixedBase=1)
    p.resetBasePositionAndOrientation(s_1, pos, [0, 0, 0, 1])