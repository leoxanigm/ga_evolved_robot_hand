import pybullet as p

import constants as c


def init_sim(conn_method: str = 'DIRECT', env='training') -> tuple[int]:
    '''Initializes pybullet simulation server and loads simulation bodies.
    Returns physics client id and ids of loaded bodies'''

    if conn_method == 'DIRECT':
        p_id = p.connect(p.DIRECT)
    elif conn_method == 'GUI':
        p_id = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=p_id)
        p.resetDebugVisualizerCamera(
            6.3, 178, -1.2, (0.108, 0.292, 3.612), physicsClientId=p_id
        )

    p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=p_id)
    # Set downward gravity
    p.setGravity(0, 0, -10, physicsClientId=p_id)

    # Create plane shape
    plane_shape = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=p_id)
    floor = p.createMultiBody(plane_shape, plane_shape, physicsClientId=p_id)

    # Load table to place target objects on
    table = p.loadURDF(c.TABLE, useFixedBase=1, physicsClientId=p_id)
    p.resetBasePositionAndOrientation(
        table, [2.82, 0, 0], [0, 0, 0, 1], physicsClientId=p_id
    )

    if env == 'training':
        # Load training target objects and separate them by a certain distance
        target_object_URDF = [c.CUBE, c.SPHERE, c.CYLINDER]
    elif env == 'testing':
        # Load testing target objects and separate them by a certain distance
        target_object_URDF = [c.BOTTLE, c.CUP]

    target_object_ids = []
    for pos, urdf_file in enumerate(target_object_URDF):
        body = p.loadURDF(urdf_file, physicsClientId=p_id)

        if pos == 0:
            p.resetBasePositionAndOrientation(
                body, [2.82, 1.45, 2], [0, 0, 0, 1], physicsClientId=p_id
            )
        else:
            p.resetBasePositionAndOrientation(
                body, [2.82, -pos, 2], [0, 0, 0, 1], physicsClientId=p_id
            )
        # Keep track of added objects
        target_object_ids.append(body)

    # Load target box
    target_box = p.loadURDF(c.TARGET_BOX, useFixedBase=1, physicsClientId=p_id)
    p.resetBasePositionAndOrientation(
        target_box, [0, -1, 0], [0, 0, 0, 1], physicsClientId=p_id
    )

    return p_id, table, target_object_ids, target_box
