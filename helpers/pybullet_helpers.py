import pybullet as p
import constants as c


def is_finger_link(body_id, joint_index):
    '''Check if a link is a finger'''
    if joint_index == -1:  # base link
        return False
    return (p.getJointInfo(body_id, joint_index)[12]).startswith(b'finger_')


def get_distance_of_bodies(body_a_id, body_b_id, p_ID=0):
    '''
    Calculates the distance between two PyBullet bodies.
    Args:
        body_a_id (int): robot hand with fingers added
        body_b_id (int): target object to be picked up
        p_ID (int): PyBullet physicsClientID
    Returns:
        List of distances of each finger on the robot hand to from
        the target object
    '''
    assert isinstance(body_a_id, int) and isinstance(body_b_id, int)
    if p.getNumJoints(body_a_id) < c.FINGER_START_INDEX:
        raise ValueError(
            'body_a_id must be an instance of a robot hand with fingers attached.'
        )

    contact_points = p.getClosestPoints(
        body_a_id, body_b_id, 1000, physicsClientId=p_ID
    )

    # Return distance at index 8
    # And we only want contact points for the fingers, not the arm
    return [
        points[8]
        for points in contact_points
        if is_finger_link(body_a_id, points[3])  # points[3] is joint index of body_a
    ]


def apply_rotation(body_id, joint_index, target_pos, p_ID=0):
    '''
    Applies rotation to either one joint or multiple joints based on
    the supplied args
    Args:
        body_id (int): robot hand
        joint_index (int | list[int]): joint(s) to apply rotation to
        target_pos (int | list[int]): target rotation angle(s)
    '''
    assert isinstance(body_id, int)
    if p.getNumJoints(body_id) < c.FINGER_START_INDEX:
        raise ValueError(
            'body_a_id must be an instance of a robot hand with fingers attached.'
        )

    if isinstance(joint_index, list):
        # If multiple joint indexes are given, target positions must also
        # a list with equal length
        assert isinstance(target_pos, list)
        assert len(joint_index) == len(target_pos)

        p.setJointMotorControlArray(
            body_id, joint_index, p.POSITION_CONTROL, target_pos, physicsClientId=p_ID
        )

    elif isinstance(joint_index, int):
        p.setJointMotorControl2(
            body_id, joint_index, p.POSITION_CONTROL, target_pos, physicsClientId=p_ID
        )

    else:
        raise TypeError('joint_index must me either an integer or list of integers')
