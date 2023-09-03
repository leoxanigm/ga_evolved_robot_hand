import pybullet as p
import constants as c
import time
import math

from .math_functions import distance_between_coordinates


def is_finger_link(body_id, joint_index):
    '''Check if a link is a finger'''
    if joint_index == -1:  # base link
        return False
    return (p.getJointInfo(body_id, joint_index)[12]).startswith(b'finger_')


def is_palm_link(body_id, joint_index):
    '''Check if a link is palm'''
    if joint_index == -1:  # base link
        return False
    return (p.getJointInfo(body_id, joint_index)[12]).startswith(b'palm_')


def get_distance_of_bodies(
    body_a_id, body_b_id, link_type, p_id=0
) -> tuple[float, int]:
    '''
    Calculates the distance between two PyBullet bodies.
    Args:
        body_a_id (int): robot hand with fingers added
        body_b_id (int): target object to be picked up
        link_type (str): which part of body_a we want distances of (fingers/palm)
        p_id (int): PyBullet physicsClientID
    Returns:
        A List tuples with distance of each phalanx from the target object
        and the joint index
    '''
    assert isinstance(body_a_id, int) and isinstance(body_b_id, int)
    if p.getNumJoints(body_a_id) < c.FINGER_START_INDEX:
        raise ValueError(
            'body_a_id must be an instance of a robot hand with fingers attached.'
        )

    # getClosestPoints() returns a list of contact points for each link.
    # Distance is at index 8, link index for body_a is at index 3
    closest_points = p.getClosestPoints(
        body_a_id, body_b_id, 1000, physicsClientId=p_id
    )

    if link_type == 'palm':
        # Distances for the palm
        return [
            (points[8], points[3])
            for points in closest_points
            if is_palm_link(body_a_id, points[3])  # points[3] is joint index of body_a
        ]
    else:  # Distances for the fingers
        return [
            (points[8], points[3])
            for points in closest_points
            if is_finger_link(
                body_a_id, points[3]
            )  # points[3] is joint index of body_a
        ]

def calculate_rotation_angle(body_a_id, body_b_id, link_length, link_index, p_id) -> float:
    '''
    Calculates angle based on length and distance
    angle = arctan(distance / length)
    Args:
        body_a_id (int): robot hand with fingers added
        body_b_id (int): target object to be picked up
        link_length (float): length from palm to base of link
        link_index (int): link index in the simulation
        p_id (int): PyBullet physicsClientID
    Returns:
        Angle in radians
    '''
    link_loc = list(p.getLinkState(body_a_id, link_index, physicsClientId=p_id)[0])  # link location at link origin
    link_loc[2] -= link_length  # get location of the link's end

    obj_loc = p.getBasePositionAndOrientation(body_b_id, physicsClientId=p_id)[0]

    target_loc = p.rayTest(link_loc, obj_loc, physicsClientId=p_id)  # cast ray from current link to cube
    target_loc = target_loc[0][3]  # 3'rd index is where the intersection loc is

    dis = distance_between_coordinates(target_loc, link_loc)
    angle = math.atan(dis / link_length)  # target angle

    return angle


def apply_rotation(body_id, joint_index, target_pos, p_id=0, prev_target_pos=None):
    '''
    Applies rotation to either one joint or multiple joints based on
    the supplied args
    Args:
        body_id (int): robot hand
        joint_index (int | list[int]): joint(s) to apply rotation to
        target_pos (int | list[int]): target rotation angle(s)
        p_id (int): physics client id
    '''
    check_id = body_id

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

        if prev_target_pos:
            smooth_joint_control(
                body_id,
                joint_index,
                prev_target_pos,
                target_pos,
                p_id,
                joint_motor_control_function=p.setJointMotorControlArray,
            )
        else:
            p.setJointMotorControlArray(
                body_id,
                joint_index,
                p.POSITION_CONTROL,
                target_pos,
                physicsClientId=p_id,
            )

        p.stepSimulation(physicsClientId=p_id)

    elif isinstance(joint_index, int):
        if prev_target_pos:
            smooth_joint_control(
                body_id, joint_index, prev_target_pos, target_pos, p_id
            )
        else:
            p.setJointMotorControl2(
                body_id,
                joint_index,
                p.POSITION_CONTROL,
                target_pos,
                physicsClientId=p_id,
            )

        p.stepSimulation(physicsClientId=p_id)

    else:
        raise TypeError('joint_index must me either an integer or list of integers')


def smooth_joint_control(
    body_id,
    joint_index,
    prev_pos,
    target_pos,
    p_id=0,
    anim_time=0.25,
    joint_motor_control_function=p.setJointMotorControl2,
):
    '''
    Smooths joint movement for position joint control.
    Works by applying a fraction of the target angle for each iteration.
    Args:
        body_id (int): robot hand
        joint_index (int | list[int]): joint(s) to apply rotation to
        target_pos (int | list[int]): target rotation angle(s)
        anim_time (float): amount of seconds to the movement takes
        p_id (int): physics client id
        joint_motor_control_function (function): joint motor control function to use
    '''

    # steps = anim_time / 100
    # while steps < anim_time:
    #     joint_motor_control_function(
    #         body_id, joint_index, p.POSITION_CONTROL, target_pos, physicsClientId=p_id
    #     )
    #     p.stepSimulation()
    #     time.sleep(steps)
    #     steps += steps
    # if isinstance(target_pos, list):
    #     step_pos = [pos / 100 for pos in target_pos]
    # else:
    #     step_pos = target_pos / 100
    steps = 2400

    if isinstance(target_pos, list):
        pos_difference = [t_pos - p_pos for t_pos, p_pos in zip(target_pos, prev_pos)]
        pos_increments = [diff / steps for diff in pos_difference]

        new_target_pos = prev_pos
        for _ in range(steps):
            new_target_pos = [
                n_pos + pos_i for n_pos, pos_i in zip(new_target_pos, pos_increments)
            ]

            joint_motor_control_function(
                body_id,
                joint_index,
                p.POSITION_CONTROL,
                new_target_pos,
                physicsClientId=p_id,
            )
            p.stepSimulation()
    else:
        new_target_pos = target_pos / steps
