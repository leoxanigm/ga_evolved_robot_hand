import pybullet as p
import constants as c
import time
import math
from collections import namedtuple

from typing import Literal

from helpers.math_functions import distance_between_coordinates
from helpers.debug_helpers import draw_debug_boundary_box, draw_debug_sphere


def is_finger_link(body_id, joint_index, p_id):
    '''Check if a link is a finger'''
    if joint_index == -1:  # base link
        return False
    link_name = p.getJointInfo(body_id, joint_index, physicsClientId=p_id)[12]
    return link_name.startswith(b'finger_')


def is_palm_link(body_id, joint_index, p_id):
    '''Check if a link is palm'''
    if joint_index == -1:  # base link
        return False
    link_name = p.getJointInfo(body_id, joint_index, physicsClientId=p_id)[12]
    return link_name.startswith(b'palm_')


def get_genome_link_indices(body_id, p_id) -> list[tuple]:
    '''
    Returns a list of tuples with the format ((finger_index, phalanx_index), link_index)
    '''

    # Update simulation state
    p.stepSimulation(physicsClientId=p_id)

    index_list = []
    Indices = namedtuple('Indices', ('genome_index', 'link_index'))
    for i in range(p.getNumJoints(body_id, physicsClientId=p_id)):
        joint_info = p.getJointInfo(body_id, i, physicsClientId=p_id)
        link_name = joint_info[12]
        if link_name.startswith(b'finger_'):
            link_name_ls = link_name.split(b'_')
            indices = Indices((int(link_name_ls[2]), int(link_name_ls[3])), i)
            index_list.append(indices)
    return index_list


def get_distance_of_bodies(body_a_id, body_b_id, link_index, p_id) -> float:
    '''
    Calculates the distance between two PyBullet bodies, from link_index
    of the first body.

    Args:
        body_a_id (int): robot hand with fingers added
        body_b_id (int): target object to be picked up
        link_index (int): link index of a phalanx on the robot hand
        p_id (int): PyBullet physicsClientID
    '''
    assert (
        isinstance(body_a_id, int)
        and isinstance(body_b_id, int)
        and isinstance(link_index, int)
    )

    # Update simulation state
    p.stepSimulation(physicsClientId=p_id)

    try:
        # phalanx_pos = p.getLinkState(body_a_id, link_index, physicsClientId=p_id)[0]
        phalanx_pos = p.getAABB(body_a_id, link_index)[0]
        target_pos = p.getBasePositionAndOrientation(body_b_id, physicsClientId=p_id)[0]
    except Exception as e:
        raise RuntimeError(
            f'{e} raised. Are you sure you are cleaning up connected servers after simulations?'
        )

    # result = p.rayTest(phalanx_pos, target_pos, physicsClientId=p_id)
    try:
        result = p.getClosestPoints(body_a_id, body_b_id, 1000, link_index)[0][6]
        distance = distance_between_coordinates(phalanx_pos, result)
        # distance = 0 if distance == 0 else 1
    except:
        distance = 0

    return distance


def check_collisions(
    body_a_id, body_b_id, body_c_id, link_index, p_id
) -> tuple[Literal[0, 1], Literal[0, 1]]:
    '''
    Checks for collision between body_a - body_b and body_a - body_c.
    The link to be checked in body_a is specified by link_index
    '''

    # Update simulation state
    p.stepSimulation(physicsClientId=p_id)

    target_contact_points = p.getContactPoints(
        body_a_id, body_b_id, link_index, physicsClientId=p_id
    )
    obstacle_contact_points = p.getContactPoints(
        body_a_id, body_c_id, link_index, physicsClientId=p_id
    )

    target_contact = int(len(target_contact_points) > 0)  # int(True) -> 1
    obstacle_contact = int(len(obstacle_contact_points) > 0)

    return target_contact, obstacle_contact


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
    if p.getNumJoints(body_id, physicsClientId=p_id) < c.FINGER_START_INDEX:
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
    else:
        raise TypeError('joint_index must me either an integer or list of integers')

    # Step simulation so that the rotations can be applied to the joints
    for _ in range(100):
        p.stepSimulation()


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
            p.stepSimulation(physicsClientId=p_id)
    else:
        new_target_pos = target_pos / steps


def check_in_target_box(body_ids: list[int], target_box_id: int, p_id: int) -> bool:
    '''Checks if a list of target objects is in the target dropping box'''

    assert isinstance(body_ids, list)

    def get_pos(id):
        return p.getBasePositionAndOrientation(id, physicsClientId=p_id)[0]

    positions = [get_pos(body_id) for body_id in body_ids]

    try:
        # Get target box bounding locations
        box_aabb = p.getAABB(target_box_id, physicsClientId=p_id)
    except:  # Target object no longer in the sim
        return False

    # # Get box boundaries
    t_min_x = box_aabb[0][0]
    t_max_x = box_aabb[1][0]
    t_min_y = box_aabb[0][1]
    t_max_y = box_aabb[1][1]

    return tuple(
        (x > t_min_x and x < t_max_x and y > t_min_y and y < t_max_y)
        for x, y, _ in positions
    )