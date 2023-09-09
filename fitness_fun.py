import pybullet as p
import numpy as np
from typing import Literal

from helpers.pybullet_helpers import check_in_target_box


class FitnessFunction:
    @staticmethod
    def get_picking_performance(
        target_obj_id: int, target_box_id: int, p_id: int
    ) -> Literal[1, 0]:
        '''
        Checks if the target object is in the desired target location

        Args:
            target_obj_id: target object id
            target_box_id: target drop box id
            p_id: connected physics client id

        Returns:
            1: object in box
            0: object not in box
        '''
        if check_in_target_box(target_obj_id, target_box_id, p_id):
            return 1
        return 0

    @staticmethod
    def get_grabbing_performance(
        phalanges: list,
    ) -> tuple[float, float, float]:
        '''
        Returns a normalized (between 0 and 1) total distance, total
        number of target collisions and total number of obstacle collisions
        for all phalanges.

        Args:
            phalanges: list of phalanx objects
        '''
        assert isinstance(phalanges, list)
        assert len(phalanges) > 0

        len_p = len(phalanges)

        distances = [p.get_performance()[0] for p in phalanges]
        t_collision = [p.get_performance()[1] for p in phalanges]
        o_collision = [p.get_performance()[2] for p in phalanges]

        return (
            sum(distances) / len_p,
            sum(t_collision) / len_p,
            sum(o_collision) / len_p,
        )

    @staticmethod
    def get_total_fitness(
        grabbing_performance: tuple[float, float, float],
        picking_performance: int,
    ) -> float:
        '''
        Returns total fitness calculated from grabbing performance normalized values
        and the object picking performance.
        It takes target collision and grabbing outputs directly and applies inverse
        proportion to distance and obstacle collision.
        '''

        distance, t_collision, o_collision = grabbing_performance

        distance = 1 / (1 + distance)  # when d = 0, d fitness = 1
        o_collision = 1 / (1 + o_collision)

        # Manually setting 6 here to normalize the fitness
        # This is because we have three inputs and three target objects
        # ToDo: move number of inputs and number of target object to config file
        return (distance + t_collision + o_collision + picking_performance) / 6
