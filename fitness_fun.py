import pybullet as p
import numpy as np
from typing import Literal

from specimen import Phalanx

from helpers.pybullet_helpers import check_in_target_box


class FitnessFunction:
    @staticmethod
    def get_picking_performance(
        target_obj_ids: list[int], target_box_id: int, p_id: int
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
        assert isinstance(target_obj_ids, list)

        if len(target_obj_ids) == 0:
            return 0

        return sum(check_in_target_box(target_obj_ids, target_box_id, p_id))

    @staticmethod
    def get_phalanx_fitness(grabbing_performance: tuple[float, float, float]) -> float:
        '''
        Returns phalanx fitness calculated from grabbing performance normalized values
        It takes target collision and grabbing outputs directly and applies inverse
        proportion to distance and obstacle collision.
        '''

        distance, t_collision, o_collision = grabbing_performance

        distance = 1 / np.square((1 + distance))  # when d = 0, d fitness = 1
        o_collision = 1 / (1 + o_collision)

        return (distance + t_collision + o_collision) / 3

    @staticmethod
    def get_total_fitness(phalanges: list[Phalanx], picking_performance: int):
        '''
        Returns total specimen fitness calculated from grabbing performance and
        the object picking performance.
        '''

        assert isinstance(phalanges, list)
        assert len(phalanges) > 0

        grabbing_performances = [
            FitnessFunction.get_phalanx_fitness(p.get_performance()) for p in phalanges
        ]
        grabbing_performances = sum(grabbing_performances) / len(phalanges)

        # Manually setting 4 here to normalize the fitness
        # This is because we have three inputs and three target objects
        # ToDo: move number of inputs and number of target object to config file
        return (grabbing_performances + picking_performance) / 4

    @staticmethod
    def get_fitness_map(phalanges: list[Phalanx], shape: tuple) -> np.ndarray:
        '''
        Returns a fitness map for the list of phalanges based on genome shape

        Args:
            phalanges: list of phalanx states
            shape: shape of genome encoding the fitness map should imitate
        '''

        fitness_map = np.zeros(shape[:2])
        for phalanx in phalanges:
            f_i = phalanx.finger_index
            j_i = phalanx.phalanx_index
            fitness_map[f_i, j_i] = FitnessFunction.get_phalanx_fitness(
                phalanx.get_performance()
            )
        print('#######################################')
        print(fitness_map)
        print('#######################################')
        return fitness_map
