import os
import numpy as np
import pickle
from copy import deepcopy

import constants as c
from constants import GeneDesc, Limits
from helpers.math_functions import normalize


class FingersPhenome:
    @staticmethod
    def genome_to_phenome(finger_genome: np.ndarray) -> np.ndarray:
        '''
        Phenotype description for the fingers.
        This is where encoded genetic information is modified to suit the traits
        we want the fingers to have. For example, each phalanx will be connected
        at the end of its parent. This class handles that modification.

        Args:
            finger_genome:  default fingers genome matrix
        '''

        phenome_matrix = deepcopy(finger_genome)

        num_pos = 8

        # Generate finger position angles from 0 to 315 in radians
        angles = []
        start = 0
        for _ in range(num_pos):
            angles.append(start)
            start += np.pi / 4

        # Convert the angles to x and y positions
        joint_origins = []
        for a in angles:
            # 0.5 is the radius of the palm
            x = round(np.sin(a), 3) * 0.5
            y = round(np.cos(a), 3) * 0.5
            joint_origins.append((x, y))

        # These joint rotation axis (x, y) correspond to position placement
        # they make sure the fingers rotate to/away from the center
        joint_axes = [
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
            (-1, -1),
        ]

        # Track which positions are taken so that we don't place two
        # fingers at the same position
        taken_indices = []

        for i in range(phenome_matrix.shape[0]):  # loop through fingers
            finger = phenome_matrix[i]

            if np.all(finger == 0):
                # No need to continue looping as the rest of array elements will be zero
                break

            joint_orig_z = 0
            parent_axes = None

            finger_axis = int(finger[0, GeneDesc.JOINT_AXIS_X] * num_pos)

            # finger_axis is already taken, search for next empty position
            # while making sure we don't get IndexError
            while finger_axis in taken_indices:
                finger_axis = (finger_axis + 1) % num_pos

            taken_indices.append(finger_axis)

            x_axis, y_axis = joint_axes[finger_axis]
            joint_orig_x, joint_orig_y = joint_origins[finger_axis]

            for j in range(phenome_matrix.shape[1]):  # loop through phalanges
                phalanx = phenome_matrix[i][j]

                if np.all(phalanx == 0):
                    break

                assert len(phalanx) == len(GeneDesc)

                # Modify link dimensions
                FingersPhenome.__set_link_dimensions(phalanx)

                # Modify joint origin
                new_joint_orig_z = FingersPhenome.__set_joint_origin(
                    phalanx, joint_orig_x, joint_orig_y, joint_orig_z, j
                )
                joint_orig_z = new_joint_orig_z

                # Modify joint axis
                parent_axes = FingersPhenome.__set_joint_axis(phalanx, x_axis, y_axis)

        return phenome_matrix

    @staticmethod
    def __set_link_dimensions(phalanx):
        '''
        Limit the lower and upper dimensions of the phalanx.
        Takes the values from predefined constants.
        '''

        phalanx[GeneDesc.RADIUS] = normalize(
            phalanx[GeneDesc.RADIUS], Limits.RADIUS_LOWER, Limits.RADIUS_UPPER
        )
        phalanx[GeneDesc.LENGTH] = normalize(
            phalanx[GeneDesc.LENGTH], Limits.LENGTH_LOWER, Limits.LENGTH_UPPER
        )

    @staticmethod
    def __set_joint_origin(phalanx, joint_orig_x, joint_orig_y, joint_orig_z, index):
        '''
        Places links at the edge of their respective parents by setting their
        joint origin to their parents' lengths.
        For the posterior phalanx, all x, y and z attributes are set to place the fingers
        at the edges of the palm, based on their pre-populated  values.
        For the rest of the phalanges, x and y values are set to 0.
        '''

        if index == 0:
            phalanx[GeneDesc.JOINT_ORIGIN_X] = joint_orig_x
            phalanx[GeneDesc.JOINT_ORIGIN_Y] = joint_orig_y

        else:
            # Other phalanges are attached to the center of their parent phalanx
            phalanx[GeneDesc.JOINT_ORIGIN_X] = 0
            phalanx[GeneDesc.JOINT_ORIGIN_Y] = 0

        # Attach link at the end of parent phalanx
        phalanx[GeneDesc.JOINT_ORIGIN_Z] = -joint_orig_z

        return phalanx[GeneDesc.LENGTH]

    @staticmethod
    def __set_joint_axis(phalanx, x_axis, y_axis):
        phalanx[GeneDesc.JOINT_AXIS_X] = x_axis
        phalanx[GeneDesc.JOINT_AXIS_Y] = y_axis

        # For now we doesn't need the phalanges to revolve around the z axis
        phalanx[GeneDesc.JOINT_AXIS_Z] = 0

        return [phalanx[GeneDesc.JOINT_AXIS_X], phalanx[GeneDesc.JOINT_AXIS_Y]]


class BrainPhenome:
    '''The brain decides the rotation direction of a phalanx: either +ve,
    -ve or no rotation. It takes a tuple of binary inputs for each phalanx
    that have structure (distance [1/0], collision with target [1/0]) and
    (collision with obstacle [1/0]). It multiples the inputs for all
    phalanges with a convolution matrix of a phalanx and outputs rotation
    direction for the said phalanx.'''

    @staticmethod
    def trajectories(brain_genome: np.ndarray, input: np.ndarray) -> np.ndarray:
        '''
        Performs convolution transform on a set of inputs based on convolution
        matrix genome encoding.

        Args:
            input: an input 3D array. The inner array has the structure:
                [distance (1/0), collision-target (1/0), collision-obstacle (1/0))]
        '''

        assert isinstance(input, np.ndarray)
        assert input.shape == brain_genome.shape[2:]

        # Perform convolution transform
        output = input * brain_genome
        output = np.sum(output, axis=(2, 3, 4))

        # Map the results to -1, 0 and 1
        map_factor = 0.2
        output[output < -map_factor] = -1
        output[output > map_factor] = 1
        output[(output >= -map_factor) & (output <= map_factor)] = 0

        return output
