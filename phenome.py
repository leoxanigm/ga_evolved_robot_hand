import os
import numpy as np
import pickle
from copy import deepcopy

import constants as c
from constants import GeneDesc, Limits, ROBOT_HAND
from genome import BrainGenome, FingersGenome
from helpers.math_functions import normalize
from helpers.misc_helpers import get_robot_palm_dims


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

        dimensions = get_robot_palm_dims(ROBOT_HAND)
        palm_dim_x = dimensions['x'] / 2
        palm_dim_y = dimensions['y'] / 2

        for i in range(phenome_matrix.shape[0]):  # loop through fingers
            finger = phenome_matrix[i]

            if np.all(finger == 0):
                # No need to continue looping as the rest of array elements will be zero
                break

            parent_dim_z = dimensions['z']  # height of the palm link
            parent_axes = None

            for j in range(phenome_matrix.shape[1]):  # loop through phalanges
                phalanx = phenome_matrix[i][j]

                if np.all(phalanx == 0):
                    break

                assert len(phalanx) == len(GeneDesc)

                # Modify link dimensions
                FingersPhenome.__set_link_dimensions(phalanx)

                # Modify joint origin
                new_parent_dim_z = FingersPhenome.__set_joint_origin(
                    phalanx, palm_dim_x, palm_dim_y, parent_dim_z, j
                )
                parent_dim_z = new_parent_dim_z

                # Modify joint axis
                parent_axes = FingersPhenome.__set_joint_axis(
                    phalanx, palm_dim_x, palm_dim_y, parent_axes
                )

        return phenome_matrix

    @staticmethod
    def __set_link_dimensions(phalanx):
        '''
        Limit the lower and upper dimensions of the phalanx.
        Takes the values from predefined constants.
        '''

        phalanx[GeneDesc.DIM_X] = normalize(
            phalanx[GeneDesc.DIM_X], Limits.DIM_X_LOWER, Limits.DIM_X_UPPER
        )
        phalanx[GeneDesc.DIM_Y] = normalize(
            phalanx[GeneDesc.DIM_Y], Limits.DIM_Y_LOWER, Limits.DIM_Y_UPPER
        )
        phalanx[GeneDesc.DIM_Z] = normalize(
            phalanx[GeneDesc.DIM_Z], Limits.DIM_Z_LOWER, Limits.DIM_Z_UPPER
        )

    @staticmethod
    def __set_joint_origin(phalanx, palm_dim_x, palm_dim_y, parent_dim_z, index):
        '''
        Places links at the edge of their respective parents by setting their
        joint origin to their parents' lengths.
        For the posterior phalanx, all x, y and z attributes are set to place the fingers
        at the edges of the palm, based on their pre-populated  values.
        For the rest of the phalanges, x and y values are set to 0.
        '''

        if index == 0:
            # Limit the range of joint origin x and y to the area of the palm
            phalanx_dim_x = normalize(
                phalanx[GeneDesc.JOINT_ORIGIN_X], -palm_dim_x, palm_dim_x
            )
            phalanx_dim_y = normalize(
                phalanx[GeneDesc.JOINT_ORIGIN_Y], -palm_dim_y, palm_dim_y
            )

            # Get sign of  location to get its quadrant
            x_quadrant = np.sign(phalanx_dim_x)
            y_quadrant = np.sign(phalanx_dim_y)

            # Get intersection point between palm edges and the  point in the
            # palm. This moves the finger attachments to the edges while
            # maintaining ness and order. The ness is chosen by
            # the value of joint origin in z axis. So for the same genome encoding,
            # we get the same attachment point each time.
            if phalanx[GeneDesc.JOINT_ORIGIN_Z] < 0.5:
                phalanx[GeneDesc.JOINT_ORIGIN_X] = phalanx_dim_x
                phalanx[GeneDesc.JOINT_ORIGIN_Y] = palm_dim_y * y_quadrant
            else:
                phalanx[GeneDesc.JOINT_ORIGIN_X] = palm_dim_x * x_quadrant
                phalanx[GeneDesc.JOINT_ORIGIN_Y] = phalanx_dim_y

        else:
            # Other phalanges are attached to the center of their parent phalanx
            phalanx[GeneDesc.JOINT_ORIGIN_X] = 0
            phalanx[GeneDesc.JOINT_ORIGIN_Y] = 0

        # Attach link at the end of parent phalanx
        phalanx[GeneDesc.JOINT_ORIGIN_Z] = parent_dim_z

        return phalanx[GeneDesc.DIM_Z]

    @staticmethod
    def __set_joint_axis(phalanx, palm_dim_x, palm_dim_y, parent_axes=None):
        if parent_axes is None:
            # The axis of rotation for the posterior phalanges should always
            # face the center of the palm. This is to ensure the fingers can
            # close and open

            # Check which edge of palm the phalanx is located
            if np.abs(phalanx[GeneDesc.JOINT_ORIGIN_X]) == np.abs(palm_dim_x):
                edge_sign = np.sign(phalanx[GeneDesc.JOINT_ORIGIN_X])
                phalanx[GeneDesc.JOINT_AXIS_X] = 0
                phalanx[GeneDesc.JOINT_AXIS_Y] = edge_sign
            else:
                edge_sign = np.sign(phalanx[GeneDesc.JOINT_ORIGIN_Y])
                phalanx[GeneDesc.JOINT_AXIS_X] = -1 * edge_sign
                phalanx[GeneDesc.JOINT_AXIS_Y] = 0
        else:
            # For the rest of the phalanges, set the axis of rotation same
            # as the posterior phalanx
            phalanx[GeneDesc.JOINT_AXIS_X], phalanx[GeneDesc.JOINT_AXIS_Y] = parent_axes

        # For now we doesn't need the phalanges to revolve around the z axis
        phalanx[GeneDesc.JOINT_AXIS_Z] = 0

        return [phalanx[GeneDesc.JOINT_AXIS_X], phalanx[GeneDesc.JOINT_AXIS_Y]]


class BrainPhenome:
    def __init__(self, brain_genome: BrainGenome = None):
        '''The brain decides the rotation direction of a phalanx: either +ve,
        -ve or no rotation. It takes a tuple of binary inputs for each phalanx
        that have structure (distance [1/0], collision with target [1/0]) and
        (collision with obstacle [1/0]). It multiples the inputs for all
        phalanges with a convolution matrix of a phalanx and outputs rotation
        direction for the said phalanx.'''

        self.brain_genome = brain_genome

    def trajectories(self, input: np.ndarray) -> np.ndarray:
        '''
        Performs convolution transform on a set of inputs based on convolution
        matrix genome encoding.

        Args:
            input: an input 3D array. The inner array has the structure:
                [distance (1/0), collision-target (1/0), collision-obstacle (1/0))]
        '''

        if self.brain_genome is None:
            raise RuntimeError(
                'Brain genome must be loaded before calculating trajectories'
            )

        assert isinstance(input, np.ndarray)
        assert input.shape == self.brain_genome.shape[2:]

        # Perform convolution transform
        output = input * self.brain_genome
        output = np.sum(output, axis=(2, 3, 4))

        # Map the results to -1, 0 and 1
        map_factor = 0.2
        output[output < -map_factor] = -1
        output[output > map_factor] = 1
        output[(output >= -map_factor) & (output <= map_factor)] = 0

        return output

    def save_genome(self, file_path: str):
        '''Saves model parameters to disk'''

        if self.brain_genome is None:
            raise RuntimeError('Brain genome needs to be loaded before saving model')

        with open(file_path, 'wb') as f:
            pickle.dump(self.brain_genome, f)

    def load_genome(self, file_path: str):
        '''Loads model parameters from disk'''
        if not os.path.exists(file_path):
            raise FileNotFoundError('Could not load model from the specified path')

        with open(file_path, 'rb') as f:
            brain_genome_array = pickle.load(f)
            self.brain_genome = brain_genome_array

    @property
    def genome(self):
        '''Returns the bran genome array'''

        if self.brain_genome is None:
            raise RuntimeError(
                'Genome array needs to be loaded before it can be accessed'
            )

        return self.brain_genome

    @genome.setter
    def genome(self, genome_array):
        '''Sets a new genome brain array'''

        assert isinstance(genome_array, np.ndarray)

        self.brain_genome = genome_array
