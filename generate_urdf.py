import xml.etree.ElementTree as ET
import copy
import os
import numpy as np

import constants as c
from constants import GeneDesc
from helpers.math_functions import calculate_cylinder_mass, moments_of_inertia_cylinder

from genome import FingersGenome
from phenome import FingersPhenome


class GenerateURDF:
    @staticmethod
    def generate_robot_fingers(
        fingers_phenome, output_file, robot_arm_file=c.ROBOT_ARM
    ) -> bool:
        '''
        Generates URDF robot file from fingers phenome

        Args:
            fingers_phenome: fingers phenome matrix
            output_file: target file of robot hands with fingers
            robot_arm_file: base robot URDF file

        Returns:
            bool
        '''

        if not os.path.exists(robot_arm_file):
            raise FileNotFoundError('Can not read the robot base URDF file.')

        robot_hand = ET.ElementTree()

        try:
            robot_hand.parse(robot_arm_file)
        except ET.ParseError:
            raise 'Could not parse URDF file, please check that it is properly formatted.'

        xml_root = robot_hand.getroot()

        GenerateURDF.__populate_links_and_joints(fingers_phenome, xml_root)

        try:
            ET.indent(robot_hand)
            robot_hand.write(output_file, encoding='utf-8', xml_declaration=True)
            return True
        except:
            return False

    @staticmethod
    def __populate_links_and_joints(fingers_phenome, xml_root):
        for i in range(len(fingers_phenome)):  # loop through fingers
            finger = fingers_phenome[i]

            if np.all(finger == 0):
                # No need to continue looping are the rest of array elements will be None
                break

            parent = 'palm_link'

            for j in range(len(finger)):  # loop through phalanges
                phalanx = fingers_phenome[i][j]

                if np.all(phalanx == 0):
                    break

                assert len(phalanx) == len(GeneDesc)

                # generate link tag
                link_tag_name = f'finger_link_{i}_{j}'
                link_tag = GenerateURDF.__generate_link_tag(phalanx, link_tag_name)

                # generate joint tag
                joint_tag_parent = parent
                joint_tag_child = link_tag_name
                joint_tag_name = f'finger_joint_{parent}_to_{link_tag_name}'

                joint_tag = GenerateURDF.__generate_joint_tag(
                    phalanx, joint_tag_name, joint_tag_parent, joint_tag_child
                )

                # add link and joint tags
                xml_root.append(link_tag)
                xml_root.append(joint_tag)

                parent = link_tag_name

    @staticmethod
    def __generate_link_tag(phalanx, name):
        # Attributes
        RADIUS = round(phalanx[GeneDesc.RADIUS], 3)
        LENGTH = round(phalanx[GeneDesc.LENGTH], 3)

        link_tag = ET.Element('link', attrib={'name': name})

        # visual tag start
        vis_tag = ET.Element('visual')

        vis_origin_tag = ET.Element(
            'origin',
            attrib={
                'xyz': f'0 0 {-LENGTH / 2}',  # Move origin to top
                'rpy': '0 0 0',
            },
        )

        vis_geom_tag = ET.Element('geometry')
        vis_box_tag = ET.Element(
            'cylinder',
            attrib={
                'radius': f'{RADIUS}',
                'length': f'{LENGTH}',
            },
        )
        vis_geom_tag.append(vis_box_tag)

        vis_tag.append(vis_origin_tag)
        vis_tag.append(vis_geom_tag)
        # visual tag end

        # collision tag start
        col_tag = ET.Element('collision')
        col_geom_tag = copy.deepcopy(vis_geom_tag)
        col_tag.append(col_geom_tag)
        # collision tag end

        # inertial tag start
        iner_tag = ET.Element('inertial')

        link_mass = calculate_cylinder_mass(RADIUS, LENGTH)
        inertial_mass_tag = ET.Element('mass', attrib={'value': f'{link_mass}'})

        mom_intertia = moments_of_inertia_cylinder(RADIUS, link_mass)
        inertial_inertia_tag = ET.Element(
            'inertia',
            attrib={
                'ixx': f'{mom_intertia}',
                'iyy': f'{mom_intertia}',
                'izz': f'{mom_intertia}',
                'ixy': '0',
                'ixz': '0',
                'iyz': '0',
            },
        )

        iner_tag.append(inertial_mass_tag)
        iner_tag.append(inertial_inertia_tag)
        # inertial tag end

        link_tag.append(vis_tag)
        link_tag.append(col_tag)
        link_tag.append(iner_tag)

        return link_tag

    @staticmethod
    def __generate_joint_tag(phalanx, name, parent, child, type='revolute'):
        # Attributes
        JOINT_ORIGIN_X = round(phalanx[GeneDesc.JOINT_ORIGIN_X], 3)
        JOINT_ORIGIN_Y = round(phalanx[GeneDesc.JOINT_ORIGIN_Y], 3)
        JOINT_ORIGIN_Z = round(phalanx[GeneDesc.JOINT_ORIGIN_Z], 3)
        JOINT_AXIS_X = round(phalanx[GeneDesc.JOINT_AXIS_X], 3)
        JOINT_AXIS_Y = round(phalanx[GeneDesc.JOINT_AXIS_Y], 3)
        JOINT_AXIS_Z = round(phalanx[GeneDesc.JOINT_AXIS_Z], 3)

        joint_tag = ET.Element('joint', attrib={'name': str(name), 'type': type})

        joint_origin_tag = ET.Element(
            'origin',
            attrib={
                'xyz': f'{JOINT_ORIGIN_X} {JOINT_ORIGIN_Y} {JOINT_ORIGIN_Z}',
                'rpy': '0 0 0',
            },
        )
        joint_parent_tag = ET.Element('parent', attrib={'link': str(parent)})
        joint_child_tag = ET.Element('child', attrib={'link': str(child)})
        joint_axis_tag = ET.Element(
            'axis',
            attrib={
                'xyz': f'{JOINT_AXIS_X} {JOINT_AXIS_Y} {JOINT_AXIS_Z}',
            },
        )
        # set joint limit from -pi to pi for now
        joint_limit_tag = ET.Element(
            'limit',
            attrib={
                'lower': '-3.142',
                'upper': '3.142',
                'effort': '10',
                'velocity': '10',
            },
        )

        joint_tag.append(joint_origin_tag)
        joint_tag.append(joint_parent_tag)
        joint_tag.append(joint_child_tag)
        joint_tag.append(joint_axis_tag)
        joint_tag.append(joint_limit_tag)

        return joint_tag
