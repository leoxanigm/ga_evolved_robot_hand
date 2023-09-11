import xml.etree.ElementTree as ET
import copy
import os
import numpy as np

from constants import GeneDesc
from helpers.math_functions import calculate_box_mass, moments_of_inertia_box

from genome import FingersGenome
from phenome import FingersPhenome


class GenerateURDF:
    '''
    Generates URDF robot file from fingers phenome

    Args:
        fingers_phenome: fingers genome matrix
    '''

    def __init__(self, fingers_phenome):
        self.fingers_phenome = fingers_phenome

    def generate_robot_fingers(self, robot_hand_file, output_file):
        '''
        Adds finger links and joint to a robot hand

        Args:
            robot_hand: base robot URDF file

        Returns:
            bool
        '''
        if not os.path.exists(robot_hand_file):
            raise FileNotFoundError('Can not read the robot base urdf file.')

        robot_hand = ET.ElementTree()

        try:
            robot_hand.parse(robot_hand_file)
        except ET.ParseError:
            raise 'Could not parse URDF file, please check that it is properly formatted.'
        except:
            raise 'Could not parse URDF file, unknown error.'

        xml_root = robot_hand.getroot()

        self.__populate_links_and_joints(xml_root)

        try:
            ET.indent(robot_hand)
            robot_hand.write(output_file, encoding='utf-8', xml_declaration=True)
            return True
        except:
            return False

    def __populate_links_and_joints(self, xml_root):
        for i in range(len(self.fingers_phenome)):  # loop through fingers
            finger = self.fingers_phenome[i]

            if np.all(finger == 0):
                # No need to continue looping are the rest of array elements will be None
                break

            parent = 'palm_link'

            for j in range(len(finger)):  # loop through phalanges
                phalanx = self.fingers_phenome[i][j]

                if np.all(phalanx == 0):
                    break

                assert len(phalanx) == len(GeneDesc)

                # generate link tag
                link_tag_name = f'finger_link_{i}_{j}'
                link_tag = self.__generate_link_tag(phalanx, link_tag_name)

                # generate joint tag
                joint_tag_parent = parent
                joint_tag_child = link_tag_name
                joint_tag_name = f'finger_joint_{parent}_to_{link_tag_name}'

                joint_tag = self.__generate_joint_tag(
                    phalanx, joint_tag_name, joint_tag_parent, joint_tag_child
                )

                # add link and joint tags
                xml_root.append(link_tag)
                xml_root.append(joint_tag)

                parent = link_tag_name

    def __generate_link_tag(self, gene_data, name):
        # Attributes
        DIM_X = round(gene_data[GeneDesc.DIM_X], 3)
        DIM_Y = round(gene_data[GeneDesc.DIM_Y], 3)
        DIM_Z = round(gene_data[GeneDesc.DIM_Z], 3)

        link_tag = ET.Element('link', attrib={'name': name})

        # visual tag start
        vis_tag = ET.Element('visual')

        vis_origin_tag = ET.Element(
            'origin',
            attrib={
                'xyz': f'0 0 {DIM_Z / 2}', # 
                'rpy': '0 0 0',
            },
        )

        vis_geom_tag = ET.Element('geometry')
        vis_box_tag = ET.Element(
            'box',
            attrib={
                'size': f'{DIM_X} {DIM_Y} {DIM_Z}',
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

        link_mass = calculate_box_mass(
            DIM_X,
            DIM_Y,
            DIM_Z,
        )
        inertial_mass_tag = ET.Element('mass', attrib={'value': f'{link_mass}'})

        (
            link_inertia_ixx,
            link_inertia_iyy,
            link_inertia_izz,
        ) = moments_of_inertia_box(
            DIM_X,
            DIM_Y,
            DIM_Z,
            link_mass,
        )
        inertial_inertia_tag = ET.Element(
            'inertia',
            attrib={
                'ixx': '0.001',
                'iyy': '0.001',
                'izz': '0.001',
                'ixy': '0',
                'ixz': '0',
                'iyz': '0',
            },
        )
        # inertial_inertia_tag = ET.Element(
        #     'inertia',
        #     attrib={
        #         'ixx': f'{link_inertia_ixx}',
        #         'iyy': f'{link_inertia_iyy}',
        #         'izz': f'{link_inertia_izz}',
        #         'ixy': '0',
        #         'ixz': '0',
        #         'iyz': '0',
        #     },
        # )

        iner_tag.append(inertial_mass_tag)
        iner_tag.append(inertial_inertia_tag)
        # inertial tag end

        link_tag.append(vis_tag)
        link_tag.append(col_tag)
        link_tag.append(iner_tag)

        return link_tag

    def __generate_joint_tag(self, gene_data, name, parent, child, type='revolute'):
        # Attributes
        JOINT_ORIGIN_X = round(gene_data[GeneDesc.JOINT_ORIGIN_X], 3)
        JOINT_ORIGIN_Y = round(gene_data[GeneDesc.JOINT_ORIGIN_Y], 3)
        JOINT_ORIGIN_Z = round(gene_data[GeneDesc.JOINT_ORIGIN_Z], 3)
        JOINT_AXIS_X = round(gene_data[GeneDesc.JOINT_AXIS_X], 3)
        JOINT_AXIS_Y = round(gene_data[GeneDesc.JOINT_AXIS_Y], 3)
        JOINT_AXIS_Z = round(gene_data[GeneDesc.JOINT_AXIS_Z], 3)

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