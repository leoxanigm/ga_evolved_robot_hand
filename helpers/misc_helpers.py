import os
import xml.etree.ElementTree as ET
from typing import TypedDict
import csv

from constants import TRAINING_DIR


class Dims(TypedDict):
    x: float
    y: float
    z: float


def get_robot_palm_dims(robot_hand_file: str) -> Dims:
    '''
    Returns a dictionary of base robots hand's palm dimensions

    Returns:
        {x: , y: , z: }
    '''
    assert robot_hand_file is not None

    if not os.path.exists(robot_hand_file):
        raise FileNotFoundError('Can not read the robot base urdf file.')

    robot = ET.parse(robot_hand_file)
    root = robot.getroot()
    palm_link_size = root.find('link[@name="palm_link"]/visual/geometry/box')

    try:
        palm_link_size = palm_link_size.attrib['size']
        size_arr = [float(d) for d in palm_link_size.split(' ')]
    except Exception as e:
        print(e)
        size_arr = [0.0, 0.0, 0.1]  # fall back incase of error


    return {
        'x': size_arr[0],
        'y': size_arr[1],
        'z': size_arr[2],
    }


def write_csv(file_path: str, data_row: list[float]):
    # assert os.path.exists(file_path)

    try:
        with open(file_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(data_row)
        return True
    except:
        return False


def clear_dir(dir_path=TRAINING_DIR):
    '''
    Deletes all files in the path specified. Helps to clear training and test files.
    Source: https://www.tutorialspoint.com/How-to-delete-all-files-in-a-directory-with-Python

    Args:
        dir_path (str): directory. Defaults to training path specified in constants.py
    '''

    files = os.listdir(dir_path)
    for file in files:
        file_path = os.path.join(dir_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
