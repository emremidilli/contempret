import json

import numpy as np

import os


def read_npy(directory: str, file_name: str) -> np.array:
    '''
    reads data from a numpy file.
    '''
    source_dir = os.path.join(
        directory,
        file_name)

    np_array = np.load(source_dir)

    return np_array


def save_npy(arr: np.array, directory: str, file_name: str) -> None:
    '''Saves the a numpy file.'''

    target_dir = directory

    os.makedirs(target_dir, exist_ok=True)

    np.save(os.path.join(target_dir, file_name), arr)


def read_json(directory: str, file_name:str):
    '''
    reads data from a json file.
    '''
    source_dir = os.path.join(
        directory,
        file_name)
    try:
        with open(source_dir, 'r') as file:
            data = json.load(file)
    except:
        data = None

    return data


def save_json(data: dict, directory: str, file_name:str) -> None:
    '''
    saves json file.
    '''
    target_dir = directory
    os.makedirs(target_dir, exist_ok=True)

    with open(os.path.join(target_dir, file_name), 'w') as file:
        json.dump(data, file, indent=4)
