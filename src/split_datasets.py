import os
import numpy as np
import shutil
import click
from pathlib import Path
import logging
from typing import Tuple


def make_wili_split(data_path, train_val):
    """
    Объединяем в один файл train и test,
    потом уже делим сообразно split_size,
    и записываем в новые файлы

    """

    x_train = open(data_path + "/x_train.txt")
    y_train = open(data_path + '/y_train.txt')
    x_test = open(data_path + "/x_test.txt")
    y_test = open(data_path + "/y_test.txt")
    x_train_split = open(data_path + '/x_train_split.txt', 'w')
    y_train_split = open(data_path + '/y_train_split.txt', 'w')
    x_valid_split = open(data_path + '/x_valid_split.txt', 'w')
    y_valid_split = open(data_path + '/y_valid_split.txt', 'w')
    x_test_split = open(data_path + '/x_test_split.txt', 'w')
    y_test_split = open(data_path + '/y_test_split.txt', 'w')

    x_train_data = x_train.readlines()
    y_train_data = y_train.readlines()
    x_test_data = x_test.readlines()
    y_test_data = y_test.readlines()

    x_train_data += x_test_data
    y_train_data += y_test_data


    train_percent = train_val[0]
    validate_percent= train_val[1]
    array = np.array(range(len(x_train_data)))
    perm = np.random.permutation(array)
    m = len(array)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = array[perm[:train_end]]
    validate = array[perm[train_end:validate_end]]
    test = array[perm[validate_end:]]

    for i in range(len(x_train_data)):
        if i in train:
            x_train_split.write(x_train_data[i])
            y_train_split.write(y_train_data[i])
        elif i in validate:
            x_valid_split.write(x_train_data[i])
            y_valid_split.write(y_train_data[i])
        else:
            x_test_split.write(x_train_data[i])
            y_test_split.write(y_train_data[i])


    x_train.close()
    y_train.close()
    x_test.close()
    y_test.close()
    x_train_split.close()
    y_train_split.close()
    x_valid_split.close()
    y_valid_split.close()
    x_test_split.close()
    y_test_split.close()



def make_taiga_split(data_path: str, path_to_move: str, train_val: Tuple):
    os.mkdir(path_to_move + 'taiga_train')
    os.mkdir(path_to_move + 'taiga_valid')
    os.mkdir(path_to_move + 'taiga_test')
    train_percent = train_val[0]
    validate_percent = train_val[1]
    files = os.listdir(data_path)
    m = len(files)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = files[:train_end]
    valid = files[train_end:validate_end]
    test = files[validate_end:]

    for item in train:
        shutil.copy(data_path + item, path_to_move + 'taiga_train')
    for item in valid:
        shutil.copy(data_path + item, path_to_move + 'taiga_valid')
    for item in test:
        shutil.copy(data_path + item, path_to_move + 'taiga_test')


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('train_percent', type=click.FLOAT)
@click.argument('valid_percent', type=click.FLOAT)
def main(input_filepath, output_filepath, train_percent, valid_percent):
    """
    Splits data to train-validation-test

    """

    make_taiga_split(input_filepath, output_filepath, (train_percent, valid_percent))
    make_wili_split(output_filepath, (train_percent, valid_percent))

    logger = logging.getLogger(__name__)
    test_percent = 1 - valid_percent - train_percent
    logger.info(f'split dataset to train-validation-test in proportion: {train_percent}-{valid_percent}-{test_percent}')


if __name__ == "__main__":
    main()
