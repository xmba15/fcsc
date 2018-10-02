#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import random
from shutil import copyfile


_DIRECTORY_ROOT = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
_ORIGINAL_DATA_PATH = os.path.join(_DIRECTORY_ROOT, "sandwich_bag1")
_DATA_PATH = os.path.join(_DIRECTORY_ROOT, "sandwich_data")
_DATA_PATH_TRAIN = os.path.join(_DATA_PATH, "train")
_DATA_PATH_VAL = os.path.join(_DATA_PATH, "val")


def main():
    for _dir in [_DATA_PATH, _DATA_PATH_TRAIN, _DATA_PATH_VAL]:
        if not os.path.exists(_dir):
            os.makedirs(_dir)
    _image_list = [f for f in os.listdir(_ORIGINAL_DATA_PATH) if f.endswith(".png")]
    random.seed(100)
    random.shuffle(_image_list)
    train_length = int(len(_image_list) * 0.8)
    train_list = _image_list[:train_length]
    val_list = _image_list[train_length:]
    for _image in train_list:
        copyfile(os.path.join(_ORIGINAL_DATA_PATH, _image), os.path.join(_DATA_PATH_TRAIN, _image))

    for _list, _dir in zip([train_list, val_list], [_DATA_PATH_TRAIN, _DATA_PATH_VAL]):
        [copyfile(os.path.join(_ORIGINAL_DATA_PATH, _image), os.path.join(_dir, _image)) for _image in _list]

if __name__ == '__main__':
    main()
