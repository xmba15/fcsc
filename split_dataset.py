#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import random


_DIRECTORY_ROOT = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
_DATA_PATH = os.path.join(_DIRECTORY_ROOT, "data")


def split(annotations, _seed=448):
    train_ann = {}
    val_ann = {}

    random.seed(_seed)
    index = list(range(len(annotations)))
    random.shuffle(index)
    num_train = round(len(annotations)*0.7)
    train_index = index[:num_train]
    val_index = index[num_train:]

    for each in train_index:
        key = list(annotations.keys())[each]
        train_ann[key] = annotations[key]
    for each in val_index:
        key = list(annotations.keys())[each]
        val_ann[key] = annotations[key]

    return train_ann, val_ann


def main():
    dataset_dir = os.path.join(_DATA_PATH, "images")
    annotations = json.load(open(os.path.join(_DATA_PATH, "via_region_data.json")))
    train_ann, val_ann, test_ann = split(annotations)


if __name__ == '__main__':
    main()
