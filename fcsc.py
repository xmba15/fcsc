#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import json
import skimage
import numpy as np
from mrcnn import model as modellib, utils
from mrcnn.config import Config


class FCSCConfig(Config):
    NAME = "fcsc"
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 4
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9


class FCSCDataset(utils.Dataset):
    CLASS_LABEL = ["bento_box", "juicebox", "onigiri", "sandwich"]
    def load_fcsc(self, dataset_dir, subset):
        for i, _class in enumerate(self.CLASS_LABEL):
            self.add_class("fcsc", i, _class)

        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())
        annotations = [a for a in annotations if a['regions']]

        for a in annotations:
            if isinstance(a['regions'], dict):
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
                names = [r['region_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]
                names = [r['region_attributes'] for r in a['regions']]
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "fcsc",
                image_id=a['filename'],
                path=image_path,
                width=width,
                height=height,
                polygons=polygons,
                names=names)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "fcsc":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def load_mask(self, image_id):
        """Generate instance masks for an image."""
        image_info = self.image_info[image_id]
        if image_info["source"] != "fcsc":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        class_names = info["names"]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        class_ids = np.zeros([len(info["polygons"])])
        for i, p in enumerate(class_names):
            class_ids[i] = self.CLASS_LABEL.index(p["name"]) + 1
        class_ids = class_ids.astype(int)

        return mask.astype(np.pool), class_ids


def main():
    pass


if __name__ == '__main__':
    main()
