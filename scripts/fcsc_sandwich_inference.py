#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import random
import numpy as np
import skimage
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from fcsc_sandwich import FCSCSandwichConfig, FCSCSandwichDataset


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
sys.path.append(ROOT_DIR)
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log


MODEL_DIR = os.path.join(ROOT_DIR, "models")
FCSCSANDWICH_WEIGHTS_PATH = os.path.join(MODEL_DIR, "mask_rcnn_fcscsandwich_0030.h5")
FCSCSANDWICH_DIR = os.path.join(ROOT_DIR, "data", "sandwich_data")
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
TEST_IMAGE = os.path.join(IMAGE_DIR, "sandwich.png")
if tf.test.is_gpu_available():
    DEVICE = "/gpu:0"
else:
    DEVICE = "/cpu:0"


class InferenceConfig(FCSCSandwichConfig):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


def get_original(ori_h, ori_w, config, image, mask, rois):
    h, w = image.shape[:2]
    top_pad = (config.IMAGE_MAX_DIM - ori_h) / 2
    left_pad = (config.IMAGE_MAX_DIM - ori_w) / 2
    image = image[top_pad:top_pad+ori_h, left_pad:left_pad+ori_w]
    mask = mask[top_pad:top_pad+ori_h, left_pad:left_pad+ori_w]
    rois = np.apply_along_axis(lambda x: x - np.array([top_pad, left_pad, top_pad, left_pad]), 1, rois)
    return image, mask, rois


def main(test_val=True):
    config = InferenceConfig()
    config.display()

    dataset = FCSCSandwichDataset()
    dataset.load_fcscsandwich(FCSCSANDWICH_DIR, "val")
    dataset.prepare()
    print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)
        model.load_weights(FCSCSANDWICH_WEIGHTS_PATH, by_name=True)

        h, w = [480, 640]
        if test_val:
            image_id = random.choice(dataset.image_ids)
            image, _, _, _, _ = modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
            info = dataset.image_info[image_id]
            print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                               dataset.image_reference(image_id)))
        else:
            image = skimage.io.imread(TEST_IMAGE)
            h, w = image.shape[:2]
            if image.ndim != 3:
                image = skimage.color.gray2rgb(image)
                # If has an alpha channel, remove it for consistency
                if image.shape[-1] == 4:
                    image = image[..., :3]
            image, _, _, _, _ = utils.resize_image(
                image,
                min_dim=config.IMAGE_MIN_DIM,
                min_scale=config.IMAGE_MIN_SCALE,
                max_dim=config.IMAGE_MAX_DIM,
                mode=config.IMAGE_RESIZE_MODE)

        results = model.detect([image], verbose=1)
        r = results[0]
        image, r['masks'], r['rois'] = get_original(h, w, config, image, r['masks'], r['rois'])
        ax = get_ax(1)
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    dataset.class_names, r['scores'], ax=ax,
                                    title="Predictions", show_mask=True)
        # plt.show()
        plt.savefig(os.path.join(IMAGE_DIR, "mask_result.jpg"))
        plt.close()
        del model


    result_img = skimage.io.imread(os.path.join(IMAGE_DIR, "mask_result.jpg"))
    skimage.io.imshow(result_img)
    skimage.io.show()


if __name__ == '__main__':
    main(test_val=False)
