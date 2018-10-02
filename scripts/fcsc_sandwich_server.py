#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import numpy as np
import tensorflow as tf
from fcsc_sandwich import FCSCSandwichConfig, FCSCSandwichDataset


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
sys.path.append(ROOT_DIR)
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib


MODEL_DIR = os.path.join(ROOT_DIR, "models")
FCSCSANDWICH_WEIGHTS_PATH = os.path.join(MODEL_DIR, "mask_rcnn_fcscsandwich_0030.h5")
FCSCSANDWICH_DIR = os.path.join(ROOT_DIR, "data", "sandwich_data")
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
if tf.test.is_gpu_available():
    DEVICE = "/gpu:0"
else:
    DEVICE = "/cpu:0"


# ros related import
import cv_bridge
from sensor_msgs.msg import Image
from fcsc.srv import ObjectMaskSrv, ObjectMaskSrvResponse
from fcsc.msg import ObjectMask, ObjectMaskArray
import rospy
from std_msgs.msg import Header


class InferenceConfig(FCSCSandwichConfig):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def get_original(ori_h, ori_w, config, image, mask, rois):
    h, w = image.shape[:2]
    top_pad = (config.IMAGE_MAX_DIM - ori_h) / 2
    left_pad = (config.IMAGE_MAX_DIM - ori_w) / 2
    image = image[top_pad:top_pad+ori_h, left_pad:left_pad+ori_w]
    mask = mask[top_pad:top_pad+ori_h, left_pad:left_pad+ori_w]
    rois = np.apply_along_axis(lambda x: x - np.array([top_pad, left_pad, top_pad, left_pad]), 1, rois)
    return image, mask, rois


class FCSCSandwichServer():
    """
    server for sandwich segmentation network
    """
    def __init__(self):
        self.config = InferenceConfig()
        self.br = cv_bridge.CvBridge()
        self.model_path = FCSCSANDWICH_WEIGHTS_PATH
        self.device = DEVICE
        self.service_queue = 0
        self.net = None

        self.s = rospy.Service("/fcsc/fcsc_sandwich_server", ObjectMaskSrv, self.handle_sandwich_parsing)
        rospy.spin()

    def handle_sandwich_parsing(self, req):
        """
        handle parsing processing
        """
        self.service_queue += 1
        if self.net == None:
            try:
                with tf.device(self.device):
                    self.net = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                                 config=self.config)
                    self.net.load_weights(self.model_path, by_name=True)
            except:
                rospy.logerr("Error, cannot load deep_net to the GPU")
                self.net =None
                self.service_queue -=1
                return ObjectMaskSrvResponse()

        try:
            image = self.br.imgmsg_to_cv2(req.rgb_img, desired_encoding="bgr8")
            h, w = image.shape[:2]
            image, _, _, _, _ = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                min_scale=self.config.IMAGE_MIN_SCALE,
                max_dim=self.config.IMAGE_MAX_DIM,
                mode=self.config.IMAGE_RESIZE_MODE)
            result = self.net.detect([image], verbose=1)
            r = result[0]
            image, r['masks'], r['rois'] = get_original(h, w, self.config, image, r['masks'], r['rois'])

            object_masks = ObjectMaskArray()
            object_masks.header.frame_id = "fcsc_sandwich"
            object_masks.header.stamp = rospy.Time.now()

            for i in range(len(r['rois'])):
                object_mask = ObjectMask()
                object_mask.class_name = "sandwich"
                _mask = r['masks'][:,:,i].astype(np.int8)
                object_mask.mask = self.br.cv2_to_imgmsg(_mask)
                object_masks.object_mask_arr.append(object_mask)

            self.service_queue -= 1
            return ObjectMaskSrvResponse(masks = object_masks)

        except cv_bridge.CvBridgeError as e:
            rospy.logerr("CvBridge exception %s", e)
            self.service_queue -= 1
            return ObjectMaskSrvResponse()


def main():
    rospy.init_node("fcsc_sandwich")
    FCSCSandwichServer()


if __name__ == '__main__':
    main()
