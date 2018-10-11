#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import cv2
# ros related import
import cv_bridge
from sensor_msgs.msg import Image
from fcsc.srv import ObjectMaskSrv, ObjectMaskSrvResponse
from fcsc.msg import ObjectMask, ObjectMaskArray
import rospy


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
sys.path.append(ROOT_DIR)
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
TEST_IMAGE = os.path.join(IMAGE_DIR, "sandwich.png")


class FCSCSandwichClient():
    """
    test fcsc sandwich server

    """
    def __init__(self, img_path=None):
        self.br = cv_bridge.CvBridge()
        self.img_path = img_path
        self.img_msg = None
        self.convert_img_msg()

    def convert_img_msg(self):
        if self.img_path:
            img = cv2.imread(self.img_path)
            self.img_msg = self.br.cv2_to_imgmsg(img, encoding="bgr8")

    def client(self):
        rospy.wait_for_service("/fcsc/fcsc_sandwich_server")
        try:
            fcsc_sandwich = rospy.ServiceProxy(
                "/fcsc/fcsc_sandwich_server", ObjectMaskSrv)
            resp = fcsc_sandwich(self.img_msg)
            rospy.logdebug("Done with fcsc sandwich parsing")
            return resp
        except rospy.ServiceException, e:
            rospy.logerr("Service call failed: %s" %e)


def main():
    fcsc_sandwich_client = FCSCSandwichClient(img_path=TEST_IMAGE)
    client = fcsc_sandwich_client.client()


if __name__ == '__main__':
    main()
