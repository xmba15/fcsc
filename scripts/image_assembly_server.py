#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import rospy


# ros related import
import message_filters
import cv_bridge
from sensor_msgs.msg import Image, CameraInfo
from fcsc.srv import ObjectMaskSrv, ObjectMaskSrvResponse
from fcsc.srv import ImageAssembly, ImageAssemblyResponse
from fcsc.msg import ObjectMask, ObjectMaskArray


class ImageAssemblyServer():
    def __init__(self):

        self.has_data = False
        self.resp_ = ImageAssemblyResponse()
        self.s = rospy.Service("/fcsc/image_assembly_server", ImageAssembly, self.handle_image_assembly)

        self.camera_info_topic = rospy.get_param("~camera_info_topic",
                                                 "/camera/depth/camera_info")
        self.rgb_img_topic = rospy.get_param("~rgb_img_topic",
                                             "/camera/color/image_raw")
        self.depth_img_topic = rospy.get_param("~depth_img_topic",
                                               "/camera/depth/image_rect_raw")
        self.approximate_sync = rospy.get_param("~approximate_sync", True)
        self.slop = rospy.get_param("~slop", 0.1)

        self.queue_size = rospy.get_param("~queue_size", 100)
        self.buff_size = rospy.get_param("~buff_size", 2**28)
        self.camera_info_msg = None

        self.camera_info_sub = rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_cb,
                                            queue_size=1, buff_size=self.buff_size)

        self.subscribers = [
            message_filters.Subscriber(
                self.rgb_img_topic, Image,
                queue_size=1, buff_size=self.buff_size),
            message_filters.Subscriber(
                self.depth_img_topic, Image,
                queue_size=1, buff_size=self.buff_size),
        ]
        if self.approximate_sync:
            sync = message_filters.ApproximateTimeSynchronizer(
                self.subscribers, queue_size=self.queue_size, slop=self.slop)
        else:
            sync = message_filters.TimeSynchronizer(
                self.subscribers, queue_size=self.queue_size)
        sync.registerCallback(self.callback)

    def camera_info_cb(self, msg):
        self.camera_info_msg = msg
        self.camera_info_sub.unregister()

    def callback(self, rgb_msg, depth_msg):
        if not self.camera_info_msg:
            rospy.logwarn("Camera info has not been received")
            return

        self.resp_.header.frame_id = "image_assembly"
        self.resp_.header.stamp = rospy.Time.now()
        self.resp_.image = rgb_msg
        self.resp_.depth = depth_msg
        self.info = self.camera_info_msg

        self.has_data = True

    def handle_image_assembly(self, req):
        if self.has_data:
            return self.resp_
        else:
            return ImageAssemblyResponse()


def main():
    rospy.init_node("image_assembly")
    ImageAssemblyServer()
    rospy.spin()


if __name__ == '__main__':
    main()
