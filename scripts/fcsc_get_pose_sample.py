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
from fcsc.msg import ObjectMask, ObjectMaskArray


class FCSCSandwichPose():
    def __init__(self):
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
        self.br = cv_bridge.CvBridge()
        self.camera_info_msg = None
        self.rgb_img = None
        self.depth_img = None

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

        # self.rgb_img_sub = rospy.Subscriber(self.rgb_img_topic, Image, self.rgb_img_cb,
        #                                     queue_size=1, buff_size=self.buff_size)
        # self.depth_img_sub = rospy.Subscriber(self.depth_img_topic, Image, self.depth_img_cb,
        #                                     queue_size=1, buff_size=self.buff_size)

    def camera_info_cb(self, msg):
        self.camera_info_msg = msg
        self.fx = self.camera_info_msg.K[0]
        self.fy = self.camera_info_msg.K[4]
        self.cx = self.camera_info_msg.K[2]
        self.cy = self.camera_info_msg.K[5]
        self.camera_info_sub.unregister()

    def callback(self, rgb_mgs, depth_msg):
        if not self.camera_info_msg:
            rospy.logwarn("Camera info has not been received")
            return
        self.rgb_msg = rgb_msgs
        self.depth_msgs = depth_msg

        try:
            self.rgb_img = self.br.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            self.depth_img = self.br.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

            if depth_msg.encoding == '16UC1':
                self.depth_img = np.asarray(self.depth_img, dtype=np.float32)
                self.depth_img /= 1000  # convert metric: mm -> m
            elif msg.encoding != '32FC1':
                rospy.logerr('Unsupported depth encoding: %s' % depth_msg.encoding)

        except cv_bridge.CvBridgeError as e:
            rospy.logerr("CvBridge exception %s", e)

    # def rgb_img_cb(self, msg):
    #     try:
    #         self.rgb_img = self.br.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    #     except cv_bridge.CvBridgeError as e:
    #         rospy.logerr("CvBridge exception %s", e)

    # def depth_img_cb(self, msg):
    #     if not self.camera_info_msg:
    #         rospy.logwarn("Camera info has not been received")
    #         return

    #     try:
    #         self.depth_img = self.br.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    #         if msg.encoding == '16UC1':
    #             self.depth_img = np.asarray(self.depth_img, dtype=np.float32)
    #             self.depth_img /= 1000  # convert metric: mm -> m
    #         elif msg.encoding != '32FC1':
    #             rospy.logerr('Unsupported depth encoding: %s' % msg.encoding)

    #     except cv_bridge.CvBridgeError as e:
    #         rospy.logerr("CvBridge exception %s", e)

    def client(self):
        if self.rgb_img and self.depth_img:
            rospy.wait_for_service("/fcsc/fcsc_sandwich_server")
            try:
                fcsc_sandwich = rospy.ServiceProxy(
                    "/fcsc/fcsc_sandwich_server", ObjectMaskSrv)
                resp = fcsc_sandwich(self.rgb_img)
                rospy.logdebug("Done with fcsc sandwich parsing")
                return resp
            except rospy.ServiceException as e:
                rospy.logerr("Service call failed: %s" %e)


def main():
    rospy.init_node("sandwich_pose_sample")
    fcsc_sandwich_pose = FCSCSandwichPose()
    # fcsc_sandwich_pose.client()
    rospy.spin()


if __name__ == '__main__':
    main()
