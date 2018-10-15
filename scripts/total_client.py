#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import cv2
import numpy as np
# ros related import
import cv_bridge
from fcsc.srv import ObjectMaskSrv, ObjectMaskSrvResponse
from fcsc.srv import ImageAssembly, ImageAssemblyResponse
from fcsc.srv import SandwichPose, SandwichPoseResponse
from fcsc.msg import ObjectMask, ObjectMaskArray, PointCloud2Array
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pcl2
import rospy


def image_assembly_client():
    rospy.wait_for_service("/fcsc/image_assembly_server")
    try:
        image_assembly = rospy.ServiceProxy(
            "/fcsc/image_assembly_server", ImageAssembly)
        resp = image_assembly()
        rospy.logdebug("Done with getting images")
    except rospy.ServiceException, e:
        rospy.logerr("Image assembly service call failed %s", %e)


def mask_client(img_msg):
    rospy.wait_for_service("/fcsc/fcsc_sandwich_server")
    try:
        fcsc_sandwich = rospy.ServiceProxy(
            "/fcsc/fcsc_sandwich_server", ObjectMaskSrv)
        resp = fcsc_sandwich(img_msg)
        rospy.logdebug("Done with fcsc sandwich parsing")
        return resp
    except rospy.ServiceException, e:
        rospy.logerr("Service call failed: %s" %e)

def pose_client(mask_num, pcl_arr_msg):
    rospy.wait_for_service("/fcsc/fcsc_sandwich_pose_server")
    try:
        sandwich_pose = rospy.ServiceProxy(
            "/fcsc/fcsc_sandwich_pose_server", SandwichPose)
        resp = sandwich_pose(mask_num, pcl_arr_msg)
        rospy.logdebug("Done with getting sandwich pose")
        return resp
    except rospy.ServiceException, e:
        rospy.logerr("Service call failed: %s" %e)


def main():
    br = cv_bridge.CvBridge()
    image_client = image_assembly_client()
    camera_info_msg = image_client.info
    rgb_msg = image_client.image
    depth_msg = image_client.depth
    fx = camera_info_msg.K[0]
    fy = camera_info_msg.K[4]
    cx = camera_info_msg.K[2]
    cy = camera_info_msg.K[5]

    depth_img = br.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
    if depth_msg.encoding == '16UC1':
        depth_img = np.asarray(self.depth_img, dtype=np.float32)
        depth_img /= 1000  # convert metric: mm -> m
    elif depth_msg.encoding != '32FC1':
        rospy.logerr('Unsupported depth encoding: %s' % depth_msg.encoding)

    maskcl = mask_client(rgb_msg)

    mask_num = len(maskcl.masks)
    _pcl_arr_msg = PointCloud2Array()

    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "pointcloud_arr"

    for _mask in maskcl.masks:
        point_list = []
        h, w = _mask.shape
        for i in range(h):
            for j in range(w):
                if _mask[i, j] == 1:
                    z = float(depth_img[int(j)][int(i)])
                    if np.isnan(z):
                        continue
                    x = (i - cx) * z / fx
                    y = (j - cy) * z / fy
                    point_list.append([x, y, z])

        pcl_msg = pcl2.create_cloud_xyz32(header, point_list)
        _pcl_arr_msg.pcl_arr.append(pcl_msg)

    _pose = pose_client(mask_num , _pcl_arr_msg)
    pose_arr = _pose.sandwich_pose


if __name__ == '__main__':
    main()
