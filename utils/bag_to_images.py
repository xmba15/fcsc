#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import argparse
import cv2
import rosbag
from cv_bridge import CvBridge


def bag_to_images(input_path, output_path, bag_file, image_topic, image_prefix="frame", down_sample=1):
    """
    extract rosbag data into sequence of images
    """
    bag_file_path = os.path.join(input_path, bag_file + ".bag")
    bag_file_reader = rosbag.Bag(bag_file_path, "r")
    bridge = CvBridge()
    count = 0
    img_num = 0

    for _, msg, _ in bag_file_reader.read_messages(topics=[image_topic]):
        if count % down_sample == 0:
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            img_file = image_prefix +"_%06i.png" % img_num
            cv2.imwrite(os.path.join(output_path, img_file), cv_img)
            print("Frame %i" % img_num)
            img_num += 1
        count += 1

    bag_file_reader.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",
                        type=str,
                        default="/home/buggy/publicWorkspace/dev/fcsc/data/rosbag")
    parser.add_argument("--output_path",
                        type=str,
                        default="/home/buggy/publicWorkspace/dev/fcsc/data")
    parser.add_argument("--image_topic",
                        help="Image topic.",
                        type=str,
                        default="/camera/color/image_raw")
    parser.add_argument("--image_prefix",
                        type=str,
                        default="bag1")
    parser.add_argument("--downsample",
                        type=int,
                        default=4)
    args = parser.parse_args()

    bag_file_list = ["2018-09-19-18-49-39",
                     "2018-09-19-19-12-57",
                     "2018-09-19-19-16-10",
                     "2018-09-19-19-10-34",
                     "2018-09-19-19-14-33",
                     "2018-09-19-19-17-38"]
    output_dir_list = ["bag1",
                       "bag2",
                       "bag3",
                       "bag4",
                       "bag5",
                       "bag6"]

    for i, bag_file in enumerate(bag_file_list):
        bag_to_images(args.input_path, os.path.join(args.output_path, output_dir_list[i]), bag_file, args.image_topic, output_dir_list[i], args.downsample)


if __name__ == '__main__':
    main()
