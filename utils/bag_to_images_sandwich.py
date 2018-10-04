#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import argparse
import cv2
from bag_to_images import bag_to_images


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
                        default="/camera/rgb/image_rect_color")
    parser.add_argument("--image_prefix",
                        type=str,
                        default="sandwich_bag1")
    parser.add_argument("--downsample",
                        type=int,
                        default=40)
    args = parser.parse_args()

    bag_file_list = ["2018-10-03-23-09-34"]
    output_dir_list = ["sandwich_bag1"]

    for i, bag_file in enumerate(bag_file_list):
        bag_to_images(args.input_path, os.path.join(args.output_path, output_dir_list[i]), bag_file, args.image_topic, output_dir_list[i], args.downsample)


if __name__ == '__main__':
    main()
