// Copyright (c) 2018
// All Rights Reserved.
#pragma once
#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/features/moment_of_inertia_estimation.h>

#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseArray.h>

#include <fcsc/SandwichPose.h>

typedef pcl::PointXYZRGB PointT;
typedef pcl::Normal NormalT;
typedef pcl::PointXYZRGBNormal PointNormalT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointCloud<NormalT> PointNormal;
typedef pcl::PointCloud<PointNormalT> PointCloudNormal;

class PCAObjectPose {
 public:
  explicit PCAObjectPose(ros::NodeHandle* nodehandle);
  geometry_msgs::Pose getPose(PointCloud::Ptr);
  bool serviceCallback(fcsc::SandwichPoseRequest&, fcsc::SandwichPoseResponse&);
 protected:
  virtual void onInit();
  virtual void subscribe();
  virtual void unsubscribe();
 private:
  ros::NodeHandle nh_;
  ros::ServiceServer service_;
};
