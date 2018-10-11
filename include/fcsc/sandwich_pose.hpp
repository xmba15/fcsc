// Copyright (c) 2018
// All Rights Reserved.
# pragma once
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <fcsc/SandwichPose.h>

typedef pcl::PointXYZRGB PointT;
typedef pcl::Normal NormalT;
typedef pcl::PointXYZRGBNormal PointNormalT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointCloud<NormalT> PointNormal;
typedef pcl::PointCloud<PointNormalT> PointCloudNormal;

namespace object_pose {

class PCAObjectPose
{
 public:
  PCAObjectPose(ros::NodeHandle* nodehandle);
  bool serviceCallback(fcsc::SandwichPoseRequest&, fcsc::SandwichPoseResponse&);
 protected:
  virtual void onInit();
  virtual void subscribe();
  virtual void unsubscribe();
 private:
  ros::NodeHandle nh_;
  ros::ServiceServer service_;
};

}  // object_pose
