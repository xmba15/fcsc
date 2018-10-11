// Copyright (c) 2018
// All Rights Reserved.
# pragma once
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

typedef pcl::PointXYZRGB PointT;
typedef pcl::Normal NormalT;
typedef pcl::PointXYZRGBNormal PointNormalT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointCloud<NormalT> PointNormal;
typedef pcl::PointCloud<PointNormalT> PointCloudNormal;

namespace fcsc {

class PCAObjectPose
{
 public:
  PCAObjectPose();

 protected:
  virtual void onInit();
  virtual void subscribe();
  virtual void unsubscribe();

  ros::NodeHandle nh_;
  ros::ServiceServer service_;
 private:
};

}  // fcsc
