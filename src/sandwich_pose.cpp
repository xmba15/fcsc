// Copyright (c) 2018
// All Rights Reserved.
#include <fcsc/sandwich_pose.hpp>

PCAObjectPose::PCAObjectPose(ros::NodeHandle* nodehandle) :
    nh_(*nodehandle) {
  this->onInit();
}

void PCAObjectPose::onInit() {
  this->subscribe();
}

void PCAObjectPose::subscribe() {
  ROS_INFO("Initialize Services");
  this->service_ = this->nh_.advertiseService("/fcsc/fcsc_sandwich_pose_server",
                                              &PCAObjectPose::serviceCallback,
                                              this);
}

void PCAObjectPose::unsubscribe() {
}

geometry_msgs::Pose PCAObjectPose::getPose(PointCloud::Ptr cloud) {
  pcl::MomentOfInertiaEstimation<PointT> feature_extractor;
  feature_extractor.setInputCloud(cloud);
  feature_extractor.compute();
  std::vector<float> moment_of_inertia;
  std::vector<float> eccentricity;

  PointT min_point_AABB;
  PointT max_point_AABB;
  PointT min_point_OBB;
  PointT max_point_OBB;
  PointT position_OBB;

  Eigen::Matrix3f rotational_matrix_OBB;
  float major_value, middle_value, minor_value;
  Eigen::Vector3f major_vector, middle_vector, minor_vector;
  Eigen::Vector3f mass_center;

  feature_extractor.getMomentOfInertia(moment_of_inertia);
  feature_extractor.getEccentricity(eccentricity);
  feature_extractor.getAABB(min_point_AABB, max_point_AABB);
  feature_extractor.getOBB(min_point_OBB, max_point_OBB,
                           position_OBB, rotational_matrix_OBB);
  feature_extractor.getEigenValues(major_value, middle_value, minor_value);
  feature_extractor.getEigenVectors(major_vector, middle_vector, minor_vector);
  feature_extractor.getMassCenter(mass_center);

  Eigen::Quaternionf quat(rotational_matrix_OBB);

  geometry_msgs::Pose pose;
  pose.position.x = mass_center(0);
  pose.position.y = mass_center(1);
  pose.position.z = mass_center(2);

  pose.orientation.x = quat.x();
  pose.orientation.y = quat.y();
  pose.orientation.z = quat.z();
  pose.orientation.w = quat.w();

  return pose;
}

bool PCAObjectPose::serviceCallback(fcsc::SandwichPoseRequest& req,
                                    fcsc::SandwichPoseResponse& res) {
  // int8 mask_num
  // sensor_msgs/PointCloud2[] sandwich_pcl
  //     ---
  // geometry_msgs/PoseArray sandwich_pose

  geometry_msgs::PoseArray posearr;
  posearr.header.stamp = ros::Time::now();
  int mask_num = req.mask_num;
  for (int i = 0; i < mask_num; i++) {
    PointCloud::Ptr cloud(new PointCloud);
    pcl::PCLPointCloud2 pcl_cloud;
    pcl_conversions::toPCL(req.sandwich_pcl.pcl_arr[i], pcl_cloud);
    pcl::fromPCLPointCloud2(pcl_cloud, *cloud);
    geometry_msgs::Pose pose = this->getPose(cloud);
    posearr.poses.push_back(pose);
  }

  res.sandwich_pose = posearr;
  return true;
}

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "sandwich_pose_node");
  ros::NodeHandle _nh;
  ROS_INFO("Instantiating a pose object");
  PCAObjectPose objectpose(&_nh);
  ros::spin();
  return 0;
}
