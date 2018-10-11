// Copyright (c) 2018
// All Rights Reserved.
#include <fcsc/sandwich_pose.hpp>

namespace fcsc_pose {

PCAObjectPose::PCAObjectPose(ros::NodeHandle* nodehandle) :
    nh_(nodehandle) {
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

bool PCAObjectPose::serviceCallback(fcsc::SandwichPoseRequest& req,
                                    fcsc::SandwichPoseResponse& res) {
  // sensor_msgs/PointCloud sandwich_pcl
  //     ---
  // geometry_msgs/Pose sandwich_pose

  req.sandwich_pcl;
  res.sandwich_pose;
}

}  // fcsc_pose

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "fcsc_sandwich_pose");
  ros::NodeHandle nh;
  ROS_INFO("Instan");
  return 0;
}
