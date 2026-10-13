cmake_minimum_required(VERSION 3.0.2)
project(bridge_sector_tuner_3d)

# Mirrors 3d_sector_tuner structure but operates on the pre-opt
# smoothed CSV instead of post-opt global_waypoints.json.

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

catkin_package(
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
