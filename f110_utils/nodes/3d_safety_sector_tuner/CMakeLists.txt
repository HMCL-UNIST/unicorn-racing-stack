cmake_minimum_required(VERSION 3.0.2)
project(safety_sector_tuner_3d)

# Friction sector tuner clone, but for raceline wall-margin sectors.
# Generates <map>/safety_sectors.yaml + /safety_sector_params/Sector{i}/...
# rosparam tree. gen_global_racing_line.py reads the rosparam tree at NLP
# setup time (no yaml file path coupling).

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  dynamic_reconfigure
)

generate_dynamic_reconfigure_options(
  cfg/dyn_safety_tuner.cfg
)

catkin_package(
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
