cmake_minimum_required(VERSION 3.0.2)
project(fast_sqp_planner)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  nav_msgs
  visualization_msgs
  f110_msgs
  dynamic_reconfigure
)

# dynamic_reconfigure .cfg
generate_dynamic_reconfigure_options(
  cfg/FastSQPPlanner.cfg
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs nav_msgs visualization_msgs f110_msgs dynamic_reconfigure
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  src/fast_sqp_planner_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
