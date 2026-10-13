cmake_minimum_required(VERSION 3.0.2)
project(global_line_3d)

find_package(catkin REQUIRED COMPONENTS rospy f110_msgs std_msgs nav_msgs sensor_msgs)
catkin_package()

# register the local racing line mux node
catkin_install_python(PROGRAMS
  local_racing_line/local_raceline_mux_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Make scripts available as ROS nodes
install(PROGRAMS scripts/run_pipeline.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install scripts
install(DIRECTORY scripts/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/scripts
  USE_SOURCE_PERMISSIONS
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
