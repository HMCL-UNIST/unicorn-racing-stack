cmake_minimum_required(VERSION 3.10)
project(vel_planner)

# Find required ROS packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
)

catkin_python_setup()

# What catkin_package() exports
catkin_package(
  CATKIN_DEPENDS roscpp std_msgs
)

# Add the include directory and catkin include paths
# include_directories(
#   include
#   ${catkin_INCLUDE_DIRS}
#   ${OpenCV_INCLUDE_DIRS}
#   ${YAML_CPP_INCLUDE_DIRS}
# )

# Library target (adjust source paths to the actual file locations)
# add_library(grid_filter_lib src/grid_filter.cpp)
# target_link_libraries(grid_filter_lib ${catkin_LIBRARIES} ${OpenCV_LIBS} yaml-cpp)

# Install the target (with an EXPORT name for exporting)
# install(TARGETS grid_filter_lib
#   EXPORT grid_filter_export
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

# Install the include directory
# install(DIRECTORY include/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
# )

# Generate the export file used by find_package() in other packages
# install(EXPORT grid_filter_export
#   FILE grid_filterTargets.cmake
#   NAMESPACE grid_filter::
#   DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
# )

catkin_install_python(PROGRAMS
  src/vel_planner/vel_planner.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)