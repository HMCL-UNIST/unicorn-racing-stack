cmake_minimum_required(VERSION 3.10)
project(polygon_filter)

# Find required ROS packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)


# What catkin_package() exports
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES polygon_filter_lib
  CATKIN_DEPENDS roscpp std_msgs
)

# Add the include directory and catkin include paths
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIRS}
)

# Library target (adjust source paths to the actual file locations)
add_library(polygon_filter_lib src/polygon_filter.cpp)
target_link_libraries(polygon_filter_lib ${catkin_LIBRARIES} ${OpenCV_LIBS} yaml-cpp)

# Install the target (with an EXPORT name for exporting)
install(TARGETS polygon_filter_lib
  EXPORT polygon_filter_export
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install the include directory
install(DIRECTORY include/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

# Generate the export file used by find_package() in other packages
install(EXPORT polygon_filter_export
  FILE polygon_filterTargets.cmake
  NAMESPACE polygon_filter::
  DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)