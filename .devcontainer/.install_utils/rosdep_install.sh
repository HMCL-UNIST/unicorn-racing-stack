#! /bin/bash

# install the rosdeps of all packages in the workspace
rosdep update
rosdep install --from-paths src --ignore-src --rosdistro=${ROS_DISTRO} -y
