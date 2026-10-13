#! /bin/bash

set -e

# source ROS setup to ensure PYTHONPATH includes genmsg etc.
source /opt/ros/noetic/setup.bash
if [ -f /home/${USER}/catkin_ws/devel/setup.bash ]; then
    source /home/${USER}/catkin_ws/devel/setup.bash
fi

exec "$@"
