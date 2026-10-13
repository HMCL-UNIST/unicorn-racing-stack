#! /bin/bash

# extend .bashrc
cat /home/${USER}/catkin_ws/src/race_stack/.devcontainer/.install_utils/bashrc_ext >> ~/.bashrc

# ensure catkin workspace always extends /opt/ros/noetic
# Without this, _setup_util.py can be generated with a CMAKE_PREFIX_PATH
# that excludes /opt/ros/noetic, causing PYTHONPATH to lose genmsg etc.
source /opt/ros/noetic/setup.bash
cd /home/${USER}/catkin_ws
catkin config --extend /opt/ros/noetic

# install dependencies
pip install ~/catkin_ws/src/race_stack/f110_utils/libs/ccma
# pip install ~/catkin_ws/src/race_stack/planner/graph_based_planner/src/GraphBasedPlanner

# staged build to avoid glim corrupting devel/_setup_util.py
# glim's CMakeLists.txt does not call find_package(catkin)/catkin_package()
# even though package.xml declares build_type=catkin. If glim runs cmake
# configure last, it bakes a CMAKE_PREFIX_PATH into devel/_setup_util.py
# without /opt/ros/noetic, which then wipes ROS PYTHONPATH on next source.
# Workaround: build all normal catkin packages first, then glim, then glim_ros.

# Step 1: build all normal catkin packages, skipping glim and glim_ros
cd /home/${USER}/catkin_ws
catkin config --skiplist glim glim_ros
catkin build

# Step 2: verify devel/_setup_util.py has the /opt/ros/noetic chain (abort if not)
if ! grep -q "/opt/ros/noetic" /home/${USER}/catkin_ws/devel/_setup_util.py; then
  echo "[ERROR] devel/_setup_util.py is missing /opt/ros/noetic chain. Aborting." >&2
  exit 1
fi

# Step 3: re-source ROS + devel to ensure clean env for glim build
source /opt/ros/noetic/setup.bash && source /home/${USER}/catkin_ws/devel/setup.bash

# Step 4: build glim by itself (clear skiplist first)
cd /home/${USER}/catkin_ws
catkin config --no-skiplist
catkin build glim

# Step 5: build glim_ros
catkin build glim_ros

# Step 6: final full build pass to catch anything missed
source /opt/ros/noetic/setup.bash && source /home/${USER}/catkin_ws/devel/setup.bash
cd /home/${USER}/catkin_ws
catkin build

# source
source /opt/ros/noetic/setup.bash && source /home/${USER}/catkin_ws/devel/setup.bash

# build FBGA (GIGI / GG::FWBW velocity profile optimizer)
# Standalone CMake library, not a catkin package. Has its own third_party deps.
FBGA_DIR=/home/${USER}/catkin_ws/src/race_stack/f110_utils/libs/FBGA
if [ -d "${FBGA_DIR}" ]; then
  cd "${FBGA_DIR}"
  if [ ! -d "third_party" ]; then
    bash third_party.sh
  fi
  bash build.sh -a -b -release
  cd /home/${USER}/catkin_ws
fi

# python privileges
find /home/${USER}/catkin_ws -type f -name "*.py" -exec chmod +x {} \;


