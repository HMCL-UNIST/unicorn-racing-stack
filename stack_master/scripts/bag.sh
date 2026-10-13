#!/bin/bash

# Create record directory if it doesn't exist
# Automatically use current user's home directory
USER_HOME="$HOME"
BAG_DIR="$USER_HOME/catkin_ws/src/race_stack/stack_master/bag"
MAPPING_DIR="$BAG_DIR/record"

mkdir -p "$RECORD_DIR"

cd "$RECORD_DIR"

rosbag record \
    /scan \
    /vesc/sensors/imu/raw \
    /vesc/low_level/ackermann_cmd_mux/output \
    /car_state/odom \
    /car_state/odom_frenet \
    /behavior_strategy \
