#! /bin/bash

# list can be manually obtained by ``catkin list --directory <all_the_directories_you_want>``
# list of packages only for the car

catkin config --skiplist \
    ackermann_cmd_mux\
    joy\
    joystick_drivers\
    racecar\
    serial\
    vesc\
    vesc_msgs\
    vesc_ackermann\
    vesc_driver\
    waypoint_logger
