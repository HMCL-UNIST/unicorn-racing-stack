#!/bin/bash

source ~/.bashrc

cd ~/catkin_ws

# remove generated Config to force regeneration (catkin clean fails on merged develspace)
rm -f devel/lib/python3/dist-packages/static_obstacle_sector_tuner_3d/cfg/static_obs_dyn_sect_tuner_3dConfig.py
rm -f devel/lib/python3/dist-packages/static_obstacle_sector_tuner_3d/cfg/__pycache__/static_obs_dyn_sect_tuner_3dConfig*.pyc
rm -f devel/include/static_obstacle_sector_tuner_3d/static_obs_dyn_sect_tuner_3dConfig.h

catkin build static_obstacle_sector_tuner_3d
