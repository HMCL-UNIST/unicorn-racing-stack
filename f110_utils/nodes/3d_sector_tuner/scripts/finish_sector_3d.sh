#!/bin/bash

source ~/.bashrc

cd ~/catkin_ws

# remove generated Config to force regeneration (catkin clean fails on merged develspace)
rm -f devel/lib/python3/dist-packages/sector_tuner_3d/cfg/dyn_sect_tuner_3dConfig.py
rm -f devel/lib/python3/dist-packages/sector_tuner_3d/cfg/__pycache__/dyn_sect_tuner_3dConfig*.pyc
rm -f devel/include/sector_tuner_3d/dyn_sect_tuner_3dConfig.h

catkin build sector_tuner_3d
