#!/bin/bash

source ~/.bashrc

cd ~/catkin_ws

# remove generated Config to force regeneration (catkin clean fails on merged develspace)
rm -f devel/lib/python3/dist-packages/overtaking_sector_tuner_3d/cfg/ot_dyn_sect_tuner_3dConfig.py
rm -f devel/lib/python3/dist-packages/overtaking_sector_tuner_3d/cfg/__pycache__/ot_dyn_sect_tuner_3dConfig*.pyc
rm -f devel/include/overtaking_sector_tuner_3d/ot_dyn_sect_tuner_3dConfig.h

catkin build overtaking_sector_tuner_3d
