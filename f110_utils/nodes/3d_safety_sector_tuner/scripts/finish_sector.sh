#!/bin/bash
# Clone of friction_sector_tuner/scripts/finish_sector.sh. Forces
# regeneration of dyn_safety_tunerConfig after the slicing GUI
# overwrites cfg/safety_sectors.yaml with a new n_sectors.

source ~/.bashrc

cd ~/catkin_ws

# remove generated Config so catkin re-runs the .cfg generator
rm -f devel/lib/python3/dist-packages/safety_sector_tuner_3d/cfg/dyn_safety_tunerConfig.py
rm -f devel/lib/python3/dist-packages/safety_sector_tuner_3d/cfg/__pycache__/dyn_safety_tunerConfig*.pyc
rm -f devel/include/safety_sector_tuner_3d/dyn_safety_tunerConfig.h

catkin build safety_sector_tuner_3d
