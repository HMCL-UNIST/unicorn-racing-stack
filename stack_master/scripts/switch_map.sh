#!/bin/bash

# This script copies the sector YAML files of the given map in stack_master/maps
# into the sector tuner packages and rebuilds them, so that their
# dynamic_reconfigure sliders match the sectors of that map.

cd "${0%/*}"
cd ../..
# This ensures the current working directory is race_stack

MAP_PATH="stack_master/maps"
NODES="f110_utils/nodes"

# <map yaml> <package folder> <catkin package> <generated Config name>
SECTOR_PKGS=(
    "speed_scaling.yaml      3d_sector_tuner                 sector_tuner_3d                 dyn_sect_tuner_3dConfig"
    "ot_sectors.yaml         3d_overtaking_sector_tuner      overtaking_sector_tuner_3d      ot_dyn_sect_tuner_3dConfig"
    "static_obs_sectors.yaml 3d_static_obstacle_sector_tuner static_obstacle_sector_tuner_3d static_obs_dyn_sect_tuner_3dConfig"
    "safety_sectors.yaml     3d_safety_sector_tuner          safety_sector_tuner_3d          dyn_safety_tunerConfig"
    "friction_scaling.yaml   friction_sector_tuner           friction_sector_tuner           dyn_sect_tunerConfig"
)

if [ $# -gt 0 ]; then
    echo "Listing contents of map $1"
    ls $MAP_PATH/$1
    echo -e "\n"

    BUILD_PKGS=()
    CONFIGS=()
    for entry in "${SECTOR_PKGS[@]}"; do
        read -r yaml folder pkg config <<< "$entry"
        if [ -f "$MAP_PATH/$1/$yaml" ]; then
            echo "Copying $yaml to $folder"
            cp "$MAP_PATH/$1/$yaml" "$NODES/$folder/cfg/$yaml"
            BUILD_PKGS+=("$pkg")
            CONFIGS+=("$pkg/$config")
        else
            echo "Skipping $folder ($yaml not found in map $1)"
        fi
    done
    echo -e "\n"

    if [ ${#BUILD_PKGS[@]} -eq 0 ]; then
        echo "No sector files found in map $1, nothing to build."
        exit 0
    fi

    echo "Building ..."
    cd ../..;   # now we are in catkin_ws
    # remove the generated Configs to force regeneration (catkin clean fails on merged develspace)
    for c in "${CONFIGS[@]}"; do
        pkg="${c%%/*}"
        config="${c##*/}"
        rm -f devel/lib/python3/dist-packages/$pkg/cfg/$config.py
        rm -f devel/lib/python3/dist-packages/$pkg/cfg/__pycache__/$config*.pyc
        rm -f devel/include/$pkg/$config.h
    done
    catkin build "${BUILD_PKGS[@]}" && \
    source ~/catkin_ws/devel/setup.bash;

    echo -e "\n"
    echo "Done! Please remember to source in your other terminals."
else
    echo "Please specify a map name for this to work."
    echo "A list of local maps in the stack_master folder:"
    ls -la $MAP_PATH
    echo -e "\n#######################################################################\n"
fi
