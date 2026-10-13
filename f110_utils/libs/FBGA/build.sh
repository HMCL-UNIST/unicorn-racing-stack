#!/bin/bash

: '
This file is used to build the GIGI package.
'

# Get the path where the user launched this script
LAUNCH_DIR=$(pwd)

# Get the absolute path to this file
ROOT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# Default values
TARGET_ALL=true
BUILD=true
BUILD_TYPE=Release

# Get the command line arguments
# -h: Print the help message
# -a: Compile everything
# -b: compile
# -r: Delete the build directory and recompile
# -c: Clean the build directory
# -release: Build in release mode
# -debug: Build in debug mode
for i in "$@"
do
    case $i in
        -h)
            echo "${YELLOW}Usage: ./compile.sh <target> <action> <other>${NORMAL}"
            echo "${YELLOW}Targets:${NORMAL}"
            echo "${YELLOW}  -a: Compile everything${NORMAL}"
            echo "${YELLOW}Actions:${NORMAL}"
            echo "${YELLOW}  -b: Compile${NORMAL}"
            echo "${YELLOW}  -r: Delete the build directory and recompile${NORMAL}"
            echo "${YELLOW}  -c: Clean the build directory${NORMAL}"
            echo "${YELLOW}Other:${NORMAL}"
            echo "${YELLOW}  -release: Build in release mode${NORMAL}"
            echo "${YELLOW}  -debug: Build in debug mode${NORMAL}"
            exit 0
            ;;
        -a)
            TARGET_ALL=true
            shift
            ;;
        -b)
            BUILD=true
            shift
            ;;
        -c)
            CLEAN=true
            shift
            ;;
        -r)
            BUILD=true
            CLEAN=true
            shift
            ;;
        -release)
            BUILD_TYPE=Release
            shift
            ;;
        -debug)
            BUILD_TYPE=Debug
            shift
            ;;
        *)
            echo "${RED}Invalid option: $i${NORMAL}"
            echo "${YELLOW}Use -h for more information${NORMAL}"
            exit 1
            ;;
    esac
done

# Source some cool colours
source "$ROOT_SCRIPT_DIR/colours.sh"

if [ "$TARGET_ALL" = true ]; then
  if [ "$CLEAN" = true ]; then
    echo "${YELLOW}Cleaning the build directory${NORMAL}"
    if [ -d "build" ]; then
      rm -rf build
    fi
  fi
  if [ "$BUILD" = true ]; then
    echo "${YELLOW}Compiling and build project GIGI${NORMAL}"
    mkdir -p build
    cd build
    cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE" ..
    make -j
    cd ..
  fi
fi

