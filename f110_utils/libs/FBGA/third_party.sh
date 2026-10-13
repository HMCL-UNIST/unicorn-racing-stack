#!/bin/bash

: '
This file is used to install the dependencies for the GIGI package.
'

# Get the path where the user launched this script
LAUNCH_DIR=$(pwd)

# Get the absolute path to this file
ROOT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# Source some cool colours
source "$ROOT_SCRIPT_DIR/colours.sh"

# Check if the third-party dependencies folder exists
if [ ! -d "$ROOT_SCRIPT_DIR/third_party" ]; then
  mkdir "$ROOT_SCRIPT_DIR/third_party"
else
  echo "${YELLOW}Third-party dependencies folder already exists${NORMAL}"
  echo "${YELLOW}If you want to reinstall the dependencies, please delete the folder${NORMAL}"
fi


cd "$ROOT_SCRIPT_DIR/third_party"


# Instrumentor

if [ ! -d "Instrumentor" ]; then
  echo "${YELLOW}Cloning and compiling the source code${NORMAL}"
  git clone --branch main --depth 1 https://github.com/mattiapzz/Instrumentor.git
  cd Instrumentor 
  echo "${YELLOW}Compiling and installing Instrumentor${NORMAL}"
  mkdir build
  cd build
  cmake ..
  make
  make install
fi

cd "$ROOT_SCRIPT_DIR/third_party"

# rapid csv

if [ ! -d "rapidcsv" ]; then
  echo "${YELLOW}Cloning and compiling the source code${NORMAL}"
  git clone --branch master --depth 1 https://github.com/d99kris/rapidcsv.git
  cd rapidcsv
  # make.sh
fi

cd "$ROOT_SCRIPT_DIR/third_party"

# nolhmann json

if [ ! -d "json" ]; then
  echo "${YELLOW}Cloning and compiling the source code${NORMAL}"
  git clone --branch v3.11.3 --depth 1 https://github.com/nlohmann/json.git
fi

# Download, compile, and install the third-party dependencies
cd "$ROOT_SCRIPT_DIR/third_party"

# cxxopts
if [ ! -d "cxxopts" ]; then
  echo "${YELLOW}Downloading cxxopts${NORMAL}"
  git clone --branch v3.1.1 --depth 1 https://github.com/jarro2783/cxxopts.git
  echo "${YELLOW}cxxopts is header-only, no need to compile${NORMAL}"
else
  echo "${YELLOW}cxxopts already installed${NORMAL}"
fi


cd "$ROOT_SCRIPT_DIR/third_party"

# download gnuplot header only https://github.com/ziotom78/gplotpp.git
if [ ! -d "gplotpp" ]; then
  echo "${YELLOW}Downloading gplotpp${NORMAL}"
  git clone --branch master --depth 1 https://github.com/ziotom78/gplotpp.git
  echo "${YELLOW}gplotpp is header-only, no need to compile${NORMAL}"
else
  echo "${YELLOW}gplotpp already installed${NORMAL}"
fi

cd "$ROOT_SCRIPT_DIR/third_party"

