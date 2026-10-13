#! /bin/bash

apt-get update

# install Livox-SDK2
git clone https://github.com/Livox-SDK/Livox-SDK2.git /tmp/Livox-SDK2
cd /tmp/Livox-SDK2 && mkdir build && cd build
cmake .. && make -j$(nproc) && make install
rm -rf /tmp/Livox-SDK2
