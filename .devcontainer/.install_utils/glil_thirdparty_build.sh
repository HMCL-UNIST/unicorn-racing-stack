#!/bin/bash
set -e

# GLIL thirdparty superbuild (GTSAM, ROBIN, gtsam_points, KISS-Matcher)
# Source is COPYed into /glil_thirdparty_src by Dockerfile
# Libraries are installed to /usr/local for system-wide availability

THIRDPARTY_DIR="/glil_thirdparty_src"
INSTALL_DIR="/usr/local"

echo "=== Building GLIL thirdparty (superbuild) ==="

# Initialize nested submodules within the copied thirdparty source
cd "${THIRDPARTY_DIR}"

mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DINSTALL_PREFIX=${INSTALL_DIR} \
    -DNUM_JOBS=$(nproc)
make -j$(nproc)
sudo make install
ldconfig

echo "=== GLIL thirdparty build complete ==="
