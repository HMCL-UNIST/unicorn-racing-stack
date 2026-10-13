#!/usr/bin/env bash
# Install HSL (Coin-HSL Archive 2024.05.15+) for IPOPT ma27 solver.
#
# Run THIS SCRIPT INSIDE THE DOCKER CONTAINER (icra2026).
# You must separately obtain the Coin-HSL tarball from
#   https://licences.stfc.ac.uk/product/coin-hsl-archive
# and place it in this folder as:
#   solver/coinhsl-<version>.tar.gz
#
# The tarball and the extracted/built artifacts are git-ignored.
# The HSL library is proprietary — do NOT redistribute to other users.
#
# Steps this script performs:
#   1. Install build dependencies (gfortran, libblas-dev, liblapack-dev, libmetis-dev, meson)
#   2. Extract the tarball
#   3. Build with Meson + Ninja
#   4. Install into solver/install/
#   5. Symlink libcoinhsl.so → libhsl.so in the CasADi package directory
#   6. Probe-test with a trivial NLP using ma27
#
# Usage:
#   cd /home/unicorn/catkin_ws/src/race_stack/planner/3d_gb_optimizer/fast_ggv_gen/solver
#   bash setup_hsl.sh            # auto-detects newest coinhsl-*.tar.gz
#   bash setup_hsl.sh coinhsl-2024.05.15.tar.gz   # or specify explicitly

set -euo pipefail

# --- Paths -------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
INSTALL_PREFIX="${SCRIPT_DIR}/install"

# --- Locate tarball ----------------------------------------------------------
if [[ $# -ge 1 ]]; then
    TARBALL="$1"
else
    # newest matching coinhsl-*.tar.gz
    TARBALL="$(ls -1t coinhsl-*.tar.gz 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "${TARBALL:-}" || ! -f "${TARBALL}" ]]; then
    echo "ERROR: No coinhsl-*.tar.gz found in ${SCRIPT_DIR}."
    echo "       Download from https://licences.stfc.ac.uk/product/coin-hsl-archive"
    echo "       and place it here, then re-run this script."
    exit 1
fi

echo "[hsl-setup] Tarball: ${TARBALL}"

# --- sudo wrapper (root in container has no sudo but also doesn't need it) ---
if [[ $EUID -eq 0 ]]; then
    SUDO=""
else
    SUDO="sudo"
    if ! command -v sudo >/dev/null 2>&1; then
        echo "ERROR: Not running as root and 'sudo' is not installed."
        echo "       Re-run inside the container as root:"
        echo "         docker exec -u 0 -it icra2026 bash ${BASH_SOURCE[0]}"
        exit 1
    fi
fi

# --- 1. System deps ----------------------------------------------------------
echo "[hsl-setup] Installing build dependencies via apt ..."
${SUDO} apt-get update -qq
${SUDO} apt-get install -y -qq \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libmetis-dev \
    ninja-build \
    pkg-config \
    python3-pip

# --- 2. Meson (need >= 0.60) -------------------------------------------------
MESON_VERSION="$(meson --version 2>/dev/null || echo 0)"
NEED_NEW_MESON=1
if [[ "${MESON_VERSION}" != 0 ]]; then
    # crude version compare: accept 1.x or 0.6x+
    if [[ "${MESON_VERSION}" =~ ^1\. ]] || [[ "${MESON_VERSION}" =~ ^0\.(6|7|8|9)[0-9]*\. ]]; then
        NEED_NEW_MESON=0
    fi
fi
if [[ ${NEED_NEW_MESON} -eq 1 ]]; then
    echo "[hsl-setup] Upgrading Meson via pip (current: ${MESON_VERSION}) ..."
    ${SUDO} pip3 install --upgrade meson
    hash -r
    echo "[hsl-setup] Meson now: $(meson --version)"
else
    echo "[hsl-setup] Meson OK: ${MESON_VERSION}"
fi

# --- 3. Extract tarball ------------------------------------------------------
TARBALL_BASENAME="$(basename "${TARBALL}" .tar.gz)"
SRC_DIR="${SCRIPT_DIR}/${TARBALL_BASENAME}"
if [[ -d "${SRC_DIR}" ]]; then
    echo "[hsl-setup] Source already extracted: ${SRC_DIR}"
else
    echo "[hsl-setup] Extracting ${TARBALL} ..."
    tar -xzf "${TARBALL}"
fi

# --- 4. Build + install -------------------------------------------------------
cd "${SRC_DIR}"
if [[ -d builddir ]]; then
    echo "[hsl-setup] Removing previous builddir ..."
    ${SUDO} rm -rf builddir
fi

echo "[hsl-setup] Meson setup ..."
meson setup builddir --buildtype=release --prefix="${INSTALL_PREFIX}"

echo "[hsl-setup] Compiling (may take 1–2 min) ..."
meson compile -C builddir

echo "[hsl-setup] Installing into ${INSTALL_PREFIX} ..."
meson install -C builddir

# --- 5. Locate libcoinhsl.so + symlink to CasADi path -------------------------
LIB="$(find "${INSTALL_PREFIX}/lib" -name 'libcoinhsl.so' | head -n 1)"
if [[ -z "${LIB}" ]]; then
    echo "ERROR: libcoinhsl.so not found after install."
    exit 1
fi
echo "[hsl-setup] Built library: ${LIB}"

# find CasADi package directory
CASADI_DIR="$(python3 -c 'import casadi, os; print(os.path.dirname(casadi.__file__))' 2>/dev/null)"
if [[ -z "${CASADI_DIR}" ]]; then
    echo "ERROR: CasADi not importable from python3. Install CasADi first."
    exit 1
fi
echo "[hsl-setup] CasADi package: ${CASADI_DIR}"

${SUDO} ln -sf "${LIB}" "${CASADI_DIR}/libhsl.so"
echo "[hsl-setup] Symlink: ${CASADI_DIR}/libhsl.so -> ${LIB}"

# --- 6. Probe-test ------------------------------------------------------------
echo "[hsl-setup] Probe: solving a trivial NLP with ma27 ..."
python3 - <<'PYEOF'
from casadi import MX, nlpsol
x = MX.sym('x')
nlp = {'x': x, 'f': (x - 1.0)**2}
s = nlpsol('hsl_probe', 'ipopt', nlp, {
    'ipopt.linear_solver': 'ma27',
    'ipopt.print_level': 0, 'print_time': 0,
})
s(x0=0.0)
ok = s.stats().get('success', False)
print(f'[hsl-setup] probe success={ok}')
if not ok:
    raise SystemExit('Probe failed — ma27 did not solve successfully.')
PYEOF

echo ""
echo "=============================================="
echo " HSL ma27 installed and verified successfully."
echo "=============================================="
echo ""
echo " Note: the symlink in CasADi's package directory is INSIDE the container."
echo "       If you recreate the container, re-run this script to restore it."
echo ""
