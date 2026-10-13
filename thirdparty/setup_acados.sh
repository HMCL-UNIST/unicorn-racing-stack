#!/usr/bin/env bash
# Install acados (NLP solver for local racing line MPC) inside the
#       icra2026 Docker container. Run this from INSIDE the container,
#       as the regular `unicorn` user (NOT root).
#
#   docker exec -it icra2026 bash          # enters as unicorn
#   bash /home/unicorn/catkin_ws/src/race_stack/thirdparty/setup_acados.sh
#
# Why unicorn user (not root)?
#   Debian/Ubuntu's pip drops root-installed packages to
#   /usr/lib/python3.8/site-packages, which is NOT on `python3` sys.path
#   (python looks at dist-packages). Installing with `pip3 install --user`
#   puts them in ~/.local/lib/python3.8/site-packages, which IS on path.
#   Fighting that with --prefix=/usr/local works but also leaves files
#   the non-root user can't clean up. User install is the simple path.
#
# Installation layout:
#   thirdparty/acados/              ← git clone (~100MB incl. submodules)
#   thirdparty/acados/build/        ← cmake build tree
#   thirdparty/acados/lib/          ← libacados.so + dependent libs
#   thirdparty/acados/bin/t_renderer← tera templates binary (Ubuntu 20.04: v0.0.34)
#   ~/.local/lib/python3.8/site-packages/acados_template.egg-link (user install)
#
# The whole thirdparty/acados/ is git-ignored; this script IS committed.
#
# Env vars appended to ~/.bashrc on success (idempotent block):
#   ACADOS_SOURCE_DIR, LD_LIBRARY_PATH

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACADOS_DIR="${SCRIPT_DIR}/acados"

echo "[acados-setup] running as: $(whoami)"
echo "[acados-setup] target dir: ${ACADOS_DIR}"

if [[ $EUID -eq 0 ]]; then
    echo "WARNING: running as root. This will install acados_template to a"
    echo "         path Debian's python3 doesn't search. Re-run as a normal"
    echo "         user (e.g. unicorn) for a clean install."
    echo "         Press Ctrl+C now to abort, or Enter to continue anyway."
    read -r _
fi

# --- 1. apt deps (need root → use sudo) ---
if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
else
    if [[ $EUID -eq 0 ]]; then
        SUDO=""
    else
        echo "ERROR: not root and sudo unavailable. Install apt deps manually:"
        echo "  build-essential cmake git pkg-config libblas-dev liblapack-dev wget"
        exit 1
    fi
fi

echo "[acados-setup] Installing apt deps ..."
${SUDO} apt-get update -qq
${SUDO} apt-get install -y -qq \
    build-essential \
    cmake \
    git \
    pkg-config \
    libblas-dev \
    liblapack-dev \
    wget

# --- 2. Clone acados if absent; otherwise pull submodules ---
if [[ ! -d "${ACADOS_DIR}/.git" ]]; then
    echo "[acados-setup] Cloning acados (with submodules) ..."
    git clone --recurse-submodules https://github.com/acados/acados.git "${ACADOS_DIR}"
else
    echo "[acados-setup] acados already cloned. Updating submodules ..."
    ( cd "${ACADOS_DIR}" && git submodule update --init --recursive )
fi

# Safety: if a previous root run left files owned by root, fix them so
# the current user can write the build tree.
if [[ $EUID -ne 0 ]] && [[ -n "$(find "${ACADOS_DIR}" -not -user "$(whoami)" -print -quit 2>/dev/null)" ]]; then
    echo "[acados-setup] fixing file ownership (previous root install detected) ..."
    ${SUDO} chown -R "$(whoami):$(whoami)" "${ACADOS_DIR}"
fi

# --- 3. Build C library ---
cd "${ACADOS_DIR}"
mkdir -p build
cd build
echo "[acados-setup] cmake ..."
cmake .. \
    -DACADOS_WITH_QPOASES=ON \
    -DACADOS_WITH_OSQP=OFF \
    -DACADOS_INSTALL_DIR="${ACADOS_DIR}"
echo "[acados-setup] make (-j$(nproc)) ..."
make install -j"$(nproc)"

# --- 4. Python wrapper (editable install, user site) ---
echo "[acados-setup] Installing Python acados_template (--user, editable) ..."
cd "${ACADOS_DIR}/interfaces/acados_template"
# Clean possible leftovers from past root installs
rm -rf .eggs *.egg-info 2>/dev/null || true
pip3 install --user -e .

# --- 5. Tera renderer (precompiled binary for code generation) ---
#     Ubuntu 20.04 ships glibc 2.31. tera v0.2.0 needs glibc ≥2.32,
#     so we pin v0.0.34 which is known-good on 20.04.
TERA_URL="https://github.com/acados/tera_renderer/releases/download/v0.0.34/t_renderer-v0.0.34-linux"
TERA_BIN="${ACADOS_DIR}/bin/t_renderer"

if [[ ! -x "${TERA_BIN}" ]]; then
    echo "[acados-setup] Fetching tera renderer v0.0.34 (Ubuntu 20.04 compatible) ..."
    mkdir -p "${ACADOS_DIR}/bin"
    wget -q "${TERA_URL}" -O "${TERA_BIN}"
    chmod +x "${TERA_BIN}"
fi

# Smoke test: we only check the binary runs without a glibc-version error.
# `--help` actually panics on v0.0.34 (expects positional args), so we
# grep stderr for glibc issues rather than expecting a zero exit code.
if "${TERA_BIN}" --help 2>&1 | grep -q "GLIBC_"; then
    echo "ERROR: t_renderer reports a missing glibc version. Your container"
    echo "       is probably newer than Ubuntu 20.04 — pick a different"
    echo "       tera release from https://github.com/acados/tera_renderer/releases"
    exit 1
fi

# --- 6. env vars in ~/.bashrc (idempotent) ---
BASHRC_TAG="# >>> acados env (setup_acados.sh) >>>"
BASHRC_END="# <<< acados env (setup_acados.sh) <<<"
BASHRC="$HOME/.bashrc"

if grep -q "${BASHRC_TAG}" "${BASHRC}" 2>/dev/null; then
    sed -i "/${BASHRC_TAG}/,/${BASHRC_END}/d" "${BASHRC}"
fi

cat >> "${BASHRC}" <<EOF
${BASHRC_TAG}
export ACADOS_SOURCE_DIR=${ACADOS_DIR}
export LD_LIBRARY_PATH=\${ACADOS_SOURCE_DIR}/lib:\${LD_LIBRARY_PATH:-}
${BASHRC_END}
EOF

export ACADOS_SOURCE_DIR="${ACADOS_DIR}"
export LD_LIBRARY_PATH="${ACADOS_DIR}/lib:${LD_LIBRARY_PATH:-}"

# --- 7. Final probe: import acados_template ---
echo "[acados-setup] Probe: import acados_template ..."
python3 - <<'PYEOF'
from acados_template import AcadosOcp, AcadosOcpSolver
import acados_template, os
print("[acados-setup] acados_template OK:", acados_template.__file__)
PYEOF

echo ""
echo "=============================================="
echo " acados installed and importable successfully."
echo "=============================================="
echo " Source       : ${ACADOS_DIR}"
echo " Libs         : ${ACADOS_DIR}/lib/"
echo " Tera bin     : ${TERA_BIN}"
echo " Python pkg   : ~/.local/lib/python3.8/site-packages/ (editable → source dir)"
echo ""
echo " Env vars added to ~/.bashrc — open a new terminal or 'source ~/.bashrc'"
echo " For the current shell:"
echo "   export ACADOS_SOURCE_DIR=${ACADOS_DIR}"
echo "   export LD_LIBRARY_PATH=${ACADOS_DIR}/lib:\$LD_LIBRARY_PATH"
