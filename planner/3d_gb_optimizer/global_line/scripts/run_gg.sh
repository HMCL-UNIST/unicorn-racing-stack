#!/bin/bash
# GG diagram generation pipeline (shell wrapper)
# Usage: ./run_gg.sh [--vehicle <name>] [--start-from <1|2|3>]
#   Step 1: GG diagram calculation (NLP solve)
#   Step 2: Diamond representation (rho → ax_max/ax_min/ay_max/gg_exponent)
#   Step 3: Plot GG diagrams

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GG_DIR="$SCRIPT_DIR/../../gg_diagram_generation"
PYTHON="python3"

# --- Parse arguments -----------------------------------------------------------
VEHICLE="rc_car_10th"
START_FROM=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --vehicle)     VEHICLE="$2"; shift 2 ;;
        --start-from)  START_FROM="$2"; shift 2 ;;
        __*) shift ;;  # ignore ROS-injected args (__name, __log, etc.)
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "======================================================"
echo " GG Diagram Pipeline"
echo "  vehicle : ${VEHICLE}"
echo "  start   : step ${START_FROM}"
echo "======================================================"

# --- Step 1: GG diagram calculation -------------------------------------------
if [[ $START_FROM -le 1 ]]; then
    echo ""
    echo "[1/3] Calculating GG diagrams (NLP solve) ..."
    $PYTHON "$GG_DIR/gen_gg_diagrams.py" --vehicle_name "$VEHICLE"
fi

# --- Step 2: Diamond representation ------------------------------------------
if [[ $START_FROM -le 2 ]]; then
    echo ""
    echo "[2/3] Generating diamond representation ..."
    $PYTHON "$GG_DIR/gen_diamond_representation.py" --vehicle_name "$VEHICLE"
fi

# --- Step 3: Plot -------------------------------------------------------------
if [[ $START_FROM -le 3 ]]; then
    echo ""
    echo "[3/3] Plotting GG diagrams ..."
    $PYTHON "$GG_DIR/plot_gg_diagrams.py" --vehicle_name "$VEHICLE"
fi

echo ""
echo "======================================================"
echo " GG Diagram Pipeline Done!"
echo "  vehicle : ${VEHICLE}"
echo "======================================================"
