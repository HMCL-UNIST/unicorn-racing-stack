#!/bin/bash
# run inside Docker container (no docker exec needed)
set -e

VEHICLE_NAME="${1:-rc_car_10th}"
RESOLUTION="${2:---fast}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

source /opt/ros/noetic/setup.bash
source "$HOME/catkin_ws/devel/setup.bash"

# opt-in tuning via TUNING=1 env var (default OFF)
#       This var is either set by direct shell usage inside container,
#       or forwarded from host by run.sh via `docker exec -e TUNING=...`.
#
#       Optional TUNING_NAME env var decouples the tuning file name
#       from --vehicle_name, enabling base × tuning combinations:
#         TUNING=1 TUNING_NAME=rc_car_10th ./run.sh rc_car_10th_v2
#           → base: params_rc_car_10th_v2.yml
#           → tuning: tuning_rc_car_10th.yml
TUNING_FLAG=""
TUNING_NAME_ARG=""
if [[ "${TUNING:-0}" == "1" ]]; then
    TUNING_FLAG="--tuning"
    if [[ -n "${TUNING_NAME:-}" ]]; then
        TUNING_NAME_ARG="--tuning_name=${TUNING_NAME}"
    fi
fi

echo "=============================="
echo " Fast GGV Pipeline (container)"
echo " Vehicle:    ${VEHICLE_NAME}"
echo " Resolution: ${RESOLUTION}"
# show tuning status in banner (includes tuning file name when ON)
if [[ -n "$TUNING_FLAG" ]]; then
    _TUNING_FILE_DISPLAY="tuning_${TUNING_NAME:-$VEHICLE_NAME}.yml"
    echo " Tuning:     ON (file: ${_TUNING_FILE_DISPLAY})"
else
    echo " Tuning:     OFF (set TUNING=1 to enable)"
fi
echo "=============================="

cd "$SCRIPT_DIR"

# 1. GGV generation
echo ""
echo "[1/3] Generating GGV diagrams..."
# pass tuning flag and optional tuning_name override
#       Empty strings collapse cleanly when unset, so defaults behave
#       identically to the original.
python3 fast_gen_gg_diagrams.py --vehicle_name $VEHICLE_NAME $RESOLUTION $TUNING_FLAG $TUNING_NAME_ARG

# 2. Diamond fitting
echo ""
echo "[2/3] Fitting diamond representation..."
# pass tuning flag + optional tuning_name to gen_diamond_representation.py
#       so the diamond post-process (gg_exp_scale, ax_*_scale, ay_scale) is
#       applied. Empty strings collapse when unset → identical to original
#       behavior.
# python3 gen_diamond_representation.py --vehicle_name $VEHICLE_NAME
python3 gen_diamond_representation.py --vehicle_name $VEHICLE_NAME $TUNING_FLAG $TUNING_NAME_ARG

# 3. Plot
echo ""
echo "[3/3] Generating plots..."
MPLBACKEND=Agg python3 plot_gg_diagrams.py --vehicle_name $VEHICLE_NAME

echo ""
echo "=============================="
echo " DONE"
echo " Output: $SCRIPT_DIR/output/$VEHICLE_NAME/"
echo "=============================="
ls -la "$SCRIPT_DIR/output/$VEHICLE_NAME/"*.png 2>/dev/null || echo " (no PNG files found)"
