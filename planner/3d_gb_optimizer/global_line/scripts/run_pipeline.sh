#!/bin/bash
# Full pipeline: bag + wall.pcd → CSV → gen3D → smooth → global racing line
# Usage: ./scripts/run_pipeline.sh --track <name> [--vehicle <name>] [--gg-mode <mode>] [--safety-distance <m>] [--smooth-step <m>] [--opt-step <m>] [--output <name>]
# Expects: bag/<name>/<name>.bag  and  pcd/<name>/wall.pcd
# Example: ./scripts/run_pipeline.sh --track localization_bridge_2 --vehicle rc_car_10th --gg-mode diamond

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

PYTHON="python3"

# --- Parse arguments -----------------------------------------------------------
MAP_DIR=""
TRACK_BASE=""
VEHICLE="rc_car_10th"
GG_MODE="diamond"
SAFETY_DISTANCE="0.3"
BIN_SIZE="0.05"
SMOOTH_STEP="0.05"
OPT_STEP="0.2"
OUTPUT="global_waypoints"
START_FROM="1"
# bridge mu scale baked into raceline (1.0=as-is)
G_WEIGHT="1.0"
# bridge centerline / heading-lock.
# Yaml at <map>/bridge_sectors.yaml stores positions only (start/end).
# Tuning knobs (pre/post/force_center/chi_max) are GLOBAL — applied to
# every bridge in the yaml. Default values reproduce legacy behavior:
# pre=post=0 (no extension), force_center=false, chi_max<0 (disabled).
# Missing yaml ⇒ silent no-op.
BRIDGE_CONSTRAINTS_ENABLE="true"
BRIDGE_PRE_M="0.0"
BRIDGE_POST_M="0.0"
BRIDGE_FORCE_CENTER="false"
BRIDGE_CHI_MAX_RAD="-1.0"
RAW_CSV="auto"
GEN3D_CSV="auto"
SMOOTHED_CSV="auto"
RACELINE_CSV="auto"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --map-dir)         MAP_DIR="$2"; shift 2 ;;
        --track)           TRACK_BASE="$2"; shift 2 ;;
        --vehicle)         VEHICLE="$2"; shift 2 ;;
        --gg-mode)         GG_MODE="$2"; shift 2 ;;
        --safety-distance) SAFETY_DISTANCE="$2"; shift 2 ;;
        --bin-size)        BIN_SIZE="$2"; shift 2 ;;
        --smooth-step)     SMOOTH_STEP="$2"; shift 2 ;;
        --opt-step)        OPT_STEP="$2"; shift 2 ;;
        --output)          OUTPUT="$2"; shift 2 ;;
        --start-from)      START_FROM="$2"; shift 2 ;;
        --g-weight)        G_WEIGHT="$2"; shift 2 ;;
        # bridge constraints + global tuning (yaml read at NLP setup)
        --bridge-constraints-enable) BRIDGE_CONSTRAINTS_ENABLE="$2"; shift 2 ;;
        --bridge-pre-m)              BRIDGE_PRE_M="$2"; shift 2 ;;
        --bridge-post-m)             BRIDGE_POST_M="$2"; shift 2 ;;
        --bridge-force-center)       BRIDGE_FORCE_CENTER="$2"; shift 2 ;;
        --bridge-chi-max-rad)        BRIDGE_CHI_MAX_RAD="$2"; shift 2 ;;
        --raw-csv)         RAW_CSV="$2"; shift 2 ;;
        --gen3d-csv)       GEN3D_CSV="$2"; shift 2 ;;
        --smoothed-csv)    SMOOTHED_CSV="$2"; shift 2 ;;
        --raceline-csv)    RACELINE_CSV="$2"; shift 2 ;;
        __*) shift ;;  # ignore ROS-injected args (__name, __log, etc.)
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Resolve map directory and track base name
if [[ -n "$MAP_DIR" ]]; then
    # map-dir mode: all I/O in the map folder
    [[ -z "$TRACK_BASE" ]] && TRACK_BASE="$(basename "$MAP_DIR")"
else
    # legacy mode: use track name with local data/ folders
    [[ -z "$TRACK_BASE" ]] && TRACK_BASE="localization_bridge_2"
    MAP_DIR=""
fi

# Vehicle config (gg_vehicle = vehicle)
GG_VEHICLE="${VEHICLE}"

if [[ -n "$MAP_DIR" ]]; then
    # -- map-dir mode: bag/pcd/csv all in MAP_DIR --
    BAG="${MAP_DIR}/${TRACK_BASE}.bag"
    WALL="${MAP_DIR}/wall.pcd"
    [[ "$RAW_CSV" == "auto" ]]      && RAW_CSV="${MAP_DIR}/${TRACK_BASE}_bounds_3d.csv"
    [[ "$GEN3D_CSV" == "auto" ]]    && GEN3D_CSV="${MAP_DIR}/${TRACK_BASE}_3d.csv"
    [[ "$SMOOTHED_CSV" == "auto" ]] && SMOOTHED_CSV="${MAP_DIR}/${TRACK_BASE}_3d_smoothed.csv"
    [[ "$RACELINE_CSV" == "auto" ]] && RACELINE_CSV="${MAP_DIR}/${TRACK_BASE}_3d_${VEHICLE}_timeoptimal.csv"
else
    # -- legacy mode: original relative paths --
    BAG="bag/${TRACK_BASE}/${TRACK_BASE}.bag"
    WALL="pcd/${TRACK_BASE}/wall.pcd"
    [[ "$RAW_CSV" == "auto" ]]      && RAW_CSV="data/raw_track_data/${TRACK_BASE}_bounds_3d.csv"
    [[ "$GEN3D_CSV" == "auto" ]]    && GEN3D_CSV="data/3d_track_data/${TRACK_BASE}_3d.csv"
    [[ "$SMOOTHED_CSV" == "auto" ]] && SMOOTHED_CSV="data/smoothed_track_data/${TRACK_BASE}_3d_smoothed.csv"
    [[ "$RACELINE_CSV" == "auto" ]] && RACELINE_CSV="data/global_racing_lines/${TRACK_BASE}_3d_${VEHICLE}_timeoptimal.csv"
    mkdir -p data/raw_track_data data/3d_track_data data/smoothed_track_data data/global_racing_lines
fi

echo "======================================================"
echo " Pipeline : ${TRACK_BASE}"
echo "  bag     : ${BAG}"
echo "  wall    : ${WALL}"
echo "  vehicle : ${VEHICLE}  (gg: ${GG_VEHICLE})"
echo "  gg_mode : ${GG_MODE}"
echo "  bin_size: ${BIN_SIZE} m"
echo "  safety  : ${SAFETY_DISTANCE} m"
echo "  smooth  : ${SMOOTH_STEP} m"
echo "  opt     : ${OPT_STEP} m"
echo "  raw_csv : ${RAW_CSV}"
echo "  gen3d   : ${GEN3D_CSV}"
echo "  smoothed: ${SMOOTHED_CSV}"
echo "  raceline: ${RACELINE_CSV}"
echo "  g_weight: ${G_WEIGHT}"
echo "  bridge  : enable=${BRIDGE_CONSTRAINTS_ENABLE}, pre=${BRIDGE_PRE_M}m, post=${BRIDGE_POST_M}m, force_center=${BRIDGE_FORCE_CENTER}, chi_max=${BRIDGE_CHI_MAX_RAD}rad"
echo "======================================================"

# --- Step 1: Bag + wall.pcd → boundary CSV -------------------------------------
# start_from lets the pipeline start from a given step
if (( START_FROM <= 1 )); then
echo ""
echo "[1/7] Generating boundary CSV from bag + wall.pcd ..."
$PYTHON track_processing/csv_generators/gen_track_boundary_csv.py \
    --bag "$BAG" \
    --wall "$WALL" \
    --pose-topic /glim_ros/imu_pose \
    --min-dist 0.25 \
    --bin-size "$BIN_SIZE" \
    --smooth-window 1 \
    --output "$RAW_CSV" \
    --no-plot
else
    echo "[1/7] Skipped (start_from=${START_FROM})"
fi

# --- Step 2: Close loop (set last row = first row) -----------------------------
if (( START_FROM <= 2 )); then
echo ""
echo "[2/7] Closing loop ..."
$PYTHON - <<PYEOF
import csv
path = '${RAW_CSV}'
with open(path) as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)
rows[-1] = dict(rows[0])
with open(path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f'  Loop closed: {path}  ({len(rows)} points)')
PYEOF
else
    echo "[2/7] Skipped (start_from=${START_FROM})"
fi

# --- Step 3: Generate 3D track data (v2: boundary-perpendicular tangent) -------
if (( START_FROM <= 3 )); then
echo ""
echo "[3/7] Generating 3D track data (v2) ..."
$PYTHON - <<PYEOF
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'track_processing')
from gen_3d_track_data_v2 import generate_3d_from_3d_track_bounds_v2
generate_3d_from_3d_track_bounds_v2(
    path='${RAW_CSV}',
    out_path='${GEN3D_CSV}',
    ignore_banking=False,
    visualize=False,
)
PYEOF
else
    echo "[3/7] Skipped (start_from=${START_FROM})"
fi

# --- Step 4: Track smoothing ----------------------------------------------------
if (( START_FROM <= 4 )); then
echo ""
echo "[4/7] Smoothing track ..."
$PYTHON - <<PYEOF
import sys
sys.path.insert(0, 'src')
from track3D import Track3D
weights = {
    'w_c': 1e5,
    'w_l': 1e5,
    'w_r': 1e5,
    'w_theta': 1e2,
    'w_mu': 1e2,
    'w_phi': 1e1,
    'w_nl': 1e-2,
    'w_nr': 1e-2,
}
track_handler = Track3D()
track_handler.smooth_track(
    out_path='${SMOOTHED_CSV}',
    weights=weights,
    in_path='${GEN3D_CSV}',
    step_size=${SMOOTH_STEP},
    visualize=False,
)
PYEOF
else
    echo "[4/7] Skipped (start_from=${START_FROM})"
fi

# --- Step 5: Global racing line optimization ------------------------------------
if (( START_FROM <= 5 )); then
echo ""
echo "[5/7] Optimizing global racing line ..."
$PYTHON - <<PYEOF
import sys, os, re
sys.path.insert(0, 'src')
sys.path.insert(0, 'global_racing_line')

with open('global_racing_line/gen_global_racing_line.py') as f:
    src = f.read()

# Override track and raceline names (use basename from CSV paths)
import os
smoothed_basename = os.path.basename('${SMOOTHED_CSV}')
raceline_basename = os.path.basename('${RACELINE_CSV}')
src = re.sub(r"'track_name'\s*:.*",    f"'track_name': '{smoothed_basename}',", src)
src = re.sub(r"'raceline_name'\s*:.*", f"'raceline_name': '{raceline_basename}',", src)
src = re.sub(r"'vehicle_name'\s*:.*",  f"'vehicle_name': '${VEHICLE}',", src)
src = re.sub(r"'gg_vehicle_name'\s*:.*", f"'gg_vehicle_name': '${GG_VEHICLE}',", src)
src = re.sub(r"'safety_distance'\s*:.*", f"'safety_distance': ${SAFETY_DISTANCE},", src)
src = re.sub(r"'gg_mode'\s*:.*", f"'gg_mode': '${GG_MODE}',", src)
src = re.sub(r"'step_size_opt'\s*:.*", f"'step_size_opt': ${OPT_STEP},", src)
# override bridge mu scale
src = re.sub(r"'g_weight'\s*:.*", f"'g_weight': ${G_WEIGHT},", src)
# forward bridge args so gen_global_racing_line.py can locate
# <map>/bridge_sectors.yaml (positions only) and apply the global
# tuning knobs. Bash bool ('true'/'false') → Python bool.
def _to_py_bool(s):
    s = s.strip().lower()
    return "True" if s == "true" else ("False" if s == "false" else s.capitalize())
_b_enable = _to_py_bool("${BRIDGE_CONSTRAINTS_ENABLE}")
_b_force  = _to_py_bool("${BRIDGE_FORCE_CENTER}")
src = re.sub(r"'bridge_constraints_enable'\s*:.*",
             f"'bridge_constraints_enable': {_b_enable},", src)
src = re.sub(r"'bridge_yaml_dir'\s*:.*",
             f"'bridge_yaml_dir': '${MAP_DIR}',", src)
src = re.sub(r"'bridge_pre_m'\s*:.*",
             f"'bridge_pre_m': ${BRIDGE_PRE_M},", src)
src = re.sub(r"'bridge_post_m'\s*:.*",
             f"'bridge_post_m': ${BRIDGE_POST_M},", src)
src = re.sub(r"'bridge_force_center'\s*:.*",
             f"'bridge_force_center': {_b_force},", src)
src = re.sub(r"'bridge_chi_max_rad'\s*:.*",
             f"'bridge_chi_max_rad': ${BRIDGE_CHI_MAX_RAD},", src)
# per-sector safety_distance is consumed via the live /safety_sector_params/...
# rosparam tree at NLP setup time (populated by safety_sector_server.py).
# No CLI arg / yaml path injection needed here — the rosparam tree is
# already populated by the time roslaunch reaches run_pipeline.sh.

# Override data directories when using map-dir mode
map_dir = '${MAP_DIR}'
if map_dir:
    src = re.sub(r"track_path\s*=\s*os\.path\.join\(data_path,\s*'smoothed_track_data'\)",
                 f"track_path = '{map_dir}'", src)
    src = re.sub(r"raceline_out_path\s*=\s*os\.path\.join\(data_path,\s*'global_racing_lines'\)",
                 f"raceline_out_path = '{map_dir}'", src)

exec(compile(src, 'gen_global_racing_line.py', 'exec'), {'__name__': '__main__', '__file__': 'global_racing_line/gen_global_racing_line.py'})
PYEOF
else
    echo "[5/7] Skipped (start_from=${START_FROM})"
fi

# --- Step 6: Export global waypoints JSON -------------------------------------
if [[ -n "$MAP_DIR" ]]; then
    WAYPOINTS_JSON="${MAP_DIR}/${OUTPUT}.json"
else
    WAYPOINTS_JSON="data/global_racing_lines/${OUTPUT}.json"
fi

if (( START_FROM <= 6 )); then
echo ""
echo "[6/7] Exporting global waypoints ..."
$PYTHON - <<PYEOF
import sys, os, re
sys.path.insert(0, 'src')
sys.path.insert(0, 'global_racing_line')

with open('global_racing_line/export_global_waypoints.py') as f:
    src = f.read()

import os
smoothed_basename = os.path.basename('${SMOOTHED_CSV}')
raceline_basename = os.path.basename('${RACELINE_CSV}')
src = re.sub(r"'track_name'\s*:.*",    f"'track_name': '{smoothed_basename}',", src)
src = re.sub(r"'raceline_name'\s*:.*", f"'raceline_name': '{raceline_basename}',", src)
src = re.sub(r"'output_name'\s*:.*",   f"'output_name': '${OUTPUT}.json',", src)

# Override data directories when using map-dir mode
map_dir = '${MAP_DIR}'
if map_dir:
    smoothed_path = '${SMOOTHED_CSV}'
    raceline_path = '${RACELINE_CSV}'
    waypoints_path = '${WAYPOINTS_JSON}'
    src = re.sub(r"track_path\s*=\s*os\.path\.join\(data_path,\s*'smoothed_track_data',.*\)",
                 f"track_path = '{smoothed_path}'", src)
    src = re.sub(r"raceline_path\s*=\s*os\.path\.join\(data_path,\s*'global_racing_lines',\s*params\['raceline_name'\]\)",
                 f"raceline_path = '{raceline_path}'", src)
    src = re.sub(r"output_path\s*=\s*os\.path\.join\(data_path,\s*'global_racing_lines',\s*params\['output_name'\]\)",
                 f"output_path = '{waypoints_path}'", src)

exec(compile(src, 'export_global_waypoints.py', 'exec'), {'__name__': '__main__', '__file__': 'global_racing_line/export_global_waypoints.py'})
PYEOF
else
    echo "[6/7] Skipped (start_from=${START_FROM})"
fi

echo ""
echo "======================================================"
echo " Done!"
echo "  Raw CSV    : ${RAW_CSV}"
echo "  3D track   : ${GEN3D_CSV}"
echo "  Smoothed   : ${SMOOTHED_CSV}"
echo "  Racing line: ${RACELINE_CSV}"
echo "  Waypoints  : ${WAYPOINTS_JSON}"
echo "======================================================"

# --- Step 7: Visualize ----------------------------------------------------------
if (( START_FROM <= 7 )); then
echo ""
echo "[7/7] Visualizing ..."
$PYTHON visualization/plot_raceline.py \
    --track    "$SMOOTHED_CSV" \
    --raceline "$RACELINE_CSV" \
    --raw      "$RAW_CSV" \
    --gen      "$GEN3D_CSV"
else
    echo "[7/7] Skipped (start_from=${START_FROM})"
fi
