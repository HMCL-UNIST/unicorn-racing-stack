#!/bin/bash
# One-shot wrapper that runs the FWBW velocity profile
# Usage:
#   ./run_fwbw.sh --track <smoothed.csv> --raceline <raceline.csv>
#   ./run_fwbw.sh --track my_track_3d_smoothed.csv --raceline my_track_3d_rc_car_10th_timeoptimal.csv
#   ./run_fwbw.sh --map-dir /path/to/map_folder   (map-dir mode)
#
# Options (all optional, with defaults):
#   --track      smoothed track CSV filename (relative to data/smoothed_track_data/)
#   --raceline   racing line CSV filename    (relative to data/global_racing_lines/)
#   --map-dir    map-dir mode: read the CSVs from a map folder, using the same layout
#                as run_pipeline.sh --map-dir (<name> = folder name):
#                  track    : <map_dir>/<name>_3d_smoothed.csv
#                  raceline : <map_dir>/<name>_3d_<vehicle>_timeoptimal.csv
#                  output   : <map_dir>/<name>_fwbw_<model>.json
#                An explicit --track / --raceline / --output-json overrides these.
#   --vehicle    vehicle name (default: rc_car_10th)
#   --params     params YAML  (default: params_rc_car_10th.yml)
#   --model      FBGA model   (default: lookup)
#   --n-laps     closed-loop laps (default: 3)
#   --spacing    resample spacing [m] (default: 0.1)
#   --no-build   skip the FBGA build

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FBGA_ROOT="$SCRIPT_DIR"
GLOBAL_LINE_DIR="$(cd "$SCRIPT_DIR/../../planner/3d_gb_optimizer/global_line" 2>/dev/null && pwd || echo "")"

# --- defaults ---------------------------------------------------------------
SKIP_BUILD=false
MAP_DIR=""
FWBW_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-build)  SKIP_BUILD=true; shift ;;
        --map-dir)   MAP_DIR="$2"; shift 2 ;;
        *)           FWBW_ARGS+=("$1"); shift ;;
    esac
done

# --- map-dir mode: derive track / raceline / output paths from the folder ---
has_fwbw_arg() {
    local a
    for a in "${FWBW_ARGS[@]}"; do
        [[ "$a" == "$1" || "$a" == "$1="* ]] && return 0
    done
    return 1
}

if [[ -n "$MAP_DIR" ]]; then
    if [[ ! -d "$MAP_DIR" ]]; then
        echo "ERROR: map directory not found: $MAP_DIR"
        exit 1
    fi
    MAP_DIR="$(cd "$MAP_DIR" && pwd)"
    MAP_NAME="$(basename "$MAP_DIR")"

    # same defaults as run_fwbw.py; picked up from the passed options if given
    MAP_VEHICLE="rc_car_10th"
    MAP_MODEL="lookup"
    for ((i = 0; i < ${#FWBW_ARGS[@]}; i++)); do
        case "${FWBW_ARGS[$i]}" in
            --vehicle)   MAP_VEHICLE="${FWBW_ARGS[$((i + 1))]}" ;;
            --vehicle=*) MAP_VEHICLE="${FWBW_ARGS[$i]#*=}" ;;
            --model)     MAP_MODEL="${FWBW_ARGS[$((i + 1))]}" ;;
            --model=*)   MAP_MODEL="${FWBW_ARGS[$i]#*=}" ;;
        esac
    done

    if ! has_fwbw_arg --track; then
        MAP_TRACK="$MAP_DIR/${MAP_NAME}_3d_smoothed.csv"
        if [[ ! -f "$MAP_TRACK" ]]; then
            echo "ERROR: track CSV not found in map dir: $MAP_TRACK"
            exit 1
        fi
        FWBW_ARGS+=(--track "$MAP_TRACK")
    fi
    if ! has_fwbw_arg --raceline; then
        MAP_RACELINE="$MAP_DIR/${MAP_NAME}_3d_${MAP_VEHICLE}_timeoptimal.csv"
        if [[ ! -f "$MAP_RACELINE" ]]; then
            echo "ERROR: raceline CSV not found in map dir: $MAP_RACELINE"
            exit 1
        fi
        FWBW_ARGS+=(--raceline "$MAP_RACELINE")
    fi
    if ! has_fwbw_arg --output-json; then
        FWBW_ARGS+=(--output-json "$MAP_DIR/${MAP_NAME}_fwbw_${MAP_MODEL}.json")
    fi
    echo "[FBGA] map-dir mode: $MAP_DIR"
fi

# --- 1. check / build the FBGA binary ---------------------------------------
RUNNER_BIN="$FBGA_ROOT/bin/GIGI_test_unicorn.exe"

if [[ ! -f "$RUNNER_BIN" ]]; then
    echo "[FBGA] binary not found -> starting build..."
    SKIP_BUILD=false
fi

if [[ "$SKIP_BUILD" == false && ! -f "$RUNNER_BIN" ]]; then
    echo "[FBGA] installing third_party dependencies..."
    cd "$FBGA_ROOT"
    if [[ -f third_party.sh ]]; then
        bash third_party.sh
    fi
    echo "[FBGA] building..."
    bash build.sh
    cd - > /dev/null

    if [[ ! -f "$RUNNER_BIN" ]]; then
        echo "ERROR: binary still missing after build: $RUNNER_BIN"
        exit 1
    fi
    echo "[FBGA] build complete: $RUNNER_BIN"
else
    echo "[FBGA] binary found: $RUNNER_BIN"
fi

# --- 2. check the global_line directory -------------------------------------
if [[ -z "$GLOBAL_LINE_DIR" || ! -d "$GLOBAL_LINE_DIR" ]]; then
    # fallback: assume FBGA lives under f110_utils/libs/
    REPO_ROOT="$(cd "$FBGA_ROOT/../../../" && pwd)"
    GLOBAL_LINE_DIR="$REPO_ROOT/planner/3d_gb_optimizer/global_line"
fi

if [[ ! -d "$GLOBAL_LINE_DIR" ]]; then
    echo "ERROR: global_line directory not found"
    echo "  tried: $GLOBAL_LINE_DIR"
    exit 1
fi

RUN_FWBW="$GLOBAL_LINE_DIR/global_racing_line/run_fwbw.py"
if [[ ! -f "$RUN_FWBW" ]]; then
    echo "ERROR: run_fwbw.py not found: $RUN_FWBW"
    exit 1
fi

echo "[FBGA] global_line: $GLOBAL_LINE_DIR"
echo "[FBGA] run_fwbw.py: $RUN_FWBW"

# --- 3. check Python dependencies --------------------------------------------
python3 -c "import numpy, pandas, yaml, scipy, matplotlib" 2>/dev/null || {
    echo "[FBGA] installing Python dependencies..."
    pip install numpy pandas pyyaml scipy matplotlib
}

# --- 4. run run_fwbw.py ------------------------------------------------------
export FBGA_ROOT="$FBGA_ROOT"

echo ""
echo "======================================================"
echo " FWBW Velocity Profiler"
echo "  FBGA_ROOT   : $FBGA_ROOT"
echo "  run_fwbw.py : $RUN_FWBW"
echo "  args        : ${FWBW_ARGS[*]}"
echo "======================================================"
echo ""

cd "$GLOBAL_LINE_DIR"
python3 global_racing_line/run_fwbw.py "${FWBW_ARGS[@]}"
