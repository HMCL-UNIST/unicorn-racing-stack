#!/bin/bash
# apply_isolation.sh — run inside the container after ROS launch is up
#
# Goal:
#   Protect the entire control/perception critical path from glim_ros's
#   ~10s-period backend bursts. Pin each critical node to dedicated cores
#   so glim cannot contend with them at the hardware (HT sibling, L1/L2)
#   level, not just the scheduler level.
#
#   Measured on the car (per-process CPU-wait aggregates):
#     - glim BURST(>1260%) -> state_machine wait avg=251.8ms (>50ms 98.7%)
#     - glim QUIET         -> state_machine wait avg= 29.9ms (>50ms  7.9%)
#     - glim CPU% vs sm wait monotonic (>=1200% -> sm avg 232ms wait)
#     - rslidar 100ms wait events 9 total, all coincident with glim burst
#       (lidar frame backed up by exactly one frame interval)
#
# Core domains (Core Ultra 7 155H, 22 threads, hybrid P+E+LP):
#     ctrl  -> cpu 0, 5         1 P-core HT pair, SCHED_FIFO
#                               (state_machine + controller_manager
#                                + vesc_driver + frenet_conversion_server)
#     lidar -> selectable via LIDAR_MODE (see below)
#     glim  -> auto-derived from LIDAR_MODE so cores don't overlap
#     cam   -> cpu 20, 21       LP E-cores, ffmpeg recorder (user's existing)
#
# LIDAR_MODE options (env var, default "lp"):
#     lp    rslidar -> cpu 20,21 (shares LP E-cores with camera).
#                      glim gets full 18 threads (cpu 1-4,6-19).
#                      Lightest footprint. Measured rslidar load on LP
#                      is ~13% of 1 core (vs ~5% on P-core, since LP
#                      clock is ~half). Combined with camera ~80% on 1
#                      LP thread, total ~90% / 200% LP capacity.
#                      OK as long as camera+rslidar peak together stays
#                      under 200%. Glim cannot encroach (different domain).
#     e     rslidar -> cpu 12 (single E-core, no HT). Clean isolation
#                      with no HT contamination. glim 17 threads
#                      (cpu 1-4,6-11,13-19). Good balance.
#     p     rslidar -> cpu 1,2 (one P-core HT pair). Fastest packet
#                      decode, lowest latency. Wastes 1 full P-core for
#                      a ~5% workload. glim 16 threads (cpu 3-4,6-19).
#                      Use if measurements show LP/E rslidar wait jitter.
#     off   no rslidar pinning. glim 18 threads (cpu 1-4,6-19).
#
# Usage (inside container):
#     sudo bash stack_master/scripts/apply_isolation.sh
#     LIDAR_MODE=p   sudo -E bash ...            # rslidar on P-core HT pair
#     LIDAR_MODE=e   sudo -E bash ...            # rslidar on E-core 12
#     LIDAR_MODE=lp  sudo -E bash ...            # rslidar with camera (default)
#     LIDAR_MODE=off sudo -E bash ...            # don't pin rslidar
#     CTRL_CORES="0,1" sudo -E bash ...          # change ctrl core set
#     LIDAR_CORES="6,7" GLIM_CORES="0-5,8-19" \  # explicit override (ignores MODE)
#         sudo -E bash ...
#     RT_PRIO="80"    sudo -E bash ...           # bump RT priority
#     NO_RT=1   sudo -E bash ...                 # skip SCHED_FIFO
#     NO_GOV=1  sudo -E bash ...                 # skip governor change
#     NO_LIDAR=1 sudo -E bash ...                # skip rslidar pinning entirely
#
# Verify with monitor:
#     - state_machine wait>50ms cumulative count stays ~0 (was ~3/min)
#     - rslidar_sdk_node wait>50ms cumulative stays 0
#     - rostopic hz /glim_ros/base_odom stays at nominal ~75Hz
#     - rostopic hz /rslidar_points stays at nominal ~10Hz

set -uo pipefail

CTRL_CORES="${CTRL_CORES:-${SM_CORES:-0,5}}"   # back-compat: SM_CORES still honored
RT_PRIO="${RT_PRIO:-50}"

# LIDAR_MODE -> default cores for rslidar + matching glim mask.
# Explicit LIDAR_CORES / GLIM_CORES env override the mode preset.
LIDAR_MODE="${LIDAR_MODE:-lp}"
case "$LIDAR_MODE" in
  lp)   _lidar_default="20,21"; _glim_default="1-4,6-19" ;;
  e)    _lidar_default="12";    _glim_default="1-4,6-11,13-19" ;;
  p)    _lidar_default="1,2";   _glim_default="3-4,6-19" ;;
  off)  _lidar_default="";      _glim_default="1-4,6-19" ;;
  *)    echo "[ERR] LIDAR_MODE must be lp|e|p|off (got: $LIDAR_MODE)" >&2; exit 2 ;;
esac
LIDAR_CORES="${LIDAR_CORES:-$_lidar_default}"
GLIM_CORES="${GLIM_CORES:-$_glim_default}"

die() { echo "[ERR] $*" >&2; exit 1; }

pin_all_threads() {  # pin_all_threads <pid> <core_list>
  local pid="$1" cores="$2" count=0
  for tid_dir in /proc/"$pid"/task/*/; do
    local t
    t=$(basename "$tid_dir")
    if taskset -cp "$cores" "$t" >/dev/null 2>&1; then
      count=$((count+1))
    fi
  done
  echo "$count"
}

# ---------- pre-check ----------

SM=$(pgrep -f 3d_state_machine_node     | head -1)
CM=$(pgrep -f controller_manager        | head -1)
GP=$(pgrep -f glim_rosnode              | head -1)
VESC=$(pgrep -f vesc_driver_node        | head -1)
FRENET=$(pgrep -f frenet_conversion_server_node | head -1)
RSL=$(pgrep -f rslidar_sdk_node         | head -1)

echo "[info] LIDAR_MODE=$LIDAR_MODE  lidar=$LIDAR_CORES  glim=$GLIM_CORES  ctrl=$CTRL_CORES"
echo "[info] sm=$SM  cm=$CM  vesc=$VESC  frenet=$FRENET  glim=$GP  rslidar=$RSL"
[ -z "$SM" ] && die "state_machine pid not found"
[ -z "$CM" ] && die "controller_manager pid not found"
[ -z "$GP" ] && die "glim_rosnode pid not found"
# vesc / frenet / rslidar are optional (warn but continue if missing)
[ -z "$VESC" ]   && echo "  [WARN] vesc_driver pid not found - skipping"
[ -z "$FRENET" ] && echo "  [WARN] frenet_conversion_server pid not found - skipping"
[ -z "$RSL" ]    && echo "  [WARN] rslidar_sdk_node pid not found - skipping"

# ---------- 1. CPU affinity ----------

echo ""
echo "[1/4] CPU affinity"

# Control critical (multi-process, all on ctrl cores)
taskset -cp "$CTRL_CORES" "$SM" > /dev/null && echo "  sm     ($SM) -> cpu $CTRL_CORES"
taskset -cp "$CTRL_CORES" "$CM" > /dev/null && echo "  cm     ($CM) -> cpu $CTRL_CORES"
[ -n "$VESC" ]   && taskset -cp "$CTRL_CORES" "$VESC"   > /dev/null && echo "  vesc   ($VESC) -> cpu $CTRL_CORES"
[ -n "$FRENET" ] && taskset -cp "$CTRL_CORES" "$FRENET" > /dev/null && echo "  frenet ($FRENET) -> cpu $CTRL_CORES"

# Lidar input — only pin if LIDAR_CORES is non-empty (mode=off skips)
if [ -n "$RSL" ] && [ "${NO_LIDAR:-0}" != "1" ] && [ -n "$LIDAR_CORES" ]; then
  n=$(pin_all_threads "$RSL" "$LIDAR_CORES")
  echo "  rslidar ($RSL, $n threads) -> cpu $LIDAR_CORES (LIDAR_MODE=$LIDAR_MODE)"
elif [ -n "$RSL" ]; then
  echo "  rslidar ($RSL) -> not pinned (LIDAR_MODE=off or NO_LIDAR=1)"
fi

# glim (all 51 threads)
n=$(pin_all_threads "$GP" "$GLIM_CORES")
echo "  glim   ($GP, $n threads) -> cpu $GLIM_CORES"

# ---------- 2. RT priority (SCHED_FIFO on control nodes only) ----------

if [ "${NO_RT:-0}" != "1" ]; then
  echo ""
  echo "[2/4] RT priority (SCHED_FIFO, prio=$RT_PRIO)"
  if chrt -f -p "$RT_PRIO" "$SM" 2>/dev/null; then
    echo "  sm     ($SM) -> SCHED_FIFO $RT_PRIO"
  else
    echo "  [WARN] chrt failed -- needs CAP_SYS_NICE or root"
    echo "         one-shot fix: setcap cap_sys_nice+ep /usr/bin/chrt"
  fi
  chrt -f -p "$RT_PRIO" "$CM"   2>/dev/null && echo "  cm     ($CM) -> SCHED_FIFO $RT_PRIO"
  [ -n "$VESC" ]   && chrt -f -p "$RT_PRIO" "$VESC"   2>/dev/null && echo "  vesc   ($VESC) -> SCHED_FIFO $RT_PRIO"
  [ -n "$FRENET" ] && chrt -f -p "$RT_PRIO" "$FRENET" 2>/dev/null && echo "  frenet ($FRENET) -> SCHED_FIFO $RT_PRIO"
else
  echo ""
  echo "[2/4] RT priority skipped (NO_RT=1)"
fi

# ---------- 3. cpufreq governor ----------

if [ "${NO_GOV:-0}" != "1" ]; then
  echo ""
  echo "[3/4] cpufreq governor → performance"
  changed=0
  for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    cur=$(cat "$c" 2>/dev/null)
    if [ "$cur" != "performance" ]; then
      echo performance > "$c" 2>/dev/null && changed=$((changed+1))
    fi
  done
  echo "  $changed cores → performance"
else
  echo ""
  echo "[3/4] governor skipped (NO_GOV=1)"
fi

# ---------- 4. verify ----------

echo ""
echo "[4/4] verify"
printf "  sm:      affinity=%s  sched=%s\n" \
  "$(taskset -p $SM 2>/dev/null | awk '{print $NF}')" \
  "$(chrt -p $SM 2>&1 | grep -i policy | awk -F: '{print $2}' | xargs)"
printf "  cm:      affinity=%s  sched=%s\n" \
  "$(taskset -p $CM 2>/dev/null | awk '{print $NF}')" \
  "$(chrt -p $CM 2>&1 | grep -i policy | awk -F: '{print $2}' | xargs)"
[ -n "$VESC" ] && printf "  vesc:    affinity=%s  sched=%s\n" \
  "$(taskset -p $VESC 2>/dev/null | awk '{print $NF}')" \
  "$(chrt -p $VESC 2>&1 | grep -i policy | awk -F: '{print $2}' | xargs)"
[ -n "$FRENET" ] && printf "  frenet:  affinity=%s  sched=%s\n" \
  "$(taskset -p $FRENET 2>/dev/null | awk '{print $NF}')" \
  "$(chrt -p $FRENET 2>&1 | grep -i policy | awk -F: '{print $2}' | xargs)"
[ -n "$RSL" ] && printf "  rslidar: affinity=%s (sample, main)\n" \
  "$(taskset -p $RSL 2>/dev/null | awk '{print $NF}')"
printf "  glim:    affinity=%s (sample, main)\n" \
  "$(taskset -p $GP 2>/dev/null | awk '{print $NF}')"
printf "  cpu0 governor=%s\n"  "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"

echo ""
echo "DONE. Verify with monitor:"
echo "  - state_machine wait>50ms : ~0/min (was ~3/min)"
echo "  - rslidar_sdk_node wait   : 0      (was 9 in 60min)"
echo "  - rostopic hz /glim_ros/base_odom : stays ~75Hz"
echo "  - rostopic hz /rslidar_points     : stays ~10Hz"
