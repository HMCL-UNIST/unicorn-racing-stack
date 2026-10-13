#!/bin/bash
# record_with_metrics.sh
# Unified rosbag + system metrics recorder.
# Designed to run inside the icra2026 docker container.
#
# Usage:
#   record_with_metrics.sh [profile] [name]
#     profile : light (default) | overtake | slam | all
#     name    : optional tag appended to filenames
#
# Outputs to <race_stack>/bag/:
#   <ts>_<name>_<profile>.bag (rotated every 60s, lz4-compressed)
#   <ts>_<name>_<profile>_metrics.csv  (1 Hz: temp / freq / throttle / disk / load)
#   <ts>_<name>_<profile>_meta.txt     (RAPL / PPD / thermald / ROS env snapshot)
#
# Stop with Ctrl-C: rosbag is signalled cleanly and metrics loop ends.

set -uo pipefail

PROFILE="${1:-light}"
NAME="${2:-run}"
TS="$(date +%Y-%m-%d-%H-%M-%S)"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RACE_STACK_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BAG_ROOT="$RACE_STACK_DIR/bag"
RUN_DIR="$BAG_ROOT/debug/${TS}_${NAME}_${PROFILE}"
mkdir -p "$RUN_DIR"

BAG_FILE="$RUN_DIR/run.bag"
METRICS_CSV="$RUN_DIR/metrics.csv"
META_TXT="$RUN_DIR/meta.txt"
ROSBAG_LOG="$RUN_DIR/rosbag.log"

# Exclude regex per profile.
# Built from analysis of recorded race bags.
case "$PROFILE" in
  light|overtake)
    # Drop: /relay/* duplicates, heavy PointCloud2 (lidar/camera/aligned), gazebo,
    #       camera, RViz-only markers, parameter_*, joint_states, blink.
    # Keep: VESC, car_state, glim_ros base_odom (control-essential), state_machine,
    #       behavior_strategy, local/global waypoint bodies, planner/{avoidance,recovery,start_wpnts}
    #       bodies, mpc_auto/*, opponent_prediction body topics, lap, joy, tf, rosout.
    # keep prior_map (latched, once), odom_corrected, lookahead, lap_*, mux vesc cmds
    #   - prior_map kept (NOT mini): /glim_ros/prior_map is latched -> recorded once in run_0.bag
    #   - odom_corrected kept: only the heavy *aligned_points*_corrected clouds are dropped,
    #     not /glim_ros/odom_corrected (was previously killed by the broad /.*_corrected rule)
    #   - lap_marker kept: narrowed /.*_marker so it no longer eats /lap_marker / /lap_marker_array
    #   - lookahead_point + vesc high_level(nav_1) + low_level output pass through -a (not excluded)
    EXCLUDE='(/relay/.*|/.*aligned_points.*_corrected|/.*/raw_aligned_points.*|/rslidar_points|/rslidar_imu_data|/livox/(lidar|imu)|/glim_ros/(points|aligned_points|mini_prior_map|raw_aligned_points)|/pcd_map|/camera/.*|/gazebo/.*|/clock|/joint_states|/unicorn/.*|.*/markers|/opponent_prediction_markerarray|.*/parameter_(descriptions|updates)|/rosout_agg|/.*/update_full|/.*/update|/opponent_predict/(beginn|end)|/blink1/blink|/diagnostics)'
    ;;
  slam)
    # Keep one lidar (rslidar_points) + one corrected aligned cloud for SLAM debugging.
    EXCLUDE='(/relay/.*|/.*/raw_aligned_points.*|/rslidar_imu_data|/livox/(lidar|imu)|/glim_ros/(points|aligned_points|prior_map|raw_aligned_points|points_corrected)|/pcd_map|/camera/.*|/gazebo/.*|/clock|/joint_states|/unicorn/.*|.*/markers|.*_marker|.*/parameter_(descriptions|updates)|/rosout_agg|/.*/update_full|/.*/update|/opponent_predict/(beginn|end)|/opponent_prediction_markerarray|/blink1/blink|/diagnostics)'
    ;;
  all)
    EXCLUDE=''
    echo "[WARN] profile=all : nothing excluded. Disk write will spike (~65 MB/s observed)."
    ;;
  *)
    echo "Unknown profile: $PROFILE" >&2
    echo "Usage: $0 [light|overtake|slam|all] [name]" >&2
    exit 1
    ;;
esac

if ! command -v rosbag >/dev/null 2>&1; then
  echo "[ERR] rosbag not found. Source ROS env first:" >&2
  echo "  source /opt/ros/noetic/setup.bash && source \$HOME/catkin_ws/devel/setup.bash" >&2
  exit 1
fi

# --- meta snapshot ---
{
  echo "=== record_with_metrics.sh meta ==="
  echo "ts=$TS"
  echo "profile=$PROFILE"
  echo "name=$NAME"
  echo "host=$(hostname)"
  echo "kernel=$(uname -r)"
  echo "cpu=$(grep -m1 '^model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
  echo "nproc=$(nproc)"
  echo "ros_master=${ROS_MASTER_URI:-unset}"
  echo "ros_ip=${ROS_IP:-unset}"
  echo "ros_hostname=${ROS_HOSTNAME:-unset}"
  echo "exclude_regex=$EXCLUDE"
  echo "--- RAPL constraints ---"
  for c in /sys/class/powercap/intel-rapl:0/constraint_0 /sys/class/powercap/intel-rapl:0/constraint_1; do
    n=$(cat ${c}_name 2>/dev/null || true)
    p=$(cat ${c}_power_limit_uw 2>/dev/null || true)
    w=$(cat ${c}_time_window_us 2>/dev/null || true)
    [ -n "${p:-}" ] && echo "  $n: limit=$((p/1000000))W window=${w}us"
  done
  echo "  ppd_profile=$(powerprofilesctl get 2>/dev/null || echo 'n/a')"
  echo "  thermald=$(systemctl is-active thermald 2>/dev/null || echo 'n/a')"
  echo "--- Throttle counters at start ---"
  echo "  package_throttle_count=$(cat /sys/devices/system/cpu/cpu0/thermal_throttle/package_throttle_count 2>/dev/null || echo n/a)"
  echo "  core_throttle_total=$(awk '{s+=$1} END{print s+0}' /sys/devices/system/cpu/cpu*/thermal_throttle/core_throttle_count 2>/dev/null)"
} > "$META_TXT"

echo "[INFO] profile = $PROFILE"
echo "[INFO] run dir = $RUN_DIR"
echo "[INFO]   bag     = run.bag (split 60s, lz4)"
echo "[INFO]   metrics = metrics.csv (1 Hz)"
echo "[INFO]   meta    = meta.txt"
echo "[INFO]   rosbag log = rosbag.log"
echo "[INFO] Ctrl-C to stop. Both rosbag and metrics loop will exit cleanly."
echo

# CSV header
echo "ts,uptime_s,pkg_C,core_max_C,nvme_C,freq_avg_MHz,freq_max_MHz,load1,cpu_busy_pct,thr_pkg,thr_pkg_dt,thr_core_total,thr_core_dt,mem_used_pct,disk_write_MBps,bag_size_MB,top1_proc,top1_pct,top2_proc,top2_pct" > "$METRICS_CSV"

# Resolve disk device for bag dir (for write rate via /proc/diskstats).
# Use lsblk to get parent block device (handles nvme0n1p2 -> nvme0n1, sda1 -> sda).
BAG_PART=$(df --output=source "$RUN_DIR" 2>/dev/null | tail -1)
BAG_DEV=$(lsblk -no PKNAME "$BAG_PART" 2>/dev/null | head -1)
[ -z "$BAG_DEV" ] && BAG_DEV=$(echo "$BAG_PART" | sed 's:/dev/::')
# Verify the device shows up in /proc/diskstats (Docker may differ); fallback to first nvme/sd.
if ! awk '{print $3}' /proc/diskstats | grep -qx "$BAG_DEV"; then
  BAG_DEV=$(awk '$3 ~ /^(nvme[0-9]+n[0-9]+|sd[a-z]+|mmcblk[0-9]+)$/{print $3; exit}' /proc/diskstats)
fi
echo "[INFO] disk    = $BAG_DEV (for write-rate from /proc/diskstats)"

# --- spawn rosbag ---
if [ -n "$EXCLUDE" ]; then
  rosbag record -O "$BAG_FILE" --lz4 --split --duration=60 -a -x "$EXCLUDE" \
    >"$ROSBAG_LOG" 2>&1 &
else
  rosbag record -O "$BAG_FILE" --lz4 --split --duration=60 -a \
    >"$ROSBAG_LOG" 2>&1 &
fi
ROSBAG_PID=$!
echo "[INFO] rosbag PID=$ROSBAG_PID  (log: $ROSBAG_LOG)"

cleanup() {
  echo
  echo "[INFO] stopping rosbag (PID=$ROSBAG_PID)..."
  kill -INT "$ROSBAG_PID" 2>/dev/null || true
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    kill -0 "$ROSBAG_PID" 2>/dev/null || break
    sleep 0.5
  done
  kill -TERM "$ROSBAG_PID" 2>/dev/null || true
  wait "$ROSBAG_PID" 2>/dev/null || true
  echo "[INFO] run dir: $RUN_DIR"
  ls -lh "$RUN_DIR"/*.bag* 2>/dev/null | awk '{print "  "$5"\t"$9}'
  echo "[INFO] metrics rows: $(($(wc -l < "$METRICS_CSV") - 1))"
  echo "[INFO] csv: $METRICS_CSV"
  exit 0
}
trap cleanup INT TERM

# --- metrics loop ---
prev_thr_pkg=$(cat /sys/devices/system/cpu/cpu0/thermal_throttle/package_throttle_count 2>/dev/null || echo 0)
prev_thr_core=$(awk '{s+=$1} END{print s+0}' /sys/devices/system/cpu/cpu*/thermal_throttle/core_throttle_count 2>/dev/null)
read _ u n s i io irq sirq st _ < /proc/stat
prev_idle=$((i+io)); prev_total=$((u+n+s+i+io+irq+sirq+st))
prev_dwrite=$(awk -v d="$BAG_DEV" '$3==d{print $10}' /proc/diskstats 2>/dev/null)
prev_dwrite=${prev_dwrite:-0}
prev_t=$(date +%s.%N)

while kill -0 "$ROSBAG_PID" 2>/dev/null; do
  now=$(date +%s.%N)
  dt=$(awk -v a="$prev_t" -v b="$now" 'BEGIN{printf "%.3f", b-a}')
  prev_t=$now
  ts=$(date +%H:%M:%S)
  uptime_s=$(awk '{printf "%d", $1}' /proc/uptime)

  # CPU package temp: prefer coretemp Package id 0, fallback x86_pkg_temp zone
  pkg=""
  for h in /sys/class/hwmon/hwmon*; do
    if [ -f "$h/name" ] && [ "$(cat $h/name 2>/dev/null)" = "coretemp" ]; then
      v=$(cat "$h/temp1_input" 2>/dev/null)
      [ -n "$v" ] && pkg=$((v/1000)) && break
    fi
  done
  if [ -z "$pkg" ]; then
    for z in /sys/class/thermal/thermal_zone*; do
      [ "$(cat $z/type 2>/dev/null)" = "x86_pkg_temp" ] || continue
      pkg=$(($(cat $z/temp 2>/dev/null || echo 0)/1000)); break
    done
  fi
  pkg=${pkg:-0}

  # Core max temp across all coretemp inputs
  core_max=0
  for f in /sys/class/hwmon/hwmon*/temp*_input; do
    [ -f "$f" ] || continue
    parent=$(dirname "$f")
    [ "$(cat $parent/name 2>/dev/null)" = "coretemp" ] || continue
    v=$(cat "$f" 2>/dev/null)
    [ -z "$v" ] && continue
    v=$((v/1000))
    [ "$v" -gt "$core_max" ] && core_max=$v
  done

  # NVMe temp (composite/temp1)
  nvme=0
  for h in /sys/class/hwmon/hwmon*; do
    n=$(cat "$h/name" 2>/dev/null || true)
    case "$n" in
      nvme*)
        v=$(cat "$h/temp1_input" 2>/dev/null)
        [ -n "$v" ] && nvme=$((v/1000)) && break
        ;;
    esac
  done

  # CPU freq across all logical cores
  freqs=$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null)
  freq_avg=$(echo "$freqs" | awk 'NF{s+=$1;n++} END{if(n)printf "%d", s/n/1000; else print 0}')
  freq_max=$(echo "$freqs" | awk 'NF{if($1>m)m=$1} END{printf "%d", m/1000}')

  load1=$(awk '{print $1}' /proc/loadavg)

  # CPU busy% from /proc/stat delta
  read _ u n s i io irq sirq st _ < /proc/stat
  idle=$((i+io)); total=$((u+n+s+i+io+irq+sirq+st))
  d_idle=$((idle-prev_idle)); d_total=$((total-prev_total))
  busy=$(awk -v ii=$d_idle -v tt=$d_total 'BEGIN{if(tt>0)printf "%.1f", 100*(tt-ii)/tt; else print "0.0"}')
  prev_idle=$idle; prev_total=$total

  # Throttle counters (delta per sample)
  thr_pkg=$(cat /sys/devices/system/cpu/cpu0/thermal_throttle/package_throttle_count 2>/dev/null || echo 0)
  thr_pkg_dt=$((thr_pkg - prev_thr_pkg)); prev_thr_pkg=$thr_pkg
  thr_core=$(awk '{s+=$1} END{print s+0}' /sys/devices/system/cpu/cpu*/thermal_throttle/core_throttle_count 2>/dev/null)
  thr_core_dt=$((thr_core - prev_thr_core)); prev_thr_core=$thr_core

  # Memory used%
  mem_used_pct=$(awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2} END{if(t)printf "%.1f", 100*(t-a)/t; else print "0.0"}' /proc/meminfo)

  # Disk write rate for bag's underlying device (sectors -> MB/s)
  dwrite=$(awk -v d="$BAG_DEV" '$3==d{print $10}' /proc/diskstats 2>/dev/null)
  dwrite=${dwrite:-0}
  d_dwrite=$((dwrite - prev_dwrite)); prev_dwrite=$dwrite
  dwrite_mbps=$(awk -v s=$d_dwrite -v t=$dt 'BEGIN{if(t>0)printf "%.2f", s*512/1048576/t; else print "0.00"}')

  # Bag total size on disk for this run (handles rotation)
  bag_mb=$(du -BM -c "$RUN_DIR"/*.bag* 2>/dev/null | tail -1 | awk '{gsub(/M/,""); print $1+0}')
  bag_mb=${bag_mb:-0}

  # Top 2 processes by CPU
  top=$(ps -eo pcpu,comm --sort=-pcpu --no-headers 2>/dev/null | head -2)
  t1c=$(echo "$top" | sed -n '1p' | awk '{print $1}'); t1c=${t1c:-0}
  t1n=$(echo "$top" | sed -n '1p' | awk '{print $2}'); t1n=${t1n:-none}
  t2c=$(echo "$top" | sed -n '2p' | awk '{print $1}'); t2c=${t2c:-0}
  t2n=$(echo "$top" | sed -n '2p' | awk '{print $2}'); t2n=${t2n:-none}

  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "$ts" "$uptime_s" "$pkg" "$core_max" "$nvme" \
    "$freq_avg" "$freq_max" "$load1" "$busy" \
    "$thr_pkg" "$thr_pkg_dt" "$thr_core" "$thr_core_dt" \
    "$mem_used_pct" "$dwrite_mbps" "$bag_mb" \
    "$t1n" "$t1c" "$t2n" "$t2c" \
    | tee -a "$METRICS_CSV"

  sleep 1
done

cleanup
