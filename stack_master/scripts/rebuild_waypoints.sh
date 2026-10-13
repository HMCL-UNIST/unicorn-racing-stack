#!/bin/bash
# Rebuild global_waypoints.json: trackbounds (1:1 d_left/d_right), markers, vel_markers
#
# Usage: ./rebuild_waypoints.sh <map_name>
#   e.g. ./rebuild_waypoints.sh gazebo_wall_2
#
# No reference file needed — marker templates are built-in.

set -e

cd "${0%/*}"
cd ../..
# Now in race_stack/

MAP_PATH="stack_master/maps"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <map_name>"
    echo "Available maps:"
    ls "$MAP_PATH"
    exit 1
fi

MAP_DIR="$MAP_PATH/$1"

if [ ! -d "$MAP_DIR" ]; then
    echo "[error] Map directory not found: $MAP_DIR"
    exit 1
fi

WPNT_FILE="$MAP_DIR/global_waypoints.json"
BACKUP_FILE="$MAP_DIR/global_waypoints_backup.json"

if [ ! -f "$WPNT_FILE" ]; then
    echo "[error] global_waypoints.json not found in $MAP_DIR"
    exit 1
fi

# Backup original
if [ ! -f "$BACKUP_FILE" ]; then
    cp "$WPNT_FILE" "$BACKUP_FILE"
    echo "[backup] Created: $BACKUP_FILE"
else
    echo "[backup] Restoring original from backup..."
    cp "$BACKUP_FILE" "$WPNT_FILE"
    echo "[backup] Restored: $WPNT_FILE"
fi

echo "[rebuild] Running rebuild on $MAP_DIR ..."

python3 - "$MAP_DIR" << 'PYEOF'
import json, math, copy, sys, os

map_dir = sys.argv[1]

with open(os.path.join(map_dir, "global_waypoints.json")) as f:
    data = json.load(f)

# -- Marker template (type=2: SPHERE, type=3: CYLINDER) --
def make_marker(mid, x, y, z, scale, color, mtype=2):
    return {
        "header": {"seq": 0, "stamp": {"secs": 0, "nsecs": 0}, "frame_id": "map"},
        "ns": "", "id": mid, "type": mtype, "action": 0,
        "pose": {
            "position": {"x": x, "y": y, "z": z},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1}
        },
        "scale": scale,
        "color": color,
        "lifetime": {"secs": 0, "nsecs": 0}, "frame_locked": False,
        "points": [], "colors": [], "text": "",
        "mesh_resource": "", "mesh_use_embedded_materials": False
    }

def velocity_color(vx, vmin, vmax):
    ratio = (vx - vmin) / (vmax - vmin) if vmax != vmin else 0.5
    ratio = max(0.0, min(1.0, ratio))
    return {"r": round(1.0 - ratio, 6), "g": round(ratio, 6), "b": 0.0, "a": 1.0}

SPHERE_SCALE = {"x": 0.05, "y": 0.05, "z": 0.05}
COLOR_PURPLE = {"r": 0.5, "g": 0.0, "b": 0.5, "a": 1.0}
COLOR_YELLOW_GREEN = {"r": 0.5, "g": 1.0, "b": 0.0, "a": 1.0}
COLOR_BLUE = {"r": 0.0, "g": 0.0, "b": 1.0, "a": 1.0}
COLOR_RED = {"r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0}
VEL_SCALE_XY = 0.1          # cylinder diameter for vel markers
VEL_SCALE_FACTOR = 0.1317   # match export_global_waypoints.py

iqp_wpnts = data["global_traj_wpnts_iqp"]["wpnts"]
sp_wpnts  = data["global_traj_wpnts_sp"]["wpnts"]
cl_wpnts  = data["centerline_waypoints"]["wpnts"]

# centerline tangent at each iqp/sp wpnt's matched s_opt (for trackbound markers)
psi_center = data.get("centerline_ref", {}).get("psi_center_rad")
if psi_center is None:
    print("  [warn] centerline_ref.psi_center_rad missing → falling back to wpnt psi_rad (less accurate at corners)")
elif len(psi_center) != len(iqp_wpnts):
    print(f"  [warn] psi_center_rad length ({len(psi_center)}) != iqp_wpnts ({len(iqp_wpnts)}) → falling back to psi_rad")
    psi_center = None

vmin_iqp = min(w["vx_mps"] for w in iqp_wpnts)
vmax_iqp = max(w["vx_mps"] for w in iqp_wpnts)
vmin_sp  = min(w["vx_mps"] for w in sp_wpnts)
vmax_sp  = max(w["vx_mps"] for w in sp_wpnts)

print(f"  IQP: {len(iqp_wpnts)} wpnts, vel {vmin_iqp:.3f}~{vmax_iqp:.3f} m/s")
print(f"  SP:  {len(sp_wpnts)} wpnts, vel {vmin_sp:.3f}~{vmax_sp:.3f} m/s")
print(f"  CL:  {len(cl_wpnts)} wpnts")

# -- Centerline markers (blue spheres) --
# preserve 3D z (was hard-coded 0.0, broken on banked/sloped tracks)
new_cl = []
for i, w in enumerate(cl_wpnts):
    new_cl.append(make_marker(i, w["x_m"], w["y_m"], w["z_m"], SPHERE_SCALE, COLOR_BLUE))
data["centerline_markers"] = {"markers": new_cl}
print(f"  centerline markers: {len(new_cl)}")

# -- Trackbounds: 1:1 with iqp wpnts, even=LEFT(purple), odd=RIGHT(yellow-green) --
# draw along CENTERLINE normal (psi_center) instead of raceline normal — d_left/d_right
#      are measured along centerline normal axis, so racing-line-tangent ± π/2 was off by
#      sin(chi_opt) at corners (~tens of cm).
new_tb = []
for i, w in enumerate(iqp_wpnts):
    psi = float(psi_center[i]) if psi_center is not None else w["psi_rad"]
    x, y, z = w["x_m"], w["y_m"], w["z_m"]
    lx = x + w["d_left"]  * math.cos(psi + math.pi / 2)
    ly = y + w["d_left"]  * math.sin(psi + math.pi / 2)
    rx = x + w["d_right"] * math.cos(psi - math.pi / 2)
    ry = y + w["d_right"] * math.sin(psi - math.pi / 2)
    new_tb.append(make_marker(i * 2,     lx, ly, z, SPHERE_SCALE, COLOR_PURPLE))
    new_tb.append(make_marker(i * 2 + 1, rx, ry, z, SPHERE_SCALE, COLOR_YELLOW_GREEN))
data["trackbounds_markers"] = {"markers": new_tb}
print(f"  trackbounds: {len(new_tb)} ({len(new_tb)//2} pairs)")

# -- IQP markers (velocity-colored spheres) --
new_iqp = []
for i, w in enumerate(iqp_wpnts):
    c = velocity_color(w["vx_mps"], vmin_iqp, vmax_iqp)
    new_iqp.append(make_marker(i, w["x_m"], w["y_m"], w["z_m"], SPHERE_SCALE, c))
data["global_traj_markers_iqp"] = {"markers": new_iqp}
print(f"  IQP markers: {len(new_iqp)}")

# -- SP markers (velocity-colored spheres) --
new_sp = []
for i, w in enumerate(sp_wpnts):
    c = velocity_color(w["vx_mps"], vmin_sp, vmax_sp)
    new_sp.append(make_marker(i, w["x_m"], w["y_m"], w["z_m"], SPHERE_SCALE, c))
data["global_traj_markers_sp"] = {"markers": new_sp}
print(f"  SP markers: {len(new_sp)}")

# -- SP velocity markers (cylinders, height = vx_mps * 0.1317, fixed red) --
# matches planner/3d_gb_optimizer/global_line/global_racing_line/export_global_waypoints.py
new_vel = []
for i, w in enumerate(sp_wpnts):
    vx = w["vx_mps"]
    height = vx * VEL_SCALE_FACTOR
    scale = {"x": VEL_SCALE_XY, "y": VEL_SCALE_XY, "z": round(height, 6)}
    new_vel.append(make_marker(i, w["x_m"], w["y_m"], round(height / 2.0, 6), scale, COLOR_RED, mtype=3))
data["global_traj_vel_markers_sp"] = {"markers": new_vel}
print(f"  SP vel markers: {len(new_vel)}")

# -- Output key order (same as eng_0410_v2 format) --
KEY_ORDER = [
    "map_info_str", "est_lap_time",
    "centerline_markers", "centerline_waypoints",
    "global_traj_markers_iqp", "global_traj_wpnts_iqp",
    "global_traj_markers_sp", "global_traj_wpnts_sp",
    "trackbounds_markers", "global_traj_vel_markers_sp",
]
ordered = {}
for k in KEY_ORDER:
    if k in data:
        ordered[k] = data[k]
# keep any extra keys at the end
for k in data:
    if k not in ordered:
        ordered[k] = data[k]

with open(os.path.join(map_dir, "global_waypoints.json"), "w") as f:
    json.dump(ordered, f, indent=2)   # match export_global_waypoints.py format
print("  Saved!")

# -- Visualization --
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    iqp_x = [w["x_m"] for w in iqp_wpnts]
    iqp_y = [w["y_m"] for w in iqp_wpnts]
    iqp_v = [w["vx_mps"] for w in iqp_wpnts]
    left_x  = [new_tb[i]["pose"]["position"]["x"] for i in range(0, len(new_tb), 2)]
    left_y  = [new_tb[i]["pose"]["position"]["y"] for i in range(0, len(new_tb), 2)]
    right_x = [new_tb[i]["pose"]["position"]["x"] for i in range(1, len(new_tb), 2)]
    right_y = [new_tb[i]["pose"]["position"]["y"] for i in range(1, len(new_tb), 2)]
    cmap = mcolors.LinearSegmentedColormap.from_list("vel", [(1,0,0),(0,1,0)])

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    sc = axes[0].scatter(iqp_x, iqp_y, c=iqp_v, cmap=cmap, s=8, zorder=3)
    axes[0].set_title(f"IQP Waypoints (n={len(iqp_wpnts)})\nvel: {min(iqp_v):.2f}~{max(iqp_v):.2f} m/s")
    axes[0].set_aspect("equal"); axes[0].grid(True, alpha=0.3)
    plt.colorbar(sc, ax=axes[0], label="velocity (m/s)")

    axes[1].scatter(left_x, left_y, c="purple", s=8, zorder=3, label="LEFT")
    axes[1].scatter(iqp_x, iqp_y, c="gray", s=2, alpha=0.3, zorder=1)
    axes[1].set_title(f"Left Boundary (n={len(left_x)})"); axes[1].set_aspect("equal")
    axes[1].grid(True, alpha=0.3); axes[1].legend()

    axes[2].scatter(right_x, right_y, c="yellowgreen", s=8, zorder=3, label="RIGHT")
    axes[2].scatter(iqp_x, iqp_y, c="gray", s=2, alpha=0.3, zorder=1)
    axes[2].set_title(f"Right Boundary (n={len(right_x)})"); axes[2].set_aspect("equal")
    axes[2].grid(True, alpha=0.3); axes[2].legend()

    plt.suptitle("IQP Waypoints & Boundaries (d_right/d_left)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(map_dir, "waypoints_boundaries_viz.png"), dpi=150, bbox_inches="tight")
    plt.close()

    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 10))
    ax2.plot(left_x+[left_x[0]], left_y+[left_y[0]], color="purple", linewidth=1.0, alpha=0.7, label="Left (purple)")
    ax2.plot(right_x+[right_x[0]], right_y+[right_y[0]], color="yellowgreen", linewidth=1.0, alpha=0.7, label="Right (yellow-green)")
    sc2 = ax2.scatter(iqp_x, iqp_y, c=iqp_v, cmap=cmap, s=12, zorder=3, label="IQP wpnts")
    ax2.set_title("Track Overview"); ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3); ax2.legend(loc="upper right")
    plt.colorbar(sc2, ax=ax2, label="velocity (m/s)")
    plt.tight_layout()
    plt.savefig(os.path.join(map_dir, "track_overview_viz.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  PNG saved to {map_dir}/")
except ImportError:
    print("  [warn] matplotlib not found, skipping visualization")
PYEOF

echo "[done] Rebuild complete for $1"
