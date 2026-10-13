#!/usr/bin/env python3
"""Generate 3D track boundary CSV from segmented wall.pcd + bag pose data.

For each centerline point, shoots rays left/right along the normal direction
and finds the nearest wall point on each side. Uses inner/outer wall split
to ensure correct wall assignment.
"""

import os
import csv
import numpy as np
import rosbag
import open3d as o3d
from scipy.spatial import cKDTree

BAG_PATH = "/media/chg/T7/roboracer/experiment_3d/bag_to_csv_1.bag"
WALL_PCD = "pcd/wall.pcd"
POSE_TOPIC = "/glim_ros/base_pose"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw_track_data")
OUTPUT_NAME = "bag_to_csv_1_bounds_3d.csv"

MIN_DIST = 0.05
MAX_SEARCH = 2.0
RAY_WIDTH = 0.3
SMOOTH_WINDOW = 3  # light smoothing on boundary coordinates


def quat_to_yaw(q):
    return np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def smooth_circular(arr, window):
    n = len(arr)
    padded = np.concatenate([arr[-window:], arr, arr[:window]], axis=0)
    smoothed = np.zeros_like(arr)
    for i in range(n):
        smoothed[i] = np.mean(padded[i:i + 2 * window + 1], axis=0)
    return smoothed


def interp_nans_circular(arr):
    valid = ~np.isnan(arr)
    if valid.all():
        return arr
    if not valid.any():
        arr[:] = 1.0
        return arr
    indices = np.arange(len(arr))
    arr[~valid] = np.interp(indices[~valid], indices[valid], arr[valid], period=len(arr))
    return arr


def find_boundary_ray(cx, cy, yaw, wall_tree, walls, direction, ray_width):
    """Shoot a ray from (cx,cy) in the given direction and find nearest wall point.

    direction: +1 for left, -1 for right
    Returns the actual wall point (x, y) or None if not found.
    """
    lnx = -np.sin(yaw) * direction
    lny = np.cos(yaw) * direction
    tx = np.cos(yaw)
    ty = np.sin(yaw)

    idx_list = wall_tree.query_ball_point([cx, cy], MAX_SEARCH)
    if not idx_list:
        return None

    nearby = walls[idx_list]
    nearby_idx = np.array(idx_list)
    dx = nearby[:, 0] - cx
    dy = nearby[:, 1] - cy

    # project onto ray direction and tangent
    ray_proj = dx * lnx + dy * lny   # distance along ray (positive = forward)
    t_proj = dx * tx + dy * ty        # distance along tangent

    # only points within strip and in front of ray
    mask = (np.abs(t_proj) < ray_width) & (ray_proj > 0.05)

    if mask.any():
        # find the closest wall point along the ray
        candidates = nearby[mask]
        ray_dists = ray_proj[mask]
        closest_idx = np.argmin(ray_dists)
        return candidates[closest_idx, :2]  # return actual wall point xy
    return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load wall
    print(f"Loading {WALL_PCD}...")
    wall_pcd = o3d.io.read_point_cloud(WALL_PCD)
    walls = np.asarray(wall_pcd.points)
    wall_tree = cKDTree(walls[:, :2])
    print(f"  {len(walls)} wall points")

    # 2. Extract & process poses
    print("Extracting poses...")
    bag = rosbag.Bag(BAG_PATH, 'r')
    raw_poses = []
    for _, msg, _ in bag.read_messages(topics=[POSE_TOPIC]):
        p = msg.pose.position
        yaw = quat_to_yaw(msg.pose.orientation)
        raw_poses.append([p.x, p.y, p.z, yaw])
    bag.close()
    raw_poses = np.array(raw_poses)

    poses = [raw_poses[0]]
    for p in raw_poses[1:]:
        if np.linalg.norm(p[:3] - poses[-1][:3]) >= MIN_DIST:
            poses.append(p)
    poses = np.array(poses)

    dists = np.linalg.norm(poses[:, :2] - poses[0, :2], axis=1)
    half = len(poses) // 2
    min_idx = half + np.argmin(dists[half:])
    poses = poses[:min_idx + 1]
    poses[-1] = poses[0].copy()
    n = len(poses)
    print(f"  {n} centerline points")

    # 3. Compute yaw from path tangent (not quaternion)
    print("Computing tangent-based yaw...")
    tangent_yaw = np.zeros(n)
    for i in range(n):
        i_next = (i + 1) % n
        i_prev = (i - 1) % n
        dx = poses[i_next, 0] - poses[i_prev, 0]
        dy = poses[i_next, 1] - poses[i_prev, 1]
        tangent_yaw[i] = np.arctan2(dy, dx)

    # 4. Find boundaries: shoot rays left and right, get actual wall points
    print("Finding boundaries...")
    left_bound = np.zeros((n, 3))
    right_bound = np.zeros((n, 3))
    left_found = np.zeros(n, dtype=bool)
    right_found = np.zeros(n, dtype=bool)

    for i in range(n):
        cx, cy, cz = poses[i, :3]
        yaw = tangent_yaw[i]

        lpt = find_boundary_ray(cx, cy, yaw, wall_tree, walls, +1, RAY_WIDTH)
        rpt = find_boundary_ray(cx, cy, yaw, wall_tree, walls, -1, RAY_WIDTH)

        if lpt is not None:
            left_bound[i] = [lpt[0], lpt[1], cz]
            left_found[i] = True

        if rpt is not None:
            right_bound[i] = [rpt[0], rpt[1], cz]
            right_found[i] = True

    # Interpolate missing points from neighbors
    for arr, found in [(left_bound, left_found), (right_bound, right_found)]:
        if found.all():
            continue
        valid_idx = np.where(found)[0]
        missing_idx = np.where(~found)[0]
        if len(valid_idx) == 0:
            continue
        for dim in range(2):  # interpolate x and y
            arr[missing_idx, dim] = np.interp(
                missing_idx, valid_idx, arr[valid_idx, dim], period=n)

    n_miss_l = (~left_found).sum()
    n_miss_r = (~right_found).sum()
    print(f"  Missing: left={n_miss_l}, right={n_miss_r} (interpolated)")

    # Smooth boundary coordinates
    if SMOOTH_WINDOW > 0:
        left_bound = smooth_circular(left_bound, SMOOTH_WINDOW)
        right_bound = smooth_circular(right_bound, SMOOTH_WINDOW)
        print(f"  Smoothed (window={SMOOTH_WINDOW})")

    # 4. Write CSV
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['right_bound_x', 'right_bound_y', 'right_bound_z',
                         'left_bound_x', 'left_bound_y', 'left_bound_z'])
        for i in range(n):
            writer.writerow([
                f"{right_bound[i,0]:.6f}", f"{right_bound[i,1]:.6f}", f"{right_bound[i,2]:.6f}",
                f"{left_bound[i,0]:.6f}", f"{left_bound[i,1]:.6f}", f"{left_bound[i,2]:.6f}"
            ])

    wl = np.linalg.norm(left_bound[:, :2] - poses[:, :2], axis=1)
    wr = np.linalg.norm(right_bound[:, :2] - poses[:, :2], axis=1)
    print(f"  Left:  {wl.min():.3f} ~ {wl.max():.3f}m (mean {wl.mean():.3f})")
    print(f"  Right: {wr.min():.3f} ~ {wr.max():.3f}m (mean {wr.mean():.3f})")
    print(f"  Saved {n} points to {output_path}")


if __name__ == '__main__':
    main()
