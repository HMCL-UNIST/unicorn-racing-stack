#!/usr/bin/env python3
"""Generate 3D track boundary CSV from rosbag.

Simple approach: at each pose, shoot rays left/right (perpendicular to heading),
find the nearest wall point on each side.
"""

import os
import csv
import struct
import numpy as np
import rosbag
from scipy.spatial import cKDTree

BAG_PATH = "/media/chg/T7/roboracer/experiment_3d/bag_to_csv_1.bag"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw_track_data")
POSE_TOPIC = "/glim_ros/base_pose"
PCL_TOPIC = "/glim_ros/aligned_points_corrected"
OUTPUT_NAME = "bag_to_csv_1_bounds_3d.csv"

MIN_DIST = 0.1
WALL_Z_MIN = 0.15   # wall = this much above LiDAR height
WALL_Z_MAX = 1.5    # ignore ceiling
MAX_SEARCH = 2.0    # max search distance for wall
RAY_WIDTH = 0.1     # ray thickness (±m along tangent)
DEFAULT_WIDTH = 1.0


def quat_to_yaw(q):
    return np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    bag = rosbag.Bag(BAG_PATH, 'r')

    # 1. Extract poses
    print("Extracting poses...")
    raw_poses = []
    for _, msg, _ in bag.read_messages(topics=[POSE_TOPIC]):
        p = msg.pose.position
        yaw = quat_to_yaw(msg.pose.orientation)
        raw_poses.append([p.x, p.y, p.z, yaw])
    raw_poses = np.array(raw_poses)
    print(f"  {len(raw_poses)} raw poses")

    # downsample
    poses = [raw_poses[0]]
    for p in raw_poses[1:]:
        if np.linalg.norm(p[:3] - poses[-1][:3]) >= MIN_DIST:
            poses.append(p)
    poses = np.array(poses)
    print(f"  {len(poses)} after downsampling")

    # close loop
    dists = np.linalg.norm(poses[:, :2] - poses[0, :2], axis=1)
    half = len(poses) // 2
    min_idx = half + np.argmin(dists[half:])
    print(f"  Closest return at idx {min_idx}, dist={dists[min_idx]:.4f}m")
    poses = poses[:min_idx + 1]
    poses[-1] = poses[0].copy()
    print(f"  Loop: {len(poses)} points")

    # 2. Accumulate pointcloud
    print("Extracting pointcloud...")
    all_points = []
    for _, msg, _ in bag.read_messages(topics=[PCL_TOPIC]):
        for i in range(msg.width):
            offset = i * msg.point_step
            x, y, z = struct.unpack_from('fff', msg.data, offset)
            if not np.isnan(x):
                all_points.append([x, y, z])
    bag.close()
    all_points = np.array(all_points)
    print(f"  {len(all_points)} total points")

    # 3. Filter wall points: above LiDAR height, below ceiling
    lidar_z = np.mean(poses[:, 2])
    wall_mask = (all_points[:, 2] > lidar_z + WALL_Z_MIN) & \
                (all_points[:, 2] < lidar_z + WALL_Z_MAX)
    walls = all_points[wall_mask]
    print(f"  Wall points: {len(walls)} (z in [{lidar_z + WALL_Z_MIN:.2f}, {lidar_z + WALL_Z_MAX:.2f}])")

    wall_tree = cKDTree(walls[:, :2])

    # 4. For each pose, find nearest wall left and right
    print("Finding boundaries...")
    n = len(poses)
    left_bound = np.zeros((n, 3))
    right_bound = np.zeros((n, 3))

    for i in range(n):
        cx, cy, cz = poses[i, :3]
        yaw = poses[i, 3]

        # left normal direction
        lnx = -np.sin(yaw)
        lny = np.cos(yaw)
        # tangent direction
        tx = np.cos(yaw)
        ty = np.sin(yaw)

        # get all wall points within search radius
        idx_list = wall_tree.query_ball_point([cx, cy], MAX_SEARCH)
        if len(idx_list) == 0:
            left_bound[i] = [cx + lnx * DEFAULT_WIDTH, cy + lny * DEFAULT_WIDTH, cz]
            right_bound[i] = [cx - lnx * DEFAULT_WIDTH, cy - lny * DEFAULT_WIDTH, cz]
            continue

        nearby = walls[idx_list]
        dx = nearby[:, 0] - cx
        dy = nearby[:, 1] - cy

        # project onto normal (left=positive) and tangent
        n_proj = dx * lnx + dy * lny
        t_proj = dx * tx + dy * ty

        # only points within thin strip along normal (ray width)
        strip = np.abs(t_proj) < RAY_WIDTH

        # left: positive n_proj, closest
        left_mask = strip & (n_proj > 0.05)
        if np.any(left_mask):
            left_d = np.min(n_proj[left_mask])
            left_bound[i] = [cx + lnx * left_d, cy + lny * left_d, cz]
        else:
            left_bound[i] = [cx + lnx * DEFAULT_WIDTH, cy + lny * DEFAULT_WIDTH, cz]

        # right: negative n_proj, closest
        right_mask = strip & (n_proj < -0.05)
        if np.any(right_mask):
            right_d = np.min(np.abs(n_proj[right_mask]))
            right_bound[i] = [cx - lnx * right_d, cy - lny * right_d, cz]
        else:
            right_bound[i] = [cx - lnx * DEFAULT_WIDTH, cy - lny * DEFAULT_WIDTH, cz]

    # 5. Write CSV
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
