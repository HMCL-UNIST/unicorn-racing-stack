#!/usr/bin/env python3
"""Generate 3D track boundary CSV from rosbag poses + wall.pcd.

V4: Use robot's actual heading from odometry to define left/right.
At each pose, find nearest wall point on robot's left and right side.
3D KD-tree for slope/bridge handling.
"""

import os
import csv
import argparse
import numpy as np
import rosbag
import open3d as o3d
from scipy.spatial import cKDTree


def quat_to_yaw(q):
    return np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def smooth_circular(arr, window):
    n = len(arr)
    if window <= 0:
        return arr
    padded = np.concatenate([arr[-window:], arr, arr[:window]], axis=0)
    smoothed = np.zeros_like(arr)
    for i in range(n):
        smoothed[i] = np.mean(padded[i:i + 2 * window + 1], axis=0)
    return smoothed


def main():
    parser = argparse.ArgumentParser(description="V4: Robot frame left/right wall detection")
    parser.add_argument("--bag", required=True)
    parser.add_argument("--wall", default="pcd/wall.pcd")
    parser.add_argument("--pose-topic", default="/glim_ros/pose_corrected")
    parser.add_argument("--output", default=None)
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--max-search", type=float, default=3.0)
    parser.add_argument("--min-dist", type=float, default=0.05)
    parser.add_argument("--density-thresh", type=int, default=5)
    parser.add_argument("--bin-size", type=float, default=0.05)
    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "raw_track_data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = args.output or os.path.join(output_dir, "localization_bridge_2_bounds_3d.csv")

    # 1. Load wall PCD
    walls = np.asarray(o3d.io.read_point_cloud(args.wall).points)
    print(f"Wall: {len(walls)} points")

    # 2. Extract poses with ACTUAL yaw from odometry
    print("Extracting poses...")
    bag = rosbag.Bag(args.bag)
    raw_poses = []
    for _, msg, _ in bag.read_messages(topics=[args.pose_topic]):
        p = msg.pose.position
        yaw = quat_to_yaw(msg.pose.orientation)
        raw_poses.append([p.x, p.y, p.z, yaw])
    bag.close()
    raw_poses = np.array(raw_poses)

    # Downsample
    poses = [raw_poses[0]]
    for p in raw_poses[1:]:
        if np.linalg.norm(p[:3] - poses[-1][:3]) >= args.min_dist:
            poses.append(p)
    poses = np.array(poses)

    # Close loop
    dists_to_start = np.linalg.norm(poses[:, :2] - poses[0, :2], axis=1)
    half = len(poses) // 2
    min_idx = half + np.argmin(dists_to_start[half:])
    poses = poses[:min_idx + 1]
    poses[-1] = poses[0].copy()
    n = len(poses)
    print(f"  Centerline: {n} points")

    # 3. Filter wall: only keep points within max_search (3D) of any pose
    pose_tree = cKDTree(poses[:, :3])
    d_to_pose, _ = pose_tree.query(walls[:, :3])
    walls = walls[d_to_pose < args.max_search]
    print(f"  Wall near track: {len(walls)} points")

    # 3D KD-tree
    wall_tree = cKDTree(walls[:, :3])

    # 4. For each pose: use robot's yaw to define left/right, find nearest wall
    print("Finding walls (robot frame)...")
    left_bound = np.zeros((n, 2))
    right_bound = np.zeros((n, 2))
    left_found = np.zeros(n, dtype=bool)
    right_found = np.zeros(n, dtype=bool)

    for i in range(n):
        cx, cy, cz = poses[i, 0], poses[i, 1], poses[i, 2]
        yaw = poses[i, 3]  # actual robot heading

        # Robot's forward direction
        fx, fy = np.cos(yaw), np.sin(yaw)

        # Get nearby wall points (3D search)
        idx_list = wall_tree.query_ball_point([cx, cy, cz], args.max_search)
        if not idx_list:
            continue

        pts = walls[idx_list]
        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy

        # Cross product with forward: positive = left, negative = right
        cross = fx * dy - fy * dx
        dot = fx * dx + fy * dy  # forward projection
        dist_2d = np.sqrt(dx**2 + dy**2)

        # Only points mostly to the side (±45° of perpendicular)
        perp_mask = np.abs(cross) > np.abs(dot)

        for direction, arr, found, sign in [
            ("left", left_bound, left_found, 1),
            ("right", right_bound, right_found, -1)
        ]:
            if sign > 0:
                side_mask = (cross > 0.01) & perp_mask
            else:
                side_mask = (cross < -0.01) & perp_mask

            if not side_mask.any():
                continue

            side_pts = pts[side_mask]
            side_dists = dist_2d[side_mask]

            # Density-based: bin by distance, find first dense bin (>=5 pts)
            bins = np.arange(0, args.max_search + args.bin_size, args.bin_size)
            counts, _ = np.histogram(side_dists, bins=bins)

            wall_pt = None
            for b in range(len(counts)):
                if counts[b] >= args.density_thresh:
                    mask = (side_dists >= bins[b]) & (side_dists < bins[b + 1])
                    wall_pt = np.median(side_pts[mask, :2], axis=0)
                    break

            if wall_pt is None:
                # No dense bin → skip, will be interpolated
                continue

            arr[i] = wall_pt
            found[i] = True

    print(f"  Found: left={left_found.sum()}/{n}, right={right_found.sum()}/{n}")

    # 5. Outlier filter
    for arr, found, label in [(left_bound, left_found, "left"),
                               (right_bound, right_found, "right")]:
        widths = np.linalg.norm(arr - poses[:, :2], axis=1)
        valid_w = widths[found]
        if len(valid_w) == 0:
            continue
        median_w = np.median(valid_w)
        max_w = max(median_w * 2.5, 1.5)
        outlier = found & (widths > max_w)
        if outlier.any():
            print(f"  [{label}] Filtered {outlier.sum()} outliers (>{max_w:.3f}m)")
            found[outlier] = False

    # 6. Interpolate missing
    for arr, found in [(left_bound, left_found), (right_bound, right_found)]:
        if found.all():
            continue
        valid_idx = np.where(found)[0]
        missing_idx = np.where(~found)[0]
        if len(valid_idx) == 0:
            continue
        for dim in range(2):
            arr[missing_idx, dim] = np.interp(
                missing_idx, valid_idx, arr[valid_idx, dim], period=n)

    print(f"  After interp: left gaps={(~left_found).sum()}, right gaps={(~right_found).sum()}")

    # 7. Smooth
    if args.smooth_window > 0:
        left_bound = smooth_circular(left_bound, args.smooth_window)
        right_bound = smooth_circular(right_bound, args.smooth_window)

    # 8. Write CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['right_bound_x', 'right_bound_y', 'right_bound_z',
                         'left_bound_x', 'left_bound_y', 'left_bound_z'])
        for i in range(n):
            writer.writerow([
                f"{right_bound[i,0]:.6f}", f"{right_bound[i,1]:.6f}", f"{poses[i,2]:.6f}",
                f"{left_bound[i,0]:.6f}", f"{left_bound[i,1]:.6f}", f"{poses[i,2]:.6f}"
            ])

    wl = np.linalg.norm(left_bound - poses[:, :2], axis=1)
    wr = np.linalg.norm(right_bound - poses[:, :2], axis=1)
    print(f"Left:  {wl.min():.3f} ~ {wl.max():.3f}m (mean {wl.mean():.3f})")
    print(f"Right: {wr.min():.3f} ~ {wr.max():.3f}m (mean {wr.mean():.3f})")
    print(f"Saved {n} points to {output_path}")

    # 9. Debug plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(walls[:, 0], walls[:, 1], s=0.1, c='gray', alpha=0.3)
    ax.plot(poses[:, 0], poses[:, 1], 'b-', linewidth=1, label='centerline')
    ax.plot(left_bound[:, 0], left_bound[:, 1], 'g-', linewidth=1.5, label='left')
    ax.plot(right_bound[:, 0], right_bound[:, 1], 'r-', linewidth=1.5, label='right')
    # Draw robot heading arrows every 20 points
    for i in range(0, n, 20):
        yaw = poses[i, 3]
        dx, dy = 0.2 * np.cos(yaw), 0.2 * np.sin(yaw)
        ax.arrow(poses[i, 0], poses[i, 1], dx, dy,
                 head_width=0.05, head_length=0.03, fc='blue', ec='blue')
    ax.scatter(left_bound[~left_found, 0], left_bound[~left_found, 1],
               c='lime', s=10, zorder=5, label=f'left interp ({(~left_found).sum()})')
    ax.scatter(right_bound[~right_found, 0], right_bound[~right_found, 1],
               c='orange', s=10, zorder=5, label=f'right interp ({(~right_found).sum()})')
    ax.set_aspect('equal')
    ax.set_title('V4: Robot frame (odometry yaw)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
