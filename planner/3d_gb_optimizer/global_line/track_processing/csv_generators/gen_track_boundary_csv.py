#!/usr/bin/env python3
"""Generate 3D track boundary CSV from rosbag odometry + segmented wall PCD.

Algorithm:
  1. Extract 1 loop of poses from rosbag (PoseStamped topic)
  2. Filter wall PCD: keep only points within max_search (3D) of any pose
  3. At each pose, use robot's actual heading (quaternion → yaw) to define left/right
     - Forward direction: (cos(yaw), sin(yaw))
     - Cross product of forward × (wall - pose): positive = left, negative = right
     - Only consider points within ±45° of perpendicular (side filter)
  4. Density-based wall edge detection:
     - Bin side-points by distance from centerline (default 5cm bins)
     - First bin with >= density_thresh points = actual wall
     - Use median position of that bin as boundary point
     - Sparse noise points (< density_thresh) are ignored
  5. Missing boundaries filled via sliding-window interpolation
  6. Outlier filter + circular smoothing

Handles:
  - Bridges/overpasses: 3D KD-tree separates by height naturally
  - Track crossings: local tangent-based left/right (no inside/outside assumption)
  - Slopes: 3D distance follows odometry z
  - Noise: density filter ignores isolated wall points

Input:
  - rosbag with PoseStamped odometry
  - wall.pcd from manual segmentation (manual_segmentation.py)

Output:
  - CSV with columns: right_bound_x/y/z, left_bound_x/y/z
"""

import os
import csv
import argparse
import numpy as np
import rosbag
import open3d as o3d
from scipy.spatial import cKDTree


def quat_to_yaw(q):
    """Convert quaternion to yaw angle."""
    return np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def smooth_circular(arr, window):
    """Circular moving average for closed-loop data."""
    n = len(arr)
    if window <= 0:
        return arr
    padded = np.concatenate([arr[-window:], arr, arr[:window]], axis=0)
    smoothed = np.zeros_like(arr)
    for i in range(n):
        smoothed[i] = np.mean(padded[i:i + 2 * window + 1], axis=0)
    return smoothed


def extract_single_loop(raw_poses, min_dist=0.05):
    """Downsample poses and extract a single closed loop."""
    poses = [raw_poses[0]]
    for p in raw_poses[1:]:
        if np.linalg.norm(p[:3] - poses[-1][:3]) >= min_dist:
            poses.append(p)
    poses = np.array(poses)

    # Find loop closure: closest point to start in second half
    dists_to_start = np.linalg.norm(poses[:, :2] - poses[0, :2], axis=1)
    half = len(poses) // 2
    min_idx = half + np.argmin(dists_to_start[half:])
    poses = poses[:min_idx + 1]
    poses[-1] = poses[0].copy()
    return poses


def find_wall_boundary(poses, walls, max_search=3.0, bin_size=0.05,
                       density_thresh=10, smooth_window=0, slide_window=5,
                       z_min_rel=-0.1, z_max_rel=1.0):
    """Find left/right wall boundaries for each pose.

    Returns:
        left_bound, right_bound: (n, 2) arrays of boundary xy coords
        left_found, right_found: boolean masks
    """
    n = len(poses)

    # Filter walls: keep only near track (3D)
    pose_tree = cKDTree(poses[:, :3])
    d_to_pose, _ = pose_tree.query(walls[:, :3])
    walls_near = walls[d_to_pose < max_search]
    wall_tree = cKDTree(walls_near[:, :3])
    print(f"  Wall near track: {len(walls_near)}/{len(walls)} points")

    left_bound = np.full((n, 2), np.nan)
    right_bound = np.full((n, 2), np.nan)
    left_found = np.zeros(n, dtype=bool)
    right_found = np.zeros(n, dtype=bool)

    for i in range(n):
        cx, cy, cz = poses[i, :3]
        yaw = poses[i, 3]
        fx, fy = np.cos(yaw), np.sin(yaw)

        idx_list = wall_tree.query_ball_point([cx, cy, cz], max_search)
        if not idx_list:
            continue

        pts = walls_near[idx_list]

        # Z filter: only keep wall points within z_band of robot height
        dz = pts[:, 2] - cz
        z_mask = (dz > z_min_rel) & (dz < z_max_rel)
        pts = pts[z_mask]
        if len(pts) == 0:
            continue

        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy

        cross = fx * dy - fy * dx  # positive = left
        dot = fx * dx + fy * dy    # forward projection
        dist_2d = np.sqrt(dx**2 + dy**2)

        # Side filter: only points within ±45° of perpendicular
        perp_mask = np.abs(cross) > np.abs(dot)

        for sign, arr, found in [(1, left_bound, left_found),
                                  (-1, right_bound, right_found)]:
            if sign > 0:
                side_mask = (cross > 0.01) & perp_mask
            else:
                side_mask = (cross < -0.01) & perp_mask

            if not side_mask.any():
                continue

            side_pts = pts[side_mask]
            side_dists = dist_2d[side_mask]

            # Density-based: first bin with >= density_thresh points
            bins = np.arange(0, max_search + bin_size, bin_size)
            counts, _ = np.histogram(side_dists, bins=bins)

            for b in range(len(counts)):
                if counts[b] >= density_thresh:
                    in_bin = (side_dists >= bins[b]) & (side_dists < bins[b + 1])
                    arr[i] = np.median(side_pts[in_bin, :2], axis=0)
                    found[i] = True
                    break

    print(f"  Detected: left={left_found.sum()}/{n}, right={right_found.sum()}/{n}")

    # Sliding window fallback for missing boundaries
    for arr, found in [(left_bound, left_found), (right_bound, right_found)]:
        for i in range(n):
            if found[i]:
                continue
            window_start = max(0, i - slide_window)
            recent_found = found[window_start:i]
            if recent_found.any():
                recent_widths = np.linalg.norm(
                    arr[window_start:i][recent_found[window_start - window_start:]] - poses[window_start:i, :2][recent_found[window_start - window_start:]],
                    axis=1)
                w = np.mean(recent_widths)
                yaw = poses[i, 3]
                sign = 1 if arr is left_bound else -1
                arr[i] = [cx + sign * (-np.sin(yaw)) * w,
                          cy + sign * np.cos(yaw) * w]
                found[i] = True

    # Interpolate remaining gaps
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
        found[missing_idx] = True

    # Outlier filter
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
            # Re-interpolate outliers
            valid_idx = np.where(found)[0]
            missing_idx = np.where(~found)[0]
            if len(valid_idx) > 0:
                for dim in range(2):
                    arr[missing_idx, dim] = np.interp(
                        missing_idx, valid_idx, arr[valid_idx, dim], period=n)

    # Smooth
    if smooth_window > 0:
        left_bound = smooth_circular(left_bound, smooth_window)
        right_bound = smooth_circular(right_bound, smooth_window)

    return left_bound, right_bound, left_found, right_found, walls_near


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D track boundary CSV from rosbag + wall PCD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (default topic and wall path)
  python gen_track_boundary_csv.py --bag my_recording.bag

  # Custom paths and parameters
  python gen_track_boundary_csv.py \\
      --bag recording.bag \\
      --wall pcd/wall.pcd \\
      --pose-topic /glim_ros/pose_corrected \\
      --output data/raw_track_data/my_track_bounds_3d.csv \\
      --max-search 2.0 \\
      --density-thresh 5

  # Narrow track with tight search
  python gen_track_boundary_csv.py --bag rec.bag --max-search 1.5 --density-thresh 3
""")
    parser.add_argument("--bag", required=True, help="Input rosbag file")
    parser.add_argument("--wall", default="pcd/wall.pcd",
                        help="Segmented wall point cloud (default: pcd/wall.pcd)")
    parser.add_argument("--pose-topic", default="/glim_ros/pose_corrected",
                        help="PoseStamped topic for odometry (default: /glim_ros/pose_corrected)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output CSV path (default: data/raw_track_data/<bag_name>_bounds_3d.csv)")
    parser.add_argument("--max-search", type=float, default=3.0,
                        help="Max 3D search radius for wall points [m] (default: 3.0)")
    parser.add_argument("--density-thresh", type=int, default=10,
                        help="Min points in distance bin to count as wall (default: 10)")
    parser.add_argument("--bin-size", type=float, default=0.05,
                        help="Distance histogram bin size [m] (default: 0.05)")
    parser.add_argument("--smooth-window", type=int, default=3,
                        help="Circular smoothing window size (default: 3)")
    parser.add_argument("--min-dist", type=float, default=0.05,
                        help="Min distance between centerline points [m] (default: 0.05)")
    parser.add_argument("--z-min-rel", type=float, default=-0.1,
                        help="Wall z min relative to robot [m] (default: -0.1)")
    parser.add_argument("--z-max-rel", type=float, default=0.1,
                        help="Wall z max relative to robot [m] (default: 1.0)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip debug visualization")
    args = parser.parse_args()

    # Default output path
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "..", "data", "raw_track_data")
    os.makedirs(output_dir, exist_ok=True)
    if args.output is None:
        bag_name = os.path.splitext(os.path.basename(args.bag))[0]
        args.output = os.path.join(output_dir, f"{bag_name}_bounds_3d.csv")

    # 1. Load wall PCD
    walls = np.asarray(o3d.io.read_point_cloud(args.wall).points)
    print(f"Wall: {len(walls)} points")

    # 2. Extract poses
    print("Extracting poses...")
    bag = rosbag.Bag(args.bag)
    raw_poses = []
    for _, msg, _ in bag.read_messages(topics=[args.pose_topic]):
        p = msg.pose.position
        yaw = quat_to_yaw(msg.pose.orientation)
        raw_poses.append([p.x, p.y, p.z, yaw])
    bag.close()
    raw_poses = np.array(raw_poses)
    print(f"  Raw: {len(raw_poses)} poses")

    poses = extract_single_loop(raw_poses, args.min_dist)
    n = len(poses)
    print(f"  1 loop: {n} points")

    # 3. Find boundaries
    left_bound, right_bound, left_found, right_found, walls_near = find_wall_boundary(
        poses, walls, args.max_search, args.bin_size,
        args.density_thresh, args.smooth_window,
        z_min_rel=args.z_min_rel, z_max_rel=args.z_max_rel)

    # 4. Write CSV
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['right_bound_x', 'right_bound_y', 'right_bound_z',
                         'left_bound_x', 'left_bound_y', 'left_bound_z',
                         'center_x', 'center_y', 'center_z'])
        for i in range(n):
            writer.writerow([
                f"{right_bound[i,0]:.6f}", f"{right_bound[i,1]:.6f}", f"{poses[i,2]:.6f}",
                f"{left_bound[i,0]:.6f}", f"{left_bound[i,1]:.6f}", f"{poses[i,2]:.6f}",
                f"{poses[i,0]:.6f}", f"{poses[i,1]:.6f}", f"{poses[i,2]:.6f}"
            ])

    wl = np.linalg.norm(left_bound - poses[:, :2], axis=1)
    wr = np.linalg.norm(right_bound - poses[:, :2], axis=1)
    print(f"Left:  {wl.min():.3f} ~ {wl.max():.3f}m (mean {wl.mean():.3f})")
    print(f"Right: {wr.min():.3f} ~ {wr.max():.3f}m (mean {wr.mean():.3f})")
    print(f"Saved {n} points to {args.output}")

    # 5. Debug plot
    if not args.no_plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.scatter(walls_near[:, 0], walls_near[:, 1], s=0.1, c='gray', alpha=0.3)
        ax.plot(poses[:, 0], poses[:, 1], 'b-', linewidth=1, label='centerline')
        ax.plot(left_bound[:, 0], left_bound[:, 1], 'g-', linewidth=1.5, label='left')
        ax.plot(right_bound[:, 0], right_bound[:, 1], 'r-', linewidth=1.5, label='right')
        for i in range(0, n, 20):
            yaw = poses[i, 3]
            ddx, ddy = 0.2 * np.cos(yaw), 0.2 * np.sin(yaw)
            ax.arrow(poses[i, 0], poses[i, 1], ddx, ddy,
                     head_width=0.05, head_length=0.03, fc='blue', ec='blue')
        ax.set_aspect('equal')
        ax.set_title('Track boundary detection')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
