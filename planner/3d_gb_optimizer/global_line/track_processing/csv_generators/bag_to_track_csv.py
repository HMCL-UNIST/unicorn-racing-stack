#!/usr/bin/env python3
"""Extract 3D centerline from rosbag and generate 3D track boundary CSV.
Uses orientation quaternion for accurate normal computation."""

import os
import csv
import numpy as np
import rosbag

BAG_PATH = "/media/chg/T7/roboracer/experiment_3d/experiment_3d_2.bag"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw_track_data")
TOPIC = "/glim_ros/base_pose"
TRACK_WIDTH = 0.5  # meters each side
MIN_DIST = 0.1  # minimum distance between points for downsampling


def quat_to_yaw(qx, qy, qz, qw):
    """Extract yaw from quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return np.arctan2(siny_cosp, cosy_cosp)


def extract_poses_3d(bag_path):
    """Extract (x, y, z, yaw) from PoseStamped messages."""
    poses = []
    bag = rosbag.Bag(bag_path, 'r')
    for _, msg, _ in bag.read_messages(topics=[TOPIC]):
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        q = msg.pose.orientation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        poses.append((x, y, z, yaw))
    bag.close()
    return np.array(poses)


def downsample_by_distance(points, min_dist):
    """Remove points that are too close together."""
    filtered = [points[0]]
    for p in points[1:]:
        d = np.linalg.norm(p[:3] - filtered[-1][:3])
        if d >= min_dist:
            filtered.append(p)
    return np.array(filtered)


def close_loop(pts):
    """Trim trailing overlap and close the loop."""
    start = pts[0]
    dists = np.linalg.norm(pts[:, :2] - start[:2], axis=1)

    half = len(pts) // 2
    min_idx = half + np.argmin(dists[half:])
    print(f"  Closest return to start at idx {min_idx}, dist={dists[min_idx]:.4f}m")

    pts = pts[:min_idx + 1]
    pts[-1] = start.copy()

    print(f"  Loop closed: {len(pts)} points")
    return pts


def compute_boundaries_from_tangent(centerline, width):
    """Compute left/right boundaries using path tangent direction.

    Uses the actual path direction (finite differences) to compute
    a consistent left/right normal, not the sensor yaw.
    Left = tangent rotated +90deg (counterclockwise), Right = -90deg.
    """
    n = len(centerline)
    left = np.zeros((n, 3))
    right = np.zeros((n, 3))

    for i in range(n):
        # tangent from neighbors (wrap for closed loop)
        i_next = (i + 1) % n
        i_prev = (i - 1) % n
        tangent = centerline[i_next, :2] - centerline[i_prev, :2]

        # normalize
        norm = np.linalg.norm(tangent)
        if norm < 1e-10:
            tangent = np.array([1.0, 0.0])
        else:
            tangent = tangent / norm

        # left normal = tangent rotated +90deg (CCW)
        nx = -tangent[1]
        ny = tangent[0]

        offset = np.array([nx * width, ny * width, 0.0])
        left[i] = centerline[i] + offset
        right[i] = centerline[i] - offset

    return left, right


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Processing {BAG_PATH} ...")
    poses = extract_poses_3d(BAG_PATH)
    print(f"  Extracted {len(poses)} raw poses (x,y,z,yaw)")

    poses = downsample_by_distance(poses, MIN_DIST)
    print(f"  After downsampling: {len(poses)} points")

    poses = close_loop(poses)

    centerline = poses[:, :3]
    yaw = poses[:, 3]

    left_bound, right_bound = compute_boundaries_from_tangent(centerline, TRACK_WIDTH)

    # write CSV
    output_path = os.path.join(OUTPUT_DIR, "experiment_3d_2_bounds_3d.csv")
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['right_bound_x', 'right_bound_y', 'right_bound_z',
                         'left_bound_x', 'left_bound_y', 'left_bound_z'])
        for i in range(len(centerline)):
            writer.writerow([
                f"{right_bound[i, 0]:.6f}", f"{right_bound[i, 1]:.6f}", f"{right_bound[i, 2]:.6f}",
                f"{left_bound[i, 0]:.6f}", f"{left_bound[i, 1]:.6f}", f"{left_bound[i, 2]:.6f}"
            ])
    print(f"  Saved {len(centerline)} points to {output_path}")


if __name__ == '__main__':
    main()
