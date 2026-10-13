#!/usr/bin/env python3
"""Visualize boundary detection process in real-time."""

import numpy as np
import matplotlib.pyplot as plt
import rosbag
import open3d as o3d
from scipy.spatial import cKDTree

BAG_PATH = "/media/chg/T7/roboracer/experiment_3d/bag_to_csv_1.bag"
WALL_PCD = "pcd/wall.pcd"
POSE_TOPIC = "/glim_ros/base_pose"

MIN_DIST = 0.05
MAX_SEARCH = 2.0
RAY_WIDTH = 0.3


def quat_to_yaw(q):
    return np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def main():
    # Load wall
    wall_pcd = o3d.io.read_point_cloud(WALL_PCD)
    walls = np.asarray(wall_pcd.points)
    wall_tree = cKDTree(walls[:, :2])

    # Extract poses
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

    # Setup plot
    plt.ion()
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.scatter(walls[:, 0], walls[:, 1], c='lightgray', s=1, alpha=0.3)
    ax.plot(poses[:, 0], poses[:, 1], 'b-', linewidth=1, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_title('Boundary Detection (real-time)')
    ax.grid(True, alpha=0.3)

    left_xs, left_ys = [], []
    right_xs, right_ys = [], []
    left_line, = ax.plot([], [], 'g-', linewidth=2, label='left')
    right_line, = ax.plot([], [], 'r-', linewidth=2, label='right')
    ray_left, = ax.plot([], [], 'g--', linewidth=0.5, alpha=0.5)
    ray_right, = ax.plot([], [], 'r--', linewidth=0.5, alpha=0.5)
    pose_dot, = ax.plot([], [], 'bo', markersize=8)
    hit_left, = ax.plot([], [], 'go', markersize=6)
    hit_right, = ax.plot([], [], 'ro', markersize=6)
    ax.legend(fontsize=10)

    fig.canvas.draw()
    fig.canvas.flush_events()

    n = len(poses)
    for i in range(n):
        cx, cy = poses[i, 0], poses[i, 1]
        yaw = poses[i, 3]

        # left ray
        lnx, lny = -np.sin(yaw), np.cos(yaw)
        tx, ty = np.cos(yaw), np.sin(yaw)

        idx_list = wall_tree.query_ball_point([cx, cy], MAX_SEARCH)
        lx, ly, rx, ry = cx, cy, cx, cy

        if idx_list:
            nearby = walls[idx_list]
            dx = nearby[:, 0] - cx
            dy = nearby[:, 1] - cy
            n_proj = dx * lnx + dy * lny
            t_proj = dx * tx + dy * ty
            strip = np.abs(t_proj) < RAY_WIDTH

            left_mask = strip & (n_proj > 0.05)
            if left_mask.any():
                vals = n_proj[left_mask]
                ld = np.percentile(vals, 5) if len(vals) >= 5 else np.min(vals)
                lx = cx + lnx * ld
                ly = cy + lny * ld

            right_mask = strip & (n_proj < -0.05)
            if right_mask.any():
                vals = np.abs(n_proj[right_mask])
                rd = np.percentile(vals, 5) if len(vals) >= 5 else np.min(vals)
                rx = cx - lnx * rd
                ry = cy - lny * rd

        left_xs.append(lx)
        left_ys.append(ly)
        right_xs.append(rx)
        right_ys.append(ry)

        # update plot
        left_line.set_data(left_xs, left_ys)
        right_line.set_data(right_xs, right_ys)
        pose_dot.set_data([cx], [cy])
        hit_left.set_data([lx], [ly])
        hit_right.set_data([rx], [ry])
        ray_left.set_data([cx, cx + lnx * MAX_SEARCH], [cy, cy + lny * MAX_SEARCH])
        ray_right.set_data([cx, cx - lnx * MAX_SEARCH], [cy, cy - lny * MAX_SEARCH])

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.02)

    plt.ioff()
    ax.set_title('Boundary Detection Complete')
    plt.show()


if __name__ == '__main__':
    main()
