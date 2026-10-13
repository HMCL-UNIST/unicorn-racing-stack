#!/usr/bin/env python3
"""Step-by-step boundary detection viewer.

Reproduces gen_track_boundary_csv.py pipeline stage by stage and shows
how each stage modifies the left/right boundary for every pose index.

Usage:
    python visualization/step_by_step_boundary.py \
        --bag localization_bridge_2.bag \
        --wall pcd/wall.pcd \
        --min-dist 0.25
"""

import argparse
import numpy as np
import rosbag
import open3d as o3d
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def quat_to_yaw(q):
    return np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def smooth_circular(arr, window):
    n = len(arr)
    if window <= 0:
        return arr.copy()
    padded = np.concatenate([arr[-window:], arr, arr[:window]], axis=0)
    smoothed = np.zeros_like(arr)
    for i in range(n):
        smoothed[i] = np.mean(padded[i:i + 2 * window + 1], axis=0)
    return smoothed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--wall", default="pcd/wall.pcd")
    parser.add_argument("--pose-topic", default="/glim_ros/pose_corrected")
    parser.add_argument("--min-dist", type=float, default=0.25)
    parser.add_argument("--max-search", type=float, default=3.0)
    parser.add_argument("--density-thresh", type=int, default=10)
    parser.add_argument("--bin-size", type=float, default=0.05)
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--z-min-rel", type=float, default=-0.1,
                        help="Wall z min relative to robot [m]")
    parser.add_argument("--z-max-rel", type=float, default=0.1,
                        help="Wall z max relative to robot [m]")
    parser.add_argument("--start-idx", type=int, default=27)
    args = parser.parse_args()

    # Load wall
    wall_pts = np.asarray(o3d.io.read_point_cloud(args.wall).points)
    wall_tree_global = cKDTree(wall_pts[:, :3])

    # Load poses
    bag = rosbag.Bag(args.bag)
    raw_poses = []
    for _, msg, _ in bag.read_messages(topics=[args.pose_topic]):
        p = msg.pose.position
        yaw = quat_to_yaw(msg.pose.orientation)
        raw_poses.append([p.x, p.y, p.z, yaw])
    bag.close()
    raw_poses = np.array(raw_poses)

    # Downsample + single loop
    poses = [raw_poses[0]]
    for p in raw_poses[1:]:
        if np.linalg.norm(p[:3] - poses[-1][:3]) >= args.min_dist:
            poses.append(p)
    poses = np.array(poses)
    dists = np.linalg.norm(poses[:, :2] - poses[0, :2], axis=1)
    half = len(poses) // 2
    mi = half + np.argmin(dists[half:])
    poses = poses[:mi + 1]
    poses[-1] = poses[0].copy()
    n = len(poses)
    print(f"Poses: {n}")

    # Filter walls near track
    pose_tree = cKDTree(poses[:, :3])
    d, _ = pose_tree.query(wall_pts[:, :3])
    walls_near = wall_pts[d < args.max_search]
    wall_tree = cKDTree(walls_near[:, :3])

    # ====== Stage 1: Detection ======
    left_bound = np.full((n, 2), np.nan)
    right_bound = np.full((n, 2), np.nan)
    left_found = np.zeros(n, dtype=bool)
    right_found = np.zeros(n, dtype=bool)

    for i in range(n):
        cx, cy, cz = poses[i, :3]
        yaw = poses[i, 3]
        fx, fy = np.cos(yaw), np.sin(yaw)
        idx_list = wall_tree.query_ball_point([cx, cy, cz], args.max_search)
        if not idx_list:
            continue
        pts = walls_near[idx_list]
        dz = pts[:, 2] - cz
        z_mask = (dz > args.z_min_rel) & (dz < args.z_max_rel)
        pts = pts[z_mask]
        if len(pts) == 0:
            continue
        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        cross = fx * dy - fy * dx
        dot = fx * dx + fy * dy
        dist_2d = np.sqrt(dx ** 2 + dy ** 2)
        perp_mask = np.abs(cross) > np.abs(dot)

        for sign, arr, found in [(1, left_bound, left_found),
                                  (-1, right_bound, right_found)]:
            sm = (cross > 0.01) & perp_mask if sign > 0 else (cross < -0.01) & perp_mask
            if not sm.any():
                continue
            sd = dist_2d[sm]
            sp = pts[sm]
            bins = np.arange(0, args.max_search + args.bin_size, args.bin_size)
            counts, _ = np.histogram(sd, bins=bins)
            for b in range(len(counts)):
                if counts[b] >= args.density_thresh:
                    ib = (sd >= bins[b]) & (sd < bins[b + 1])
                    arr[i] = np.median(sp[ib, :2], axis=0)
                    found[i] = True
                    break

    stages = {}
    stages['1_detect'] = (left_bound.copy(), right_bound.copy(),
                          left_found.copy(), right_found.copy())

    # ====== Stage 2: Sliding window fallback ======
    slide_window = 5
    for arr, found in [(left_bound, left_found), (right_bound, right_found)]:
        for i in range(n):
            if found[i]:
                continue
            ws = max(0, i - slide_window)
            rf = found[ws:i]
            if rf.any():
                rw = np.linalg.norm(
                    arr[ws:i][rf] - poses[ws:i, :2][rf], axis=1)
                w = np.mean(rw)
                yaw = poses[i, 3]
                cx, cy = poses[i, 0], poses[i, 1]
                sign = 1 if arr is left_bound else -1
                arr[i] = [cx + sign * (-np.sin(yaw)) * w,
                          cy + sign * np.cos(yaw) * w]
                found[i] = True

    stages['2_fallback'] = (left_bound.copy(), right_bound.copy(),
                            left_found.copy(), right_found.copy())

    # ====== Stage 3: Interpolation ======
    for arr, found in [(left_bound, left_found), (right_bound, right_found)]:
        if found.all():
            continue
        vi = np.where(found)[0]
        mi2 = np.where(~found)[0]
        if len(vi) == 0:
            continue
        for dim in range(2):
            arr[mi2, dim] = np.interp(mi2, vi, arr[vi, dim], period=n)
        found[mi2] = True

    stages['3_interp'] = (left_bound.copy(), right_bound.copy(),
                          left_found.copy(), right_found.copy())

    # ====== Stage 4: Outlier filter ======
    for arr, found, label in [(left_bound, left_found, "left"),
                               (right_bound, right_found, "right")]:
        widths = np.linalg.norm(arr - poses[:, :2], axis=1)
        vw = widths[found]
        if len(vw) == 0:
            continue
        mw = np.median(vw)
        max_w = max(mw * 2.5, 1.5)
        outlier = found & (widths > max_w)
        if outlier.any():
            print(f"  [{label}] Outlier: {outlier.sum()} pts (>{max_w:.3f}m)")
            found[outlier] = False
            vi = np.where(found)[0]
            mi2 = np.where(~found)[0]
            if len(vi) > 0:
                for dim in range(2):
                    arr[mi2, dim] = np.interp(mi2, vi, arr[vi, dim], period=n)

    stages['4_outlier'] = (left_bound.copy(), right_bound.copy(),
                           left_found.copy(), right_found.copy())

    # ====== Stage 5: Smoothing (multiple window sizes) ======
    for sw in [1, 2, 3, 5]:
        sl = smooth_circular(left_bound, sw)
        sr = smooth_circular(right_bound, sw)
        stages[f'5_smooth_w{sw}'] = (sl, sr, left_found.copy(), right_found.copy())

    stage_names = list(stages.keys())
    stage_colors = {
        '1_detect': ('cyan', 'magenta'),
        '2_fallback': ('deepskyblue', 'salmon'),
        '3_interp': ('dodgerblue', 'tomato'),
        '4_outlier': ('blue', 'red'),
        '5_smooth_w1': ('green', 'orange'),
        '5_smooth_w2': ('darkgreen', 'darkorange'),
        '5_smooth_w3': ('darkblue', 'darkred'),
        '5_smooth_w5': ('purple', 'brown'),
    }

    # ====== Interactive viewer ======
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    plt.subplots_adjust(bottom=0.12)
    fig.canvas.manager.set_window_title('Step-by-step boundary detection')

    def draw(idx):
        idx = int(idx)
        for ax in axes:
            ax.cla()

        cx, cy, cz = poses[idx, :3]
        yaw = poses[idx, 3]
        fx, fy = np.cos(yaw), np.sin(yaw)

        # === Left: spatial view ===
        ax = axes[0]

        # Nearby wall points
        idx_list = wall_tree.query_ball_point([cx, cy, cz], args.max_search)
        if idx_list:
            pts = walls_near[idx_list]
            ax.scatter(pts[:, 0], pts[:, 1], c='lightgray', s=1, alpha=0.3)

        # Pose + heading
        ax.plot(cx, cy, 'ko', ms=8, zorder=20)
        ax.arrow(cx, cy, fx * 0.3, fy * 0.3,
                 head_width=0.08, head_length=0.05, fc='k', ec='k', zorder=20)

        # Each stage boundary
        for sname in stage_names:
            sl, sr, lf, rf = stages[sname]
            cl, cr = stage_colors[sname]
            if not np.isnan(sl[idx, 0]):
                ax.plot(sl[idx, 0], sl[idx, 1], 'o', color=cl, ms=8,
                        label=f'{sname} L={np.sqrt((sl[idx,0]-cx)**2+(sl[idx,1]-cy)**2):.3f}m',
                        zorder=10)
            if not np.isnan(sr[idx, 0]):
                ax.plot(sr[idx, 0], sr[idx, 1], 's', color=cr, ms=8,
                        label=f'{sname} R={np.sqrt((sr[idx,0]-cx)**2+(sr[idx,1]-cy)**2):.3f}m',
                        zorder=10)

        # Search radius
        th = np.linspace(0, 2 * np.pi, 100)
        ax.plot(cx + args.max_search * np.cos(th),
                cy + args.max_search * np.sin(th), 'g--', lw=0.5, alpha=0.3)

        ax.set_aspect('equal')
        ax.set_xlim(cx - args.max_search * 1.1, cx + args.max_search * 1.1)
        ax.set_ylim(cy - args.max_search * 1.1, cy + args.max_search * 1.1)
        ax.set_title(f'idx={idx}/{n - 1}', fontsize=13)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)

        # === Right: width per stage (bar chart) ===
        ax2 = axes[1]
        labels = []
        left_widths = []
        right_widths = []
        for sname in stage_names:
            sl, sr, lf, rf = stages[sname]
            lw = np.sqrt((sl[idx, 0] - cx) ** 2 + (sl[idx, 1] - cy) ** 2) if not np.isnan(sl[idx, 0]) else 0
            rw = np.sqrt((sr[idx, 0] - cx) ** 2 + (sr[idx, 1] - cy) ** 2) if not np.isnan(sr[idx, 0]) else 0
            labels.append(sname)
            left_widths.append(lw)
            right_widths.append(rw)

        x = np.arange(len(labels))
        w = 0.35
        ax2.bar(x - w / 2, left_widths, w, color='blue', alpha=0.7, label='Left')
        ax2.bar(x + w / 2, right_widths, w, color='red', alpha=0.7, label='Right')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=30, fontsize=9)
        ax2.set_ylabel('Width [m]')
        ax2.set_title(f'Boundary width at each pipeline stage (idx={idx})')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        fig.canvas.draw_idle()

    ax_sl = plt.axes([0.15, 0.02, 0.7, 0.03])
    slider = Slider(ax_sl, 'Index', 0, n - 1, valinit=args.start_idx, valstep=1)
    slider.on_changed(draw)

    def on_key(event):
        cur = int(slider.val)
        if event.key == 'right':
            slider.set_val(min(cur + 1, n - 1))
        elif event.key == 'left':
            slider.set_val(max(cur - 1, 0))

    fig.canvas.mpl_connect('key_press_event', on_key)

    draw(args.start_idx)
    plt.show()


if __name__ == '__main__':
    main()
