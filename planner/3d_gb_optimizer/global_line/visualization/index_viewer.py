#!/usr/bin/env python3
"""Interactive index viewer: Raw CSV vs Gen3D boundary with index labels.

Usage:
    python visualization/index_viewer.py
    python visualization/index_viewer.py --gen-path data/3d_track_data/localization_bridge_2_3d.csv
"""
import argparse
import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from track3D import Track3D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw_track_data/localization_bridge_2_bounds_3d.csv")
    parser.add_argument("--gen-path", default="data/3d_track_data/localization_bridge_2_3d.csv")
    parser.add_argument("--wall", default="pcd/wall.pcd")
    args = parser.parse_args()

    with open(args.raw) as f:
        raw = list(csv.DictReader(f))
    raw_lx = np.array([float(r['left_bound_x']) for r in raw])
    raw_ly = np.array([float(r['left_bound_y']) for r in raw])
    raw_rx = np.array([float(r['right_bound_x']) for r in raw])
    raw_ry = np.array([float(r['right_bound_y']) for r in raw])
    raw_cx = np.array([float(r['center_x']) for r in raw])
    raw_cy = np.array([float(r['center_y']) for r in raw])

    track_gen = Track3D(path=args.gen_path)
    left_gen, right_gen = track_gen.get_track_bounds()

    wall_pts = np.asarray(o3d.io.read_point_cloud(args.wall).points)

    fig, axes = plt.subplots(1, 2, figsize=(24, 11))
    fig.canvas.manager.set_window_title('Raw vs Gen3D - index view')

    for ax, title, cx, cy, lx, ly, rx, ry in [
        (axes[0], 'Raw CSV', raw_cx, raw_cy, raw_lx, raw_ly, raw_rx, raw_ry),
        (axes[1], 'Gen3D', track_gen.x, track_gen.y,
         left_gen[0], left_gen[1], right_gen[0], right_gen[1]),
    ]:
        ax.scatter(wall_pts[:, 0], wall_pts[:, 1], c='lightgray', s=0.3, alpha=0.2)
        ax.plot(np.append(lx, lx[0]), np.append(ly, ly[0]), 'b-', lw=1.5, label='Left')
        ax.plot(np.append(rx, rx[0]), np.append(ry, ry[0]), 'r-', lw=1.5, label='Right')
        ax.plot(np.append(cx, cx[0]), np.append(cy, cy[0]), 'k--', lw=0.7, alpha=0.4)
        for i in range(len(cx)):
            ax.plot([lx[i], rx[i]], [ly[i], ry[i]], 'g-', lw=0.3, alpha=0.3)
            ax.annotate(str(i), (cx[i], cy[i]), fontsize=6, fontweight='bold',
                        color='darkgreen', ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.7, ec='none'))
        ax.scatter(lx, ly, c='blue', s=5, zorder=5)
        ax.scatter(rx, ry, c='red', s=5, zorder=5)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
