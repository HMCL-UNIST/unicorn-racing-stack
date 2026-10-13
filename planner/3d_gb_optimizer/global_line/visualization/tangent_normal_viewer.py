#!/usr/bin/env python3
"""Visualize tangent vs normal perpendicularity on the track.

Shows boundary lines colored by |tangent · normal|:
  green = perpendicular (good), orange = slightly off, red = bad
Arrows show tangent (black) and normal (blue) at worst points.

Usage:
    python visualization/tangent_normal_viewer.py
    python visualization/tangent_normal_viewer.py --raw data/raw_track_data/localization_bridge_2_bounds_3d.csv
"""
import argparse
import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import open3d as o3d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw_track_data/localization_bridge_2_bounds_3d.csv")
    parser.add_argument("--wall", default="pcd/wall.pcd")
    parser.add_argument("--arrow-thresh", type=float, default=0.25,
                        help="Show arrows for |t·n| above this threshold")
    args = parser.parse_args()

    with open(args.raw) as f:
        raw = list(csv.DictReader(f))
    lx = np.array([float(r['left_bound_x']) for r in raw])
    ly = np.array([float(r['left_bound_y']) for r in raw])
    rx = np.array([float(r['right_bound_x']) for r in raw])
    ry = np.array([float(r['right_bound_y']) for r in raw])

    # gen_3d centerline = (left+right)/2
    gen_cx = (lx + rx) / 2
    gen_cy = (ly + ry) / 2

    # Tangent from centerline diff
    tang_x = np.diff(gen_cx, append=gen_cx[0])
    tang_y = np.diff(gen_cy, append=gen_cy[0])
    tang_len = np.sqrt(tang_x ** 2 + tang_y ** 2)
    tang_x /= tang_len
    tang_y /= tang_len

    # Normal from boundary
    w_left = np.sqrt((lx - gen_cx) ** 2 + (ly - gen_cy) ** 2)
    norm_x = (lx - gen_cx) / w_left
    norm_y = (ly - gen_cy) / w_left

    # tang dot normal
    tdotn = tang_x * norm_x + tang_y * norm_y

    wall_pts = np.asarray(o3d.io.read_point_cloud(args.wall).points)

    # Print summary
    print(f"Points: {len(lx)}")
    print(f"|t·n| max: {np.abs(tdotn).max():.3f}, mean: {np.abs(tdotn).mean():.3f}")
    print(f"Points |t·n|>0.1: {(np.abs(tdotn)>0.1).sum()}")
    print(f"Points |t·n|>0.3: {(np.abs(tdotn)>0.3).sum()}")

    fig, ax = plt.subplots(figsize=(16, 14))
    fig.canvas.manager.set_window_title('Tangent vs Normal perpendicularity')

    ax.scatter(wall_pts[:, 0], wall_pts[:, 1], c='lightgray', s=0.3, alpha=0.15)

    # Track boundaries
    ax.plot(np.append(lx, lx[0]), np.append(ly, ly[0]), 'b-', lw=1, alpha=0.5)
    ax.plot(np.append(rx, rx[0]), np.append(ry, ry[0]), 'r-', lw=1, alpha=0.5)

    # Boundary lines colored by |t·n|
    for i in range(len(lx)):
        val = abs(tdotn[i])
        if val > 0.3:
            color = 'red'; lw = 2; alpha = 0.9
        elif val > 0.15:
            color = 'orange'; lw = 1.5; alpha = 0.7
        else:
            color = 'green'; lw = 0.8; alpha = 0.4
        ax.plot([lx[i], rx[i]], [ly[i], ry[i]], color=color, lw=lw, alpha=alpha)

    # Arrows at worst points
    worst = np.where(np.abs(tdotn) > args.arrow_thresh)[0]
    scale = 0.4
    for i in worst:
        ax.arrow(gen_cx[i], gen_cy[i], tang_x[i] * scale, tang_y[i] * scale,
                 head_width=0.06, head_length=0.03, fc='black', ec='black', zorder=10)
        ax.arrow(gen_cx[i], gen_cy[i], norm_x[i] * scale, norm_y[i] * scale,
                 head_width=0.06, head_length=0.03, fc='blue', ec='blue', zorder=10)
        ax.annotate(f'{i} ({tdotn[i]:.2f})', (gen_cx[i], gen_cy[i]), fontsize=6,
                    color='red', fontweight='bold', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.8, ec='none'))

    legend_elements = [
        Line2D([0], [0], color='green', lw=2, label='|t·n| < 0.15 (OK)'),
        Line2D([0], [0], color='orange', lw=2, label='|t·n| 0.15~0.3'),
        Line2D([0], [0], color='red', lw=2, label='|t·n| > 0.3 (bad)'),
        Line2D([0], [0], color='black', lw=2, marker='>', label='tangent (diff centerline)'),
        Line2D([0], [0], color='blue', lw=2, marker='>', label='normal (from boundary)'),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc='upper left')
    ax.set_aspect('equal')
    ax.set_title('Tangent vs Normal perpendicularity\n(red = boundary not perpendicular to tangent)',
                 fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
