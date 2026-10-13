#!/usr/bin/env python3
"""Generate 3D track boundary CSV from wall.pcd + ground.pcd only.

1. Ground → 2D occupancy grid → distance transform → skeleton → centerline
2. Centerline + wall.pcd → ray casting → boundary (same as pcd_to_track_csv.py)
"""

import os
import csv
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import splprep, splev
import cv2


RESOLUTION = 0.05  # grid resolution in meters


def smooth_circular(arr, window):
    n = len(arr)
    if window <= 0:
        return arr
    padded = np.concatenate([arr[-window:], arr, arr[:window]], axis=0)
    smoothed = np.zeros_like(arr)
    for i in range(n):
        smoothed[i] = np.mean(padded[i:i + 2 * window + 1], axis=0)
    return smoothed


def ground_to_centerline(ground_pts, n_points=200):
    """Extract centerline from ground using distance transform + skeletonization."""
    pts_2d = ground_pts[:, :2]
    xmin, ymin = pts_2d.min(axis=0) - 0.5
    xmax, ymax = pts_2d.max(axis=0) + 0.5

    # 1. Create occupancy grid
    nx = int((xmax - xmin) / RESOLUTION) + 1
    ny = int((ymax - ymin) / RESOLUTION) + 1
    grid = np.zeros((ny, nx), dtype=bool)

    ix = ((pts_2d[:, 0] - xmin) / RESOLUTION).astype(int)
    iy = ((pts_2d[:, 1] - ymin) / RESOLUTION).astype(int)
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)
    grid[iy, ix] = True

    # dilate to fill sparse holes, then erode back
    img_raw = (grid * 255).astype(np.uint8)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grid = cv2.morphologyEx(img_raw, cv2.MORPH_CLOSE, k3) > 0
    print(f"  Grid: {nx}x{ny}, occupied: {grid.sum()}")

    # 2. Distance transform for erosion sizing
    dist = distance_transform_edt(grid)

    # Erode ground to remove branches, then thin to get clean skeleton
    img = (grid * 255).astype(np.uint8)

    # erode to shrink ground — removes thin branches
    kernel_size = max(3, int(dist.max() * 0.5)) | 1  # odd number, ~half track width
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv2.erode(img, kernel)
    print(f"  Erode kernel: {kernel_size}px, remaining pixels: {(eroded > 0).sum()}")

    # thin the eroded result — should be a clean loop
    skel = cv2.ximgproc.thinning(eroded)

    # 3. Find contours — pick the one with highest avg distance transform value
    contours, _ = cv2.findContours(skel, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    best_contour = None
    best_dist_avg = 0
    for c in contours:
        c = c.squeeze()
        if c.ndim < 2 or len(c) < 20:
            continue
        avg_d = np.mean([dist[pt[1], pt[0]] for pt in c])
        if avg_d > best_dist_avg:
            best_dist_avg = avg_d
            best_contour = c
    contour = best_contour
    print(f"  Best contour: {len(contour)} points (avg dist={best_dist_avg:.1f}px, from {len(contours)} contours)")

    # Convert to world coords
    ordered = np.zeros((len(contour), 2))
    ordered[:, 0] = contour[:, 0] * RESOLUTION + xmin
    ordered[:, 1] = contour[:, 1] * RESOLUTION + ymin

    # 6. Spline smooth + resample
    tck, u = splprep([ordered[:, 0], ordered[:, 1]], s=0.1, per=True)
    u_new = np.linspace(0, 1, n_points, endpoint=False)
    x_new, y_new = splev(u_new, tck)

    # Z from ground
    ground_tree = cKDTree(ground_pts[:, :2])
    z_new = np.zeros(n_points)
    for i in range(n_points):
        _, gi = ground_tree.query([x_new[i], y_new[i]])
        z_new[i] = ground_pts[gi, 2]

    cl = np.column_stack([x_new, y_new, z_new])
    cl = np.vstack([cl, cl[0]])  # close loop
    return cl


def find_boundary_ray(cx, cy, yaw, wall_tree, walls, direction, ray_width=0.3, max_search=2.0):
    """Shoot ray, return actual wall point or None."""
    lnx = -np.sin(yaw) * direction
    lny = np.cos(yaw) * direction
    tx = np.cos(yaw)
    ty = np.sin(yaw)

    idx_list = wall_tree.query_ball_point([cx, cy], max_search)
    if not idx_list:
        return None

    nearby = walls[idx_list]
    dx = nearby[:, 0] - cx
    dy = nearby[:, 1] - cy
    ray_proj = dx * lnx + dy * lny
    t_proj = dx * tx + dy * ty
    mask = (np.abs(t_proj) < ray_width) & (ray_proj > 0.05)

    if mask.any():
        return nearby[mask][np.argmin(ray_proj[mask]), :2]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wall", default="pcd/wall.pcd")
    parser.add_argument("--ground", default="pcd/ground.pcd")
    parser.add_argument("--output", default=None)
    parser.add_argument("--n-points", type=int, default=200)
    parser.add_argument("--smooth-window", type=int, default=3)
    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "raw_track_data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = args.output or os.path.join(output_dir, "pcd_only_bounds_3d.csv")

    walls = np.asarray(o3d.io.read_point_cloud(args.wall).points)
    ground = np.asarray(o3d.io.read_point_cloud(args.ground).points)
    print(f"Wall: {len(walls)}, Ground: {len(ground)}")

    # 1. Centerline from ground skeleton
    print("Extracting centerline...")
    centerline = ground_to_centerline(ground, n_points=args.n_points)
    n = len(centerline)
    print(f"  Centerline: {n} points")

    # 2. Tangent yaw
    tangent_yaw = np.zeros(n)
    for i in range(n):
        dx = centerline[(i + 1) % n, 0] - centerline[(i - 1) % n, 0]
        dy = centerline[(i + 1) % n, 1] - centerline[(i - 1) % n, 1]
        tangent_yaw[i] = np.arctan2(dy, dx)

    # 3. Boundary = ray cast to wall (same as pcd_to_track_csv.py)
    print("Finding boundaries...")
    wall_tree = cKDTree(walls[:, :2])
    left_bound = np.zeros((n, 3))
    right_bound = np.zeros((n, 3))
    left_found = np.zeros(n, dtype=bool)
    right_found = np.zeros(n, dtype=bool)

    for i in range(n):
        cx, cy, cz = centerline[i]
        yaw = tangent_yaw[i]

        lpt = find_boundary_ray(cx, cy, yaw, wall_tree, walls, +1)
        rpt = find_boundary_ray(cx, cy, yaw, wall_tree, walls, -1)

        if lpt is not None:
            left_bound[i] = [lpt[0], lpt[1], cz]
            left_found[i] = True
        if rpt is not None:
            right_bound[i] = [rpt[0], rpt[1], cz]
            right_found[i] = True

    # Filter outlier widths: if width suddenly jumps (>2x median or >2x neighbor),
    # treat as missing and interpolate from neighbors
    for arr, found, label in [(left_bound, left_found, "left"),
                               (right_bound, right_found, "right")]:
        widths = np.linalg.norm(arr[:, :2] - centerline[:, :2], axis=1)
        valid_widths = widths[found]
        if len(valid_widths) == 0:
            continue
        median_w = np.median(valid_widths)
        max_w = max(median_w * 2.5, 1.5)  # allow up to 2.5x median or 1.5m
        outlier = found & (widths > max_w)
        if outlier.any():
            print(f"  [{label}] Filtering {outlier.sum()} outlier points (width > {max_w:.3f}m)")
            found[outlier] = False

    n_miss_before = ((~left_found).sum(), (~right_found).sum())

    # Interpolate missing (not found + outlier filtered)
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

    print(f"  Missing+outlier: left={n_miss_before[0]}, right={n_miss_before[1]}")

    if args.smooth_window > 0:
        left_bound = smooth_circular(left_bound, args.smooth_window)
        right_bound = smooth_circular(right_bound, args.smooth_window)

    # 4. Write CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['right_bound_x', 'right_bound_y', 'right_bound_z',
                         'left_bound_x', 'left_bound_y', 'left_bound_z'])
        for i in range(n):
            writer.writerow([
                f"{right_bound[i,0]:.6f}", f"{right_bound[i,1]:.6f}", f"{right_bound[i,2]:.6f}",
                f"{left_bound[i,0]:.6f}", f"{left_bound[i,1]:.6f}", f"{left_bound[i,2]:.6f}"
            ])

    wl = np.linalg.norm(left_bound[:, :2] - centerline[:, :2], axis=1)
    wr = np.linalg.norm(right_bound[:, :2] - centerline[:, :2], axis=1)
    print(f"Left:  {wl.min():.3f} ~ {wl.max():.3f}m (mean {wl.mean():.3f})")
    print(f"Right: {wr.min():.3f} ~ {wr.max():.3f}m (mean {wr.mean():.3f})")
    print(f"Saved {n} points to {output_path}")


if __name__ == '__main__':
    main()
