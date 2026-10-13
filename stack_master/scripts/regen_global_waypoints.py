#!/usr/bin/env python3
"""Regenerate stack_master/maps/<map>/global_waypoints.json from
the raw smoothed CSV + time-optimal raceline CSV.

Uses the same pipeline as
  planner/3d_gb_optimizer/global_line/global_racing_line/export_global_waypoints.py
so `psi_rad` comes from the (periodic) CubicSpline derivative and is always
self-consistent with point ordering — fixes cases where the existing JSON
has a seam-flipped psi (e.g. gazebo_wall_2 idx 0 was +88° instead of -103°).

Boundaries (`d_left`, `d_right`) come from Track3D's interpolators and are
therefore perpendicular to the smoothed centerline tangent, not to the
resampled raceline tangent. Good enough for planners; keeps the Track3D
contract intact.

Usage (in docker container):
  python3 stack_master/scripts/regen_global_waypoints.py gazebo_wall_2
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MAPS_DIR = os.path.join(REPO_ROOT, 'stack_master', 'maps')
TRACK3D_SRC = os.path.join(
    REPO_ROOT, 'planner', '3d_gb_optimizer', 'global_line', 'src'
)


# -- Marker helpers (1:1 with rebuild_waypoints.sh defaults) --------------

def _marker(mid, x, y, z, scale, color, mtype=2):
    return {
        'header': {'seq': 0, 'stamp': {'secs': 0, 'nsecs': 0}, 'frame_id': 'map'},
        'ns': '', 'id': int(mid), 'type': int(mtype), 'action': 0,
        'pose': {
            'position': {'x': float(x), 'y': float(y), 'z': float(z)},
            'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1},
        },
        'scale': scale,
        'color': color,
        'lifetime': {'secs': 0, 'nsecs': 0},
        'frame_locked': False,
        'points': [], 'colors': [], 'text': '',
        'mesh_resource': '', 'mesh_use_embedded_materials': False,
    }


SPHERE_SCALE = {'x': 0.05, 'y': 0.05, 'z': 0.05}
COLOR_BLUE = {'r': 0.0, 'g': 0.0, 'b': 1.0, 'a': 1.0}
COLOR_PURPLE = {'r': 0.5, 'g': 0.0, 'b': 0.5, 'a': 1.0}
COLOR_YELLOWGREEN = {'r': 0.5, 'g': 1.0, 'b': 0.0, 'a': 1.0}
COLOR_RED = {'r': 1.0, 'g': 0.0, 'b': 0.0, 'a': 1.0}
VEL_SCALE_XY = 0.1
VEL_SCALE_FACTOR = 0.1317


def _vel_color(v, vmin, vmax):
    t = 0.5 if vmax <= vmin else (v - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    return {'r': round(1.0 - t, 6), 'g': round(t, 6), 'b': 0.0, 'a': 1.0}


def _centerline_markers(track):
    return [
        _marker(i, float(track.x[i]), float(track.y[i]), 0.0,
                SPHERE_SCALE, COLOR_BLUE)
        for i in range(len(track.s))
    ]


def _raceline_sphere_markers(xs, ys, vs):
    vmin, vmax = float(min(vs)), float(max(vs))
    return [
        _marker(i, float(xs[i]), float(ys[i]), 0.0,
                SPHERE_SCALE, _vel_color(float(vs[i]), vmin, vmax))
        for i in range(len(xs))
    ]


def _raceline_vel_cylinders(xs, ys, vs):
    out = []
    for i in range(len(xs)):
        h = float(vs[i]) * VEL_SCALE_FACTOR
        out.append(_marker(
            i, float(xs[i]), float(ys[i]), round(h / 2.0, 6),
            {'x': VEL_SCALE_XY, 'y': VEL_SCALE_XY, 'z': round(h, 6)},
            COLOR_RED, mtype=3,
        ))
    return out


def _trackbounds_markers_wpnt_aligned(waypoints):
    """1:1 w/ raceline wpnts, LEFT=purple, RIGHT=yellowgreen, normal = psi±π/2."""
    out = []
    for i, w in enumerate(waypoints):
        x, y, psi = w['x_m'], w['y_m'], w['psi_rad']
        dl, dr = w['d_left'], w['d_right']
        lx = x + dl * math.cos(psi + math.pi / 2)
        ly = y + dl * math.sin(psi + math.pi / 2)
        rx = x + dr * math.cos(psi - math.pi / 2)
        ry = y + dr * math.sin(psi - math.pi / 2)
        out.append(_marker(2 * i,     lx, ly, 0.0, SPHERE_SCALE, COLOR_PURPLE))
        out.append(_marker(2 * i + 1, rx, ry, 0.0, SPHERE_SCALE, COLOR_YELLOWGREEN))
    return out


# -- centerline_waypoints (unchanged contract) ----------------------------

def _centerline_waypoints(track):
    n = len(track.s)
    wpnts = []
    for k in range(n):
        wpnts.append({
            'id': k,
            's_m': float(track.s[k]),
            'd_m': 0.0,
            'x_m': float(track.x[k]),
            'y_m': float(track.y[k]),
            'z_m': float(track.z[k]),
            'd_right': float(abs(track.w_tr_right[k])),
            'd_left': float(abs(track.w_tr_left[k])),
            'psi_rad': float(track.theta[k]),
            'kappa_radpm': float(track.Omega_z[k]),
            'vx_mps': 0.0,
            'ax_mps2': 0.0,
        })
    return {
        'header': {'seq': 0, 'stamp': {'secs': 0, 'nsecs': 0}, 'frame_id': ''},
        'wpnts': wpnts,
    }


# -- main pipeline --------------------------------------------------------

def _find_csv(map_dir, suffix):
    hits = [f for f in os.listdir(map_dir) if f.endswith(suffix)]
    if not hits:
        raise FileNotFoundError(f'no file ending with {suffix} in {map_dir}')
    if len(hits) > 1:
        print(f'[warn] multiple {suffix} in {map_dir}; picking {hits[0]}')
    return os.path.join(map_dir, hits[0])


def _dedup_close(arr_list, tol=1e-7):
    """Drop indices where consecutive xyz gap < tol (prevents spline blow-up).

    arr_list = [x, y, z, *others]; returns the filtered arrays.
    """
    x, y, z = arr_list[0], arr_list[1], arr_list[2]
    keep = [0]
    for i in range(1, len(x)):
        j = keep[-1]
        if (x[i] - x[j])**2 + (y[i] - y[j])**2 + (z[i] - z[j])**2 > tol**2:
            keep.append(i)
    keep = np.asarray(keep)
    return [a[keep] for a in arr_list]


def regen(map_name, spacing=0.1, dry_run=False):
    map_dir = os.path.join(MAPS_DIR, map_name)
    if not os.path.isdir(map_dir):
        raise FileNotFoundError(map_dir)

    sys.path.insert(0, TRACK3D_SRC)
    from track3D import Track3D  # noqa: E402

    track_csv = _find_csv(map_dir, '_3d_smoothed.csv')
    rl_csv = _find_csv(map_dir, '_timeoptimal.csv')
    print(f'[load] track  = {track_csv}')
    print(f'[load] racing = {rl_csv}')

    track = Track3D(path=track_csv)

    rl = pd.read_csv(rl_csv)
    s_opt = rl['s_opt'].to_numpy()
    v_opt = rl['v_opt'].to_numpy()
    n_opt = rl['n_opt'].to_numpy()
    ax_opt = rl['ax_opt'].to_numpy()
    laptime = float(rl['laptime'].iloc[0])

    # Step 1: curvilinear → Cartesian via Track3D
    n = len(s_opt)
    x_raw = np.empty(n); y_raw = np.empty(n); z_raw = np.empty(n)
    for k in range(n):
        cart = track.sn2cartesian(s_opt[k], n_opt[k])
        x_raw[k], y_raw[k], z_raw[k] = float(cart[0]), float(cart[1]), float(cart[2])

    # Drop duplicate seam (last == first). Raceline CSV typically has one.
    closure = math.hypot(x_raw[-1] - x_raw[0], y_raw[-1] - y_raw[0])
    if closure < 1e-3:
        x_r, y_r, z_r = x_raw[:-1], y_raw[:-1], z_raw[:-1]
        v_r, ax_r, s_opt_r, n_opt_r = v_opt[:-1], ax_opt[:-1], s_opt[:-1], n_opt[:-1]
        print(f'[seam] duplicate endpoint stripped (closure={closure:.2e} m)')
    else:
        x_r, y_r, z_r = x_raw, y_raw, z_raw
        v_r, ax_r, s_opt_r, n_opt_r = v_opt, ax_opt, s_opt, n_opt

    # Safety dedup (CSV may have stacked rows); keep the parallel arrays aligned.
    x_r, y_r, z_r, v_r, ax_r, s_opt_r, n_opt_r = _dedup_close(
        [x_r, y_r, z_r, v_r, ax_r, s_opt_r, n_opt_r], tol=1e-6,
    )
    if len(x_r) < len(x_raw) - 1:
        print(f'[seam] dedup removed {len(x_raw) - 1 - len(x_r)} degenerate rows')

    # Step 2: arc-length parameterization (Cartesian) + closure gap
    ds = np.sqrt(np.diff(x_r)**2 + np.diff(y_r)**2 + np.diff(z_r)**2)
    ds_close = math.sqrt(
        (x_r[0] - x_r[-1])**2 + (y_r[0] - y_r[-1])**2 + (z_r[0] - z_r[-1])**2
    )
    arc = np.empty(len(x_r) + 1)
    arc[0] = 0.0
    arc[1:-1] = np.cumsum(ds)
    arc[-1] = arc[-2] + ds_close
    total = float(arc[-1])

    # Step 3: periodic CubicSpline (value@end == value@start)
    cs_x = CubicSpline(arc, np.append(x_r, x_r[0]), bc_type='periodic')
    cs_y = CubicSpline(arc, np.append(y_r, y_r[0]), bc_type='periodic')
    cs_z = CubicSpline(arc, np.append(z_r, z_r[0]), bc_type='periodic')

    # Step 4: uniform 0.1 m resample + seam duplicate (2D gb_optimizer convention)
    n_inner = int(round(total / spacing))
    arc_inner = np.linspace(0.0, total, n_inner, endpoint=False)
    arc_new = np.concatenate([arc_inner, [total]])
    n_new = len(arc_new)

    x_new = cs_x(arc_new)
    y_new = cs_y(arc_new)
    z_new = cs_z(arc_new)

    # Linear interp for v/ax/s_opt/n_opt along arc (periodic wrap)
    arc_r_inner = arc[:-1]
    v_new = np.interp(arc_new, arc_r_inner, v_r, period=total)
    ax_new = np.interp(arc_new, arc_r_inner, ax_r, period=total)
    s_opt_new = np.interp(arc_new, arc_r_inner, s_opt_r, period=total)
    n_opt_new = np.interp(arc_new, arc_r_inner, n_opt_r, period=total)

    # Step 5: psi/kappa/mu from spline derivatives (guaranteed monotone at seam)
    dxdt = cs_x(arc_new, 1)
    dydt = cs_y(arc_new, 1)
    dzdt = cs_z(arc_new, 1)
    d2x = cs_x(arc_new, 2)
    d2y = cs_y(arc_new, 2)

    psi = np.arctan2(dydt, dxdt)
    denom = (dxdt**2 + dydt**2)**1.5
    denom = np.where(denom < 1e-9, 1e-9, denom)
    kappa = (dxdt * d2y - dydt * d2x) / denom
    mu = -np.arcsin(np.clip(
        dzdt / np.sqrt(dxdt**2 + dydt**2 + dzdt**2 + 1e-12), -1.0, 1.0,
    ))

    # Step 6: boundaries from Track3D interpolators (perpendicular to centerline)
    waypoints = []
    for k in range(n_new):
        w_tr_l = float(track.w_tr_left_interpolator(s_opt_new[k]))
        w_tr_r = float(track.w_tr_right_interpolator(s_opt_new[k]))
        d_left = w_tr_l - float(n_opt_new[k])
        d_right = -w_tr_r + float(n_opt_new[k])
        waypoints.append({
            'id': k,
            's_m': float(arc_new[k]),
            'd_m': 0.0,
            'x_m': float(x_new[k]),
            'y_m': float(y_new[k]),
            'z_m': float(z_new[k]),
            'd_right': float(abs(d_right)),
            'd_left': float(abs(d_left)),
            'psi_rad': float(psi[k]),
            'kappa_radpm': float(kappa[k]),
            'vx_mps': float(v_new[k]),
            'ax_mps2': float(ax_new[k]),
            'mu_rad': float(mu[k]),
        })

    # Step 7: markers
    cl_markers = _centerline_markers(track)
    rl_sphere = _raceline_sphere_markers(x_new, y_new, v_new)
    rl_cyl = _raceline_vel_cylinders(x_new, y_new, v_new)
    tb_markers = _trackbounds_markers_wpnt_aligned(waypoints)

    # Step 8: assemble JSON matching existing key order
    info_str = (
        f'estimated lap time: {laptime:.4f}s; '
        f'maximum speed: {float(v_new.max()):.4f}m/s; '
    )
    output = {
        'map_info_str': {'data': info_str},
        'est_lap_time': {'data': laptime},
        'centerline_markers': {'markers': cl_markers},
        'centerline_waypoints': _centerline_waypoints(track),
        'global_traj_markers_iqp': {'markers': rl_sphere},
        'global_traj_wpnts_iqp': {
            'header': {'seq': 0, 'stamp': {'secs': 0, 'nsecs': 0}, 'frame_id': ''},
            'wpnts': waypoints,
        },
        'global_traj_markers_sp': {'markers': rl_sphere},
        'global_traj_wpnts_sp': {
            'header': {'seq': 1, 'stamp': {'secs': 0, 'nsecs': 0}, 'frame_id': ''},
            'wpnts': waypoints,
        },
        'trackbounds_markers': {'markers': tb_markers},
        'global_traj_vel_markers_sp': {'markers': rl_cyl},
    }

    out_path = os.path.join(map_dir, 'global_waypoints.json')
    backup_path = os.path.join(map_dir, 'global_waypoints_backup.json')

    if not os.path.exists(backup_path) and os.path.exists(out_path):
        import shutil
        shutil.copy2(out_path, backup_path)
        print(f'[backup] {out_path} → {backup_path}')
    elif os.path.exists(backup_path):
        print(f'[backup] already exists: {backup_path} (not overwritten)')

    # sanity report
    ds_check = np.sqrt(np.diff(x_new)**2 + np.diff(y_new)**2 + np.diff(z_new)**2)
    psi_fd = np.arctan2(np.diff(y_new), np.diff(x_new))
    mism = (psi[:-1] - psi_fd + np.pi) % (2 * np.pi) - np.pi
    print(f'[stats] waypoints={n_new}  total_arc={total:.4f} m  '
          f'laptime={laptime:.3f}s')
    print(f'[stats] v=[{v_new.min():.2f}, {v_new.max():.2f}] m/s  '
          f'z=[{z_new.min():.3f}, {z_new.max():.3f}]')
    print(f'[stats] ds_xyz mean={ds_check.mean():.5f}  '
          f'std={ds_check.std():.2e}  min={ds_check.min():.5f}  '
          f'max={ds_check.max():.5f}')
    print(f'[stats] psi vs finite-diff: max={np.degrees(np.abs(mism)).max():.3f}°  '
          f'mismatches>45°={int((np.abs(mism) > np.pi / 4).sum())}')

    if dry_run:
        print('[dry-run] not writing; pass --no-dry-run to commit')
        return

    with open(out_path, 'w') as f:
        json.dump(output, f)
    print(f'[write] {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('map_name', help='map folder under stack_master/maps/')
    ap.add_argument('--spacing', type=float, default=0.1)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    regen(args.map_name, spacing=args.spacing, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
