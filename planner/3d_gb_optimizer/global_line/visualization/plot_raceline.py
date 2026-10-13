#!/usr/bin/env python3
"""Visualize full pipeline result: track boundaries + racing line + profiles.

Usage:
    python visualization/plot_raceline.py \
        --track data/smoothed_track_data/localization_bridge_2_3d_smoothed.csv \
        --raceline data/global_racing_lines/localization_bridge_2_3d_rc_car_10th_timeoptimal.csv

    Also overlay raw boundary CSV:
    python visualization/plot_raceline.py \
        --track data/smoothed_track_data/localization_bridge_2_3d_smoothed.csv \
        --raceline data/global_racing_lines/localization_bridge_2_3d_rc_car_10th_timeoptimal.csv \
        --raw data/raw_track_data/localization_bridge_2_bounds_3d.csv \
        --gen data/3d_track_data/localization_bridge_2_3d.csv
"""
# work around the numpy 1.17.4 + pandas 2.0.3 version mismatch
import sys as _sys
_sys.path[:] = [p for p in _sys.path if '/.local/' not in p]
import numpy as _np
_np.__version__ = '1.22.4'
if not hasattr(_np.random, 'BitGenerator'):
    from numpy.random.bit_generator import BitGenerator as _BG
    _np.random.BitGenerator = _BG

import argparse
import os
import sys
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from track3D import Track3D

# -----------------------------------------------------------------------------
def _close(arr):
    """Append first element to close loop for plotting."""
    return np.append(arr, arr[0])


def _speed_lc(x, y, v, lw=3.0, cmap='RdYlGn'):
    """Create a LineCollection colored by speed."""
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, linewidth=lw)
    lc.set_array(v)
    lc.set_clim(v.min(), v.max())
    return lc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--track',    required=True, help='Smoothed 3D track CSV')
    parser.add_argument('--raceline', required=True, help='Raceline CSV')
    parser.add_argument('--raw',      default=None,  help='Raw boundary CSV (optional, for comparison)')
    parser.add_argument('--gen',      default=None,  help='Gen3D track CSV before smoothing (optional)')
    args = parser.parse_args()

    # -- Load smoothed track & raceline ---------------------------------------
    track = Track3D(path=args.track)
    left_b, right_b = track.get_track_bounds()

    df      = pd.read_csv(args.raceline, sep=',')
    s_opt   = df['s_opt'].to_numpy()
    v_opt   = df['v_opt'].to_numpy()
    n_opt   = df['n_opt'].to_numpy()
    chi_opt = df['chi_opt'].to_numpy()
    ax_opt  = df['ax_opt'].to_numpy()
    ay_opt  = df['ay_opt'].to_numpy()
    jx_opt  = df['jx_opt'].to_numpy()
    jy_opt  = df['jy_opt'].to_numpy()
    laptime = df['laptime'].iloc[0]

    nv = track.get_normal_vector_numpy(theta=track.theta, mu=track.mu, phi=track.phi)

    # Raceline may have different resolution than track (opt_step != smooth_step).
    # Interpolate track quantities at raceline s_opt positions.
    if len(n_opt) != len(track.x):
        trk_x  = np.interp(s_opt, track.s, track.x)
        trk_y  = np.interp(s_opt, track.s, track.y)
        trk_z  = np.interp(s_opt, track.s, track.z)
        nv0    = np.interp(s_opt, track.s, nv[0])
        nv1    = np.interp(s_opt, track.s, nv[1])
        nv2    = np.interp(s_opt, track.s, nv[2])
        trk_phi = np.interp(s_opt, track.s, track.phi)
        trk_mu  = np.interp(s_opt, track.s, track.mu)
        trk_wl  = np.interp(s_opt, track.s, track.w_tr_left)
        trk_wr  = np.interp(s_opt, track.s, track.w_tr_right)
    else:
        trk_x, trk_y, trk_z = track.x, track.y, track.z
        nv0, nv1, nv2 = nv[0], nv[1], nv[2]
        trk_phi, trk_mu = track.phi, track.mu
        trk_wl, trk_wr = track.w_tr_left, track.w_tr_right

    rl_x = trk_x + nv0 * n_opt
    rl_y = trk_y + nv1 * n_opt

    # -- Load optional raw / gen3d data ---------------------------------------
    raw_data = None
    if args.raw and os.path.exists(args.raw):
        with open(args.raw) as f:
            rows = list(csv.DictReader(f))
        raw_data = {
            'lx': np.array([float(r['left_bound_x']) for r in rows]),
            'ly': np.array([float(r['left_bound_y']) for r in rows]),
            'rx': np.array([float(r['right_bound_x']) for r in rows]),
            'ry': np.array([float(r['right_bound_y']) for r in rows]),
        }

    gen_data = None
    if args.gen and os.path.exists(args.gen):
        gen_track = Track3D(path=args.gen)
        gl, gr = gen_track.get_track_bounds()
        gen_data = {'left': gl, 'right': gr, 'x': gen_track.x, 'y': gen_track.y}

    # ========================================================================
    #  FIGURE 1: Track map with racing line (speed-colored)
    # ========================================================================
    fig1 = plt.figure('Track + Racing Line', figsize=(12, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[40, 1], wspace=0.03)
    ax_map = fig1.add_subplot(gs[0])
    ax_cb  = fig1.add_subplot(gs[1])

    # Track boundary fill (light gray between walls)
    from matplotlib.patches import Polygon
    left_xy  = np.column_stack([_close(left_b[0]),  _close(left_b[1])])
    right_xy = np.column_stack([_close(right_b[0]), _close(right_b[1])])
    track_poly = np.vstack([left_xy, right_xy[::-1]])
    ax_map.add_patch(Polygon(track_poly, closed=True, fc='#e8e8e8', ec='none', zorder=0))

    # Boundaries
    ax_map.plot(_close(left_b[0]),  _close(left_b[1]),  color='#333', lw=1.8, zorder=2)
    ax_map.plot(_close(right_b[0]), _close(right_b[1]), color='#333', lw=1.8, zorder=2)

    # Centerline (faint)
    ax_map.plot(_close(track.x), _close(track.y), '--', color='#999', lw=0.6, zorder=1)

    # Racing line colored by speed
    lc = _speed_lc(rl_x, rl_y, v_opt, lw=3.5, cmap='RdYlGn')
    ax_map.add_collection(lc)
    plt.colorbar(lc, cax=ax_cb, label='Speed [m/s]')

    # Start marker
    ax_map.plot(rl_x[0], rl_y[0], 'o', color='white', ms=8, mec='black', mew=2, zorder=10)
    ax_map.annotate('START', (rl_x[0], rl_y[0]), fontsize=8, fontweight='bold',
                    xytext=(8, 8), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.2', fc='yellow', ec='k', alpha=0.9))

    ax_map.set_aspect('equal')
    ax_map.set_title(f'Laptime: {laptime:.3f}s  |  V: {v_opt.min():.1f}~{v_opt.max():.1f} m/s', fontsize=13, fontweight='bold')
    ax_map.grid(True, alpha=0.15)
    ax_map.set_xlabel('x [m]')
    ax_map.set_ylabel('y [m]')
    fig1.tight_layout()

    # ========================================================================
    #  FIGURE 4: 3D Track + Racing Line (speed-colored)
    # ========================================================================
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    rl_z = trk_z + nv2 * n_opt

    fig4 = plt.figure('3D Track + Racing Line', figsize=(14, 10))
    ax3d = fig4.add_subplot(111, projection='3d')

    # Track boundaries in 3D
    left_x, left_y, left_z = left_b[0], left_b[1], left_b[2]
    right_x, right_y, right_z = right_b[0], right_b[1], right_b[2]
    ax3d.plot(_close(left_x), _close(left_y), _close(left_z),
              color='#555', lw=1.0, alpha=0.5)
    ax3d.plot(_close(right_x), _close(right_y), _close(right_z),
              color='#555', lw=1.0, alpha=0.5)

    # Centerline (faint)
    ax3d.plot(_close(track.x), _close(track.y), _close(track.z),
              '--', color='#aaa', lw=0.5, alpha=0.3)

    # Racing line colored by speed (3D LineCollection)
    pts3d = np.column_stack([rl_x, rl_y, rl_z]).reshape(-1, 1, 3)
    segs3d = np.concatenate([pts3d[:-1], pts3d[1:]], axis=1)
    lc3d = Line3DCollection(segs3d, cmap='RdYlGn', linewidth=3.0)
    lc3d.set_array(v_opt)
    lc3d.set_clim(v_opt.min(), v_opt.max())
    ax3d.add_collection3d(lc3d)

    # Cross-bars every N points
    n_bars = 30
    step = max(1, len(track.x) // n_bars)
    for i in range(0, len(track.x), step):
        ax3d.plot([left_x[i], right_x[i]],
                  [left_y[i], right_y[i]],
                  [left_z[i], right_z[i]],
                  '-', color='green', lw=0.3, alpha=0.3)

    # Start marker
    ax3d.scatter([rl_x[0]], [rl_y[0]], [rl_z[0]], c='yellow', s=80,
                 edgecolors='black', linewidths=2, zorder=10)

    # Equal aspect ratio for all 3 axes
    all_x = np.concatenate([left_x, right_x, rl_x])
    all_y = np.concatenate([left_y, right_y, rl_y])
    all_z = np.concatenate([left_z, right_z, rl_z])
    max_range = max(all_x.ptp(), all_y.ptp(), all_z.ptp()) / 2.0
    mid_x, mid_y, mid_z = all_x.mean(), all_y.mean(), all_z.mean()
    ax3d.set_xlim(mid_x - max_range, mid_x + max_range)
    ax3d.set_ylim(mid_y - max_range, mid_y + max_range)
    ax3d.set_zlim(mid_z - max_range, mid_z + max_range)

    fig4.colorbar(lc3d, ax=ax3d, shrink=0.6, label='Speed [m/s]')
    ax3d.set_xlabel('x [m]')
    ax3d.set_ylabel('y [m]')
    ax3d.set_zlabel('z [m]')
    ax3d.set_title(f'3D Racing Line  —  Laptime {laptime:.3f}s\n'
                   f'V: {v_opt.min():.1f}~{v_opt.max():.1f} m/s',
                   fontsize=13, fontweight='bold')
    fig4.tight_layout()

    # ========================================================================
    #  FIGURE 5: 3D Profiles (z, speed, accel along track)
    # ========================================================================
    # fig5, axes5 = plt.subplots(4, 1, figsize=(14, 9), sharex=True, num='3D Profiles')

    # # Elevation
    # ax_z = axes5[0]
    # ax_z.plot(s_opt, trk_z, color='C1', lw=1.2, label='centerline z')
    # ax_z.plot(s_opt, rl_z, color='C0', lw=1.5, label='raceline z')
    # ax_z.set_ylabel('z [m]')
    # ax_z.legend(fontsize=7, loc='upper right')

    # # Banking angle (phi)
    # ax_phi = axes5[1]
    # ax_phi.plot(s_opt, np.degrees(trk_phi), color='C4', lw=1.2)
    # ax_phi.set_ylabel(r'Banking $\phi$ [deg]')
    # ax_phi.axhline(0, color='k', lw=0.4)

    # # Pitch (mu)
    # ax_mu = axes5[2]
    # ax_mu.plot(s_opt, np.degrees(trk_mu), color='C5', lw=1.2)
    # ax_mu.set_ylabel(r'Pitch $\mu$ [deg]')
    # ax_mu.axhline(0, color='k', lw=0.4)

    # # Speed (repeated for context with 3D data)
    # ax_v3 = axes5[3]
    # ax_v3.fill_between(s_opt, 0, v_opt, alpha=0.15, color='C0')
    # ax_v3.plot(s_opt, v_opt, color='C0', lw=1.5)
    # ax_v3.set_ylabel('V [m/s]')
    # ax_v3.set_xlabel('s [m]')
    # ax_v3.set_ylim(bottom=0)

    # for ax in axes5:
    #     ax.grid(True, alpha=0.2)
    #     ax.tick_params(labelsize=8)

    # fig5.suptitle(f'3D Track Profiles  —  Laptime {laptime:.3f}s', fontsize=13, fontweight='bold')
    # fig5.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()
