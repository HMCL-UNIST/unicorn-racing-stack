import os
import sys
import struct
import argparse
import subprocess

import json

import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import CubicSpline
from matplotlib import pyplot as plt


# ----------------------------------------------------------------------
# Path setup — same pattern as plot_racing_line_full.py
# ----------------------------------------------------------------------
dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, '..', 'data')
sys.path.append(os.path.join(dir_path, '..', 'src'))

from track3D import Track3D  # noqa: E402

# reuse the marker/waypoint builders from export_global_waypoints (no duplicated code)
from export_global_waypoints import (  # noqa: E402
    _build_sphere_markers,
    _build_speed_sphere_markers,
    _build_cylinder_markers,
    _build_trackbounds_markers,
    _build_centerline_waypoints,
)

# default inputs (override on the CLI)
DEFAULT_TRACK    = 'gazebo_wall_2_3d_smoothed.csv'
DEFAULT_RACELINE = 'gazebo_wall_2_3d_rc_car_10th_timeoptimal.csv'
DEFAULT_VEHICLE  = 'rc_car_10th'
DEFAULT_PARAMS   = 'params_rc_car_10th.yml'
DEFAULT_MODEL    = 'lookup'

# C++ runner path
# FBGA_ROOT env var takes precedence; otherwise use f110_utils/libs/FBGA of this repository
FBGA_ROOT  = os.environ.get('FBGA_ROOT', os.path.normpath(
    os.path.join(dir_path, '..', '..', '..', '..', 'f110_utils', 'libs', 'FBGA')))
RUNNER_BIN = os.path.join(FBGA_ROOT, 'bin', 'GIGI_test_unicorn.exe')

# work folder / figure folder
WORK_DIR   = os.path.join(data_path, 'fwbw_work')
FIGURE_DIR = os.path.join(data_path, 'figure')


# ----------------------------------------------------------------------
# Step A. NLP racing line CSV → Cartesian resampling + curvature/pitch
#   Same logic as export_global_waypoints.py.
#
#   1) (s_opt, n_opt) → Cartesian (x, y, z)  via Track3D.sn2cartesian
#   2) arc length including the closure gap
#   3) resample at uniform spacing (default 0.1m) with a periodic CubicSpline (x, y, z)
#   4) 1st/2nd spline derivatives → kappa = (x'·y'' − y'·x'') / (x'² + y'²)^1.5
#   5) spline z derivative → mu = -arcsin(dz / |dr|)
#   6) interpolate v_opt onto the new arc-length grid as well
#
#   Returns: dict with s, x, y, z, kappa, mu, v
# ----------------------------------------------------------------------
def resample_raceline_cartesian(track: Track3D, rl_df: pd.DataFrame,
                                spacing: float = 0.1) -> dict:
    s_opt = rl_df['s_opt'].to_numpy()
    n_opt = rl_df['n_opt'].to_numpy()
    v_opt = rl_df['v_opt'].to_numpy()

    # -- 1) curvilinear → Cartesian (sn2cartesian per point) --
    n_pts = len(s_opt)
    x_raw = np.zeros(n_pts)
    y_raw = np.zeros(n_pts)
    z_raw = np.zeros(n_pts)
    for k in range(n_pts):
        cart = track.sn2cartesian(s_opt[k], n_opt[k])
        x_raw[k] = float(cart[0])
        y_raw[k] = float(cart[1])
        z_raw[k] = float(cart[2])

    # -- 2) drop the duplicated end point + arc length including the closure gap --
    x_r = x_raw[:-1]; y_r = y_raw[:-1]; z_r = z_raw[:-1]
    v_r       = v_opt[:-1]
    s_opt_r   = s_opt[:-1]
    n_opt_r   = n_opt[:-1]

    ds_r     = np.sqrt(np.diff(x_r)**2 + np.diff(y_r)**2 + np.diff(z_r)**2)
    ds_close = np.sqrt((x_r[0]-x_r[-1])**2 + (y_r[0]-y_r[-1])**2 + (z_r[0]-z_r[-1])**2)
    arc_r    = np.zeros(len(x_r) + 1)
    arc_r[1:-1] = np.cumsum(ds_r)
    arc_r[-1]   = arc_r[-2] + ds_close
    total_loop  = arc_r[-1]

    # -- 3) periodic CubicSpline resampling --
    cs_x = CubicSpline(arc_r, np.append(x_r, x_r[0]), bc_type='periodic')
    cs_y = CubicSpline(arc_r, np.append(y_r, y_r[0]), bc_type='periodic')
    cs_z = CubicSpline(arc_r, np.append(z_r, z_r[0]), bc_type='periodic')

    n_new   = int(round(total_loop / spacing))  # numpy 1.17 round() returns float64 → cast to int
    arc_new = np.linspace(0, total_loop, n_new, endpoint=False)

    x_new = cs_x(arc_new)
    y_new = cs_y(arc_new)
    z_new = cs_z(arc_new)

    # -- 4) heading + kappa (Cartesian formula) --
    dx_dt   = cs_x(arc_new, 1)
    dy_dt   = cs_y(arc_new, 1)
    d2x_dt2 = cs_x(arc_new, 2)
    d2y_dt2 = cs_y(arc_new, 2)
    psi   = np.arctan2(dy_dt, dx_dt)
    kappa = (dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (dx_dt**2 + dy_dt**2)**1.5

    # -- 5) mu (pitch) --
    # mu from the spline z-derivative shows ringing artifacts that are not in the raw data
    #      (e.g. on one test track dmu/ds spiked to -0.45 around s≈36m).
    #      → the spline-based mu is kept as a fallback,
    #        but (6) below replaces it with Track3D.mu_interpolator.
    dz_dt = cs_z(arc_new, 1)
    mu_spline = -np.arcsin(np.clip(
        dz_dt / np.sqrt(dx_dt**2 + dy_dt**2 + dz_dt**2),
        -1.0, 1.0))

    # -- 6) interpolate v, s_opt (centerline), n_opt (same as export_global_waypoints) --
    arc_r_inner = arc_r[:-1]
    v_new     = np.interp(arc_new, arc_r_inner, v_r,     period=total_loop)
    s_opt_new = np.interp(arc_new, arc_r_inner, s_opt_r, period=total_loop)
    n_opt_new = np.interp(arc_new, arc_r_inner, n_opt_r, period=total_loop)

    # -- 6b) interpolate the raw mu (Track3D raw values) directly at the centerline s,
    #       then compute dmu/ds by central differences on the raceline arc-length grid.
    #       phi (bank) is practically 0 on these tracks and is ignored.
    mu = np.array([float(track.mu_interpolator(s_opt_new[k])) for k in range(n_new)])

    # central diff (periodic wrap)
    mu_wrap = np.concatenate([[mu[-1]], mu, [mu[0]]])
    ds_grid = (arc_new[-1] - arc_new[0]) / (n_new - 1) if n_new > 1 else spacing
    # arc_new uses endpoint=False, so the wrap-around ds at both ends can be assumed equal to ds_grid
    dmu_ds = (mu_wrap[2:] - mu_wrap[:-2]) / (2.0 * ds_grid)

    return {
        's':         arc_new,
        'x':         x_new,
        'y':         y_new,
        'z':         z_new,
        'psi':       psi,
        'kappa':     kappa,
        'mu':        mu,         # raw centerline mu (Track3D linear interp)
        'mu_spline': mu_spline,  # for comparison / debugging
        'dmu_ds':    dmu_ds,     # raw mu central diff
        'v':         v_new,
        's_opt_new': s_opt_new,   # centerline arc length (interpolated)
        'n_opt_new': n_opt_new,   # lateral offset (interpolated)
        'n_pts':     n_new,
        'total_length': total_loop,
    }


# ----------------------------------------------------------------------
# Step B. N-laps trick — handling the closed loop
#
#   FBGA's FWBW is open loop (fixed start v0, free end boundary), but
#   the track is a closed loop. To remove the effect of v0 the input is stacked N times,
#   solved, and a single lap is extracted as the result → close to steady state.
#
#   stack_laps:          stacks one lap (s, kappa, g_tilde) N times
#   extract_middle_lap:  cuts out the middle lap and restarts s at 0
#                        (avoids the boundary effects at both ends, closest to steady state)
# ----------------------------------------------------------------------
def stack_laps(s: np.ndarray, kappa: np.ndarray, g_tilde: np.ndarray,
               n_laps: int) -> tuple:
    """
    Returns: (s_stack, k_stack, g_stack, lap_length, n_pts_per_lap)
    """
    n_pts = len(s)
    # total lap length: arc_new uses endpoint=False, so up to the last point + one spacing
    # spacing: s is assumed uniform, so the mean ds is used
    ds  = (s[-1] - s[0]) / (n_pts - 1)
    lap_length = s[-1] - s[0] + ds          # full lap (including the closure)

    s_stack = np.concatenate([s + i * lap_length for i in range(n_laps)])
    k_stack = np.tile(kappa,   n_laps)
    g_stack = np.tile(g_tilde, n_laps)
    return s_stack, k_stack, g_stack, lap_length, n_pts


def extract_middle_lap(fwbw_df: pd.DataFrame, n_laps: int,
                       lap_length: float, n_pts_per_lap: int) -> pd.DataFrame:
    """
    Extract the middle lap from the fwbw result (N laps) and restart s at 0.

    Reason: FWBW is open loop, so
      - lap 1   : affected by the forward boundary (v0)
      - lap N   : affected by the backward boundary (v_end)
      - middle  : free of both boundary effects, closest to steady state.
    With 3 laps the middle is lap 2 (0-indexed = 1).
    """
    middle = n_laps // 2
    start  = middle * n_pts_per_lap
    end    = start  + n_pts_per_lap
    out = fwbw_df.iloc[start:end].reset_index(drop=True).copy()
    out['s'] = out['s'] - middle * lap_length
    return out


# ----------------------------------------------------------------------
# NaN handling helper
#   - if the FWBW BW pass fails to converge, v/ax of some segments become NaN.
#   - NaN in the JSON can break the ROS controller.
#   - fill them by linear interpolation between the neighboring valid values.
#   - fail fast if more than 5% are NaN.
# ----------------------------------------------------------------------
def fill_nan_interp(arr: np.ndarray) -> np.ndarray:
    nan_mask = np.isnan(arr)
    if not nan_mask.any():
        return arr
    valid_idx = np.where(~nan_mask)[0]
    if len(valid_idx) == 0:
        return np.zeros_like(arr)
    out = arr.copy()
    out[nan_mask] = np.interp(
        np.where(nan_mask)[0],
        valid_idx,
        arr[valid_idx],
    )
    return out


def sanitize_nan(arr: np.ndarray, name: str,
                 max_nan_ratio: float = 0.05) -> np.ndarray:
    n_nan = int(np.isnan(arr).sum())
    if n_nan == 0:
        return arr
    ratio = n_nan / len(arr)
    if ratio > max_nan_ratio:
        raise RuntimeError(
            f'{name}: too many NaNs ({n_nan}/{len(arr)} = {ratio:.1%}), '
            f'threshold {max_nan_ratio:.0%}'
        )
    print(f'  WARNING: {n_nan} NaN(s) in {name} ({ratio:.2%}), interpolating')
    return fill_nan_interp(arr)


# ----------------------------------------------------------------------
# Step C. apparent gravity g̃(s, V) — including the 3D correction
#
#   g_tilde(s, V) = 9.81 · cos(mu(s))  −  V(s)² · dmu/ds(s)
#
#   Sign note — Track3D's mu convention is the opposite of the standard one.
#     Track3D system dynamics: dz/ds = -sin(mu)
#     → mu > 0 = "downhill" (z decreasing),  mu < 0 = "uphill"
#     → over a crest: mu_neg → mu_pos → dmu/ds > 0
#     → through a dip: mu_pos → mu_neg → dmu/ds < 0
#
#   Derivation (Track3D convention, bank phi=0 assumed):
#     - standard vertical curvature κ_v = -dmu/ds_track3D
#     - Newton, normal direction: g_tilde = g·cos(mu) + V²·κ_v
#                                = g·cos(mu) - V²·dmu/ds_track3D
#     - crest (dmu/ds > 0): g_tilde decreases → tires unload
#     - dip   (dmu/ds < 0): g_tilde increases → tires load up
#
#   This matches the output of Track3D.calc_apparent_accelerations
#   (V_omega term: g_tilde -= Omega_y · V², Omega_y ≈ dmu/ds_track3D).
#
#   V and g_tilde depend on each other → solved by fixed-point iteration in main().
#   the bank (phi) term is left out since |phi| < 1° on these tracks.
# ----------------------------------------------------------------------
def compute_g_tilde(mu: np.ndarray, v: np.ndarray, dmu_ds: np.ndarray,
                    mode: str = '3d') -> np.ndarray:
    if mode == 'flat':
        # pitch only (no V² term)
        return 9.81 * np.cos(mu) * np.ones_like(v)
    elif mode == '3d':
        # minus sign to match the Track3D mu convention
        return 9.81 * np.cos(mu) - v**2 * dmu_ds
    else:
        raise ValueError(f'unknown g_tilde mode: {mode}')


# ----------------------------------------------------------------------
# Step 5. write fwbw_input.csv
#   - input CSV read by the C++ runner
#   - columns: s, kappa, g_tilde
# ----------------------------------------------------------------------
def save_input_csv(s: np.ndarray, kappa: np.ndarray, g_tilde: np.ndarray,
                   path: str) -> None:
    df = pd.DataFrame({'s': s, 'kappa': kappa, 'g_tilde': g_tilde})
    df.to_csv(path, index=False)
    print(f'  saved {path}  ({len(df)} rows)')


# ----------------------------------------------------------------------
# Step 6. params YAML → params.txt
#   - plain key=value text read by the C++ runner
#   - returns the cfg dict (h is needed for g_tilde)
# ----------------------------------------------------------------------
def convert_params_yaml_to_txt(yaml_path: str, txt_path: str) -> dict:
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    vp = cfg['vehicle_params']
    tp = cfg['tire_params']

    # the lookup model does not use P_brake (ax_min comes from the GG table directly).
    # add it back when friction_circle/aero models are added.
    with open(txt_path, 'w') as f:
        f.write(f"m={vp['m']}\n")
        f.write(f"P_max={vp['P_max']}\n")
        f.write(f"mu_x={tp['p_Dx_1']}\n")
        f.write(f"mu_y={tp['p_Dy_1']}\n")
        f.write(f"v_max={vp['v_max']}\n")

    print(f'  saved {txt_path}')
    return cfg


# ----------------------------------------------------------------------
# Step 7. GG diagram .npy → gg.bin
#   - the C++ runner reads binary (bit-exact double precision)
#   - format:
#       [uint32 n_v][uint32 n_g]
#       [double × n_v]              v_list
#       [double × n_g]              g_list
#       [double × n_v × n_g]        ax_max  (row-major: idx = iv*n_g + ig)
#       [double × n_v × n_g]        ax_min
#       [double × n_v × n_g]        ay_max
#       [double × n_v × n_g]        gg_exp
# ----------------------------------------------------------------------
def convert_gg_npy_to_bin(npy_dir: str, bin_path: str) -> None:
    v_list = np.load(os.path.join(npy_dir, 'v_list.npy')).astype(np.float64)
    g_list = np.load(os.path.join(npy_dir, 'g_list.npy')).astype(np.float64)
    ax_max = np.load(os.path.join(npy_dir, 'ax_max.npy')).astype(np.float64)
    ax_min = np.load(os.path.join(npy_dir, 'ax_min.npy')).astype(np.float64)
    ay_max = np.load(os.path.join(npy_dir, 'ay_max.npy')).astype(np.float64)
    gg_exp = np.load(os.path.join(npy_dir, 'gg_exponent.npy')).astype(np.float64)

    with open(bin_path, 'wb') as f:
        f.write(struct.pack('II', len(v_list), len(g_list)))
        v_list.tofile(f)
        g_list.tofile(f)
        ax_max.tofile(f)
        ax_min.tofile(f)
        ay_max.tofile(f)
        gg_exp.tofile(f)

    print(f'  saved {bin_path}  '
          f'(v_list:{len(v_list)}, g_list:{len(g_list)})')


# ----------------------------------------------------------------------
# Step 8. run the C++ runner as a subprocess
# ----------------------------------------------------------------------
def run_unicorn(model: str, input_csv: str, params_txt: str,
                gg_bin: str, output_csv: str, v0: float) -> None:
    cmd = [
        RUNNER_BIN,
        '--model',  model,
        '--input',  input_csv,
        '--params', params_txt,
        '--gg',     gg_bin,
        '--output', output_csv,
        '--v0',     f'{v0}',
    ]
    print(f'  $ {" ".join(cmd)}')
    subprocess.run(cmd, check=True)


# ----------------------------------------------------------------------
# Step 9. load fwbw_output_<model>.csv
#   - the first lines are # metadata (laptime, model, v0, ...)
# ----------------------------------------------------------------------
def load_fwbw_output(path: str):
    meta = {}
    with open(path, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                break
            key_val = line.lstrip('#').strip()
            if '=' in key_val:
                k, v = key_val.split('=', 1)
                meta[k.strip()] = v.strip()

    df = pd.read_csv(path, comment='#')
    return df, meta


# ----------------------------------------------------------------------
# Step E2. write the FWBW result in the same JSON structure as export_global_waypoints
#   - the resampled line (rl) and the FWBW result (fwbw_last) share the same grid,
#     so vx_mps/ax_mps2 are filled 1:1 without interpolation.
#   - the marker/centerline builders are imported from export_global_waypoints.
# ----------------------------------------------------------------------
def export_fwbw_to_json(track: Track3D, rl: dict, fwbw_last: pd.DataFrame,
                        laptime_fwbw: float, model_name: str,
                        output_path: str) -> None:
    n_new = rl['n_pts']
    s_new = rl['s']
    x_new = rl['x']
    y_new = rl['y']
    z_new = rl['z']
    psi   = rl['psi']
    kappa = rl['kappa']
    mu    = rl['mu']
    s_opt_new = rl['s_opt_new']
    n_opt_new = rl['n_opt_new']

    # FWBW result (same grid)
    v_new  = fwbw_last['v'].to_numpy()
    ax_new = fwbw_last['ax'].to_numpy()

    # clean up NaN (raise if more than 5%)
    v_new  = sanitize_nan(v_new,  'fwbw v')
    ax_new = sanitize_nan(ax_new, 'fwbw ax')

    # check the lengths match
    assert len(v_new) == n_new, \
        f'fwbw_last length {len(v_new)} != rl n_pts {n_new}'

    # -- waypoints (same structure as export_global_waypoints) --
    waypoints = []
    for k in range(n_new):
        s = s_opt_new[k]
        n = n_opt_new[k]
        w_tr_left  = float(track.w_tr_left_interpolator(s))
        w_tr_right = float(track.w_tr_right_interpolator(s))
        d_left  = w_tr_left - n
        d_right = -w_tr_right + n

        waypoints.append({
            'id': k,
            's_m':         float(s_new[k]),
            'd_m':         0.0,
            'x_m':         float(x_new[k]),
            'y_m':         float(y_new[k]),
            'z_m':         float(z_new[k]),
            'd_right':     float(abs(d_right)),
            'd_left':      float(abs(d_left)),
            'psi_rad':     float(psi[k]),
            'kappa_radpm': float(kappa[k]),
            'vx_mps':      float(v_new[k]),
            'ax_mps2':     float(ax_new[k]),
            'mu_rad':      float(mu[k]),
        })

    # -- markers (built with the imported helpers) --
    centerline_markers   = _build_sphere_markers(
        track.x, track.y, track.z, r=0.0, g=0.0, b=1.0, scale=0.05)
    raceline_markers     = _build_speed_sphere_markers(
        x_new, y_new, z_new, v_new, scale=0.05)
    raceline_vel_markers = _build_cylinder_markers(
        x_new, y_new, z_new, v_new, r=1.0, g=0.0, b=0.0)
    trackbounds_markers  = _build_trackbounds_markers(track)

    # -- JSON output (same structure as export_global_waypoints) --
    output = {
        'map_info_str': {
            'data': f'estimated lap time: {laptime_fwbw:.4f}s; '
                    f'maximum speed: {v_new.max():.4f}m/s; '
                    f'(FWBW {model_name} post-processed)'
        },
        'est_lap_time': {'data': float(laptime_fwbw)},
        'centerline_markers': {'markers': centerline_markers},
        'centerline_waypoints': _build_centerline_waypoints(track),
        'global_traj_markers_iqp':  {'markers': raceline_markers},
        'global_traj_wpnts_iqp': {
            'header': {'seq': 0, 'stamp': {'secs': 0, 'nsecs': 0}, 'frame_id': ''},
            'wpnts':  waypoints,
        },
        'global_traj_markers_sp':     {'markers': raceline_markers},
        'global_traj_vel_markers_sp': {'markers': raceline_vel_markers},
        'global_traj_wpnts_sp': {
            'header': {'seq': 1, 'stamp': {'secs': 0, 'nsecs': 0}, 'frame_id': ''},
            'wpnts':  waypoints,
        },
        'trackbounds_markers': {'markers': trackbounds_markers},
        'centerline_ref': {
            's_center_m': [float(s_opt_new[k]) for k in range(n_new)],
            'n_center_m': [float(n_opt_new[k]) for k in range(n_new)],
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'  saved {output_path}')


# ----------------------------------------------------------------------
# Step D. plot NLP (resampled) vs FWBW
#   - 2 subplots: speed profile / ΔV
#   - NLP is resampled the same way as export_global_waypoints (as used on the car)
#   - FWBW is the extracted lap
# ----------------------------------------------------------------------
def visualize(s_nlp: np.ndarray, v_nlp: np.ndarray,
              fwbw_df: pd.DataFrame,
              meta: dict, laptime_nlp: float, laptime_fwbw: float,
              track_name: str, model_name: str,
              save_path: str) -> None:

    s_fwbw = fwbw_df['s'].to_numpy()
    v_fwbw = fwbw_df['v'].to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(
        f'{track_name} — NLP(resampled) vs FWBW({model_name})\n'
        f'NLP {laptime_nlp:.3f}s    '
        f'FWBW {laptime_fwbw:.3f}s    '
        f'Δ {laptime_fwbw - laptime_nlp:+.3f}s',
        fontsize=13
    )

    # (1) Velocity profile
    ax1 = axes[0]
    ax1.plot(s_nlp,  v_nlp,  color='blue', linewidth=1.5, label='NLP (resampled)')
    ax1.plot(s_fwbw, v_fwbw, color='red',  linewidth=1.5,
             linestyle='--', label=f'FWBW ({model_name})')
    ax1.fill_between(s_nlp,  v_nlp,  alpha=0.15, color='blue')
    ax1.fill_between(s_fwbw, v_fwbw, alpha=0.15, color='red')
    ax1.set_ylabel('V [m/s]')
    ax1.set_ylim(bottom=0)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9)
    ax1.set_title('Velocity Profile')

    # (2) ΔV = FWBW − NLP (interpolated onto the NLP grid)
    ax2 = axes[1]
    v_fwbw_on_nlp = np.interp(s_nlp, s_fwbw, v_fwbw)
    dv = v_fwbw_on_nlp - v_nlp
    ax2.fill_between(s_nlp, dv, where=dv >= 0, alpha=0.4,
                     color='green', label='FWBW > NLP')
    ax2.fill_between(s_nlp, dv, where=dv < 0,  alpha=0.4,
                     color='red',   label='FWBW < NLP')
    ax2.plot(s_nlp, dv, color='black', linewidth=0.8)
    ax2.axhline(0, color='k', linewidth=0.5)
    ax2.set_ylabel(r'$\Delta V$ [m/s]')
    ax2.set_xlabel('s [m]')
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9)
    ax2.set_title('Velocity Difference (FWBW − NLP)')

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'  saved {save_path}')

    plt.show()


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='FWBW runner (Phase 3, JSON-equivalent)')
    parser.add_argument('--model',    default=DEFAULT_MODEL,
                        choices=['lookup'])
    parser.add_argument('--track',    default=DEFAULT_TRACK,
                        help='track CSV name in data/smoothed_track_data/')
    parser.add_argument('--raceline', default=DEFAULT_RACELINE,
                        help='racing line CSV name in data/global_racing_lines/')
    parser.add_argument('--vehicle',  default=DEFAULT_VEHICLE,
                        help='vehicle name (gg_diagrams subfolder)')
    parser.add_argument('--params',   default=DEFAULT_PARAMS,
                        help='vehicle params YAML in data/vehicle_params/')
    parser.add_argument('--n-laps',   type=int, default=3,
                        help='number of laps to stack for closed-loop trick')
    parser.add_argument('--spacing',  type=float, default=0.1,
                        help='resample spacing [m] (matches export_global_waypoints)')
    parser.add_argument('--mode',     default='3d',
                        choices=['flat', '3d'],
                        help='g_tilde mode: flat=g*cos(mu), 3d=g*cos(mu)-V^2*dmu/ds')
    # --output-json: write the result JSON to this path
    parser.add_argument('--output-json', default=None,
                        help='output JSON path (overrides the default path)')
    args = parser.parse_args()

    # resolve paths: absolute paths are used as is, file names are looked up under data/
    track_csv    = args.track    if os.path.isabs(args.track)    else os.path.join(data_path, 'smoothed_track_data', args.track)
    raceline_csv = args.raceline if os.path.isabs(args.raceline) else os.path.join(data_path, 'global_racing_lines', args.raceline)
    params_yaml  = args.params   if os.path.isabs(args.params)   else os.path.join(data_path, 'vehicle_params',      args.params)
    gg_npy_dir   = os.path.join(data_path, 'gg_diagrams', args.vehicle, 'velocity_frame')

    # prepare the work folder
    os.makedirs(WORK_DIR, exist_ok=True)
    fwbw_input_csv  = os.path.join(WORK_DIR, 'fwbw_input.csv')
    params_txt      = os.path.join(WORK_DIR, 'params.txt')
    gg_bin          = os.path.join(WORK_DIR, 'gg.bin')
    fwbw_output_csv = os.path.join(WORK_DIR, f'fwbw_output_{args.model}.csv')

    # figure output path
    track_name = os.path.splitext(os.path.basename(args.track))[0].replace('_3d_smoothed', '')
    figure_path = os.path.join(FIGURE_DIR,
                               f'{track_name}_fwbw_{args.model}_{args.mode}_vs_nlp.png')

    print('==== run_fwbw ====')
    print(f'  model    = {args.model}')
    print(f'  track    = {track_csv}')
    print(f'  raceline = {raceline_csv}')
    print(f'  params   = {params_yaml}')
    print(f'  gg_dir   = {gg_npy_dir}')
    print(f'  n_laps   = {args.n_laps}')
    print(f'  spacing  = {args.spacing} m')

    # --- Step 1. load Track3D + NLP racing line ---------------------
    print('[1/8] Loading Track3D + NLP racing line...')
    track  = Track3D(path=track_csv)
    nlp_df = pd.read_csv(raceline_csv)

    # --- Step 2. resample (Cartesian) → (s, kappa, mu, dmu_ds, v) --
    print('[2/8] Resampling racing line (Cartesian, periodic CubicSpline)...')
    rl = resample_raceline_cartesian(track, nlp_df, spacing=args.spacing)
    s       = rl['s']
    kappa   = rl['kappa']
    mu      = rl['mu']        # raw centerline mu (Track3D linear interp)
    dmu_ds  = rl['dmu_ds']    # raw mu central diff
    v_nlp_resampled = rl['v']
    print(f'  resampled N  = {rl["n_pts"]} (total length {rl["total_length"]:.2f} m)')
    print(f'  kappa  range : [{kappa.min():.4f}, {kappa.max():.4f}]')
    print(f'  mu     range : [{np.degrees(mu).min():+.2f}°, {np.degrees(mu).max():+.2f}°]')
    print(f'  dmu/ds range : [{dmu_ds.min():+.4f}, {dmu_ds.max():+.4f}] rad/m')
    print(f'  v_nlp  range : [{v_nlp_resampled.min():.2f}, {v_nlp_resampled.max():.2f}] m/s')

    # --- Step 3. params YAML → params.txt --------------------------
    print('[3/8] Converting params YAML → params.txt...')
    convert_params_yaml_to_txt(params_yaml, params_txt)

    # --- Step 4. .npy → gg.bin -------------------------------------
    print('[4/8] Converting GG diagram .npy → binary...')
    convert_gg_npy_to_bin(gg_npy_dir, gg_bin)

    # GG g_list bounds (for clamping)
    g_list = np.load(os.path.join(gg_npy_dir, 'g_list.npy'))
    g_min, g_max = float(g_list.min()), float(g_list.max())
    print(f'  GG g_list range: [{g_min:.4f}, {g_max:.4f}] m/s^2  (n={len(g_list)})')

    # --- Step 5–7. fixed-point iteration over (g_tilde, FWBW v) ----
    # initial v = NLP v.   every iteration: recompute g_tilde → FWBW → update v.
    max_iter = 10
    tol      = 0.05         # m/s
    alpha    = 0.7          # under-relaxation (0.7 = a bit more aggressive)
    print(f'[5-7] Fixed-point iteration (mode={args.mode}, '
          f'max_iter={max_iter}, tol={tol} m/s, alpha={alpha})')

    v_prev = v_nlp_resampled.copy()
    fwbw_full = None
    meta = {}
    fwbw_last = None
    lap_length = None
    n_pts_per_lap = None

    for it in range(max_iter):
        # (a) g_tilde from previous v
        g_tilde_raw = compute_g_tilde(mu, v_prev, dmu_ds, mode=args.mode)
        n_low  = int((g_tilde_raw < g_min).sum())
        n_high = int((g_tilde_raw > g_max).sum())
        g_tilde = np.clip(g_tilde_raw, g_min, g_max)

        # (b) stack N laps
        s_stack, k_stack, g_stack, lap_length, n_pts_per_lap = stack_laps(
            s, kappa, g_tilde, n_laps=args.n_laps)
        save_input_csv(s_stack, k_stack, g_stack, fwbw_input_csv)

        # (c) run C++ runner
        v0 = float(v_prev[0])
        run_unicorn(args.model, fwbw_input_csv, params_txt, gg_bin,
                    fwbw_output_csv, v0)

        # (d) load + middle lap
        fwbw_full, meta = load_fwbw_output(fwbw_output_csv)
        fwbw_last = extract_middle_lap(fwbw_full, args.n_laps, lap_length, n_pts_per_lap)
        v_new_raw = sanitize_nan(fwbw_last['v'].to_numpy(), 'fwbw v (iter)')

        # (e) convergence + under-relaxation
        delta = float(np.max(np.abs(v_new_raw - v_prev)))
        print(f'  iter {it}:'
              f'  g_tilde=[{g_tilde_raw.min():+.3f}, {g_tilde_raw.max():+.3f}]'
              f'  clamp(low/high)={n_low}/{n_high}'
              f'  max|Δv|={delta:.4f} m/s')

        if delta < tol:
            print(f'  → converged at iter {it}')
            break
        v_prev = alpha * v_new_raw + (1.0 - alpha) * v_prev
    else:
        print(f'  → max_iter reached (max|Δv|={delta:.4f}); using last result')

    # --- Step 8. final result ------------------------------------
    print('[8/8] Finalizing FWBW middle lap, visualizing...')

    # middle-lap lap time = sum( ds / v_avg )
    # fwbw_last['v'] may still contain NaN from C++, so sanitize it
    fwbw_last['v'] = sanitize_nan(fwbw_last['v'].to_numpy(), 'fwbw v (final)')
    s_last = fwbw_last['s'].to_numpy()
    v_last = fwbw_last['v'].to_numpy()
    ds_last = np.diff(s_last)
    v_avg   = 0.5 * (v_last[:-1] + v_last[1:])
    laptime_fwbw = float(np.sum(ds_last / np.maximum(v_avg, 1e-3)))
    laptime_nlp  = float(nlp_df['laptime'].iloc[0])

    print(f'  total runner laptime = {meta.get("laptime", "nan")} s '
          f'({args.n_laps} laps stacked)')
    print(f'  FWBW middle-lap      = {laptime_fwbw:.3f} s')
    print(f'  NLP  laptime         = {laptime_nlp:.3f} s')
    print(f'  Δ                    = {laptime_fwbw - laptime_nlp:+.3f} s')

    # --- FWBW result → JSON (same structure as export_global_waypoints) ----
    # --output-json takes precedence, otherwise the default path below
    if args.output_json:
        json_path = args.output_json
    else:
        json_path = os.path.join(
            data_path, 'global_racing_lines',
            f'global_waypoints_fwbw_{args.model}.json'
        )
    print('  exporting FWBW waypoints JSON...')
    export_fwbw_to_json(track, rl, fwbw_last, laptime_fwbw,
                        args.model, json_path)

    visualize(s, v_nlp_resampled, fwbw_last, meta,
              laptime_nlp, laptime_fwbw,
              track_name, args.model, figure_path)

    print('==== done ====')


if __name__ == '__main__':
    main()
