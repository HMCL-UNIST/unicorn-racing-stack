#!/usr/bin/env python3
# 3D speed-only optimizer — path fixed via REDUCED state NLP
#
# Mathematically equivalent to gen_global_racing_line.py, but with (n, chi)
# treated as INPUT FUNCTIONS of s (not decision variables).
#
# State vector    : [V, ax]            (was [V, n, chi, ax, ay])
# Control         : [jx]               (was [jx, jy])
# Parameters      : n_fixed(s), chi_fixed(s), dchi_ds_fixed(s)
# Dynamics        : dV/ds = ax/s_dot,  dax/ds = jx/s_dot
# Algebraic ay    : ay = V*s_dot*(dchi_ds + Omega_z)
# Constraints     : GGV diamond on (ax_tilde, ay_tilde)  — same as original
# Cost            : min lap time + jerk regularization   — same as original
#
# Workflow:
#   1. Subscribe to /global_waypoints (latched WpntArray)
#   2. Extract fixed path from message (s_m, d_m, psi_rad, vx_mps)
#   3. Compute chi(s) = psi_rad(s) - theta_centerline(s)  (from Track3D csv)
#   4. Solve reduced NLP
#   5. RE-PUBLISH to same topic /global_waypoints with new vx_mps, ax_mps2
#   6. Spin forever (latched) but only solve ONCE
#
# Usage:
#   rosrun stack_master 3d_optimized_vel_planner.py --map gazebo_wall_2 \
#       --vehicle_name rc_car_10th_fast1 --gg_vehicle_name rc_car_10th

# --- pandas/numpy version bypass ---
import sys as _sys
_sys.path[:] = [p for p in _sys.path if '/.local/' not in p]
import numpy as _np
_np.__version__ = '1.22.4'
if not hasattr(_np.random, 'BitGenerator'):
    from numpy.random.bit_generator import BitGenerator as _BG
    _np.random.BitGenerator = _BG
# --- end bypass ---

import os
import sys
import argparse
import threading
import time
import numpy as np
import pandas as pd
import yaml
import casadi as ca

import rospy
from f110_msgs.msg import WpntArray, Wpnt
# Trigger service for /velopt/reload (hot-reload from gg_tuner_3d)
from std_srvs.srv import Trigger, TriggerResponse

# --- locate 3d_gb_optimizer modules ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_GB_OPT_DIR = os.path.abspath(os.path.join(_THIS_DIR, '..', '..', 'planner', '3d_gb_optimizer'))
_SRC_DIR = os.path.join(_GB_OPT_DIR, 'global_line', 'src')
_DATA_DIR = os.path.join(_GB_OPT_DIR, 'global_line', 'data')

sys.path.insert(0, _SRC_DIR)
from track3D import Track3D
from ggManager import GGManager


# Probe HSL ma27; fall back to MUMPS if libhsl.so is not loadable.
def _select_linear_solver():
    try:
        _x = ca.MX.sym('x')
        _probe = ca.nlpsol('hsl_probe', 'ipopt',
                           {'x': _x, 'f': (_x - 1.0) ** 2},
                           {'ipopt.linear_solver': 'ma27',
                            'ipopt.print_level': 0, 'print_time': 0})
        _probe(x0=0.0)
        if _probe.stats().get('success', False):
            print('[velopt] linear_solver: ma27 (HSL)')
            return 'ma27'
    except Exception:
        pass
    print('[velopt] linear_solver: mumps (HSL not available, fallback)')
    return 'mumps'

_LINEAR_SOLVER = _select_linear_solver()


# per-sector multi-GGV support.
#   Priority: /gg_tuner/sector_ggv_map/sector<i> (snapshot name) →
#   fallback to friction-value lookup <base>_f{NNN}/velocity_frame/.
def _read_friction_sectors():
    """Read friction sectors from rosparam (friction_sector_server).
    Returns [{sector_idx, start, end, friction}, ...] or [].
    """
    try:
        n_sec = rospy.get_param('/friction_map_params/n_sectors', 0)
        if n_sec <= 0:
            return []
        sectors = []
        for i in range(n_sec):
            sectors.append({
                'sector_idx': i,
                'start': int(rospy.get_param(f'/friction_map_params/Sector{i}/start', 0)),
                'end':   int(rospy.get_param(f'/friction_map_params/Sector{i}/end', 0)),
                'friction': float(rospy.get_param(f'/friction_map_params/Sector{i}/friction', -1.0)),
                # raceline-s (m) bounds for real arclen sector->grid mapping
                's_start': float(rospy.get_param(f'/friction_map_params/Sector{i}/s_start', -1.0)),
                's_end':   float(rospy.get_param(f'/friction_map_params/Sector{i}/s_end', -1.0)),
                # per-sector mu_scale_x / mu_scale_y on top of sector mu.
                # Default 1.0 if not set ⇒ legacy isotropic behavior.
                'mu_scale_x': float(rospy.get_param(f'/friction_map_params/Sector{i}/mu_scale_x', 1.0)),
                'mu_scale_y': float(rospy.get_param(f'/friction_map_params/Sector{i}/mu_scale_y', 1.0)),
            })
        return sectors
    except Exception:
        return []


def _build_friction_gg_map(sectors, n_waypoints, gg_base_dir, gg_margin,
                           base_p_Dx_1):
    """Build per-waypoint GGManager mapping.

    Per sector source priority:
      1) /gg_tuner/sector_ggv_map/sector<i> → load gg_diagrams/<snap>/velocity_frame/
      2) friction value → <base>_f{NNN}/velocity_frame/  (or base dir if fric==base)

    Returns (gg_list, gg_idx) or (None, None) if single-GGV mode suffices / fails.
    """
    if not sectors:
        return None, None

    valid = [s for s in sectors if s['friction'] > 0]
    if not valid:
        return None, None

    gg_base_parent = os.path.dirname(os.path.dirname(gg_base_dir))   # gg_diagrams/
    gg_base_name = os.path.basename(os.path.dirname(gg_base_dir))    # e.g. rc_car_10th_latest

    def _resolve(sec_idx, fric):
        """Return (source_key, load_dir) for a given sector index + friction."""
        snap = ''
        try:
            snap = rospy.get_param(
                f'/gg_tuner/sector_ggv_map/sector{sec_idx}', '')
        except Exception:
            snap = ''
        snap = (snap or '').strip()
        if snap:
            snap_dir = os.path.join(gg_base_parent, snap, 'velocity_frame')
            if os.path.exists(snap_dir):
                return (f'snap:{snap}', snap_dir)
            rospy.logwarn(
                f'[velopt] sector{sec_idx} snapshot not found: {snap_dir} '
                f'→ fallback to friction lookup')
        if abs(fric - base_p_Dx_1) < 1e-4:
            return ('base', gg_base_dir)
        fric_int = int(round(fric * 100))
        fric_dir = os.path.join(
            gg_base_parent, f'{gg_base_name}_f{fric_int:03d}', 'velocity_frame')
        return (f'fric:{fric:.3f}', fric_dir)

    # Resolve each valid sector to a source, dedupe by key
    sector_keys = []  # parallel to valid
    key_to_dir = {}
    for sec in valid:
        key, load_dir = _resolve(sec['sector_idx'], sec['friction'])
        sector_keys.append(key)
        key_to_dir.setdefault(key, load_dir)

    unique_keys = list(dict.fromkeys(sector_keys))  # preserve order

    # Single source matching base → no multi-GGV
    if len(unique_keys) == 1 and unique_keys[0] == 'base':
        return None, None

    gg_list = []
    key_to_idx = {}
    for key in unique_keys:
        load_dir = key_to_dir[key]
        if not os.path.exists(load_dir):
            rospy.logwarn(
                f'[velopt] GGV not found for {key}: {load_dir} '
                f'→ skipping multi-GGV')
            return None, None
        key_to_idx[key] = len(gg_list)
        gg_list.append(GGManager(gg_path=load_dir, gg_margin=gg_margin))
        rospy.loginfo(f'[velopt] loaded GGV [{key}]: {load_dir}')

    # Default fallback: base source if present, else first source
    default_idx = key_to_idx.get('base', 0)
    gg_idx = np.full(n_waypoints, default_idx, dtype=int)

    total_original = max(s['end'] for s in valid) + 1
    for sec, key in zip(valid, sector_keys):
        idx = key_to_idx[key]
        grid_start = int(round(sec['start'] / total_original * n_waypoints))
        grid_end = int(round((sec['end'] + 1) / total_original * n_waypoints))
        grid_start = max(0, min(grid_start, n_waypoints - 1))
        grid_end = max(0, min(grid_end, n_waypoints))
        gg_idx[grid_start:grid_end] = idx

    rospy.loginfo(
        f'[velopt] multi-GGV: {len(gg_list)} sources, keys={unique_keys}')
    return gg_list, gg_idx

def build_and_solve(track, gg, vehicle_params,
                    n_fixed, chi_fixed, v_init, ax_init,
                    w_T=1.0, w_jx=1e-2, V_min=0.0, RK4_steps=1, sol_opt=None,
                    gg_list=None, gg_idx=None,
                    # per-point friction scaling applied at NLP layer.
                    # Cheaper than rebuilding per-friction GGV
                    # directories for short patches (e.g. 2 m bridge).
                    # `mu_scale_k` is a length-N float array
                    # multiplying (ax_max, ay_max, |ax_min|) at each
                    # grid point k. None ⇒ no scaling (legacy).
                    mu_scale_k=None,
                    # extra global multipliers split by direction.
                    # Friction is theoretically isotropic but in
                    # practice (e.g. on bridges) ax-grip and ay-grip
                    # can degrade differently. Defaults 1.0 ≡ isotropic
                    # (legacy behavior). Applied on top of mu_scale_k.
                    # per-sector mu_scale_x / mu_scale_y arrays (length N,
                    # one per grid point). Built from
                    # /friction_map_params/Sector<i>/mu_scale_x|y in the
                    # same sector→grid mapping used for mu_scale_k.
                    # None ⇒ no per-direction scaling (legacy isotropic).
                    mu_scale_x_k=None,
                    mu_scale_y_k=None,
                    w_jx_acc=None, w_jx_brk=None,
                    # === New (tanh-based) ===
                    w_jx_curv_alpha_acc=0.0, w_jx_curv_alpha_brk=0.0,
                    w_jx_curv_k_alpha=0.2,
                    w_jx_curv_beta_acc=0.0, w_jx_curv_beta_brk=0.0,
                    w_jx_curv_k_beta=0.5,
                    # direct ax_pos^2 corner penalty (independent k from k_alpha).
                    # Catches steady-state acceleration inside corners that the
                    # jerk-based weights cannot see (jx=0 once ax is constant).
                    w_ax_corner_acc=0.0,
                    w_ax_corner_k=0.2,
                    # === Legacy (deprecated) ===
                    w_jx_curv_alpha=0.0):
    """Reduced-state NLP. All arrays must be on track.s (resampled grid)."""
    # asymmetric jerk defaults — fall back to symmetric w_jx
    if w_jx_acc is None:
        w_jx_acc = w_jx
    if w_jx_brk is None:
        w_jx_brk = w_jx

    if w_jx_curv_alpha > 0.0 and (w_jx_curv_alpha_acc + w_jx_curv_alpha_brk) == 0.0:
        rospy.logwarn(
            f'[velopt] legacy w_jx_curv_alpha={w_jx_curv_alpha} → '
            f'mapped to alpha_acc=alpha_brk (deprecated, use _acc/_brk).')
        w_jx_curv_alpha_acc = w_jx_curv_alpha
        w_jx_curv_alpha_brk = w_jx_curv_alpha
    h = vehicle_params['h']
    N = track.s.size
    ds = track.ds
    V_max_gg = float(gg.V_max)

    # Periodic CasADi interpolants for fixed path
    def concat_arr(a):
        return np.concatenate((a, a[1:], a[1:]))
    s_aug = np.concatenate((track.s, track.s[-1] + track.s[1:], 2 * track.s[-1] + track.s[1:]))

    n_fn = ca.interpolant('n_fix', 'linear', [s_aug], concat_arr(n_fixed))
    chi_fn = ca.interpolant('chi_fix', 'linear', [s_aug], concat_arr(chi_fixed))

    chi_unwrapped = np.unwrap(chi_fixed)
    dchi_ds = (np.roll(chi_unwrapped, -1) - np.roll(chi_unwrapped, 1)) / (2.0 * ds)
    dchi_ds_fn = ca.interpolant('dchi_ds_fix', 'linear', [s_aug], concat_arr(dchi_ds))

    Omega_z_arr = np.array([float(track.Omega_z_interpolator(float(si)))
                            for si in track.s])
    dOmega_z_ds_arr = np.gradient(Omega_z_arr, ds)
    dOmega_z_ds_fn = ca.interpolant('dOmega_z_ds_fix', 'linear',
                                     [s_aug], concat_arr(dOmega_z_ds_arr))

    # Symbolic states: [V, ax]
    V = ca.MX.sym('V')
    ax = ca.MX.sym('ax')
    x = ca.vertcat(V, ax)
    nx_ = x.shape[0]

    jx = ca.MX.sym('jx')
    u = jx
    nu_ = 1

    s_sym = ca.MX.sym('s')

    n_s = n_fn(s_sym)
    chi_s = chi_fn(s_sym)
    dchi_ds_s = dchi_ds_fn(s_sym)
    Omega_z_s = track.Omega_z_interpolator(s_sym)

    s_dot = (V * ca.cos(chi_s)) / (1.0 - n_s * Omega_z_s)
    ay_alg = V * s_dot * (dchi_ds_s + Omega_z_s)

    # Apparent accelerations via Track3D (symbolic MX input OK)
    ax_tilde, ay_tilde, g_tilde = track.calc_apparent_accelerations(
        V=V, n=n_s, chi=chi_s, ax=ax, ay=ay_alg, s=s_sym, h=h,
        neglect_w_omega_y=True, neglect_w_omega_x=True,
        neglect_euler=True, neglect_centrifugal=True,
        neglect_w_dot=False, neglect_V_omega=False,
    )

    dV = ax / s_dot
    dax = jx / s_dot
    dx = ca.vertcat(dV, dax)

    L_t = w_T * 1.0 / s_dot

    dOmega_z_ds_s = dOmega_z_ds_fn(s_sym)
    omega_term  = ca.tanh(ca.fabs(Omega_z_s)     / w_jx_curv_k_alpha)
    domega_term = ca.tanh(ca.fabs(dOmega_z_ds_s) / w_jx_curv_k_beta)

    # jerk-side α/β reverted (w_T must not affect jerk smoothness).
    weight_acc = (1.0
                  + w_jx_curv_alpha_acc * omega_term
                  + w_jx_curv_beta_acc  * domega_term)
    weight_brk = (1.0
                  + w_jx_curv_alpha_brk * omega_term
                  + w_jx_curv_beta_brk  * domega_term)

    jx_pos = ca.fmax(jx, 0.0)
    jx_neg = ca.fmax(-jx, 0.0)
    L_reg = (weight_acc * w_jx_acc * jx_pos ** 2
           + weight_brk * w_jx_brk * jx_neg ** 2) / (s_dot ** 2)

    ax_corner_term = ca.tanh(ca.fabs(Omega_z_s) / w_ax_corner_k)
    ax_pos = ca.fmax(ax, 0.0)
    # only ax² corner cap scales with w_T (not a jerk term).
    # L_reg = L_reg + w_ax_corner_acc * ax_corner_term * (ax_pos ** 2) / (s_dot ** 2)
    w_T_corner = max(float(w_T), 1e-6)
    L_reg = L_reg + w_T_corner * w_ax_corner_acc * ax_corner_term * (ax_pos ** 2) / (s_dot ** 2)

    # RK4 (same as original)
    M = RK4_steps
    ds_rk = ds / M
    f = ca.Function('f', [x, u, s_sym], [dx, L_t, L_reg])
    X0 = ca.MX.sym('X0', nx_)
    U = ca.MX.sym('U', nu_)
    S0 = ca.MX.sym('S0')
    X = X0
    S = S0
    Q_t = 0
    Q_reg = 0
    for j in range(M):
        k1, k1_qt, k1_qr = f(X, U, S)
        k2, k2_qt, k2_qr = f(X + ds_rk/2 * k1, U, S + ds_rk/2)
        k3, k3_qt, k3_qr = f(X + ds_rk/2 * k2, U, S + ds_rk/2)
        k4, k4_qt, k4_qr = f(X + ds_rk * k3, U, S + ds_rk)
        X = X + ds_rk/6 * (k1 + 2*k2 + 2*k3 + k4)
        Q_t = Q_t + ds_rk/6 * (k1_qt + 2*k2_qt + 2*k3_qt + k4_qt)
        Q_reg = Q_reg + ds_rk/6 * (k1_qr + 2*k2_qr + 2*k3_qr + k4_qr)
        S = S + ds_rk
    F = ca.Function('F', [X0, U, S0], [X, Q_t, Q_reg], ['x0','u','s0'], ['xf','q_t','q_reg'])

    # Build NLP
    w = []
    w0 = []
    lbw = []
    ubw = []
    J_t = 0.0
    J_reg = 0.0
    g = []
    lbg = []
    ubg = []

    Xk = ca.MX.sym('X0', nx_)
    w += [Xk]
    lbw += [V_min, -np.inf]
    ubw += [V_max_gg, np.inf]
    w0 += [max(float(v_init[0]), max(V_min, 0.5)), float(ax_init[0])]

    for k in range(N):
        s_k = k * ds
        n_k = float(n_fixed[k])
        chi_k = float(chi_fixed[k])
        dchi_k = float(dchi_ds[k])
        Om_z_k = float(track.Omega_z_interpolator(s_k))

        s_dot_k = (Xk[0] * ca.cos(chi_k)) / (1.0 - n_k * Om_z_k)
        ay_k = Xk[0] * s_dot_k * (dchi_k + Om_z_k)

        axt_k, ayt_k, gt_k = track.calc_apparent_accelerations(
            V=Xk[0], n=n_k, chi=chi_k, ax=Xk[1], ay=ay_k, s=s_k, h=h,
            neglect_w_omega_y=True, neglect_w_omega_x=True,
            neglect_euler=True, neglect_centrifugal=True,
            neglect_w_dot=False, neglect_V_omega=False,
        )

        if gg_list is not None and gg_idx is not None:
            gg_k = gg_list[int(gg_idx[k])]
        else:
            gg_k = gg
        gg_exp, ax_min, ax_max, ay_max = ca.vertsplit(
            gg_k.acc_interpolator(ca.vertcat(Xk[0], gt_k))
        )
        # per-point friction scaling (cheap alternative to per-friction
        # GGV directories). Applies to *all three* envelope outputs so
        # the diamond constraint stays self-consistent: scaling only
        # ax_max / ay_max while leaving ax_min unscaled would shrink
        # accel side while leaving brake side at full grip, which is
        # physically wrong for a mu drop. ax_min is signed-negative
        # from the LUT, so we scale by mu directly (|ax_min|·mu is the
        # reduced brake force at that point). gg_exp untouched —
        # friction changes envelope size, not envelope shape.
        # sx_k / sy_k = final long/lat envelope scale (mu_k drop)
        if mu_scale_x_k is not None or mu_scale_y_k is not None:
            sx_k = float(mu_scale_x_k[k]) if mu_scale_x_k is not None else 1.0
            sy_k = float(mu_scale_y_k[k]) if mu_scale_y_k is not None else 1.0
            ax_max = ax_max * sx_k
            ay_max = ay_max * sy_k
            ax_min = ax_min * sx_k
        g += [ay_max - ca.fabs(ayt_k)]
        lbg += [0.0]; ubg += [np.inf]

        g += [ca.fabs(ax_min) * ca.power(
            ca.fmax(1.0 - ca.power(ca.fmin(ca.fabs(ayt_k) / ay_max, 1.0), gg_exp), 1e-3),
            1.0 / gg_exp) - ca.fabs(axt_k)]
        lbg += [0.0]; ubg += [np.inf]

        g += [ax_max - axt_k]
        lbg += [0.0]; ubg += [np.inf]

        if k == N - 1:
            break

        Uk = ca.MX.sym('U_' + str(k), nu_)
        w += [Uk]
        lbw += [-np.inf] * nu_
        ubw += [np.inf] * nu_
        w0 += [0.0] * nu_

        Fk = F(x0=Xk, u=Uk, s0=s_k)
        Xk_end = Fk['xf']
        J_t = J_t + Fk['q_t']
        J_reg = J_reg + Fk['q_reg']

        Xk = ca.MX.sym('X_' + str(k+1), nx_)
        w += [Xk]
        lbw += [V_min, -np.inf]
        ubw += [V_max_gg, np.inf]
        w0 += [max(float(v_init[k+1]), max(V_min, 0.5)), float(ax_init[k+1])]

        g += [Xk_end - Xk]
        lbg += [0.0] * nx_
        ubg += [0.0] * nx_

    g += [w[0] - Xk]
    lbg += [0.0] * nx_
    ubg += [0.0] * nx_

    w_vec = ca.vertcat(*w)
    g_vec = ca.vertcat(*g)
    w0_vec = ca.vertcat(*w0)
    lbw_vec = ca.vertcat(*lbw)
    ubw_vec = ca.vertcat(*ubw)
    lbg_vec = ca.vertcat(*lbg)
    ubg_vec = ca.vertcat(*ubg)

    if sol_opt is None:

        sol_opt = {
            'ipopt.max_iter': 100,
            'ipopt.hessian_approximation': 'limited-memory',
            'ipopt.line_search_method': 'cg-penalty',
            'ipopt.tol':                        1e-4,
            'ipopt.dual_inf_tol':               1e-1,
            'ipopt.constr_viol_tol':            1e-4,
            'ipopt.compl_inf_tol':              1e-4,
            'ipopt.acceptable_tol':             1e-3,
            'ipopt.acceptable_dual_inf_tol':    5.0,
            'ipopt.acceptable_constr_viol_tol': 1e-3,
            'ipopt.acceptable_iter':            10,
            'ipopt.linear_solver': _LINEAR_SOLVER,  # ma27 if HSL available, else mumps
        }
    sol_opt = dict(sol_opt)
    sol_opt.setdefault('print_time', 0)
    sol_opt.setdefault('ipopt.print_level', 5)  # verbose — shows per-iteration log

    nlp = {'f': J_t + J_reg, 'x': w_vec, 'g': g_vec}
    rospy.loginfo(f'[velopt] NLP: {int(w_vec.shape[0])} vars, {int(g_vec.shape[0])} constraints, {N} points')

    t_build = time.time()
    solver = ca.nlpsol('solver', 'ipopt', nlp, sol_opt)
    t_build = time.time() - t_build
    rospy.loginfo(f'[velopt] solver built in {t_build:.2f}s, solving...')

    t_solve = time.time()
    sol = solver(x0=w0_vec, lbx=lbw_vec, ubx=ubw_vec, lbg=lbg_vec, ubg=ubg_vec)
    t_solve = time.time() - t_solve

    # J_t = w_T * laptime_real; divide out w_T for real seconds.
    # laptime = float(ca.Function('f_laptime', [w_vec], [J_t])(sol['x']))
    # rospy.loginfo(f'[velopt] IPOPT: {t_solve:.2f}s, success={success}, laptime={laptime:.4f}s')
    J_t_val = float(ca.Function('f_Jt', [w_vec], [J_t])(sol['x']))
    laptime = J_t_val / max(float(w_T), 1e-6)
    success = solver.stats()['success']
    rospy.loginfo(
        f'[velopt] IPOPT: {t_solve:.2f}s, success={success}, '
        f'laptime={laptime:.4f}s, J_t={J_t_val:.4f} (w_T={float(w_T):.3f})')

    sol_x = np.array(sol['x']).flatten()
    stride = nx_ + nu_
    V_opt = np.zeros(N)
    ax_opt = np.zeros(N)
    for k in range(N):
        off = k * stride
        V_opt[k] = sol_x[off]
        ax_opt[k] = sol_x[off + 1]

    return V_opt, ax_opt, laptime, success


# same-ruler NLP on the raceline grid (kappa from wpnts, 1:1 output).
# no banking: ay=V^2*kappa, ax_tilde=ax-g*sin(mu), g_tilde=g*cos(mu)-V^2*dmu_ds.
def build_and_solve_rl(s_grid, kappa_arr, mu_arr, gg, vehicle_params,
                       v_init, ax_init,
                       w_T=1.0, w_jx=1e-2, V_min=0.0, RK4_steps=1, sol_opt=None,
                       mu_scale_x_k=None, mu_scale_y_k=None,
                       w_jx_acc=None, w_jx_brk=None,
                       w_jx_curv_alpha_acc=0.0, w_jx_curv_alpha_brk=0.0,
                       w_jx_curv_k_alpha=0.2,
                       w_jx_curv_beta_acc=0.0, w_jx_curv_beta_brk=0.0,
                       w_jx_curv_k_beta=0.5,
                       w_ax_corner_acc=0.0, w_ax_corner_k=0.2):
    G = 9.81
    if w_jx_acc is None:
        w_jx_acc = w_jx
    if w_jx_brk is None:
        w_jx_brk = w_jx
    N = s_grid.size
    ds = float(np.mean(np.diff(s_grid)))
    V_max_gg = float(gg.V_max)

    def concat_arr(a):
        return np.concatenate((a, a[1:], a[1:]))
    s_aug = np.concatenate((s_grid, s_grid[-1] + s_grid[1:], 2 * s_grid[-1] + s_grid[1:]))
    dmu_arr = np.gradient(mu_arr, ds)
    dkappa_arr = np.gradient(kappa_arr, ds)
    kappa_fn = ca.interpolant('kappa', 'linear', [s_aug], concat_arr(kappa_arr))
    dkappa_fn = ca.interpolant('dkappa', 'linear', [s_aug], concat_arr(dkappa_arr))

    # dynamics: s_dot = V (integrate along raceline arc length)
    V = ca.MX.sym('V'); ax = ca.MX.sym('ax')
    x = ca.vertcat(V, ax); nx_ = 2
    jx = ca.MX.sym('jx'); u = jx; nu_ = 1
    s_sym = ca.MX.sym('s')
    s_dot = V
    dV = ax / s_dot
    dax = jx / s_dot
    dx = ca.vertcat(dV, dax)
    L_t = w_T * 1.0 / s_dot
    kap_s = kappa_fn(s_sym)
    dkap_s = dkappa_fn(s_sym)
    omega_term  = ca.tanh(ca.fabs(kap_s)  / w_jx_curv_k_alpha)
    domega_term = ca.tanh(ca.fabs(dkap_s) / w_jx_curv_k_beta)
    weight_acc = 1.0 + w_jx_curv_alpha_acc * omega_term + w_jx_curv_beta_acc * domega_term
    weight_brk = 1.0 + w_jx_curv_alpha_brk * omega_term + w_jx_curv_beta_brk * domega_term
    jx_pos = ca.fmax(jx, 0.0); jx_neg = ca.fmax(-jx, 0.0)
    L_reg = (weight_acc * w_jx_acc * jx_pos ** 2
           + weight_brk * w_jx_brk * jx_neg ** 2) / (s_dot ** 2)
    ax_corner_term = ca.tanh(ca.fabs(kap_s) / w_ax_corner_k)
    ax_pos = ca.fmax(ax, 0.0)
    w_T_corner = max(float(w_T), 1e-6)
    L_reg = L_reg + w_T_corner * w_ax_corner_acc * ax_corner_term * (ax_pos ** 2) / (s_dot ** 2)

    M = RK4_steps; ds_rk = ds / M
    f = ca.Function('f', [x, u, s_sym], [dx, L_t, L_reg])
    X0 = ca.MX.sym('X0', nx_); U = ca.MX.sym('U', nu_); S0 = ca.MX.sym('S0')
    X = X0; S = S0; Q_t = 0; Q_reg = 0
    for j in range(M):
        k1, k1t, k1r = f(X, U, S)
        k2, k2t, k2r = f(X + ds_rk/2 * k1, U, S + ds_rk/2)
        k3, k3t, k3r = f(X + ds_rk/2 * k2, U, S + ds_rk/2)
        k4, k4t, k4r = f(X + ds_rk * k3, U, S + ds_rk)
        X = X + ds_rk/6 * (k1 + 2*k2 + 2*k3 + k4)
        Q_t = Q_t + ds_rk/6 * (k1t + 2*k2t + 2*k3t + k4t)
        Q_reg = Q_reg + ds_rk/6 * (k1r + 2*k2r + 2*k3r + k4r)
        S = S + ds_rk
    F = ca.Function('F', [X0, U, S0], [X, Q_t, Q_reg], ['x0','u','s0'], ['xf','q_t','q_reg'])

    w=[]; w0=[]; lbw=[]; ubw=[]; J_t=0.0; J_reg=0.0; g=[]; lbg=[]; ubg=[]
    Xk = ca.MX.sym('X0', nx_)
    w += [Xk]; lbw += [V_min, -np.inf]; ubw += [V_max_gg, np.inf]
    w0 += [max(float(v_init[0]), max(V_min, 0.5)), float(ax_init[0])]
    for k in range(N):
        s_k = k * ds
        kap_k = float(kappa_arr[k]); mu_k = float(mu_arr[k]); dmu_k = float(dmu_arr[k])
        Vk = Xk[0]
        ay_k = Vk * Vk * kap_k
        axt_k = Xk[1] - G * np.sin(mu_k)
        ayt_k = ay_k
        gt_k = ca.fmax(G * np.cos(mu_k) - Vk * Vk * dmu_k, 0.0)
        gg_exp, ax_min, ax_max, ay_max = ca.vertsplit(
            gg.acc_interpolator(ca.vertcat(Vk, gt_k)))
        if mu_scale_x_k is not None or mu_scale_y_k is not None:
            sx = float(mu_scale_x_k[k]) if mu_scale_x_k is not None else 1.0
            sy = float(mu_scale_y_k[k]) if mu_scale_y_k is not None else 1.0
            ax_max = ax_max * sx; ay_max = ay_max * sy; ax_min = ax_min * sx
        g += [ay_max - ca.fabs(ayt_k)]; lbg += [0.0]; ubg += [np.inf]
        g += [ca.fabs(ax_min) * ca.power(
            ca.fmax(1.0 - ca.power(ca.fmin(ca.fabs(ayt_k) / ay_max, 1.0), gg_exp), 1e-3),
            1.0 / gg_exp) - ca.fabs(axt_k)]
        lbg += [0.0]; ubg += [np.inf]
        g += [ax_max - axt_k]; lbg += [0.0]; ubg += [np.inf]
        if k == N - 1:
            break
        Uk = ca.MX.sym('U_' + str(k), nu_)
        w += [Uk]; lbw += [-np.inf] * nu_; ubw += [np.inf] * nu_; w0 += [0.0] * nu_
        Fk = F(x0=Xk, u=Uk, s0=s_k)
        Xk_end = Fk['xf']; J_t = J_t + Fk['q_t']; J_reg = J_reg + Fk['q_reg']
        Xk = ca.MX.sym('X_' + str(k+1), nx_)
        w += [Xk]; lbw += [V_min, -np.inf]; ubw += [V_max_gg, np.inf]
        w0 += [max(float(v_init[k+1]), max(V_min, 0.5)), float(ax_init[k+1])]
        g += [Xk_end - Xk]; lbg += [0.0] * nx_; ubg += [0.0] * nx_
    g += [w[0] - Xk]; lbg += [0.0] * nx_; ubg += [0.0] * nx_

    w_vec = ca.vertcat(*w); g_vec = ca.vertcat(*g)
    w0_vec = ca.vertcat(*w0); lbw_vec = ca.vertcat(*lbw); ubw_vec = ca.vertcat(*ubw)
    lbg_vec = ca.vertcat(*lbg); ubg_vec = ca.vertcat(*ubg)
    if sol_opt is None:
        sol_opt = {
            'ipopt.max_iter': 100,
            'ipopt.hessian_approximation': 'limited-memory',
            'ipopt.line_search_method': 'cg-penalty',
            'ipopt.tol': 1e-4, 'ipopt.dual_inf_tol': 1e-1,
            'ipopt.constr_viol_tol': 1e-4, 'ipopt.compl_inf_tol': 1e-4,
            'ipopt.acceptable_tol': 1e-3, 'ipopt.acceptable_dual_inf_tol': 5.0,
            'ipopt.acceptable_constr_viol_tol': 1e-3, 'ipopt.acceptable_iter': 10,
            'ipopt.linear_solver': _LINEAR_SOLVER,
        }
    sol_opt = dict(sol_opt)
    sol_opt.setdefault('print_time', 0)
    sol_opt.setdefault('ipopt.print_level', 5)
    nlp = {'f': J_t + J_reg, 'x': w_vec, 'g': g_vec}
    rospy.loginfo(f'[velopt-rl] NLP: {int(w_vec.shape[0])} vars, '
                  f'{int(g_vec.shape[0])} constraints, {N} points')
    solver = ca.nlpsol('solver', 'ipopt', nlp, sol_opt)
    t_solve = time.time()
    sol = solver(x0=w0_vec, lbx=lbw_vec, ubx=ubw_vec, lbg=lbg_vec, ubg=ubg_vec)
    t_solve = time.time() - t_solve
    J_t_val = float(ca.Function('f_Jt', [w_vec], [J_t])(sol['x']))
    laptime = J_t_val / max(float(w_T), 1e-6)
    success = solver.stats()['success']
    rospy.loginfo(f'[velopt-rl] IPOPT: {t_solve:.2f}s, success={success}, '
                  f'laptime={laptime:.4f}s')
    sol_x = np.array(sol['x']).flatten(); stride = nx_ + nu_
    V_opt = np.zeros(N); ax_opt = np.zeros(N)
    for k in range(N):
        off = k * stride
        V_opt[k] = sol_x[off]; ax_opt[k] = sol_x[off + 1]
    return V_opt, ax_opt, laptime, success


class VelOptNode:

    def __init__(self, map_name, raceline_variant, vehicle_yml_file, gg_dir_name,
                 step_size_opt=0.2, V_min=0.0, gg_margin=0.0):
        rospy.init_node('vel_opt_3d')
        # instance state for reload (constructor args saved verbatim)
        self.map_name = map_name
        self.raceline_variant = raceline_variant
        self.vehicle_yml_file = vehicle_yml_file
        self.gg_dir_name = gg_dir_name
        self.step_size_opt = step_size_opt
        self.V_min = V_min
        self.gg_margin = gg_margin

        # load+solve delegated so reload_cb can reuse it
        self._load_and_solve()

        self.process_lock = threading.Lock()
        self.last_wpnts_msg = None
        self.pub = rospy.Publisher('/global_waypoints', WpntArray, queue_size=1, latch=True)
        self._processed = False
        self.sub = rospy.Subscriber('/global_waypoints', WpntArray, self._cb, queue_size=1)
        self.reload_srv = rospy.Service('/velopt/reload', Trigger, self.reload_cb)
        rospy.loginfo('[velopt] /velopt/reload service ready')
        rospy.loginfo('[velopt] waiting for /global_waypoints template message ...')


    def _load_and_solve(self):
        # rosparam override (used by gg_tuner_3d cold-start and reload)
        self.map_name         = rospy.get_param('~map',            self.map_name)
        racecar               = rospy.get_param('~racecar',        None)
        # racecar acts as shortcut when individual overrides are absent
        default_variant = racecar if racecar else self.raceline_variant
        default_yml     = f'params_{racecar}.yml' if racecar else self.vehicle_yml_file
        default_gg      = racecar if racecar else self.gg_dir_name
        self.raceline_variant = rospy.get_param('~raceline',     default_variant)
        self.vehicle_yml_file = rospy.get_param('~vehicle_yml',  default_yml)
        self.gg_dir_name      = rospy.get_param('~gg_dir',       default_gg)
        self.step_size_opt    = float(rospy.get_param('~step_size_opt', self.step_size_opt))
        self.V_min            = float(rospy.get_param('~V_min',         self.V_min))
        self.gg_margin        = float(rospy.get_param('~gg_margin',     self.gg_margin))
        # NLP cost weights — default to build_and_solve() defaults on first call
        if not hasattr(self, 'w_T'):
            self.w_T = 1.0
        if not hasattr(self, 'w_jx'):
            self.w_jx = 1e-2
        self.w_T  = float(rospy.get_param('~w_T',  self.w_T))
        self.w_jx = float(rospy.get_param('~w_jx', self.w_jx))

        if not hasattr(self, 'w_jx_acc_ratio'):
            self.w_jx_acc_ratio = 1.0
        if not hasattr(self, 'w_jx_brk_ratio'):
            self.w_jx_brk_ratio = 1.0
        if not hasattr(self, 'w_jx_curv_alpha_acc'):
            self.w_jx_curv_alpha_acc = 0.0
        if not hasattr(self, 'w_jx_curv_alpha_brk'):
            self.w_jx_curv_alpha_brk = 0.0
        if not hasattr(self, 'w_jx_curv_k_alpha'):
            self.w_jx_curv_k_alpha = 0.2
        if not hasattr(self, 'w_jx_curv_beta_acc'):
            self.w_jx_curv_beta_acc = 0.0
        if not hasattr(self, 'w_jx_curv_beta_brk'):
            self.w_jx_curv_beta_brk = 0.0
        if not hasattr(self, 'w_jx_curv_k_beta'):
            self.w_jx_curv_k_beta = 0.5
        if not hasattr(self, 'w_jx_curv_alpha'):
            self.w_jx_curv_alpha = 0.0  # legacy, deprecated
        # direct ax_pos^2 corner penalty (see build_and_solve)
        if not hasattr(self, 'w_ax_corner_acc'):
            self.w_ax_corner_acc = 0.0
        if not hasattr(self, 'w_ax_corner_k'):
            self.w_ax_corner_k = 0.2
        self.w_jx_acc_ratio       = float(rospy.get_param('~w_jx_acc_ratio',       self.w_jx_acc_ratio))
        self.w_jx_brk_ratio       = float(rospy.get_param('~w_jx_brk_ratio',       self.w_jx_brk_ratio))
        self.w_jx_curv_alpha_acc  = float(rospy.get_param('~w_jx_curv_alpha_acc',  self.w_jx_curv_alpha_acc))
        self.w_jx_curv_alpha_brk  = float(rospy.get_param('~w_jx_curv_alpha_brk',  self.w_jx_curv_alpha_brk))
        self.w_jx_curv_k_alpha    = float(rospy.get_param('~w_jx_curv_k_alpha',    self.w_jx_curv_k_alpha))
        self.w_jx_curv_beta_acc   = float(rospy.get_param('~w_jx_curv_beta_acc',   self.w_jx_curv_beta_acc))
        self.w_jx_curv_beta_brk   = float(rospy.get_param('~w_jx_curv_beta_brk',   self.w_jx_curv_beta_brk))
        self.w_jx_curv_k_beta     = float(rospy.get_param('~w_jx_curv_k_beta',     self.w_jx_curv_k_beta))
        self.w_jx_curv_alpha      = float(rospy.get_param('~w_jx_curv_alpha',      self.w_jx_curv_alpha))
        self.w_ax_corner_acc      = float(rospy.get_param('~w_ax_corner_acc',      self.w_ax_corner_acc))
        self.w_ax_corner_k        = float(rospy.get_param('~w_ax_corner_k',        self.w_ax_corner_k))
        self.w_jx_acc = self.w_jx * self.w_jx_acc_ratio
        self.w_jx_brk = self.w_jx * self.w_jx_brk_ratio
        # bridge_effect (shared with fbga; fed into Track3D as g_weight)
        self.bridge_effect = float(rospy.get_param('~bridge_effect', 1.0))
        # per-point friction scaling (cheap alt to multi-GGV).
        # When True, friction sector mu is applied as a per-grid-point
        # scalar multiplier inside the NLP — no per-friction GGV
        # directories are needed. Multi-GGV directory lookup is
        # bypassed so the two paths don't double-scale.
        self.use_friction_scaling = bool(
            rospy.get_param('~use_friction_scaling', False))
        # extra per-direction scale on top of sector mu. Default 1.0
        # preserves the (theoretically isotropic) sector-mu behavior.
        # Re-read inside _load_and_solve so every hot-reload picks up
        # live rqt slider changes without restarting the node.
        # self.mu_scale_x = float(rospy.get_param('~mu_scale_x', 1.0))
        # self.mu_scale_y = float(rospy.get_param('~mu_scale_y', 1.0))

        # Resolve paths — folder structure fixed, filenames derived from state
        self.map_dir = os.path.abspath(os.path.join(_THIS_DIR, '..', 'maps', self.map_name))
        self.track_csv = os.path.join(self.map_dir, f'{self.map_name}_3d_smoothed.csv')
        self.raceline_csv = os.path.join(
            self.map_dir, f'{self.map_name}_3d_{self.raceline_variant}_timeoptimal.csv')
        self.vehicle_yml = os.path.join(_DATA_DIR, 'vehicle_params', self.vehicle_yml_file)
        self.gg_path = os.path.join(_DATA_DIR, 'gg_diagrams', self.gg_dir_name, 'velocity_frame')

        for p, label in [(self.track_csv, 'track csv'),
                         (self.raceline_csv, 'raceline csv'),
                         (self.vehicle_yml, 'vehicle yml'),
                         (self.gg_path, 'gg path')]:
            if not os.path.exists(p):
                raise FileNotFoundError(f'{label} not found: {p}')

        rospy.loginfo(f'[velopt] map={self.map_name}')
        rospy.loginfo(f'[velopt] track    : {self.track_csv}')
        rospy.loginfo(f'[velopt] raceline : {self.raceline_csv}')
        rospy.loginfo(f'[velopt] vehicle  : {self.vehicle_yml}')
        rospy.loginfo(f'[velopt] gg       : {self.gg_path}')

        with open(self.vehicle_yml) as f:
            self.vehicle_params = yaml.safe_load(f)['vehicle_params']

        self.track = Track3D(path=self.track_csv, g_weight=self.bridge_effect)
        L_track = float(self.track.s[-1] + self.track.ds)
        N_target = max(10, int(round(L_track / self.step_size_opt)))
        actual_step = L_track / N_target
        rospy.loginfo(f'[velopt] L_track={L_track:.4f}m, '
                      f'desired_step={self.step_size_opt:.4f}, '
                      f'actual_step={actual_step:.6f} (N={N_target})')
        self.track.resample(actual_step)
        self.gg = GGManager(gg_path=self.gg_path, gg_margin=self.gg_margin)

        with open(self.vehicle_yml) as _f:
            _full_params = yaml.safe_load(_f)
        _base_p_Dx_1 = _full_params.get('tire_params', {}).get('p_Dx_1', 0.56)

        # load fixed path + build arclen remap BEFORE friction, so the
        # friction sector->grid mapping reuses the same real arclen map.
        rl = pd.read_csv(self.raceline_csv)
        s_rl = rl['s_opt'].to_numpy()
        n_rl = rl['n_opt'].to_numpy()
        chi_rl = rl['chi_opt'].to_numpy()
        v_rl = rl['v_opt'].to_numpy() if 'v_opt' in rl.columns else None
        ax_rl = rl['ax_opt'].to_numpy() if 'ax_opt' in rl.columns else None
        s_period = s_rl[-1] + (s_rl[-1] - s_rl[-2])
        s_q = self.track.s % s_period
        self.n_fixed = np.interp(s_q, s_rl, n_rl)
        self.chi_fixed = np.interp(s_q, s_rl, np.unwrap(chi_rl))
        self.v_init = np.interp(s_q, s_rl, v_rl) if v_rl is not None else np.full_like(self.track.s, 3.0)
        self.ax_init = np.interp(s_q, s_rl, ax_rl) if ax_rl is not None else np.zeros_like(self.track.s)
        self._build_raceline_arclen_map()

        friction_sectors = _read_friction_sectors()
        # when per-point friction scaling is on, skip multi-GGV
        # directory lookup entirely (avoid double-application). Sector
        # data is reused below to build mu_scale_k.
        if self.use_friction_scaling:
            self.gg_list, self.gg_idx = None, None
            rospy.loginfo('[velopt] multi-GGV bypassed (use_friction_scaling=True)')
        else:
            self.gg_list, self.gg_idx = _build_friction_gg_map(
                friction_sectors, self.track.s.size, self.gg_path,
                self.gg_margin, _base_p_Dx_1)
            if self.gg_list is not None:
                rospy.loginfo(f'[velopt] multi-GGV active: {len(self.gg_list)} GGVs, '
                              f'base p_Dx_1={_base_p_Dx_1:.3f}')
            else:
                rospy.loginfo('[velopt] multi-GGV inactive (single GGV)')
        # build per-point mu_scale_k from friction sectors.
        # Map sector index ranges (defined on the original raceline
        # grid) onto the resampled NLP grid the same way
        # _build_friction_gg_map does — proportional rescale. Sectors
        # with friction <= 0 are skipped (treated as "no override").
        # Reference for mu=1 is base_p_Dx_1 so sectors set to that
        # value leave the envelope unchanged.
        self.mu_scale_k = None
        # per-sector mu_scale_x / mu_scale_y arrays built from the same
        # sector→grid mapping used for mu_scale_k. None ⇒ legacy
        # isotropic (no extra scaling).
        self.mu_scale_x_k = None
        self.mu_scale_y_k = None
        if self.use_friction_scaling and friction_sectors:
            self.mu_scale_k = np.ones(self.track.s.size, dtype=float)
            # per-sector x/y scale arrays, default 1.0
            self.mu_scale_x_k = np.ones(self.track.s.size, dtype=float)
            self.mu_scale_y_k = np.ones(self.track.s.size, dtype=float)
            n_grid = self.track.s.size
            valid = [s for s in friction_sectors if s['friction'] > 0]
            if valid:
                total_original = max(s['end'] for s in valid) + 1
                for sec in valid:
                    ratio = sec['friction'] / max(_base_p_Dx_1, 1e-6)
                    # real arclen sector->grid
                    grid_start, grid_end = self._sector_grid_range(sec, n_grid, total_original)
                    self.mu_scale_k[grid_start:grid_end] = ratio
                    # fill per-sector x/y scale in the same grid range
                    self.mu_scale_x_k[grid_start:grid_end] = float(sec.get('mu_scale_x', 1.0))
                    self.mu_scale_y_k[grid_start:grid_end] = float(sec.get('mu_scale_y', 1.0))
                rospy.loginfo(
                    f'[velopt] mu_scale_k built from {len(valid)} sector(s), '
                    f'range=[{self.mu_scale_k.min():.3f}, '
                    f'{self.mu_scale_k.max():.3f}] '
                    f'(base p_Dx_1={_base_p_Dx_1:.3f})')
                # per-sector x/y scale ranges (info)
                rospy.loginfo(
                    f'[velopt] mu_scale_x_k range=[{self.mu_scale_x_k.min():.3f}, '
                    f'{self.mu_scale_x_k.max():.3f}], '
                    f'mu_scale_y_k range=[{self.mu_scale_y_k.min():.3f}, '
                    f'{self.mu_scale_y_k.max():.3f}]')
            else:
                rospy.loginfo('[velopt] use_friction_scaling=True but no valid '
                              'friction sectors found — mu_scale_k ≡ 1')


        rospy.loginfo(f'[velopt] Track3D + GGManager + raceline ready, grid={self.track.s.size} pts')
        rospy.loginfo(f'[velopt] fixed n  : [{self.n_fixed.min():.3f}, {self.n_fixed.max():.3f}] m')
        rospy.loginfo(f'[velopt] fixed chi: [{self.chi_fixed.min():.3f}, {self.chi_fixed.max():.3f}] rad')

        # same-ruler solve on raceline grid (was centerline _solve_once)
        # self._solve_once()
        self._solve_once_rl()

    def _solve_once_rl(self):
        """Same-ruler solve: NLP on the raceline grid from /global_waypoints."""
        rospy.loginfo('[velopt-rl] waiting for /global_waypoints template ...')
        msg = rospy.wait_for_message('/global_waypoints', WpntArray)
        wp = msg.wpnts
        s_wp  = np.array([p.s_m for p in wp], dtype=float)
        kap_wp = np.array([p.kappa_radpm for p in wp], dtype=float)
        mu_wp  = np.array([p.mu_rad for p in wp], dtype=float)
        vx_wp  = np.array([p.vx_mps for p in wp], dtype=float)
        ax_wp  = np.array([p.ax_mps2 for p in wp], dtype=float)
        L = s_wp[-1] + (s_wp[-1] - s_wp[-2])
        N = max(10, int(round(L / self.step_size_opt)))
        s_grid = np.linspace(0.0, L, N, endpoint=False)
        ds_rl = L / N
        kappa_arr = np.interp(s_grid, s_wp, kap_wp, period=L)
        mu_arr = np.interp(s_grid, s_wp, mu_wp, period=L)
        # smooth mu (periodic moving avg) to stabilise dmu/ds (crest/dip term)
        win = max(3, int(round(1.5 / ds_rl)) | 1)
        ker = np.ones(win) / win
        pad = np.concatenate([mu_arr[-win:], mu_arr, mu_arr[:win]])
        mu_arr = np.convolve(pad, ker, 'same')[win:-win]
        v_init = np.clip(np.interp(s_grid, s_wp, vx_wp, period=L), max(self.V_min, 0.5), None)
        ax_init = np.interp(s_grid, s_wp, ax_wp, period=L)

        # friction sectors -> mu_scale on raceline grid (1:1 via s_start/s_end)
        mu_scale_x_k = mu_scale_y_k = None
        if self.use_friction_scaling:
            valid = [s for s in _read_friction_sectors() if s['friction'] > 0]
            if valid:
                mu_scale_x_k = np.ones(N); mu_scale_y_k = np.ones(N)
                for sec in valid:
                    s0, s1 = sec.get('s_start', -1.0), sec.get('s_end', -1.0)
                    if s0 < 0 or s1 < 0:
                        continue
                    gs = max(0, min(int(round(s0 / ds_rl)), N - 1))
                    ge = max(0, min(int(round(s1 / ds_rl)) + 1, N))
                    mu_scale_x_k[gs:ge] = float(sec.get('mu_scale_x', 1.0))
                    mu_scale_y_k[gs:ge] = float(sec.get('mu_scale_y', 1.0))
                rospy.loginfo(f'[velopt-rl] friction mu_scale_x '
                              f'[{mu_scale_x_k.min():.3f},{mu_scale_x_k.max():.3f}]')

        self.s_grid_rl = s_grid
        self.V_opt, self.ax_opt, laptime, success = build_and_solve_rl(
            s_grid, kappa_arr, mu_arr, self.gg, self.vehicle_params,
            v_init, ax_init,
            w_T=self.w_T, w_jx=self.w_jx, V_min=self.V_min,
            mu_scale_x_k=mu_scale_x_k, mu_scale_y_k=mu_scale_y_k,
            w_jx_acc=self.w_jx_acc, w_jx_brk=self.w_jx_brk,
            w_jx_curv_alpha_acc=self.w_jx_curv_alpha_acc,
            w_jx_curv_alpha_brk=self.w_jx_curv_alpha_brk,
            w_jx_curv_k_alpha=self.w_jx_curv_k_alpha,
            w_jx_curv_beta_acc=self.w_jx_curv_beta_acc,
            w_jx_curv_beta_brk=self.w_jx_curv_beta_brk,
            w_jx_curv_k_beta=self.w_jx_curv_k_beta,
            w_ax_corner_acc=self.w_ax_corner_acc,
            w_ax_corner_k=self.w_ax_corner_k)
        rospy.loginfo(f'[velopt-rl] solved: laptime={laptime:.3f}s '
                      f'V[{self.V_opt.min():.2f},{self.V_opt.max():.2f}] success={success}')

    def _solve_once(self):
        """Run the reduced-state NLP once using csv-based fixed path."""
        rospy.loginfo(
            f'[velopt] solving NLP ... (w_T={self.w_T}, w_jx={self.w_jx}, '
            f'ratio_acc/brk={self.w_jx_acc_ratio}/{self.w_jx_brk_ratio}, '
            f'α_acc/brk={self.w_jx_curv_alpha_acc}/{self.w_jx_curv_alpha_brk} '
            f'(k={self.w_jx_curv_k_alpha} [1/m]), '
            f'β_acc/brk={self.w_jx_curv_beta_acc}/{self.w_jx_curv_beta_brk} '
            f'(k={self.w_jx_curv_k_beta} [1/m^2]), '
            f'legacy_alpha={self.w_jx_curv_alpha}, '
            f'V_min={self.V_min}, gg_margin={self.gg_margin}, '
            f'bridge_effect={self.bridge_effect})')
        self.V_opt, self.ax_opt, laptime, success = build_and_solve(
            track=self.track, gg=self.gg, vehicle_params=self.vehicle_params,
            n_fixed=self.n_fixed, chi_fixed=self.chi_fixed,
            v_init=self.v_init, ax_init=self.ax_init, V_min=self.V_min,
            w_T=self.w_T, w_jx=self.w_jx,
            gg_list=self.gg_list, gg_idx=self.gg_idx,
            # per-point friction scaling (cheap alt to multi-GGV)
            mu_scale_k=self.mu_scale_k,
            # extra per-direction scale on top of sector mu
            # mu_scale_x=self.mu_scale_x,
            # mu_scale_y=self.mu_scale_y,
            # per-sector mu_scale_x / mu_scale_y arrays (replace scalars)
            mu_scale_x_k=self.mu_scale_x_k,
            mu_scale_y_k=self.mu_scale_y_k,
            w_jx_acc=self.w_jx_acc, w_jx_brk=self.w_jx_brk,
            # tanh-saturated curvature weighting (α corner + β transition)
            w_jx_curv_alpha_acc=self.w_jx_curv_alpha_acc,
            w_jx_curv_alpha_brk=self.w_jx_curv_alpha_brk,
            w_jx_curv_k_alpha=self.w_jx_curv_k_alpha,
            w_jx_curv_beta_acc=self.w_jx_curv_beta_acc,
            w_jx_curv_beta_brk=self.w_jx_curv_beta_brk,
            w_jx_curv_k_beta=self.w_jx_curv_k_beta,
            # direct ax_pos^2 corner penalty
            w_ax_corner_acc=self.w_ax_corner_acc,
            w_ax_corner_k=self.w_ax_corner_k,
            w_jx_curv_alpha=self.w_jx_curv_alpha,  # legacy
        )
        rospy.loginfo(f'[velopt] laptime={laptime:.4f}s  '
                      f'V range [{self.V_opt.min():.2f}, {self.V_opt.max():.2f}] m/s  '
                      f'success={success}')

    def _cb(self, msg):
        with self.process_lock:
            if self._processed:
                return
            self._processed = True
            try:
                self.sub.unregister()
            except Exception:
                pass
            self.last_wpnts_msg = msg
            self._publish_solution(msg)

    # reconstruct raceline (centerline + n_fixed) on the NLP grid and
    # integrate its cumulative arc length, paired with centerline s.
    # Inverting this maps a wpnt's raceline-s to its true centerline-s.
    def _build_raceline_arclen_map(self):
        try:
            cx = np.asarray(self.track.x, dtype=float)
            cy = np.asarray(self.track.y, dtype=float)
            th = np.asarray(self.track.theta, dtype=float)
            cs = np.asarray(self.track.s, dtype=float)
            n = np.asarray(self.n_fixed, dtype=float)
            # raceline xy = centerline + n * left-normal(-sin, cos)
            rx = cx - n * np.sin(th)
            ry = cy + n * np.cos(th)
            seg = np.hypot(np.diff(rx, append=rx[0]), np.diff(ry, append=ry[0]))
            S_rl = np.concatenate(([0.0], np.cumsum(seg)[:-1]))
            L_rl = float(S_rl[-1] + seg[-1])
            self._S_rl_wrap = np.concatenate((S_rl, [L_rl]))          # raceline s
            self._track_s_wrap = np.concatenate((cs, [cs[-1] + self.track.ds]))  # center s
            rospy.loginfo(f'[velopt] arclen map: L_raceline={L_rl:.2f}m '
                          f'L_center={cs[-1] + self.track.ds:.2f}m')
        except Exception as e:
            self._S_rl_wrap = None
            self._track_s_wrap = None
            rospy.logwarn(f'[velopt] arclen map build failed: {e} -> linear fallback')

    # map a friction sector to NLP grid indices via real raceline
    # arclen (raceline-s -> centerline-s -> grid). Falls back to the
    # legacy proportional-index mapping if s bounds / map unavailable.
    def _sector_grid_range(self, sec, n_grid, total_original):
        s0, s1 = sec.get('s_start', -1.0), sec.get('s_end', -1.0)
        if s0 >= 0.0 and s1 >= 0.0 and getattr(self, '_S_rl_wrap', None) is not None:
            cs0 = float(np.interp(s0, self._S_rl_wrap, self._track_s_wrap))
            cs1 = float(np.interp(s1, self._S_rl_wrap, self._track_s_wrap))
            gs = int(round(cs0 / self.track.ds))
            ge = int(round(cs1 / self.track.ds)) + 1
        else:
            gs = int(round(sec['start'] / total_original * n_grid))
            ge = int(round((sec['end'] + 1) / total_original * n_grid))
        return max(0, min(gs, n_grid - 1)), max(0, min(ge, n_grid))

    def _publish_solution(self, msg):
        # Determine the message's own track length (for periodic wrap of our V_opt)
        s_msg = np.array([w.s_m for w in msg.wpnts], dtype=np.float64)
        L_msg = s_msg[-1] + (s_msg[-1] - s_msg[-2])  # total track length from wpnts spacing

        # same-ruler — V_opt is on the raceline grid, same ruler as the
        # wpnts, so map 1:1 by raceline-s (no remap, no corner shift).
        ds_rl = self.s_grid_rl[1] - self.s_grid_rl[0]
        s_wrap = np.concatenate((self.s_grid_rl, [self.s_grid_rl[-1] + ds_rl]))
        V_wrap = np.concatenate((self.V_opt, [self.V_opt[0]]))
        ax_wrap = np.concatenate((self.ax_opt, [self.ax_opt[0]]))
        V_out = np.interp(s_msg, s_wrap, V_wrap)
        ax_out = np.interp(s_msg, s_wrap, ax_wrap)

        out = WpntArray()
        out.header = msg.header
        out.header.stamp = rospy.Time.now()
        for i, wpnt_in in enumerate(msg.wpnts):
            wo = Wpnt()
            wo.id = wpnt_in.id
            wo.s_m = wpnt_in.s_m
            wo.d_m = wpnt_in.d_m
            wo.x_m = wpnt_in.x_m
            wo.y_m = wpnt_in.y_m
            wo.z_m = wpnt_in.z_m
            wo.d_right = wpnt_in.d_right
            wo.d_left = wpnt_in.d_left
            wo.psi_rad = wpnt_in.psi_rad
            wo.kappa_radpm = wpnt_in.kappa_radpm
            wo.mu_rad = wpnt_in.mu_rad
            wo.vx_mps = float(V_out[i])
            wo.ax_mps2 = float(ax_out[i])
            out.wpnts.append(wo)

        self.pub.publish(out)
        rospy.loginfo(f'[velopt] published /global_waypoints '
                      f'(msg L={L_msg:.2f}m, grid L={s_wrap[-1]:.2f}m, '
                      f'V[0]={V_out[0]:.2f}, V[-1]={V_out[-1]:.2f})')

    def reload_cb(self, req):
        rospy.loginfo('[velopt] /velopt/reload received')
        try:
            with self.process_lock:
                # Re-read rosparams, reload files, re-solve NLP
                self._load_and_solve()

                # Re-subscribe: discard stale cache, wait for fresh latched msg
                self._processed = False
                try:
                    self.sub.unregister()
                except Exception:
                    pass
                self.sub = rospy.Subscriber(
                    '/global_waypoints', WpntArray, self._cb, queue_size=1)
                fresh_msg = None
                try:
                    fresh_msg = rospy.wait_for_message(
                        '/global_waypoints', WpntArray, timeout=2.0)
                except rospy.ROSException:
                    pass
                if fresh_msg is not None:
                    self.last_wpnts_msg = fresh_msg
                    # mark processed so _cb won't re-publish from the same msg
                    self._processed = True
                    try:
                        self.sub.unregister()
                    except Exception:
                        pass
                    self._publish_solution(fresh_msg)
                    return TriggerResponse(
                        success=True, message='reloaded with fresh template')
                elif self.last_wpnts_msg is not None:
                    self._publish_solution(self.last_wpnts_msg)
                    return TriggerResponse(
                        success=True, message='reloaded with cached template')
                else:
                    return TriggerResponse(
                        success=False, message='no template message yet')
        except Exception as e:
            import traceback
            rospy.logerr(f'[velopt] reload failed: {e}')
            rospy.logerr(traceback.format_exc())
            return TriggerResponse(success=False, message=str(e)[:200])


def main():
    ap = argparse.ArgumentParser(
        description='3D speed-only optimizer — re-optimizes vx/ax on a fixed racing line '
                    'and republishes /global_waypoints with the new velocity profile.')

    ap.add_argument('--map', default=None,
                    help='Map folder name under stack_master/maps/ (e.g. "eng_0410_v5"). '
                         'Track csv is auto-derived as <map>_3d_smoothed.csv. '
                         'If omitted, ~map rosparam is used.')

    ap.add_argument('--racecar', default=None,
                    help='Shortcut: set raceline/gg_dir/vehicle_yml to this '
                         'variant in one go (e.g. "rc_car_10th_v7"). '
                         'Individually-specified flags override this.')

    # --- raceline/vehicle/gg: None means "inherit from --racecar, else default" ---
    # Left as None so we can tell whether the user explicitly set them.
    ap.add_argument('--raceline', default=None,
                    help='Raceline variant — filename becomes '
                         '<map>_3d_<raceline>_timeoptimal.csv '
                         '(default: --racecar if set, else "rc_car_10th")')
    ap.add_argument('--vehicle_yml', default=None,
                    help='Vehicle params yml filename inside '
                         '3d_gb_optimizer/.../vehicle_params/ '
                         '(default: "params_<racecar>.yml" if --racecar set, '
                         'else "params_rc_car_10th.yml")')
    ap.add_argument('--gg_dir', default=None,
                    help='GG diagrams folder name inside '
                         '3d_gb_optimizer/.../gg_diagrams/ '
                         '(default: --racecar if set, else "rc_car_10th")')

    ap.add_argument('--step_size_opt', type=float, default=0.2,
                    help='Desired NLP grid spacing in meters (default: 0.2)')
    ap.add_argument('--V_min', type=float, default=1.0,
                    help='Minimum velocity bound in m/s (default: 1.0)')
    ap.add_argument('--gg_margin', type=float, default=0.0,
                    help='GGV shrink margin (default: 0.0)')

    # parse_known_args so ROS remapping args (__name:=, __log:=) don't choke argparse
    args, _ = ap.parse_known_args()

    _racecar = args.racecar or 'rc_car_10th'
    args.raceline    = args.raceline    or _racecar
    args.gg_dir      = args.gg_dir      or _racecar
    args.vehicle_yml = args.vehicle_yml or f'params_{_racecar}.yml'
    map_name_init = args.map if args.map else ''  # empty str → _load_and_solve will read ~map

    VelOptNode(
        map_name=map_name_init,
        raceline_variant=args.raceline,
        vehicle_yml_file=args.vehicle_yml,
        gg_dir_name=args.gg_dir,
        step_size_opt=args.step_size_opt,
        V_min=args.V_min,
        gg_margin=args.gg_margin,
    )
    rospy.spin()


if __name__ == '__main__':
    main()
