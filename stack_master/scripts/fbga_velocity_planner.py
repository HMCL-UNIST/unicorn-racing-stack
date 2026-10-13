#!/usr/bin/env python3
# FBGA-based 3D velocity planner ROS node

import rospy
import os
import subprocess
import tempfile
import numpy as np
import json
import yaml
import struct

# hot-reload support (threading lock + Trigger service)
import threading
from std_srvs.srv import Trigger, TriggerResponse

# SG filter on mu/dmu_ds/kappa
try:
    from scipy.signal import savgol_filter
    HAS_SCIPY_SG = True
except ImportError:
    HAS_SCIPY_SG = False

from f110_msgs.msg import WpntArray, Wpnt
import trajectory_planning_helpers as tph
# rqt sliders are owned by gg_tuner_node (GGTuner.cfg FBGA group);
# gg_tuner forwards values via /fbga_planner/* rosparam + /fbga/reload.


class FBGAVelocityPlanner:

    def __init__(self):
        rospy.loginfo("[FBGA] Initializing...")

        # derive race_stack root from this script's path to avoid hardcoded /home/unicorn
        race_stack = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.fbga_bin = rospy.get_param(
            "~fbga_bin",
            os.path.join(race_stack, 'f110_utils', 'libs', 'FBGA', 'bin', 'GIGI_test_unicorn.exe'))

        # default paths → *_latest (auto-updated by gg_tuner_node)
        #       gg_tuner creates v<N> files and copies/symlinks to *_latest,
        #       so these defaults always resolve to the most recent GGV + params.
        # (previous: hardcoded to rc_car_10th_fast4 — stale after gg_tuner runs)
        # gg_bin_default = os.path.join(
        #     race_stack, 'planner', '3d_gb_optimizer', 'global_line', 'data',
        #     'gg_diagrams', 'rc_car_10th_fast4', 'velocity_frame', 'gg.bin')
        # params_yml_default = os.path.join(
        #     race_stack, 'planner', '3d_gb_optimizer', 'global_line', 'data',
        #     'vehicle_params', 'params_rc_car_10th_fast4.yml')
        gg_bin_default = os.path.join(
            race_stack, 'planner', '3d_gb_optimizer', 'global_line', 'data',
            'gg_diagrams', 'rc_car_10th_latest', 'velocity_frame', 'gg.bin')
        self.gg_bin = rospy.get_param("~gg_bin", gg_bin_default)

        params_yml_default = os.path.join(
            race_stack, 'planner', '3d_gb_optimizer', 'global_line', 'data',
            'vehicle_params', 'params_rc_car_10th_latest.yml')
        params_yml = rospy.get_param("~params_yml", params_yml_default)

        for name, path in [('fbga_bin', self.fbga_bin), ('params_yml', params_yml)]:
            if not os.path.exists(path):
                rospy.logerr(f"[FBGA] File not found: {name}={path}")
                raise FileNotFoundError(path)

        if not os.path.exists(self.gg_bin):
            self._generate_gg_bin(self.gg_bin)

        self.params_txt = os.path.join(tempfile.gettempdir(), 'fbga_params.txt')
        self._convert_params_yml(params_yml)

        self.n_laps = rospy.get_param("~n_laps", 3)
        self.max_iter = rospy.get_param("~max_iter", 50)
        self.tol = rospy.get_param("~tol", 0.05)        # m/s
        # under-relax to break limit-cycle oscillation on bridge
        #       (g_tilde clip + binary nonlinearity → 2-cycle without alpha<1)
        self.alpha = rospy.get_param("~alpha", 0.5)
        self.v0 = rospy.get_param("~v0", 1.0)
        self.enable_mu = rospy.get_param("~enable_mu", True)

        # legacy g_weight kept only for backward compat (not applied)
        self.g_weight = float(rospy.get_param("~g_weight", 1.0))
        if self.g_weight != 1.0:
            rospy.logwarn(
                "[FBGA] ~g_weight is legacy; bridge_effect now controls slope effect")

        # bridge_effect = 2.5d_vel_planner slope_correction
        # 1.0 = physics, 0.0 = treat as flat, >1.0 = more conservative.
        # Applied in C++ FWBW (sin(mu)*slope_corr) via --slope-corr CLI.
        self.bridge_effect = float(rospy.get_param("~bridge_effect", 1.0))

        # softplus floor sharpness for g_tilde (1/beta = transition width)
        self.g_tilde_soft_beta = float(rospy.get_param("~g_tilde_soft_beta", 20.0))

        # SG filter params (origin/main f19f8fe defaults)
        self.smooth_mu = bool(rospy.get_param("~smooth_mu", True))
        self.mu_smooth_window = int(rospy.get_param("~mu_smooth_window", 21))
        self.mu_smooth_polyorder = int(rospy.get_param("~mu_smooth_polyorder", 3))
        self.smooth_kappa = bool(rospy.get_param("~smooth_kappa", True))
        self.kappa_smooth_window = int(rospy.get_param("~kappa_smooth_window", 21))
        self.kappa_smooth_polyorder = int(rospy.get_param("~kappa_smooth_polyorder", 3))

        # pre-slope brake (off by default, 2.5d_vel_planner pattern)
        self.slope_brake_margin = float(rospy.get_param("~slope_brake_margin", 0.0))
        self.slope_brake_vmax = float(rospy.get_param("~slope_brake_vmax", 5.0))

        if self.smooth_mu and not HAS_SCIPY_SG:
            rospy.logwarn("[FBGA] scipy unavailable — SG smoothing disabled")
            self.smooth_mu = False
            self.smooth_kappa = False

        self.g_min, self.g_max = self._read_g_range()

        # === Pub/Sub ===
        self.processed = False
        self.pub = rospy.Publisher('/global_waypoints', WpntArray, queue_size=10)
        rospy.Subscriber('/global_waypoints', WpntArray, self.wpnts_callback)

        # hot-reload — cache last input wpnts + mutex + /fbga/reload service
        self.last_wpnts_msg = None
        self.process_lock = threading.Lock()
        self.reload_srv = rospy.Service('/fbga/reload', Trigger, self.reload_cb)

        rospy.loginfo(f"[FBGA] Ready. bin={self.fbga_bin}")
        rospy.loginfo(f"[FBGA] gg.bin={self.gg_bin}")
        rospy.loginfo(f"[FBGA] n_laps={self.n_laps}, max_iter={self.max_iter}, tol={self.tol}")
        rospy.loginfo(
            f"[FBGA] bridge_effect={self.bridge_effect:.2f}, "
            f"g_tilde_soft_beta={self.g_tilde_soft_beta:.1f}, "
            f"smooth_mu={self.smooth_mu}/win{self.mu_smooth_window}, "
            f"smooth_kappa={self.smooth_kappa}/win{self.kappa_smooth_window}, "
            f"slope_brake_margin={self.slope_brake_margin:.2f}m")

    def _generate_gg_bin(self, bin_path):
        npy_dir = os.path.dirname(bin_path)
        rospy.loginfo(f"[FBGA] gg.bin not found, generating from {npy_dir}")

        v_list = np.load(os.path.join(npy_dir, 'v_list.npy')).astype(np.float64)
        g_list = np.load(os.path.join(npy_dir, 'g_list.npy')).astype(np.float64)
        ax_max = np.load(os.path.join(npy_dir, 'ax_max.npy')).astype(np.float64)
        ax_min = np.load(os.path.join(npy_dir, 'ax_min.npy')).astype(np.float64)
        ay_max = np.load(os.path.join(npy_dir, 'ay_max.npy')).astype(np.float64)
        gg_exp = np.load(os.path.join(npy_dir, 'gg_exponent.npy')).astype(np.float64)

        nv, ng = len(v_list), len(g_list)
        with open(bin_path, 'wb') as f:
            f.write(struct.pack('II', nv, ng))
            for arr in [v_list, g_list, ax_max, ax_min, ay_max, gg_exp]:
                arr.tofile(f)

        rospy.loginfo(f"[FBGA] gg.bin generated: nv={nv}, ng={ng}, size={os.path.getsize(bin_path)} bytes")

    def _convert_params_yml(self, yml_path):
        with open(yml_path) as f:
            cfg = yaml.safe_load(f)
        vp = cfg['vehicle_params']
        tp = cfg['tire_params']
        with open(self.params_txt, 'w') as f:
            f.write(f"m={vp['m']}\n")
            f.write(f"P_max={vp['P_max']}\n")
            f.write(f"mu_x={tp['p_Dx_1']}\n")
            f.write(f"mu_y={tp['p_Dy_1']}\n")
            f.write(f"v_max={vp['v_max']}\n")
        # cache v_max for publish-time hard clamp
        self.v_max = float(vp['v_max'])
        rospy.loginfo(f"[FBGA] params.txt saved: m={vp['m']}, P_max={vp['P_max']}, v_max={vp['v_max']}")

    def _read_g_range(self):
        file_size = os.path.getsize(self.gg_bin)
        with open(self.gg_bin, 'rb') as f:
            nv, ng = struct.unpack('II', f.read(8))
            expected_2d = 8 + 8 * (nv + ng) + 4 * 8 * nv * ng
            if file_size > expected_2d:
                # 3D format: read ns next
                ns = struct.unpack('I', f.read(4))[0]
                v_list = np.frombuffer(f.read(nv * 8), dtype=np.float64)
                g_list = np.frombuffer(f.read(ng * 8), dtype=np.float64)
                rospy.loginfo(
                    f"[FBGA] GGV range (3D): v=[{v_list.min():.1f},{v_list.max():.1f}], "
                    f"g=[{g_list.min():.2f},{g_list.max():.2f}], ns={ns}")
            else:
                v_list = np.frombuffer(f.read(nv * 8), dtype=np.float64)
                g_list = np.frombuffer(f.read(ng * 8), dtype=np.float64)
                rospy.loginfo(
                    f"[FBGA] GGV range (2D): v=[{v_list.min():.1f},{v_list.max():.1f}], "
                    f"g=[{g_list.min():.2f},{g_list.max():.2f}]")
        return float(g_list.min()), float(g_list.max())

    def _compute_g_tilde(self, mu, v, dmu_ds):
        """g_tilde = 9.81*cos(mu) - v^2 * dmu_ds, with softplus floor at g_min."""
        mu_eff = mu
        dmu_ds_eff = dmu_ds
        gt = 9.81 * np.cos(mu_eff) - v**2 * dmu_ds_eff
        # softplus lower bound (smooth replacement of np.clip's hard floor)
        beta = self.g_tilde_soft_beta
        gt_soft = self.g_min + np.log1p(np.exp(beta * (gt - self.g_min))) / beta
        return np.minimum(gt_soft, self.g_max)

    def _initial_speed_estimate(self, kappa, mu, dmu_ds):
        ay_max = 4.5
        v_max = 12.0

        radius = np.where(np.abs(kappa) > 1e-4, 1.0 / np.abs(kappa), 1e4)
        v_lat = np.clip(np.sqrt(ay_max * radius), 0, v_max)

        mu_eff = mu
        dmu_ds_eff = dmu_ds
        v_vert = np.full_like(kappa, v_max)
        crest = dmu_ds_eff > 1e-4
        v_vert[crest] = np.clip(
            np.sqrt(9.81 * np.cos(mu_eff[crest]) / dmu_ds_eff[crest]), 0.5, v_max)

        return np.minimum(v_lat, v_vert)

    def _smooth_kappa(self, kappa):
        """SG-smooth a *copy* of kappa for vel calc only (path is preserved)."""
        if not self.smooth_kappa or not HAS_SCIPY_SG:
            return kappa
        win = self.kappa_smooth_window | 1   # force odd
        if len(kappa) < win:
            return kappa
        poly = max(1, min(self.kappa_smooth_polyorder, win - 1))
        pad = win // 2
        k_pad = np.concatenate([kappa[-pad:], kappa, kappa[:pad]])
        return savgol_filter(k_pad, win, poly)[pad:-pad]

    def _stack_laps(self, s, kappa, g_tilde, mu, dmu_ds):
        n_pts = len(s)
        ds = s[1] - s[0] if n_pts > 1 else 0.1
        lap_length = s[-1] - s[0] + ds

        s_stack = np.concatenate([s + i * lap_length for i in range(self.n_laps)])
        k_stack = np.tile(kappa, self.n_laps)
        g_stack = np.tile(g_tilde, self.n_laps)
        mu_stack = np.tile(mu, self.n_laps)
        dmu_stack = np.tile(dmu_ds, self.n_laps) 
        return s_stack, k_stack, g_stack, mu_stack, dmu_stack, lap_length, n_pts

    def _run_fbga(self, s, kappa, g_tilde, mu, dmu_ds, v0):
        """Run FBGA C++ binary on a single stacked-lap input."""
        input_csv = os.path.join(tempfile.gettempdir(), 'fbga_input.csv')
        output_csv = os.path.join(tempfile.gettempdir(), 'fbga_output.csv')

        mu_csv = mu
        dmu_ds_csv = dmu_ds
        with open(input_csv, 'w') as f:
            if self.enable_mu:
                f.write('s,kappa,g_tilde,mu,dmu_ds\n')
                for i in range(len(s)):
                    f.write(f'{s[i]:.6f},{kappa[i]:.8f},{g_tilde[i]:.6f},{mu_csv[i]:.8f},{dmu_ds_csv[i]:.8f}\n')
            else:
                f.write('s,kappa,g_tilde\n')
                for i in range(len(s)):
                    f.write(f'{s[i]:.6f},{kappa[i]:.8f},{g_tilde[i]:.6f}\n')

        cmd = [
            self.fbga_bin,
            '--model', 'lookup',
            '--input', input_csv,
            '--params', self.params_txt,
            '--gg', self.gg_bin,
            '--output', output_csv,
            '--v0', f'{v0:.4f}',
            '--slope-corr', f'{self.bridge_effect:.4f}',
        ]

        try:
            rospy.loginfo(f"[FBGA] Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=10)
            rospy.loginfo(f"[FBGA] exe stdout: {result.stdout[-200:]}")
        except subprocess.CalledProcessError as e:
            rospy.logerr(f"[FBGA] exe failed (rc={e.returncode}): {e.stderr[:500]}")
            return None
        except subprocess.TimeoutExpired:
            rospy.logerr("[FBGA] exe timeout")
            return None
        except Exception as e:
            rospy.logerr(f"[FBGA] exe error: {e}")
            return None

        v_out = []
        ax_out = []
        with open(output_csv) as f:
            for line in f:
                if line.startswith('#') or line.startswith('s,'):
                    continue
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    v_out.append(float(parts[1]))
                    ax_out.append(float(parts[2]))

        try:
            os.remove(input_csv)
            os.remove(output_csv)
        except OSError:
            pass

        return np.array(v_out), np.array(ax_out)

    def _extract_middle_lap(self, v_full, ax_full, n_pts_per_lap):
        middle = self.n_laps // 2
        start = middle * n_pts_per_lap
        end = start + n_pts_per_lap
        return v_full[start:end], ax_full[start:end]

    def wpnts_callback(self, msg):
        with self.process_lock:
            if self.processed:
                return
            self.last_wpnts_msg = msg
            self.processed = True
            self._process_and_publish(msg)

    def reload_cb(self, req):
        try:
            new_gg = rospy.get_param('~gg_bin', self.gg_bin)
            new_params = rospy.get_param('~params_yml', None)
            new_enable_mu = rospy.get_param('~enable_mu', self.enable_mu)

            # refresh runtime knobs from rosparam (rqt sliders also write here)
            new_bridge_effect = float(rospy.get_param('~bridge_effect', self.bridge_effect))
            new_g_tilde_soft_beta = float(rospy.get_param('~g_tilde_soft_beta', self.g_tilde_soft_beta))
            new_smooth_mu = bool(rospy.get_param('~smooth_mu', self.smooth_mu))
            new_mu_smooth_window = int(rospy.get_param('~mu_smooth_window', self.mu_smooth_window))
            new_mu_smooth_polyorder = int(rospy.get_param('~mu_smooth_polyorder', self.mu_smooth_polyorder))
            new_smooth_kappa = bool(rospy.get_param('~smooth_kappa', self.smooth_kappa))
            new_kappa_smooth_window = int(rospy.get_param('~kappa_smooth_window', self.kappa_smooth_window))
            new_kappa_smooth_polyorder = int(rospy.get_param('~kappa_smooth_polyorder', self.kappa_smooth_polyorder))
            new_slope_brake_margin = float(rospy.get_param('~slope_brake_margin', self.slope_brake_margin))
            new_slope_brake_vmax = float(rospy.get_param('~slope_brake_vmax', self.slope_brake_vmax))

            with self.process_lock:
                self.gg_bin = new_gg
                self.enable_mu = new_enable_mu
                self.bridge_effect = new_bridge_effect
                self.g_tilde_soft_beta = new_g_tilde_soft_beta
                self.smooth_mu = new_smooth_mu and HAS_SCIPY_SG
                self.mu_smooth_window = new_mu_smooth_window | 1
                self.mu_smooth_polyorder = new_mu_smooth_polyorder
                self.smooth_kappa = new_smooth_kappa and HAS_SCIPY_SG
                self.kappa_smooth_window = new_kappa_smooth_window | 1
                self.kappa_smooth_polyorder = new_kappa_smooth_polyorder
                self.slope_brake_margin = new_slope_brake_margin
                self.slope_brake_vmax = new_slope_brake_vmax
                self.g_min, self.g_max = self._read_g_range()
                if new_params and os.path.exists(new_params):
                    self._convert_params_yml(new_params)

                rospy.loginfo(
                    f"[FBGA] Reloaded: bridge_effect={self.bridge_effect:.2f}, "
                    f"soft_beta={self.g_tilde_soft_beta:.1f}, "
                    f"smooth_mu={self.smooth_mu}/win{self.mu_smooth_window}, "
                    f"smooth_kappa={self.smooth_kappa}/win{self.kappa_smooth_window}, "
                    f"slope_brake_margin={self.slope_brake_margin:.2f}m")

                if self.last_wpnts_msg is not None:
                    self._process_and_publish(self.last_wpnts_msg)
                    return TriggerResponse(
                        success=True, message="reloaded and reprocessed")
                else:
                    return TriggerResponse(
                        success=True, message="reloaded (no cached waypoints)")
        except Exception as e:
            rospy.logerr(f"[FBGA] reload failed: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            return TriggerResponse(success=False, message=str(e)[:200])

    def _process_and_publish(self, msg):
        wpnts = msg.wpnts
        n = len(wpnts)

        s = np.array([wp.s_m for wp in wpnts])
        kappa = np.array([wp.kappa_radpm for wp in wpnts])
        mu = np.array([wp.mu_rad for wp in wpnts])
        v_existing = np.array([wp.vx_mps for wp in wpnts])

        ds_grid = s[1] - s[0] if n > 1 else 0.1

        # SG filter on mu, deriv=1 on padded mu for dmu_ds.
        ##   Replaces the central diff that overestimates derivatives at the seam
        ##   and leaks cm-scale PCD noise into v at curvature inflections.
        if self.smooth_mu and HAS_SCIPY_SG and n >= self.mu_smooth_window:
            win = self.mu_smooth_window | 1
            poly = max(1, min(self.mu_smooth_polyorder, win - 1))
            pad = win // 2
            mu_pad = np.concatenate([mu[-pad:], mu, mu[:pad]])
            mu = savgol_filter(mu_pad, win, poly)[pad:-pad]
            dmu_ds = savgol_filter(mu_pad, win, poly,
                                   deriv=1, delta=ds_grid)[pad:-pad]
        else:
            mu_pad = np.concatenate([mu[-1:], mu, mu[:1]])
            dmu_ds = (mu_pad[2:] - mu_pad[:-2]) / (2.0 * ds_grid)

        # SG filter on kappa for vel calc only
        ##   wpnts[i].kappa_radpm (path) is preserved.
        kappa_for_fbga = self._smooth_kappa(kappa)

        if np.any(v_existing > 0.1):
            v_prev = v_existing.copy()
            rospy.loginfo("[FBGA] Using existing waypoint speeds as initial estimate")
        else:
            v_prev = self._initial_speed_estimate(kappa_for_fbga, mu, dmu_ds)
            rospy.loginfo("[FBGA] Using curvature+slope initial speed estimate")

        # === Fixed-point iteration ===
        for it in range(self.max_iter):
            g_tilde = self._compute_g_tilde(mu, v_prev, dmu_ds)

            s_stack, k_stack, g_stack, mu_stack, dmu_stack, lap_length, n_pts = self._stack_laps(s, kappa_for_fbga, g_tilde, mu, dmu_ds)

            v0 = max(float(v_prev[0]), 1.0)

            result = self._run_fbga(s_stack, k_stack, g_stack,
                                    mu_stack, dmu_stack, v0)
            if result is None:
                rospy.logwarn("[FBGA] Failed, keeping existing speeds")
                return
            v_full, ax_full = result
            v_new, ax_new = self._extract_middle_lap(v_full, ax_full, n_pts)

            nan_mask = np.isnan(v_new)
            if nan_mask.any():
                n_nan = nan_mask.sum()
                if n_nan / n > 0.05:
                    rospy.logwarn(f"[FBGA] Too many NaNs: {n_nan}/{n}")
                    return
                valid = np.where(~nan_mask)[0]
                v_new[nan_mask] = np.interp(np.where(nan_mask)[0], valid, v_new[valid])

            delta = float(np.max(np.abs(v_new - v_prev)))
            rospy.loginfo(f"[FBGA] iter {it}: max|dv|={delta:.4f} m/s, "
                          f"g_tilde=[{g_tilde.min():.2f},{g_tilde.max():.2f}]")

            if delta < self.tol:
                rospy.loginfo(f"[FBGA] Converged at iter {it}")
                break

            v_prev = self.alpha * v_new + (1.0 - self.alpha) * v_prev

        # === waypoint update ===
        ax_nan_mask = np.isnan(ax_new)
        if ax_nan_mask.any():
            valid_ax = np.where(~ax_nan_mask)[0]
            ax_new[ax_nan_mask] = np.interp(np.where(ax_nan_mask)[0], valid_ax, ax_new[valid_ax])

        # hard v_max clamp (belt-and-suspenders; GGV grid already saturates)
        v_new = np.minimum(v_new, self.v_max)

        # optional pre-slope brake (same pattern as 2.5d_vel_planner).
        ##   off when slope_brake_margin == 0. Heuristic safety cap that
        ##   forces v <= slope_brake_vmax from `margin` meters before slope
        ##   entry through slope exit (slope = |mu| > 2 deg).
        if self.slope_brake_margin > 0:
            in_slope = np.abs(mu) > np.radians(2.0)
            diffs = np.diff(in_slope.astype(int))
            entries = np.where(diffs == 1)[0] + 1
            exits = np.where(diffs == -1)[0] + 1
            margin_pts = max(1, int(round(self.slope_brake_margin / ds_grid)))
            for entry in entries:
                ex = exits[exits > entry]
                end = ex[0] if len(ex) > 0 else n
                start = max(0, entry - margin_pts)
                v_new[start:end] = np.minimum(v_new[start:end], self.slope_brake_vmax)

        for i in range(n):
            wpnts[i].vx_mps = float(v_new[i])
            wpnts[i].ax_mps2 = float(ax_new[i])

        msg.wpnts = wpnts
        rospy.loginfo(f"[FBGA] Publishing: v=[{v_new.min():.2f},{v_new.max():.2f}] m/s "
                      f"(v_max={self.v_max})")
        self.pub.publish(msg)


if __name__ == '__main__':
    rospy.init_node("fbga_velocity_planner")
    node = FBGAVelocityPlanner()
    rospy.spin()
