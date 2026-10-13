#!/usr/bin/env python3
# 3D version of global_velocity_planner — uses vel_planner_25d with g_tilde

import rospy
import os
import math
import numpy as np
import json
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Joy
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float64
import trajectory_planning_helpers as tph
from rospkg import RosPack
from f110_msgs.msg import ObstacleArray, OTWpntArray, WpntArray, Wpnt
from visualization_msgs.msg import Marker, MarkerArray
from vel_planner_25d.vel_planner import calc_vel_profile

from copy import deepcopy
import configparser
import shutil
import yaml
from dynamic_reconfigure.server import Server
from stack_master.cfg import VelPlanner3DConfig

# optional SG filter for kappa-only smoothing (for vel calc, path preserved)
try:
    from scipy.signal import savgol_filter
    HAS_SCIPY_SG = True
except ImportError:
    HAS_SCIPY_SG = False

class VelocityPlanner:

    SAVE_CONFIG = False  # legacy CSV save flag (now controlled via rqt save_csv)

    def __init__(self):
        """Initialize the node, subscribe to topics, create publishers and set up member variables."""

        self.racecar_version = rospy.get_param("/racecar_version")

        # Velocity Planning
        parser = configparser.ConfigParser()
        self.pars = {}
        if not parser.read(os.path.join(RosPack().get_path('stack_master'), 'config', self.racecar_version, 'racecar_f110.ini')):
            raise ValueError('Specified config file does not exist or is empty!')
        self.pars["veh_params"] = json.loads(parser.get('GENERAL_OPTIONS', 'veh_params'))
        self.pars["vel_calc_opts"] = json.loads(parser.get('GENERAL_OPTIONS', 'vel_calc_opts'))
        ggv_path = os.path.join(RosPack().get_path('stack_master'), 'config', self.racecar_version, "veh_dyn_info", "ggv.csv")
        ax_max_path = os.path.join(RosPack().get_path('stack_master'), 'config', self.racecar_version, "veh_dyn_info", "ax_max_machines.csv")
        b_ax_max_path = os.path.join(RosPack().get_path('stack_master'), 'config', self.racecar_version, "veh_dyn_info", "b_ax_max_machines.csv")
        self.ggv, self.ax_max_machines = tph.import_veh_dyn_info.\
            import_veh_dyn_info(ggv_import_path=ggv_path,
                                ax_max_machines_import_path=ax_max_path)

        _, self.b_ax_max_machines = tph.import_veh_dyn_info.\
            import_veh_dyn_info(ggv_import_path=ggv_path,
                                ax_max_machines_import_path=b_ax_max_path)


        self.v_max = self.pars["veh_params"]["v_max"]
        self.drag_coeff = self.pars["veh_params"]["dragcoeff"]
        self.m_veh = self.pars["veh_params"]["mass"]
        self.filt_window = self.pars["vel_calc_opts"]["vel_profile_conv_filt_window"]
        self.dyn_model_exp = self.pars["vel_calc_opts"]["dyn_model_exp"]

        # Load vel_planner tuning params from yaml
        self.vel_planner_yaml = os.path.join(
            RosPack().get_path('stack_master'), 'config', self.racecar_version, 'vel_planner.yaml')
        self._load_vel_planner_params()

        # Apply tuning params to ggv tables
        self._apply_params_to_ggv()

        # CoG height for g_tilde calculation
        self.h_cog = self.pars["veh_params"].get("cog_z", 0.074)

        # Publishers
        self.glb_wpnts_pub = rospy.Publisher('/global_waypoints', WpntArray, queue_size=10)
        self.smart_static_wpnts_pub = rospy.Publisher('/planner/avoidance/smart_static_otwpnts', OTWpntArray, queue_size=10)
        # 2D (no 3D correction) vel markers for comparison
        self.vel_markers_2d_pub = rospy.Publisher('/global_waypoints/vel_markers_2d', MarkerArray, queue_size=10)

        # Subscribers
        rospy.Subscriber("/global_waypoints", WpntArray, self.wpnts_callback)
        rospy.Subscriber("/planner/avoidance/smart_static_otwpnts", OTWpntArray, self.smart_static_wpnts_callback)

        # Dynamic reconfigure server (rqt real-time tuning)
        self._init_phase = True               # block all changed=True during init
        self.dyn_reconfig_initialized = False  # skip first callback (Server auto-calls on init)
        self.dyn_reconfig_changed = False
        self.dyn_srv = Server(VelPlanner3DConfig, self._dyn_reconfig_callback)

        # Push yaml values to dynamic_reconfigure server (overrides cfg defaults)
        self.dyn_srv.update_configuration({
            'v_max': self.v_max,
            'ax_max_motor': self.ax_max_motor,
            'ax_max_brake': self.ax_max_brake,
            'dyn_model_exp': self.dyn_model_exp,
            'a_y_max': self.a_y_max,
            'a_x_max': self.a_x_max,
            'grip_scale_exp': self.grip_scale_exp,
            'slope_correction': self.slope_correction,
            'slope_brake_margin': self.slope_brake_margin,
            'slope_brake_vmax': self.slope_brake_vmax,
            # SG mu smoothing toggle
            'smooth_mu': self.smooth_mu,
            'mu_smooth_window': self.mu_smooth_window,
            'mu_smooth_polyorder': self.mu_smooth_polyorder,
            # SG kappa smoothing (vel-only)
            'smooth_kappa': self.smooth_kappa,
            'kappa_smooth_window': self.kappa_smooth_window,
            'kappa_smooth_polyorder': self.kappa_smooth_polyorder,
        })
        self.dyn_reconfig_changed = False

        # Friction sector subscriber
        from dynamic_reconfigure.msg import Config as DynConfig
        rospy.Subscriber("/dyn_sector_tuner/friction/parameter_updates", DynConfig, self.friction_changed_cb)

        # Init complete — now allow rqt changes to trigger recalculation
        self._init_phase = False
        self.dyn_reconfig_changed = False

    def friction_changed_cb(self, msg):
        """Friction sector changed — trigger vel_planner recalculation"""
        if not hasattr(self, '_friction_first_skipped'):
            self._friction_first_skipped = True
            return
        self.dyn_reconfig_changed = True

    # locate ggv velocity_frame dir for slope_ax_scale.npy meta
    def _resolve_ggv_meta_dir(self):
        base = os.path.join(
            RosPack().get_path('stack_master'),
            '..', '..', 'planner', '3d_gb_optimizer', 'global_line', 'data',
            'gg_diagrams', self.racecar_version, 'velocity_frame')
        return os.path.normpath(base)

    def _load_vel_planner_params(self):
        """Load vel planner params from yaml file"""
        params = {}
        if os.path.exists(self.vel_planner_yaml):
            with open(self.vel_planner_yaml) as f:
                params = yaml.safe_load(f) or {}
            self.v_max = params.get('v_max', 12.0)
            self.ax_max_motor = params.get('ax_max_motor', 5.0)
            self.ax_max_brake = params.get('ax_max_brake', 7.0)
            self.dyn_model_exp = params.get('dyn_model_exp', 1.0)
            self.a_y_max = params.get('a_y_max', 4.5)
            self.a_x_max = params.get('a_x_max', 5.0)
            self.grip_scale_exp = params.get('grip_scale_exp', 0.7)
            self.slope_correction = params.get('slope_correction', 1.0)
            self.slope_brake_margin = params.get('slope_brake_margin', 0.0)
            self.slope_brake_vmax = params.get('slope_brake_vmax', 5.0)
            # SG mu smoothing toggle (passed to vel_planner_25d)
            self.smooth_mu = params.get('smooth_mu', True)
            self.mu_smooth_window = params.get('mu_smooth_window', 21)
            self.mu_smooth_polyorder = params.get('mu_smooth_polyorder', 3)
            # SG kappa smoothing (vel-only, path preserved)
            self.smooth_kappa = params.get('smooth_kappa', True)
            self.kappa_smooth_window = params.get('kappa_smooth_window', 21)
            self.kappa_smooth_polyorder = params.get('kappa_smooth_polyorder', 3)
            rospy.loginfo(f"[VelPlanner3D] Loaded params from {self.vel_planner_yaml}")
        else:
            rospy.logwarn(f"[VelPlanner3D] {self.vel_planner_yaml} not found, using defaults")
            self.v_max = 12.0
            self.ax_max_motor = 5.0
            self.ax_max_brake = 7.0
            self.dyn_model_exp = 1.0
            self.a_y_max = 4.5
            self.a_x_max = 5.0
            self.grip_scale_exp = 0.7
            self.slope_correction = 1.0
            self.slope_brake_margin = 0.0
            self.slope_brake_vmax = 5.0
            # SG mu smoothing defaults
            self.smooth_mu = True
            self.mu_smooth_window = 21
            self.mu_smooth_polyorder = 3
            # SG kappa smoothing defaults (vel-only)
            self.smooth_kappa = True
            self.kappa_smooth_window = 21
            self.kappa_smooth_polyorder = 3

        # auto-pair slope_correction with ggv meta if yaml does not specify
        if 'slope_correction' not in params:
            try:
                meta_dir = self._resolve_ggv_meta_dir()
                sas_path = os.path.join(meta_dir, 'slope_ax_scale.npy')
                es_path  = os.path.join(meta_dir, 'enable_slope.npy')
                if os.path.isfile(sas_path):
                    sas = float(np.load(sas_path))
                    self.slope_correction = max(0.0, 1.0 - sas)
                    rospy.loginfo(
                        f'[VelPlanner3D] auto-paired: slope_ax_scale={sas:.2f} '
                        f'-> slope_correction={self.slope_correction:.2f}')
                elif os.path.isfile(es_path):
                    self.slope_correction = 0.0
                    rospy.loginfo('[VelPlanner3D] partial meta -> slope_correction=0.0')
            except Exception as e:
                rospy.logwarn(f'[VelPlanner3D] auto-pair failed: {e} -> keep default')

        # auto-pair slope_correction with ggv meta if yaml does not specify
        if 'slope_correction' not in params:
            try:
                meta_dir = self._resolve_ggv_meta_dir()
                sas_path = os.path.join(meta_dir, 'slope_ax_scale.npy')
                es_path  = os.path.join(meta_dir, 'enable_slope.npy')
                if os.path.isfile(sas_path):
                    sas = float(np.load(sas_path))
                    self.slope_correction = max(0.0, 1.0 - sas)
                    rospy.loginfo(
                        f'[VelPlanner3D] auto-paired: slope_ax_scale={sas:.2f} '
                        f'-> slope_correction={self.slope_correction:.2f}')
                elif os.path.isfile(es_path):
                    self.slope_correction = 0.0
                    rospy.loginfo('[VelPlanner3D] partial meta -> slope_correction=0.0')
            except Exception as e:
                rospy.logwarn(f'[VelPlanner3D] auto-pair failed: {e} -> keep default')

    def _apply_params_to_ggv(self):
        """Apply current params to ggv/ax_max tables"""
        self.ggv[:,1] = self.a_x_max
        self.ggv[:,2] = self.a_y_max
        self.ax_max_machines[:,1] = self.ax_max_motor
        self.b_ax_max_machines[:,1] = self.ax_max_brake

    def _save_vel_planner_params(self):
        """Save current params to yaml file"""
        params = {
            'v_max': float(self.v_max),
            'ax_max_motor': float(self.ax_max_motor),
            'ax_max_brake': float(self.ax_max_brake),
            'dyn_model_exp': float(self.dyn_model_exp),
            'a_y_max': float(self.a_y_max),
            'a_x_max': float(self.a_x_max),
            'grip_scale_exp': float(self.grip_scale_exp),
            'slope_correction': float(self.slope_correction),
            'slope_brake_margin': float(self.slope_brake_margin),
            'slope_brake_vmax': float(self.slope_brake_vmax),
            # SG mu smoothing
            'smooth_mu': bool(self.smooth_mu),
            'mu_smooth_window': int(self.mu_smooth_window),
            'mu_smooth_polyorder': int(self.mu_smooth_polyorder),
            # SG kappa smoothing (vel-only)
            'smooth_kappa': bool(self.smooth_kappa),
            'kappa_smooth_window': int(self.kappa_smooth_window),
            'kappa_smooth_polyorder': int(self.kappa_smooth_polyorder),
        }
        with open(self.vel_planner_yaml, 'w') as f:
            yaml.dump(params, f, default_flow_style=False)
        rospy.loginfo(f"[VelPlanner3D] Saved params to {self.vel_planner_yaml}")

    def _dyn_reconfig_callback(self, config, level):
        """Dynamic reconfigure callback — update params on rqt change"""
        if not self.dyn_reconfig_initialized:
            # First callback is auto-called by Server init — skip to preserve existing speeds
            self.dyn_reconfig_initialized = True
            rospy.loginfo("[VelPlanner3D] Dynamic reconfigure initialized (no publish)")
            return config

        # Mark as changed so wpnts_callback recalculates (skip during init)
        if not getattr(self, '_init_phase', True):
            self.dyn_reconfig_changed = True

        self.v_max = config.v_max
        self.ax_max_motor = config.ax_max_motor
        self.ax_max_brake = config.ax_max_brake
        self.dyn_model_exp = config.dyn_model_exp
        self.a_y_max = config.a_y_max
        self.a_x_max = config.a_x_max
        self.grip_scale_exp = config.grip_scale_exp
        self.slope_correction = config.slope_correction
        self.slope_brake_margin = config.slope_brake_margin
        self.slope_brake_vmax = config.slope_brake_vmax
        # SG mu smoothing
        self.smooth_mu = config.smooth_mu
        self.mu_smooth_window = int(config.mu_smooth_window)
        self.mu_smooth_polyorder = int(config.mu_smooth_polyorder)
        # SG kappa smoothing (vel-only)
        self.smooth_kappa = config.smooth_kappa
        self.kappa_smooth_window = int(config.kappa_smooth_window)
        self.kappa_smooth_polyorder = int(config.kappa_smooth_polyorder)

        self._apply_params_to_ggv()

        if config.save_config:
            self._save_vel_planner_params()
            config.save_config = False

        if config.load_yaml:
            self._load_vel_planner_params()
            self._apply_params_to_ggv()
            config.v_max = self.v_max
            config.ax_max_motor = self.ax_max_motor
            config.ax_max_brake = self.ax_max_brake
            config.dyn_model_exp = self.dyn_model_exp
            config.a_y_max = self.a_y_max
            config.a_x_max = self.a_x_max
            config.grip_scale_exp = self.grip_scale_exp
            config.slope_correction = self.slope_correction
            config.slope_brake_margin = self.slope_brake_margin
            config.slope_brake_vmax = self.slope_brake_vmax
            # SG mu smoothing
            config.smooth_mu = self.smooth_mu
            config.mu_smooth_window = self.mu_smooth_window
            config.mu_smooth_polyorder = self.mu_smooth_polyorder
            # SG kappa smoothing (vel-only)
            config.smooth_kappa = self.smooth_kappa
            config.kappa_smooth_window = self.kappa_smooth_window
            config.kappa_smooth_polyorder = self.kappa_smooth_polyorder
            config.load_yaml = False
            rospy.loginfo("[VelPlanner3D] Reloaded from yaml")

        if config.save_csv:
            ggv_path = os.path.join(RosPack().get_path('stack_master'), 'config', self.racecar_version, "veh_dyn_info", "ggv.csv")
            ax_max_path = os.path.join(RosPack().get_path('stack_master'), 'config', self.racecar_version, "veh_dyn_info", "ax_max_machines.csv")
            b_ax_max_path = os.path.join(RosPack().get_path('stack_master'), 'config', self.racecar_version, "veh_dyn_info", "b_ax_max_machines.csv")
            self._save_modified_config(ggv_path, ax_max_path, b_ax_max_path)
            config.save_csv = False

        rospy.loginfo(f"[VelPlanner3D] Reconfig: v_max={self.v_max}, ax={self.a_x_max}, ay={self.a_y_max}, "
                      f"motor={self.ax_max_motor}, brake={self.ax_max_brake}, "
                      f"dyn_exp={self.dyn_model_exp}, grip_exp={self.grip_scale_exp}, "
                      f"slope_corr={self.slope_correction}, "
                      f"brake_margin={self.slope_brake_margin}m, brake_vmax={self.slope_brake_vmax}, "
                      f"smooth_mu={self.smooth_mu} (win={self.mu_smooth_window}, poly={self.mu_smooth_polyorder}), "
                      f"smooth_kappa={self.smooth_kappa} (win={self.kappa_smooth_window}, poly={self.kappa_smooth_polyorder})")
        return config

    def _save_modified_config(self, ggv_path, ax_max_path, b_ax_max_path):
        """
        Save modified ggv and ax_max_machines parameters to CSV files.
        Backup original files to 'original/' folder (only if not already backed up).
        """
        veh_dyn_dir = os.path.dirname(ggv_path)
        original_dir = os.path.join(veh_dyn_dir, 'original')

        if not os.path.exists(original_dir):
            os.makedirs(original_dir)
            rospy.loginfo(f"[VelPlanner3D] Created backup directory: {original_dir}")

        files_to_process = [
            (ggv_path, 'ggv.csv'),
            (ax_max_path, 'ax_max_machines.csv'),
            (b_ax_max_path, 'b_ax_max_machines.csv')
        ]

        # Backup originals (only if not already backed up)
        for file_path, filename in files_to_process:
            backup_path = os.path.join(original_dir, filename)
            backup_exists = os.path.exists(backup_path) and os.path.getsize(backup_path) > 0
            if not backup_exists:
                if os.path.exists(file_path):
                    shutil.copy2(file_path, backup_path)
                    rospy.loginfo(f"[VelPlanner3D] Backed up original: {filename} -> original/{filename}")
                else:
                    rospy.logwarn(f"[VelPlanner3D] Original file not found: {file_path}")
            else:
                rospy.loginfo(f"[VelPlanner3D] Original already backed up: {filename} (skipping)")

        # Save modified files
        try:
            np.savetxt(ggv_path, self.ggv, delimiter=',', fmt='%.6f',
                      header='v_mps,ax_max_mps2,ay_max_mps2')
            rospy.loginfo(f"[VelPlanner3D] Saved ggv.csv: ax_max={self.a_x_max}, ay_max={self.a_y_max}")
        except Exception as e:
            rospy.logerr(f"[VelPlanner3D] Failed to save ggv.csv: {e}")

        try:
            np.savetxt(ax_max_path, self.ax_max_machines, delimiter=',', fmt='%.6f',
                      header='vx_mps,ax_max_machines_mps2')
            rospy.loginfo(f"[VelPlanner3D] Saved ax_max_machines.csv: ax_max_motor={self.ax_max_motor}")
        except Exception as e:
            rospy.logerr(f"[VelPlanner3D] Failed to save ax_max_machines.csv: {e}")

        try:
            np.savetxt(b_ax_max_path, self.b_ax_max_machines, delimiter=',', fmt='%.6f',
                      header='vx_mps,ax_max_machines_mps2')
            rospy.loginfo(f"[VelPlanner3D] Saved b_ax_max_machines.csv: ax_max_brake={self.ax_max_brake}")
        except Exception as e:
            rospy.logerr(f"[VelPlanner3D] Failed to save b_ax_max_machines.csv: {e}")

    # SG-smooth kappa for vel calc only (path kappa_radpm is preserved)
    # Reason: SP raceline csv carries kappa noise spikes near the closed-loop
    # seam (i=0 ~ i=N-1) and at sharp transitions, which makes v_kappa = sqrt(ay/|k|)
    # plummet at otherwise-easy points. Keep the geometric path untouched,
    # smooth a *copy* of kappa just for the velocity profile.
    # Toggleable via rqt (smooth_kappa).
    def _smooth_kappa(self, kappa):
        if not self.smooth_kappa or not HAS_SCIPY_SG:
            return kappa
        win = self.kappa_smooth_window
        if win % 2 == 0:
            win += 1
        if len(kappa) < win:
            return kappa
        poly = max(1, min(self.kappa_smooth_polyorder, win - 1))
        # closed loop wrap pad so SG sees continuity at the seam
        pad = win // 2
        k_pad = np.concatenate([kappa[-pad:], kappa, kappa[:pad]])
        return savgol_filter(k_pad, win, poly)[pad:-pad]

    def _build_track_3d_params(self, wpnts):
        """
        Extract slope, compute angular rates via Euler->body Jacobian,
        build track_3d_params dict from waypoint mu_rad.
        """
        n = len(wpnts)
        mu = np.array([wp.mu_rad for wp in wpnts])
        kappa = np.array([wp.kappa_radpm for wp in wpnts])
        s = np.array([wp.s_m for wp in wpnts])

        # Check if 3D data exists
        if np.all(np.abs(mu) < 1e-8):
            return None, mu  # flat track, no 3D params needed

        slope = mu
        dmu_ds = np.gradient(slope, s)

        # phi = 0 for no-bank tracks
        phi = np.zeros(n)

        # Angular rates via Euler->body Jacobian
        # J = [[1, 0, -sin(mu)], [0, cos(phi), cos(mu)*sin(phi)], [0, -sin(phi), cos(mu)*cos(phi)]]
        # omega = J . [dphi/ds, dmu/ds, dtheta/ds], dtheta/ds = kappa, dphi/ds ~ 0
        dphi_ds = np.gradient(phi, s)
        dtheta_ds = kappa

        omega_x = dphi_ds - np.sin(mu) * dtheta_ds
        omega_y = np.cos(phi) * dmu_ds + np.cos(mu) * np.sin(phi) * dtheta_ds
        omega_z = -np.sin(phi) * dmu_ds + np.cos(mu) * np.cos(phi) * dtheta_ds

        d_omega_x = np.gradient(omega_x, s)
        d_omega_y = np.gradient(omega_y, s)
        d_omega_z = np.gradient(omega_z, s)

        track_3d_params = {
            'mu': mu,
            'phi': phi,
            'omega_x': omega_x,
            'omega_y': omega_y,
            'omega_z': omega_z,
            'd_omega_x': d_omega_x,
            'd_omega_y': d_omega_y,
            'd_omega_z': d_omega_z,
            'dmu_ds': dmu_ds,
            'h': self.h_cog,
            'slope_correction': self.slope_correction,
            'slope_brake_margin': self.slope_brake_margin,
            'slope_brake_vmax': self.slope_brake_vmax,
        }

        return track_3d_params, slope

    def wpnts_callback(self, msg):
        wpnts = msg.wpnts

        # Only recalculate when rqt vel_planner or friction has been changed
        if not self.dyn_reconfig_changed:
            return
        self.dyn_reconfig_changed = False

        kappa = np.array([wp.kappa_radpm for wp in wpnts])
        el_lengths = 0.1 * np.ones(len(kappa))
        track_3d_params, slope = self._build_track_3d_params(wpnts)
        # kappa for velocity calc only — path kappa_radpm in wpnts is untouched
        kappa_for_vel = self._smooth_kappa(kappa)

        # 3D velocity profile
        vx_profile = calc_vel_profile(ggv=self.ggv,
                                      ax_max_machines=self.ax_max_machines,
                                      b_ax_max_machines=self.b_ax_max_machines,
                                      v_max=self.v_max,
                                      kappa=kappa_for_vel,
                                      el_lengths=el_lengths,
                                      closed=True,
                                      filt_window=self.filt_window,
                                      dyn_model_exp=self.dyn_model_exp,
                                      drag_coeff=self.drag_coeff,
                                      m_veh=self.m_veh,
                                      slope=slope,
                                      track_3d_params=track_3d_params,
                                      grip_scale_exp=self.grip_scale_exp,
                                      # SG mu smoothing toggle
                                      smooth_mu=self.smooth_mu,
                                      mu_smooth_window=self.mu_smooth_window,
                                      mu_smooth_polyorder=self.mu_smooth_polyorder)

        for i in range(len(vx_profile)):
            wpnts[i].vx_mps = vx_profile[i]

        vx_profile_opt_cl = np.append(vx_profile, vx_profile[0])
        ax_profile = tph.calc_ax_profile.calc_ax_profile(vx_profile=vx_profile_opt_cl,
                                                         el_lengths=el_lengths,
                                                         eq_length_output=False)
        for i in range(len(ax_profile)):
            wpnts[i].ax_mps2 = ax_profile[i]

        msg.wpnts = wpnts
        print("NEW 3D Vel Profile Pub")
        self.glb_wpnts_pub.publish(msg)

        # 2D velocity profile (no 3D correction, with friction) for comparison markers
        # reuse smoothed kappa here too — same vel-calc principle
        vx_profile_2d = calc_vel_profile(ggv=self.ggv,
                                          ax_max_machines=self.ax_max_machines,
                                          b_ax_max_machines=self.b_ax_max_machines,
                                          v_max=self.v_max,
                                          kappa=kappa_for_vel,
                                          el_lengths=el_lengths,
                                          closed=True,
                                          filt_window=self.filt_window,
                                          dyn_model_exp=self.dyn_model_exp,
                                          drag_coeff=self.drag_coeff,
                                          m_veh=self.m_veh)
        self._publish_2d_vel_markers(wpnts, vx_profile_2d)

    def smart_static_wpnts_callback(self, msg):
        """Callback for smart_static_otwpnts — velocity profile with 3D correction"""
        if not self.dyn_reconfig_changed:
            return
        wpnts = msg.wpnts

        kappa = np.array([wp.kappa_radpm for wp in wpnts])
        el_lengths = 0.1 * np.ones(len(kappa))

        track_3d_params, slope = self._build_track_3d_params(wpnts)
        # kappa for velocity calc only — path kappa_radpm in wpnts is untouched
        kappa_for_vel = self._smooth_kappa(kappa)

        vx_profile = calc_vel_profile(ggv=self.ggv,
                                      ax_max_machines=self.ax_max_machines,
                                      b_ax_max_machines=self.b_ax_max_machines,
                                      v_max=self.v_max,
                                      kappa=kappa_for_vel,
                                      el_lengths=el_lengths,
                                      closed=True,
                                      filt_window=self.filt_window,
                                      dyn_model_exp=self.dyn_model_exp,
                                      drag_coeff=self.drag_coeff,
                                      m_veh=self.m_veh,
                                      slope=slope,
                                      track_3d_params=track_3d_params,
                                      grip_scale_exp=self.grip_scale_exp,
                                      # SG mu smoothing toggle
                                      smooth_mu=self.smooth_mu,
                                      mu_smooth_window=self.mu_smooth_window,
                                      mu_smooth_polyorder=self.mu_smooth_polyorder)

        for i in range(len(vx_profile)):
            wpnts[i].vx_mps = vx_profile[i]

        vx_profile_opt_cl = np.append(vx_profile, vx_profile[0])
        ax_profile = tph.calc_ax_profile.calc_ax_profile(vx_profile=vx_profile_opt_cl,
                                                         el_lengths=el_lengths,
                                                         eq_length_output=False)

        for i in range(len(ax_profile)):
            wpnts[i].ax_mps2 = ax_profile[i]

        msg.wpnts = wpnts
        msg.header.stamp = rospy.Time.now()
        print("NEW 3D Smart Static Vel Profile Pub")
        self.smart_static_wpnts_pub.publish(msg)

    def _publish_2d_vel_markers(self, wpnts, vx_2d):
        """Publish 2D velocity profile as yellow cylinder markers (same style as vel_markers)"""
        VEL_SCALE = 0.1317  # same scale factor as vel_markers
        marker_array = MarkerArray()
        for i, wp in enumerate(wpnts):
            height = vx_2d[i] * VEL_SCALE
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = ""
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = wp.x_m
            marker.pose.position.y = wp.y_m
            marker.pose.position.z = height * 0.5
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = height
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.5
            marker_array.markers.append(marker)
        self.vel_markers_2d_pub.publish(marker_array)


if __name__ == '__main__':
    rospy.init_node("global_velplanner_3d")
    simple_mux = VelocityPlanner()
    rospy.spin()
