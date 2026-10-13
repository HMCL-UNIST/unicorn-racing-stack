#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.interpolate import UnivariateSpline
import rospy
from std_msgs.msg import Bool, Float32, Float32MultiArray, Header
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from tf.transformations import euler_from_quaternion
from rospkg import RosPack

import trajectory_planning_helpers as tph
from ccma import CCMA
from f110_msgs.msg import (
    Wpnt, WpntArray, Obstacle, ObstacleArray,
    OTWpntArray, OpponentTrajectory, OppWpnt, BehaviorStrategy,
)
from frenet_converter.frenet_converter import FrenetConverter

from sqp_casadi import CasadiSQPSolver, SQPProblem
from warm_start import shift_solution
from velocity_profiler import VelocityProfiler
from abort_checker import AbortChecker, AbortConfig, AbortReason
# nlp velocity mode
from nlp_velocity import solve_velocity_nlp


class FastSQPPlannerNode:

    def __init__(self):
        rospy.init_node('fast_sqp_planner_node')

        # ---- params -------------------------------------------------------
        pp = rospy.get_param
        self.rate_hz = pp('~rate_hz', 20.0)
        self.rate = rospy.Rate(self.rate_hz)
        self.dt = 1.0 / self.rate_hz

        self.racecar_version = os.environ.get('CAR_NAME', pp('~racecar_version', 'SRX1'))

        # SQP / shape params (same defaults as the SQP planner)
        self.lookahead = pp('~lookahead', 15.0)
        self.width_car = pp('~width_car', 0.30)
        self.evasion_dist = pp('~evasion_dist', 0.65)
        self.spline_bound_mindist = pp('~spline_bound_mindist', 0.20)
        self.avoidance_resolution = int(pp('~avoidance_resolution', 20))
        self.back_to_raceline_after = pp('~back_to_raceline_after', 5.0)
        self.obs_traj_tresh = pp('~obs_traj_tresh', 1.5)
        self.prediction_horizon_s = pp('~prediction_horizon_s', 1.0)
        # obs_min += sigma_k * sqrt(d_var)
        self.obs_sigma_k = pp('~obs_sigma_k', 2.0)

        # Rolling-horizon specific
        self.lambda_reg = pp('~regularization/lambda_reg', 1.0)
        self.homotopy_lock = bool(pp('~regularization/homotopy_lock', True))

        # SQP cost weights (tune via yaml to shape overtake path)
        self.lambda_smooth = pp('~weights/lambda_smooth', 500.0)
        self.lambda_start_heading = pp('~weights/lambda_start_heading', 1000.0)
        self.lambda_apex_bias = pp('~weights/lambda_apex_bias', 0.0)
        self.lambda_side = pp('~weights/lambda_side', 300.0)
        self.lambda_jerk = pp('~weights/lambda_jerk', 0.0)
        # online rolling mode
        self.online_mode = bool(pp('~rolling_mode', False))
        self.T_horizon = pp('~T_horizon', 1.0)
        self.s_min_horizon = pp('~s_min_horizon', 3.0)
        self.lambda_term = pp('~weights/lambda_term', 0.0)
        # all-soft penalty weights
        self.lambda_obs = pp('~weights/lambda_obs', 10000.0)
        self.lambda_kappa = pp('~weights/lambda_kappa', 5000.0)
        self.lambda_gg = pp('~weights/lambda_gg', 5000.0)
        self.lambda_near_reg = pp('~weights/lambda_near_reg', 0.0)
        self.lambda_obs_smooth = pp('~weights/lambda_obs_smooth', 5000.0)
        self.near_knots_K = int(pp('~near_knots_K', 0))
        self.obs_ramp_knots = int(pp('~obs_ramp_knots', 5))
        #post-solve freeze — lock near-term path to previous solution
        self.freeze_distance_m = pp('~freeze_distance_m', 3.0)
        self.freeze_blend_m = pp('~freeze_blend_m', 2.0)
        # joint velocity optimization (optional)
        self.optimize_velocity = bool(pp('~optimize_velocity', False))
        self.lambda_progress = pp('~weights/lambda_progress', 1.0)
        # velocity_mode selector (2_5d / 3d / nlp) — live via rqt dynamic_reconfigure
        self.velocity_mode = pp('~velocity_mode', '2_5d')
        # GGV data for velocity optimization
        self.ggv_data = None  # loaded in _load_ggv()

        # real C++ FBGA params (velocity_mode='3d')
        self.fbga_enable        = bool(pp('~fbga_enable', False))
        self.fbga_bin           = pp('~fbga_bin', '')
        self.fbga_gg_bin        = pp('~fbga_gg_bin', '')
        self.fbga_params_yml    = pp('~fbga_params_yml', '')
        self.fbga_bridge_effect = float(pp('~bridge_effect', 1.0))
        self.fbga_enable_mu     = bool(pp('~fbga_enable_mu', True))
        self.fbga_max_iter      = int(pp('~fbga_max_iter', 5))   # local cycle → small
        self.fbga_tol           = float(pp('~fbga_tol', 0.05))
        self.fbga_alpha         = float(pp('~fbga_alpha', 0.5))
        self._fbga_runner       = None  # built lazily in _init_fbga_runner()

        # Abort
        self.abort = AbortChecker(AbortConfig(
            performance_margin_s=pp('~abort/performance_margin_s', 0.1),
            consecutive_cycles=int(pp('~abort/consecutive_cycles', 3)),
            cooldown_s=pp('~abort/cooldown_s', 2.0),
            safety_sigma_multiplier=pp('~abort/safety_sigma_multiplier', 3.0),
        ))
        # hold-last-valid fallback
        self.hold_last_valid = bool(pp('~fallback/hold_last_valid', False))
        self.max_fail_hold_cycles = int(pp('~fallback/max_fail_hold_cycles', 5))
        self.fail_streak = 0

        self.measure = rospy.get_param('/measure', False)

        # ---- state caches -------------------------------------------------
        self.frenet_state = Odometry()
        self.cur_x = 0.0
        self.cur_y = 0.0
        self.cur_v = 0.0
        self.cur_yaw = 0.0
        self.cur_s = 0.0
        self.current_d = 0.0

        self.scaled_wpnts = None
        self.scaled_wpnts_msg = WpntArray()
        self.scaled_vmax = None
        self.scaled_max_idx = None
        self.scaled_max_s = None
        self.scaled_delta_s = None

        self.wpnts_updated = None
        self.max_s_updated = None
        self.max_idx_updated = None

        self.obs = ObstacleArray()
        self.obs_perception = ObstacleArray()
        self.obs_predict = ObstacleArray()
        self.avoid_static_obs = True

        self.opponent_waypoints = []
        self.max_opp_idx = None
        self.opponent_wpnts_sm = None
        self.opponent_wpnts_d = None
        self.opponent_wpnts_dvar = None
        self.opponent_wpnts_vs = None

        self.ot_section_check = False
        self.smart_static_active = False
        self.local_wpnts = None

        self.ccma = CCMA(w_ma=10, w_cc=3)
        self.global_waypoints = None
        self.converter = None

        # Solver + rolling state
        self.solver = CasadiSQPSolver()
        self._pending_msg = None
        self.prev_d = None      # np.ndarray
        self.prev_s = None      # np.ndarray
        self.prev_v = None      # np.ndarray (velocity warm-start)
        self.last_ot_side = ''  # 'left' | 'right'
        self.last_desired_side = 'any'
        # previous solve latency (for adaptive start_av shift)
        self.last_solve_ms = 0.0

        # Velocity profiler (loads ggv CSVs once)
        cfg_dir = os.path.join(RosPack().get_path('stack_master'), 'config')
        self.vp = VelocityProfiler(cfg_dir, self.racecar_version)

        # ---- topics -------------------------------------------------------
        # Subscribers — same as the SQP planner
        rospy.Subscriber('/tracking/obstacles', ObstacleArray, self._obs_perception_cb)
        rospy.Subscriber('/opponent_prediction/obstacles', ObstacleArray, self._obs_prediction_cb)
        rospy.Subscriber('/car_state/odom_frenet', Odometry, self._state_frenet_cb)
        rospy.Subscriber('/car_state/odom', Odometry, self._state_cartesian_cb)
        rospy.Subscriber('/global_waypoints_scaled', WpntArray, self._scaled_wpnts_cb)
        rospy.Subscriber('/behavior_strategy', BehaviorStrategy, self._behavior_cb)
        rospy.Subscriber('/global_waypoints', WpntArray, self._gb_cb)
        rospy.Subscriber('/global_waypoints_updated', WpntArray, self._updated_wpnts_cb)
        rospy.Subscriber('/opponent_trajectory', OpponentTrajectory, self._opp_traj_cb)
        rospy.Subscriber('/ot_section_check', Bool, self._ot_section_cb)
        rospy.Subscriber('/planner/avoidance/smart_static_active', Bool, self._smart_static_cb)

        # Publishers — remapped by launch onto /planner/avoidance/*
        self.evasion_pub = rospy.Publisher('/planner/fast_sqp_planner/otwpnts', OTWpntArray, queue_size=10)
        self.mrks_pub = rospy.Publisher('/planner/fast_sqp_planner/markers', MarkerArray, queue_size=10)
        self.merger_pub = rospy.Publisher('/planner/fast_sqp_planner/merger', Float32MultiArray, queue_size=10)
        self.debug_pub = rospy.Publisher('/planner/fast_sqp_planner/debug', Float32MultiArray, queue_size=10)
        # diagnostic: data = [obs_center_min, obs_center_max, obs_center_std,
        #                     n_constraints, n_considered, cur_s, start_av, end_av,
        #                     d_init_std, d_opt_std]
        self.diag_pub = rospy.Publisher('/planner/fast_sqp_planner/diag', Float32MultiArray, queue_size=10)
        # collision flag (ego-obstacle physical overlap)
        self.collision_pub = rospy.Publisher('/planner/fast_sqp_planner/collision', Bool, queue_size=1)
        # signal state_machine that fast_sqp_planner is active
        self.active_pub = rospy.Publisher('/fast_sqp_planner/active', Bool, queue_size=1, latch=True)
        if self.measure:
            self.measure_pub = rospy.Publisher('/planner/fast_sqp_planner/latency', Float32, queue_size=10)

        self.converter = self._init_converter()
        self._load_ggv()  # GGV for FBGA velocity profiling
        # real C++ FBGA runner (only if fbga_enable=true)
        self._init_fbga_runner()

        # final velocity profile uniform scale (rqt-tunable, applied at publish only)
        self.v_scale = float(pp('~v_scale', 1.0))

        rospy.loginfo('[FastSQP] initialized. rate=%.1fHz car=%s lambda_reg=%.3f',
                      self.rate_hz, self.racecar_version, self.lambda_reg)

        # dynamic_reconfigure server for rqt velocity_mode switching
        from dynamic_reconfigure.server import Server as DynServer
        from fast_sqp_planner.cfg import FastSQPPlannerConfig
        self._dyn_srv = DynServer(FastSQPPlannerConfig, self._dyn_reconfigure_cb)

        # state_machine hyst override (online mode only)
        self._state_dr_client = None
        self._state_dr_original_hyst = None
        if self.online_mode:
            self._override_state_hyst_timer()
            rospy.on_shutdown(self._restore_state_hyst_timer)

    # dynamic_reconfigure callback — live parameter tuning
    def _dyn_reconfigure_cb(self, config, level):
        # reload GGV from disk when checkbox checked
        if getattr(config, 'reload_ggv', False):
            rospy.loginfo('[FastSQP] reload_ggv triggered — reloading from disk')
            self._load_ggv()        # 2D .npy + GGManager
            self.vp.reload()        # 2D ggv.csv + ax/b_ax_max_machines
            # invalidate warm-start so next tick is not pinned to prev-GGV solution
            self.prev_d = None
            self.prev_s = None
            self.prev_v = None
            config.reload_ggv = False
        if config.velocity_mode != self.velocity_mode:
            rospy.loginfo('[FastSQP] velocity_mode: %s → %s',
                          self.velocity_mode, config.velocity_mode)
        self.velocity_mode      = config.velocity_mode
        self.lookahead          = config.lookahead
        self.evasion_dist       = config.evasion_dist
        self.prediction_horizon_s = config.prediction_horizon_s
        self.obs_sigma_k        = config.obs_sigma_k
        self.T_horizon          = config.T_horizon
        self.s_min_horizon      = config.s_min_horizon
        self.lambda_reg         = config.lambda_reg
        self.lambda_smooth      = config.lambda_smooth
        self.lambda_jerk        = config.lambda_jerk
        self.lambda_obs         = config.lambda_obs
        self.lambda_obs_smooth  = config.lambda_obs_smooth
        self.lambda_kappa       = config.lambda_kappa
        self.lambda_gg          = config.lambda_gg
        self.lambda_near_reg    = config.lambda_near_reg
        self.lambda_side        = config.lambda_side
        self.lambda_start_heading = config.lambda_start_heading
        self.lambda_term        = config.lambda_term
        self.near_knots_K       = config.near_knots_K
        self.obs_ramp_knots     = config.obs_ramp_knots
        if hasattr(config, 'v_scale'):
            self.v_scale = float(config.v_scale)
        return config

    # state_machine hyst override helpers
    def _override_state_hyst_timer(self):
        """Force state_machine to refresh cached overtake path every tick."""
        try:
            import dynamic_reconfigure.client
            # 3d_state_machine_node.py uses /dyn_planners_statemachine/...
            ns = '/dyn_planners_statemachine/dynamic_avoidance_planner'
            self._state_dr_client = dynamic_reconfigure.client.Client(
                ns, timeout=3.0)
            self._state_dr_original_hyst = rospy.get_param(
                ns + '/hyst_timer_sec', None)
            self._state_dr_client.update_configuration(
                {'hyst_timer_sec': 0.01})
            rospy.loginfo(
                '[FastSQP] state_machine hyst_timer_sec -> 0.01 '
                '(orig=%s)', str(self._state_dr_original_hyst))
        except Exception as e:   # noqa: BLE001
            rospy.logwarn('[FastSQP] dyn_recfg override failed: %s', e)

    def _restore_state_hyst_timer(self):
        """Restore state_machine hyst_timer_sec on node shutdown."""
        try:
            if (self._state_dr_client is not None
                    and self._state_dr_original_hyst is not None):
                self._state_dr_client.update_configuration(
                    {'hyst_timer_sec': self._state_dr_original_hyst})
                rospy.loginfo(
                    '[FastSQP] restored hyst_timer_sec=%s',
                    str(self._state_dr_original_hyst))
        except Exception:   # noqa: BLE001
            pass

    # =====================================================================
    # Callbacks (mirror the SQP planner)
    # =====================================================================
    def _obs_perception_cb(self, data: ObstacleArray):
        self.obs_perception = data
        self.obs.header = data.header
        self.obs.obstacles = list(data.obstacles) + self.obs_predict.obstacles
        rospy.loginfo_throttle(2.0,
            '[FastSQP] perc_cb: raw=%d pred=%d total=%d',
            len(data.obstacles),
            len(self.obs_predict.obstacles), len(self.obs.obstacles))

    def _obs_prediction_cb(self, data: ObstacleArray):
        self.obs_predict = data
        self.obs = self.obs_predict
        if self.avoid_static_obs:
            self.obs.obstacles = self.obs.obstacles + self.obs_perception.obstacles
        # detailed obstacle type logging
        n_pred_static = sum(1 for o in data.obstacles if o.is_static)
        n_pred_dyn = len(data.obstacles) - n_pred_static
        rospy.loginfo_throttle(2.0,
            '[FastSQP] pred_cb: n_pred=%d (static=%d dyn=%d) '
            'n_perc_static=%d n_total=%d',
            len(data.obstacles), n_pred_static, n_pred_dyn,
            len(self.obs_perception.obstacles),
            len(self.obs.obstacles))

    def _state_frenet_cb(self, data: Odometry):
        self.frenet_state = data
        q = data.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.cur_yaw = yaw

    def _state_cartesian_cb(self, msg: Odometry):
        self.cur_x = msg.pose.pose.position.x
        self.cur_y = msg.pose.pose.position.y
        self.cur_v = msg.twist.twist.linear.x

    def _gb_cb(self, data: WpntArray):
        self.global_waypoints = np.array([[w.x_m, w.y_m, w.z_m] for w in data.wpnts])

    def _scaled_wpnts_cb(self, data: WpntArray):
        self.scaled_wpnts = np.array([[w.s_m, w.d_m] for w in data.wpnts])
        self.scaled_wpnts_msg = data
        vmax = np.max(np.array([w.vx_mps for w in data.wpnts]))
        if self.scaled_vmax != vmax:
            self.scaled_vmax = float(vmax)
            self.scaled_max_idx = data.wpnts[-1].id
            self.scaled_max_s = data.wpnts[-1].s_m
            self.scaled_delta_s = data.wpnts[1].s_m - data.wpnts[0].s_m
        # fallback: if /global_waypoints_updated never publishes (no publisher
        # in this launch), reuse scaled as the "updated" waypoints
        if self.wpnts_updated is None:
            self.wpnts_updated = data.wpnts[:-1]
            self.max_s_updated = self.wpnts_updated[-1].s_m
            self.max_idx_updated = self.wpnts_updated[-1].id

    def _updated_wpnts_cb(self, data: WpntArray):
        self.wpnts_updated = data.wpnts[:-1]
        self.max_s_updated = self.wpnts_updated[-1].s_m
        self.max_idx_updated = self.wpnts_updated[-1].id

    def _behavior_cb(self, data: BehaviorStrategy):
        self.local_wpnts = np.array([[w.s_m, w.d_m] for w in data.local_wpnts])

    def _opp_traj_cb(self, data: OpponentTrajectory):
        self.opponent_waypoints = data.oppwpnts
        if len(data.oppwpnts) == 0:
            return
        self.max_opp_idx = len(data.oppwpnts) - 1
        new_sm = np.array([w.s_m for w in data.oppwpnts])
        new_d = np.array([w.d_m for w in data.oppwpnts])
        # EMA on opponent d to suppress GP jitter
        alpha = 0.7
        if self.opponent_wpnts_d is not None and len(self.opponent_wpnts_d) == len(new_d):
            self.opponent_wpnts_d = alpha * new_d + (1.0 - alpha) * self.opponent_wpnts_d
        else:
            self.opponent_wpnts_d = new_d
        self.opponent_wpnts_sm = new_sm
        self.opponent_wpnts_dvar = np.array([w.d_var for w in data.oppwpnts])
        self.opponent_wpnts_vs = np.array([w.proj_vs_mps for w in data.oppwpnts])

    def _ot_section_cb(self, data: Bool):
        self.ot_section_check = data.data

    def _smart_static_cb(self, data: Bool):
        self.smart_static_active = data.data

    # =====================================================================
    # Utilities
    # =====================================================================
    def _init_converter(self) -> FrenetConverter:
        rospy.wait_for_message('/global_waypoints', WpntArray)
        conv = FrenetConverter(self.global_waypoints[:, 0],
                               self.global_waypoints[:, 1],
                               self.global_waypoints[:, 2])
        rospy.loginfo('[FastSQP] FrenetConverter initialized')
        return conv

    def _load_ggv(self, vehicle_name: str = 'rc_car_10th_latest') -> None:
        """Load precomputed 3D GGV (velocity_frame) for NLP velocity opt."""
        # resolve relative to this file: fast_sqp_planner/src/ → ../../3d_gb_optimizer/global_line/
        _this_dir = os.path.dirname(os.path.abspath(__file__))
        gg_base = os.path.join(
            _this_dir, '..', '..', '3d_gb_optimizer', 'global_line',
            'data', 'gg_diagrams', vehicle_name, 'velocity_frame')
        if not os.path.isdir(gg_base):
            rospy.logwarn('[FastSQP] GGV dir not found: %s', gg_base)
            return
        self.ggv_data = {
            'v_list': np.load(os.path.join(gg_base, 'v_list.npy')),
            'g_list': np.load(os.path.join(gg_base, 'g_list.npy')),
            'ax_max': np.load(os.path.join(gg_base, 'ax_max.npy')),
            'ax_min': np.load(os.path.join(gg_base, 'ax_min.npy')),
            'ay_max': np.load(os.path.join(gg_base, 'ay_max.npy')),
        }
        rospy.loginfo('[FastSQP] GGV loaded from %s (v=[%.1f..%.1f], g=%s)',
                      gg_base, self.ggv_data['v_list'][0],
                      self.ggv_data['v_list'][-1],
                      str(self.ggv_data['g_list']))
        # load GGManager for nlp velocity mode (CasADi acc_interpolator)
        self.gg_manager = None
        try:
            _gb_src = os.path.join(
                _this_dir, '..', '..', '3d_gb_optimizer', 'global_line', 'src')
            if _gb_src not in sys.path:
                sys.path.insert(0, _gb_src)
            from ggManager import GGManager
            self.gg_manager = GGManager(gg_path=gg_base, gg_margin=0.0)
            rospy.loginfo('[FastSQP] GGManager loaded (V_max=%.1f)',
                          self.gg_manager.V_max)
        except Exception as exc:
            rospy.logwarn('[FastSQP] GGManager load failed: %s', exc)

    # real C++ FBGA runner init (only when fbga_enable=true)
    def _init_fbga_runner(self) -> None:
        if not self.fbga_enable:
            return
        _this_dir = os.path.dirname(os.path.abspath(__file__))
        # default paths (mirror fbga_velocity_planner.py: rc_car_10th_latest)
        race_stack = os.path.abspath(os.path.join(_this_dir, '..', '..', '..'))
        if not self.fbga_bin:
            self.fbga_bin = os.path.join(
                race_stack, 'f110_utils', 'libs', 'FBGA', 'bin',
                'GIGI_test_unicorn.exe')
        if not self.fbga_gg_bin:
            self.fbga_gg_bin = os.path.join(
                race_stack, 'planner', '3d_gb_optimizer', 'global_line', 'data',
                'gg_diagrams', 'rc_car_10th_latest', 'velocity_frame', 'gg.bin')
        if not self.fbga_params_yml:
            self.fbga_params_yml = os.path.join(
                race_stack, 'planner', '3d_gb_optimizer', 'global_line', 'data',
                'vehicle_params', 'params_rc_car_10th_latest.yml')
        try:
            from fbga_runner import FBGARunner
            self._fbga_runner = FBGARunner(
                fbga_bin=self.fbga_bin,
                gg_bin=self.fbga_gg_bin,
                params_yml=self.fbga_params_yml,
                bridge_effect=self.fbga_bridge_effect,
                enable_mu=self.fbga_enable_mu,
                alpha=self.fbga_alpha,
                max_iter=self.fbga_max_iter,
                tol=self.fbga_tol,
                logger=rospy.loginfo,
            )
            rospy.loginfo('[FastSQP] FBGA runner ready')
        except Exception as exc:
            rospy.logwarn(
                '[FastSQP] FBGA runner init failed; velocity_mode=3d will fallback: %s',
                exc)
            self._fbga_runner = None

    # add optional g_values override for 3d mode g_tilde iteration
    def _ggv_lookup(self, v_arr: np.ndarray, mu_arr: np.ndarray,
                    g_values: np.ndarray | None = None):
        """Per-knot ax_max, ay_max from precomputed GGV.
        g_values: if given, use as effective-g directly (skips 9.81*cos(mu))
        """
        if self.ggv_data is None:
            n = len(v_arr)
            return np.full(n, 5.0), np.full(n, 4.5)
        from scipy.interpolate import RegularGridInterpolator
        v_list = self.ggv_data['v_list']
        g_list = self.ggv_data['g_list']
        g_eff = g_values if g_values is not None else 9.81 * np.cos(mu_arr)
        v_c = np.clip(v_arr, v_list[0], v_list[-1])
        g_c = np.clip(g_eff, g_list[0], g_list[-1])
        ax_interp = RegularGridInterpolator(
            (v_list, g_list), self.ggv_data['ax_max'], bounds_error=False, fill_value=None)
        ay_interp = RegularGridInterpolator(
            (v_list, g_list), self.ggv_data['ay_max'], bounds_error=False, fill_value=None)
        pts = np.column_stack([v_c, g_c])
        return ax_interp(pts), ay_interp(pts)

    def _ggv_ax_at_knots(self, idxs: np.ndarray) -> np.ndarray:
        """Per-knot ax_max from GGV at raceline speed + track slope."""
        if not self.optimize_velocity or self.ggv_data is None:
            return np.full(len(idxs), 5.0)
        v_arr = np.array([self.scaled_wpnts_msg.wpnts[i].vx_mps for i in idxs])
        mu_arr = np.array([getattr(self.scaled_wpnts_msg.wpnts[i], 'mu_rad', 0.0) for i in idxs])
        ax, _ = self._ggv_lookup(v_arr, mu_arr)
        return ax

    def _ggv_ay_at_knots(self, idxs: np.ndarray) -> np.ndarray:
        """Per-knot ay_max from GGV at raceline speed + track slope."""
        if not self.optimize_velocity or self.ggv_data is None:
            return np.full(len(idxs), 4.5)
        v_arr = np.array([self.scaled_wpnts_msg.wpnts[i].vx_mps for i in idxs])
        mu_arr = np.array([getattr(self.scaled_wpnts_msg.wpnts[i], 'mu_rad', 0.0) for i in idxs])
        _, ay = self._ggv_lookup(v_arr, mu_arr)
        return ay

    def _more_space(self, obstacle: Obstacle, gb_wpnts, gb_idxs):
        # use track boundaries at obs s_center + fixed car width (stable)
        idx_at_obs = int(np.abs(self.scaled_wpnts[:, 0]
                                - obstacle.s_center % self.scaled_max_s).argmin())
        w = gb_wpnts[idx_at_obs]
        track_left = w.d_left
        track_right = w.d_right  # positive distance from centerline
        obs_d = obstacle.d_center
        obs_hw = self.width_car / 2.0  # fixed half-width

        left_gap = track_left - (obs_d + obs_hw)
        right_gap = (obs_d - obs_hw) + track_right
        min_space = self.evasion_dist + self.spline_bound_mindist
        if right_gap > min_space and left_gap < min_space:
            apex = obs_d - obs_hw - self.evasion_dist
            return 'right', min(apex, 0.0)
        if left_gap > min_space and right_gap < min_space:
            apex = obs_d + obs_hw + self.evasion_dist
            return 'left', max(apex, 0.0)
        cand_l = obs_d + obs_hw + self.evasion_dist
        cand_r = obs_d - obs_hw - self.evasion_dist
        # both sides feasible — prefer inside of curve
        kappa_at_obs = gb_wpnts[idx_at_obs].kappa_radpm
        if abs(kappa_at_obs) > 0.05:
            if kappa_at_obs > 0:  # right turn → inside = right
                return 'right', min(cand_r, 0.0)
            return 'left', max(cand_l, 0.0)
        if left_gap >= right_gap:
            return 'left', max(cand_l, 0.0)
        return 'right', min(cand_r, 0.0)

    @staticmethod
    def _group_objects(obstacles):
        agg = deepcopy(obstacles[0])
        for o in obstacles:
            agg.d_left = max(agg.d_left, o.d_left)
            agg.d_right = min(agg.d_right, o.d_right)
            agg.s_start = min(agg.s_start, o.s_start)
            agg.s_end = max(agg.s_end, o.s_end)
        agg.s_center = (agg.s_start + agg.s_end) / 2.0
        return agg

    def _clear_markers(self):
        mrks = MarkerArray()
        del_mrk = Marker(header=Header(stamp=rospy.Time.now()))
        del_mrk.ns = 'rolling_path'
        del_mrk.action = Marker.DELETEALL
        mrks.markers = [del_mrk]
        self.mrks_pub.publish(mrks)

    def _republish_shifted_prev(self):
        """Shift prev solution toward GB raceline and re-publish. Stop when close enough."""
        cur_s = float(self.frenet_state.pose.pose.position.x)
        cur_d = float(self.frenet_state.pose.pose.position.y)
        end_s = float(self.prev_s[-1])
        if end_s <= cur_s or self.scaled_delta_s is None:
            self.prev_d = None
            self.prev_s = None
            return
        n = max(20, int((end_s - cur_s) / self.scaled_delta_s))
        s_new = np.linspace(cur_s, end_s, n)
        d_new = np.interp(s_new, self.prev_s, self.prev_d)
        d_new[0] = cur_d
        #blend toward raceline for publishing only — don't corrupt prev_d
        blend_rate = 0.15
        d_publish = d_new * (1.0 - blend_rate)
        d_publish[0] = cur_d
        #if close enough to raceline, stop publishing and clear prev
        if np.max(np.abs(d_publish)) < 0.05:
            self.prev_d = None
            self.prev_s = None
            return
        #prev_d keeps original (unblended) for warm-start
        self.prev_s = s_new.copy()
        d_new = d_publish
        s_wrap = np.mod(s_new, self.scaled_max_s)
        try:
            xyz = self.converter.get_cartesian_3d(s_wrap, d_new).T
        except Exception:
            return
        coords = np.column_stack((xyz[:, 0], xyz[:, 1]))
        el = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        el = np.where(el < 1e-4, 1e-4, el)
        psi, kap = tph.calc_head_curv_num.calc_head_curv_num(
            path=coords, el_lengths=el, is_closed=False)
        psi = psi + np.pi / 2
        sw = self.scaled_wpnts_msg.wpnts
        s_ref = [w.s_m for w in sw]
        v_ref = [w.vx_mps for w in sw]
        v_pro = np.interp(s_wrap, s_ref, v_ref)
        msg = OTWpntArray(header=Header(stamp=rospy.Time.now(), frame_id='map'))
        for i in range(len(s_wrap)):
            msg.wpnts.append(Wpnt(
                id=i, s_m=float(s_wrap[i]), d_m=float(d_new[i]),
                x_m=float(xyz[i, 0]), y_m=float(xyz[i, 1]), z_m=float(xyz[i, 2]),
                psi_rad=float(psi[i]), kappa_radpm=float(kap[i]),
                vx_mps=float(v_pro[i])))
        msg.ot_side = self.last_ot_side or 'trail'
        self.evasion_pub.publish(msg)

    def _converge_to_raceline(self):
        """SQP solve without obstacles to smoothly converge to raceline."""
        if (self.prev_d is None or self.prev_s is None
                or self.scaled_max_s is None or self.converter is None):
            self.prev_d = None
            self.prev_s = None
            return
        n = len(self.prev_d)
        cur_d = float(self.frenet_state.pose.pose.position.y)
        cur_s = float(self.frenet_state.pose.pose.position.x)
        # rebuild s_av from current position
        delta_s = float(self.prev_s[1] - self.prev_s[0]) if n > 1 else 0.3
        s_av = np.array([cur_s + i * delta_s for i in range(n)])
        # jump detection
        s_gap = abs((cur_s - self.prev_s[0]) % self.scaled_max_s)
        if s_gap > self.scaled_max_s / 2:
            s_gap = self.scaled_max_s - s_gap
        if s_gap > 1.0:
            d_init = np.zeros(n)
        else:
            d_init = shift_solution(self.prev_d, self.prev_s, s_av,
                                    delta_s_shift=self.cur_v * self.dt)
        d_init[0] = cur_d
        n_src = len(self.scaled_wpnts_msg.wpnts)
        idxs = [int(np.round(s / self.scaled_delta_s)) % n_src for s in s_av]
        d_lb = np.array([-(self.scaled_wpnts_msg.wpnts[i].d_right - self.spline_bound_mindist)
                         for i in idxs])
        d_ub = np.array([(self.scaled_wpnts_msg.wpnts[i].d_left - self.spline_bound_mindist)
                         for i in idxs])
        d_init = np.clip(d_init, d_lb, d_ub)
        speed = max(self.cur_v, 1.0)
        min_r = float(np.interp(speed, [1, 3, 5, 7], [1.0, 2.0, 3.0, 4.0]))
        prob = SQPProblem(
            n_knots=n, delta_s=float(s_av[1] - s_av[0]) if n > 1 else 0.3,
            d_init=d_init, current_d=cur_d,
            bounds_lower=d_lb, bounds_upper=d_ub,
            obs_center_d=np.zeros(n), obs_min_dist=np.zeros(n),
            desired_side='any', kappa_limit=1.0 / min_r,
            lambda_reg=self.lambda_reg,
            lambda_smooth=self.lambda_smooth,
            lambda_jerk=self.lambda_jerk,
            lambda_obs=0.0,
            lambda_kappa=self.lambda_kappa,
            lambda_near_reg=self.lambda_near_reg,
            near_knots_K=self.near_knots_K,
            obs_ramp_knots=self.obs_ramp_knots,
        )
        try:
            d_opt, info = self.solver.solve(prob)
        except Exception:
            self.prev_d = None
            self.prev_s = None
            return
        self.prev_d = d_opt.copy()
        self.prev_s = s_av.copy()
        if np.max(np.abs(d_opt)) < 0.05:
            self.prev_d = None
            self.prev_s = None
            self._pending_msg = OTWpntArray(
                header=Header(stamp=rospy.Time.now(), frame_id='map'), wpnts=[])
            return
        # dense interpolation (same as _rolling_step)
        n_dense = max(n, int((s_av[-1] - s_av[0]) / self.scaled_delta_s))
        s_dense = np.linspace(s_av[0], s_av[-1], n_dense)
        d_dense = np.interp(s_dense, s_av, d_opt)
        s_wrap = np.mod(s_dense, self.scaled_max_s)
        xyz = self.converter.get_cartesian_3d(s_wrap, d_dense).T
        n_src = len(self.scaled_wpnts_msg.wpnts)
        msg = OTWpntArray(header=Header(stamp=rospy.Time.now(), frame_id='map'))
        out = []
        for k in range(n_dense):
            idx = int(np.round(s_wrap[k] / self.scaled_delta_s)) % n_src
            src = self.scaled_wpnts_msg.wpnts[idx]
            w = Wpnt()
            w.id = k
            w.s_m = float(s_dense[k])
            w.d_m = float(d_dense[k])
            w.x_m = float(xyz[k, 0])
            w.y_m = float(xyz[k, 1])
            w.z_m = float(xyz[k, 2])
            w.psi_rad = float(src.psi_rad)
            w.kappa_radpm = float(src.kappa_radpm)
            w.vx_mps = float(src.vx_mps)
            out.append(w)
        msg.wpnts = out
        msg.ot_side = 'converge'
        msg.ot_line = 'converge'
        self._pending_msg = msg

    def _publish_empty_otwpnts(self, clear_markers=True):
        msg = OTWpntArray(header=Header(stamp=rospy.Time.now(), frame_id='map'))
        msg.wpnts = []
        self.evasion_pub.publish(msg)
        if clear_markers:
            self._clear_markers()

    #failure handler — do NOT publish on failure; SM keeps prev path.
    def _handle_failure(self, dry_run: bool, is_abort: bool = False) -> None:
        if is_abort or not self.hold_last_valid:
            self.prev_d = None
            self.prev_s = None
            self.prev_v = None
            self.fail_streak = 0
            return
        self.fail_streak += 1
        if self.fail_streak >= self.max_fail_hold_cycles:
            self.prev_d = None
            self.prev_s = None
            self.prev_v = None
            self.fail_streak = 0



    def _publish_debug(self, solve_ms: float, success: bool,
                       t_ot: float, t_trail: float, abort_reason: AbortReason):
        msg = Float32MultiArray()
        flag_map = {AbortReason.NONE: 0.0,
                    AbortReason.SAFETY: 1.0,
                    AbortReason.PERFORMANCE: 2.0}
        msg.data = [solve_ms,
                    1.0 if success else 0.0,
                    t_ot,
                    t_trail,
                    flag_map[abort_reason]]
        self.debug_pub.publish(msg)

    def _visualize(self, s_arr, d_arr, x_arr, y_arr, v_arr, dry_run=False):
        if len(s_arr) == 0:
            return
        mrks = MarkerArray()
        del_mrk = Marker(header=Header(stamp=rospy.Time.now()))
        del_mrk.ns = 'rolling_path'
        del_mrk.action = Marker.DELETEALL
        mrks.markers.append(del_mrk)
        scale_factor = 0.1317  # match vel_markers_tuned scale
        ttl = rospy.Duration(max(0.2, 3.0 / self.rate_hz))
        for i in range(len(s_arr)):
            height = v_arr[i] * scale_factor
            m = Marker(header=Header(stamp=rospy.Time.now(), frame_id='map'))
            m.ns = 'rolling_path'
            m.type = m.CYLINDER
            m.scale.x = 0.1
            m.scale.y = 0.1
            m.scale.z = max(height, 0.05)
            m.color.a = 0.45 if dry_run else 1.0
            if dry_run:
                m.color.r, m.color.g, m.color.b = 1.0, 0.55, 0.0   # orange ghost
            else:
                m.color.r, m.color.g, m.color.b = 0.10, 0.70, 0.85  # cyan active
            m.id = i
            m.pose.position.x = x_arr[i]
            m.pose.position.y = y_arr[i]
            m.pose.position.z = height / 2.0
            m.pose.orientation.w = 1.0
            m.lifetime = ttl
            mrks.markers.append(m)
        self.mrks_pub.publish(mrks)

    # visualize GP-based obstacle avoidance band in RViz
    def _visualize_obs_band(self, s_av, obs_center, obs_min):
        """Draw upper/lower boundary + center of the obstacle band as LINE_STRIP markers."""
        active = obs_min > 0.0
        if not np.any(active):
            # publish delete to clear stale markers
            mrks = MarkerArray()
            d = Marker(header=Header(stamp=rospy.Time.now()))
            d.ns = 'obs_band'
            d.action = Marker.DELETEALL
            mrks.markers = [d]
            self.mrks_pub.publish(mrks)
            return
        idxs = np.where(active)[0]
        s_band = s_av[idxs]
        c_band = obs_center[idxs]
        m_band = obs_min[idxs]
        s_wrap = np.mod(s_band, self.scaled_max_s)
        try:
            xyz_upper = self.converter.get_cartesian_3d(s_wrap, c_band + m_band).T
            xyz_center = self.converter.get_cartesian_3d(s_wrap, c_band).T
            xyz_lower = self.converter.get_cartesian_3d(s_wrap, c_band - m_band).T
        except Exception:
            return
        mrks = MarkerArray()
        del_mrk = Marker(header=Header(stamp=rospy.Time.now()))
        del_mrk.ns = 'obs_band'
        del_mrk.action = Marker.DELETEALL
        mrks.markers.append(del_mrk)
        ttl = rospy.Duration(max(0.2, 3.0 / self.rate_hz))
        hdr = Header(stamp=rospy.Time.now(), frame_id='map')
        # upper boundary (red)
        m_up = Marker(header=hdr)
        m_up.ns = 'obs_band'
        m_up.id = 0
        m_up.type = Marker.LINE_STRIP
        m_up.scale.x = 0.04
        m_up.color.r, m_up.color.g, m_up.color.b, m_up.color.a = 1.0, 0.2, 0.2, 0.8
        m_up.lifetime = ttl
        m_up.points = [Point(x=xyz_upper[i, 0], y=xyz_upper[i, 1], z=xyz_upper[i, 2] + 0.05)
                       for i in range(len(xyz_upper))]
        mrks.markers.append(m_up)
        # lower boundary (red)
        m_lo = Marker(header=hdr)
        m_lo.ns = 'obs_band'
        m_lo.id = 1
        m_lo.type = Marker.LINE_STRIP
        m_lo.scale.x = 0.04
        m_lo.color.r, m_lo.color.g, m_lo.color.b, m_lo.color.a = 1.0, 0.2, 0.2, 0.8
        m_lo.lifetime = ttl
        m_lo.points = [Point(x=xyz_lower[i, 0], y=xyz_lower[i, 1], z=xyz_lower[i, 2] + 0.05)
                       for i in range(len(xyz_lower))]
        mrks.markers.append(m_lo)
        # center line (yellow)
        m_ct = Marker(header=hdr)
        m_ct.ns = 'obs_band'
        m_ct.id = 2
        m_ct.type = Marker.LINE_STRIP
        m_ct.scale.x = 0.03
        m_ct.color.r, m_ct.color.g, m_ct.color.b, m_ct.color.a = 1.0, 1.0, 0.0, 0.9
        m_ct.lifetime = ttl
        m_ct.points = [Point(x=xyz_center[i, 0], y=xyz_center[i, 1], z=xyz_center[i, 2] + 0.05)
                       for i in range(len(xyz_center))]
        mrks.markers.append(m_ct)
        self.mrks_pub.publish(mrks)

    def _publish_status_text(self, n_obs, ot_check, smart_static,
                             solve_ms, solve_ok, side, abort_reason, dry_run):
        if self.cur_x == 0.0 and self.cur_y == 0.0:
            return
        mrks = MarkerArray()
        m = Marker(header=Header(stamp=rospy.Time.now(), frame_id='map'))
        m.type = m.TEXT_VIEW_FACING
        m.id = 999
        m.ns = 'rolling_status'
        m.pose.position.x = self.cur_x
        m.pose.position.y = self.cur_y
        m.pose.position.z = 1.2
        m.pose.orientation.w = 1.0
        m.scale.z = 0.3
        m.color.a = 1.0
        m.lifetime = rospy.Duration(1.0)
        if abort_reason != AbortReason.NONE:
            m.color.r, m.color.g, m.color.b = 1.0, 0.2, 0.2
        elif dry_run:
            m.color.r, m.color.g, m.color.b = 1.0, 0.6, 0.1
        elif solve_ok:
            m.color.r, m.color.g, m.color.b = 0.2, 1.0, 0.4
        else:
            m.color.r, m.color.g, m.color.b = 0.8, 0.8, 0.8
        mode = ('ABORT:' + abort_reason.value) if abort_reason != AbortReason.NONE \
            else ('ACTIVE' if (solve_ok and not dry_run) else
                  ('DRY' if dry_run else 'IDLE'))
        m.text = (f'{mode} | obs={n_obs} ot={int(ot_check)} static={int(smart_static)} '
                  f'solve={solve_ms:.1f}ms side={side}')
        mrks.markers = [m]
        self.mrks_pub.publish(mrks)

    # =====================================================================
    # Main loop
    # =====================================================================
    def _prewarm_solver(self):
        """Force CasADi JIT build so the first real cycle isn't ~500ms."""
        n = int(self.avoidance_resolution)
        obs_center = np.zeros(n)
        obs_min = np.zeros(n)
        obs_center[n // 3 : 2 * n // 3] = 0.1
        obs_min[n // 3 : 2 * n // 3] = 0.55
        try:
            _, info = self.solver.solve(SQPProblem(
                n_knots=n, delta_s=0.5,
                d_init=np.zeros(n), current_d=0.0,
                bounds_lower=np.full(n, -1.0), bounds_upper=np.full(n, 1.0),
                obs_center_d=obs_center,
                obs_min_dist=obs_min,
                desired_side='right', kappa_limit=0.5,
                lambda_reg=self.lambda_reg,
                lambda_smooth=self.lambda_smooth,
                lambda_start_heading=self.lambda_start_heading,
                lambda_apex_bias=self.lambda_apex_bias,
                lambda_side=self.lambda_side,
                lambda_obs=self.lambda_obs,
                lambda_kappa=self.lambda_kappa,
                lambda_gg=self.lambda_gg,
                lambda_obs_smooth=self.lambda_obs_smooth,
            ))
            rospy.loginfo('[FastSQP] solver prewarmed (status=%s, iters=%d)',
                          info.get('status'), info.get('iter_count', 0))
        except Exception as exc:   # noqa: BLE001
            rospy.logwarn('[FastSQP] prewarm failed (non-fatal): %s', exc)

    def loop(self):
        rospy.loginfo('[FastSQP] waiting for upstream topics...')
        rospy.wait_for_message('/global_waypoints_scaled', WpntArray)
        rospy.wait_for_message('/car_state/odom', Odometry)
        rospy.wait_for_message('/behavior_strategy', BehaviorStrategy)
        self._prewarm_solver()
        # notify state_machine that fast_sqp_planner is active
        self.active_pub.publish(Bool(data=True))
        rospy.on_shutdown(lambda: self.active_pub.publish(Bool(data=False)))
        rospy.loginfo('[FastSQP] ready')

        while not rospy.is_shutdown():
            t0 = time.perf_counter()

            # snapshot
            obs_now = deepcopy(self.obs)
            fr = self.frenet_state
            self.current_d = fr.pose.pose.position.y
            self.cur_s = fr.pose.pose.position.x
            rospy.loginfo_throttle(2.0,
                '[FastSQP] loop: cur_s=%.2f n_obs=%d max_s=%s wpnts_upd=%s',
                self.cur_s, len(obs_now.obstacles),
                str(self.scaled_max_s is not None),
                str(self.wpnts_updated is not None))

            # filter considered obstacles — use all tracking data (incl. coast).
            # Only cap forward lookahead; behind obstacles stay as long as
            # tracking publishes them (coast_duration handles expiry).
            sorted_obs = sorted(obs_now.obstacles, key=lambda o: o.s_start)
            considered = []
            if self.scaled_max_s is not None:
                for o in sorted_obs:
                    s_fwd = (o.s_center - self.cur_s) % self.scaled_max_s
                    if s_fwd > self.scaled_max_s / 2.0:
                        s_fwd -= self.scaled_max_s
                    if (abs(o.d_center) < self.obs_traj_tresh
                            and s_fwd < self.lookahead):
                        considered.append(o)

            have_inputs = (self.scaled_max_s is not None
                           and self.wpnts_updated is not None)
            have_obs = len(considered) > 0
            if have_obs and not have_inputs:
                rospy.logwarn_throttle(2.0,
                    '[FastSQP] have_obs=%d have_inputs=0 → solve SKIPPED '
                    '(scaled_max_s=%s wpnts_updated=%s)',
                    len(considered),
                    str(self.scaled_max_s is not None),
                    str(self.wpnts_updated is not None))
            # diag: why is _rolling_step not firing?
            if not have_obs and len(sorted_obs) > 0:
                o0 = sorted_obs[0]
                ds = ((o0.s_start - self.cur_s) % self.scaled_max_s
                      if self.scaled_max_s is not None else -1.0)
                rospy.loginfo_throttle(2.0,
                    '[FastSQP] filter cut: n_raw=%d d=%.2f ds=%.2f (tresh=%.2f look=%.2f)',
                    len(sorted_obs), o0.d_center, ds,
                    self.obs_traj_tresh, self.lookahead)
            # ot_section_check always respected (even in online/rolling mode)
            if self.online_mode:
                publish_to_sm = (have_inputs and have_obs and self.ot_section_check)
            else:
                publish_to_sm = (have_inputs and have_obs
                                 and self.ot_section_check
                                 and not self.smart_static_active)
            dry_run = (have_inputs and have_obs and not publish_to_sm)

            # leave-OT cleanup
            if not publish_to_sm and self.prev_d is not None:
                self.abort.reset()
            # outside ot sector → clear warm-start
            if not self.ot_section_check:
                self.prev_d = None
                self.prev_s = None

            solve_ok = False
            t_ot = 0.0
            t_trail = 0.0
            abort_reason = AbortReason.NONE
            side = self.last_desired_side
            try:
                if have_obs and have_inputs:
                    solve_ok, t_ot, t_trail, abort_reason, side = \
                        self._rolling_step(considered, dry_run=dry_run)
                elif not have_obs and have_inputs and self.prev_d is not None:
                    self._converge_to_raceline()
            except Exception as exc:
                rospy.logwarn('[FastSQP] rolling_step raised: %s', exc)
                self._handle_failure(dry_run)

            solve_ms = (time.perf_counter() - t0) * 1000.0
            # publish pending msg only if solve was fast enough (skip cold start)
            if hasattr(self, '_pending_msg') and self._pending_msg is not None:
                if solve_ms < 500.0:
                    self.evasion_pub.publish(self._pending_msg)
                else:
                    rospy.logwarn('[FastSQP] cold start skipped (%.0fms)', solve_ms)
                    self._clear_markers()
                self._pending_msg = None
            if solve_ok:
                self.last_solve_ms = solve_ms
            self._publish_debug(solve_ms, solve_ok, t_ot, t_trail, abort_reason)
            self._publish_status_text(
                n_obs=len(considered),
                ot_check=self.ot_section_check,
                smart_static=self.smart_static_active,
                solve_ms=solve_ms, solve_ok=solve_ok,
                side=side, abort_reason=abort_reason, dry_run=dry_run)
            if self.measure:
                self.measure_pub.publish(Float32(time.perf_counter() - t0))
            self.rate.sleep()

    # ---------------------------------------------------------------------
    def _rolling_step(self, considered_obs, dry_run=False):
        """One cycle: build RoC, SQP refine d(s), FB velocity, abort check, publish.

        If dry_run=True, skip publishing /planner/fast_sqp_planner/otwpnts to state_machine
        and draw candidate markers in ghost color instead. Used to preview what
        the planner would produce outside the OT section.

        Returns (success, t_ot, t_trail, abort_reason, side_str).
        """
        # ---- build RoC knots and bounds (mirrors the SQP planner's sqp_solver) ----
        # limit dynamic obstacle to prediction_horizon_s
        dyn_obs = sorted([o for o in considered_obs if not o.is_static],
                         key=lambda o: o.s_start)
        static_obs = [o for o in considered_obs if o.is_static]
        dyn_swept = []
        if len(dyn_obs) >= 1:
            opp_now = dyn_obs[0]
            opp_speed = self._estimate_opponent_speed()
            horizon_s_end = (opp_now.s_center
                             + opp_speed * self.prediction_horizon_s
                             + 0.5)  # +half car length
            compact = deepcopy(opp_now)
            compact.s_end = horizon_s_end
            compact.is_static = False
            dyn_swept = [compact]
        clusters = static_obs + dyn_swept
        n_clusters = len(clusters)

        # gb_idxs = union of per-cluster s-ranges
        gb_idx_set = set()
        for c in clusters:
            i_start = int(np.abs(self.scaled_wpnts[:, 0] - c.s_start).argmin())
            i_end = int(np.abs(self.scaled_wpnts[:, 0] - c.s_end).argmin())
            span = (i_end - i_start) % self.scaled_max_idx
            for k in range(span + 1):
                gb_idx_set.add((i_start + k) % self.scaled_max_idx)
        gb_idxs = np.array(sorted(gb_idx_set), dtype=int)
        if len(gb_idxs) < 20:
            # fallback: contiguous 20-knot window around first cluster center
            s_c = clusters[0].s_center if hasattr(clusters[0], 's_center') \
                else 0.5 * (clusters[0].s_start + clusters[0].s_end)
            gb_idxs = np.array([int(s_c / self.scaled_delta_s + i) % self.scaled_max_idx
                                for i in range(20)], dtype=int)

        # per-cluster side decision
        cluster_sides = []  # list of (cluster, side_str, apex)
        for c in clusters:
            c_side, c_apex = self._more_space(c, self.scaled_wpnts_msg.wpnts, gb_idxs)
            cluster_sides.append((c, c_side, c_apex))

        # initial_apex from ego-nearest cluster (drives warm-start direction)
        def _s_gap(c):
            return (c.s_start - self.cur_s) % self.scaled_max_s
        nearest_idx = int(np.argmin([_s_gap(c) for c, _, _ in cluster_sides]))
        first_side = cluster_sides[nearest_idx][1]
        initial_apex = cluster_sides[nearest_idx][2]

        # side lock — keep current side unless blocked (3x hysteresis)
        if self.last_desired_side in ('left', 'right'):
            c_near = cluster_sides[nearest_idx][0]
            left_mean = np.mean([self.scaled_wpnts_msg.wpnts[i].d_left for i in gb_idxs])
            right_mean = np.mean([self.scaled_wpnts_msg.wpnts[i].d_right for i in gb_idxs])
            lg = abs(left_mean - c_near.d_left)
            rg = abs(right_mean + c_near.d_right)
            if self.last_desired_side == 'right' and lg > rg * 1.5:
                first_side = 'left'
            elif self.last_desired_side == 'left' and rg > lg * 1.5:
                first_side = 'right'
            else:
                first_side = self.last_desired_side
        self.last_desired_side = first_side
        # single string 'side' kept for debug/status UIs; reflects first cluster
        side = first_side

        kappas = np.array([self.scaled_wpnts_msg.wpnts[i].kappa_radpm for i in gb_idxs])
        max_kappa = np.max(np.abs(kappas))
        outside = 'left' if np.sum(kappas) < 0 else 'right'
        # extend s_end for outside-curve overtaking
        for c, c_side, _ in cluster_sides:
            if c_side != outside:
                continue
            extend = (c.s_end - c.s_start) % self.max_s_updated \
                * max_kappa * (self.width_car + self.evasion_dist)
            c.s_end = c.s_end + extend
            # static clusters are single-obstacle references into considered_obs;
            # dyn_swept are deepcopies so only the cluster envelope shifts

        max_s_obs_end = max(c.s_end for c in clusters)

        # re-read ego s right before solve
        start_av = self.frenet_state.pose.pose.position.x
        if self.online_mode:
            horizon_m = max(self.cur_v * self.T_horizon, self.s_min_horizon)
            end_av = start_av + horizon_m
        else:
            end_av = max_s_obs_end + self.back_to_raceline_after

        s_av = np.linspace(start_av, end_av, self.avoidance_resolution)
        delta_s = float(s_av[1] - s_av[0])

        idxs = np.array([int(np.abs(self.scaled_wpnts[:, 0] - (s % self.scaled_max_s)).argmin())
                         for s in s_av])
        corr = [self.scaled_wpnts_msg.wpnts[i] for i in idxs]
        bounds = np.array([(-w.d_right + self.spline_bound_mindist,
                            w.d_left - self.spline_bound_mindist) for w in corr])
        d_lb = bounds[:, 0]
        d_ub = bounds[:, 1]

        # obstacle RoC knots
        # obs arrays (length n_knots, inactive = 0). pre_buffer reserves
        # knots 0..1 for smooth transition from current_d.
        n_knots = len(s_av)
        obs_center = np.zeros(n_knots)
        obs_min = np.zeros(n_knots)
        pre_buffer = 2
        for o in clusters:
            i0 = int(np.abs(s_av - o.s_start).argmin())
            i1 = int(np.abs(s_av - o.s_end).argmin())
            if i0 >= len(s_av) - 2:
                continue
            i0 = max(i0, pre_buffer)
            if i1 < i0:
                continue
            if o.is_static or i1 == i0:
                if i1 == i0:
                    i1 = i0 + 1
                for ii in range(i0, i1 + 1):
                    obs_center[ii] = (o.d_left + o.d_right) / 2.0
                    obs_min[ii] = ((o.d_left - o.d_right) / 2.0
                                   + self.width_car + self.evasion_dist)
            else:
                # per-knot GP d(s) lookup
                has_gp = (self.opponent_wpnts_sm is not None
                          and self.opponent_wpnts_sm.size > 0
                          and self.opponent_wpnts_d is not None)
                fallback_center = (o.d_left + o.d_right) / 2.0
                # has_gp also requires d_var for variance-scaled obs_min
                has_dvar = (has_gp
                            and self.opponent_wpnts_dvar is not None
                            and self.opponent_wpnts_dvar.size > 0)
                for ii in range(i0, i1 + 1):
                    if has_gp:
                        s_at_knot = s_av[ii] % self.scaled_max_s
                        obs_center[ii] = float(np.interp(
                            s_at_knot, self.opponent_wpnts_sm, self.opponent_wpnts_d))
                    else:
                        obs_center[ii] = fallback_center
                    # obs_min mode toggle
                    #   (A) fixed width: uncomment the line below + comment out the block.
                    #       Equivalent to setting obs_sigma_k=0 in rqt.
                    #   (B) variance-scaled (default): obs_min = base + sigma_k * sqrt(d_var)
                    # obs_min[ii] = self.width_car + self.evasion_dist
                    base_min = self.width_car + self.evasion_dist
                    if has_dvar and self.obs_sigma_k > 0:
                        s_at_knot = s_av[ii] % self.scaled_max_s
                        dvar = float(np.interp(
                            s_at_knot, self.opponent_wpnts_sm, self.opponent_wpnts_dvar))
                        base_min += self.obs_sigma_k * np.sqrt(max(dvar, 0.0))
                    obs_min[ii] = base_min

        # visualize obs band in RViz
        self._visualize_obs_band(s_av, obs_center, obs_min)

        # curvature constraint: interpolated min-radius like the SQP planner
        clipped_v = max(self.cur_v, 1.0)
        first_upd_v = self.wpnts_updated[idxs[0] % self.max_idx_updated].vx_mps
        radius_speed = min(clipped_v, first_upd_v)
        min_radius = float(np.interp(radius_speed, [1, 3, 5, 7], [1.0, 2.0, 3.0, 4.0]))
        kappa_limit = 1.0 / min_radius

        # ---- warm-start ---------------------------------------------------
        cur_d_now = float(self.frenet_state.pose.pose.position.y)
        if self.prev_d is not None and self.prev_s is not None:
            s_gap = abs((self.cur_s - self.prev_s[0]) % self.scaled_max_s)
            if s_gap > self.scaled_max_s / 2:
                s_gap = self.scaled_max_s - s_gap
            if s_gap > 1.0:
                d_init = np.zeros(len(s_av))
            else:
                d_init = shift_solution(self.prev_d, self.prev_s, s_av,
                                        delta_s_shift=self.cur_v * self.dt)
        else:
            d_init = np.zeros(len(s_av))
        # anchor start to current ego d
        d_init[0] = cur_d_now
        d_init = np.clip(d_init, d_lb, d_ub)

        # ---- SQP solve ----------------------------------------------------
        # drop side bias when >=2 clusters
        if n_clusters >= 2:
            eff_lambda_side = 0.0
            eff_desired_side = 'any'
        else:
            eff_lambda_side = self.lambda_side
            eff_desired_side = side if self.homotopy_lock else 'any'

        problem = SQPProblem(
            n_knots=len(s_av),
            delta_s=delta_s,
            d_init=d_init,
            current_d=self.current_d,
            bounds_lower=d_lb,
            bounds_upper=d_ub,
            #obs_indices removed — obs_center/obs_min are length n_knots.
            obs_center_d=obs_center,
            obs_min_dist=obs_min,
            desired_side=eff_desired_side,
            kappa_limit=kappa_limit,
            lambda_reg=self.lambda_reg,
            lambda_smooth=self.lambda_smooth,
            lambda_start_heading=self.lambda_start_heading,
            lambda_apex_bias=self.lambda_apex_bias,
            lambda_side=eff_lambda_side,
            lambda_jerk=self.lambda_jerk,
            lambda_term=self.lambda_term,
            #all-soft — obs, curvature, GG as penalties
            lambda_obs=self.lambda_obs,
            lambda_kappa=self.lambda_kappa,
            lambda_gg=self.lambda_gg,
            lambda_near_reg=self.lambda_near_reg,
            lambda_obs_smooth=self.lambda_obs_smooth,
            near_knots_K=self.near_knots_K,
            obs_ramp_knots=self.obs_ramp_knots,
            optimize_velocity=self.optimize_velocity,
            v_init=self.prev_v,
            v_current=max(self.cur_v, 0.5),
            v_max_arr=np.array([self.scaled_wpnts_msg.wpnts[i].vx_mps for i in idxs]),
            kappa_ref=kappas,
            ax_max_arr=self._ggv_ax_at_knots(idxs),
            ay_max_arr=self._ggv_ay_at_knots(idxs),
            lambda_progress=self.lambda_progress,
        )
        d_opt, info = self.solver.solve(problem)

        #zigzag reject removed — lambda_smooth/jerk/near_reg handle stability

        # diag: obs_center wiggle + d stats — confirms GP d-curve drives oscillation
        #fixed-slot arrays — stat on active slots only.
        active_mask = obs_min > 0.0
        if np.any(active_mask):
            oc_active = obs_center[active_mask]
            oc_min = float(oc_active.min())
            oc_max = float(oc_active.max())
            oc_std = float(oc_active.std())
            n_active = int(active_mask.sum())
        else:
            oc_min = oc_max = oc_std = 0.0
            n_active = 0
        d_opt_std = float(np.std(d_opt)) if info['success'] else 0.0
        #add n_clusters + d_opt bump amplitude for shape diagnosis
        d_opt_min = float(np.min(d_opt)) if info['success'] else 0.0
        d_opt_max = float(np.max(d_opt)) if info['success'] else 0.0
        diag = Float32MultiArray()
        diag.data = [oc_min, oc_max, oc_std,
                     float(n_active), float(len(considered_obs)),
                     float(self.cur_s), float(start_av), float(end_av),
                     float(np.std(d_init)), d_opt_std,
                     float(n_clusters), d_opt_min, d_opt_max,
                     float(self.current_d), float(d_lb[0]), float(d_ub[0]),
                     0.0, float(self.last_solve_ms),
                     float(self.cur_v)]
        self.diag_pub.publish(diag)

        if not info['success']:
            rospy.logwarn_throttle(1.0,
                '[FastSQP] SQP failed (%s, iters=%d). skipping cycle.',
                info['status'], info['iter_count'])
            self._handle_failure(dry_run)
            return False, 0.0, 0.0, AbortReason.NONE, side

        #freeze near-term path to previous solution.
        #  Prevents IPOPT local-minimum hopping from causing near-term jitter.
        #  Collision override: if frozen d hits obstacle, keep solver's d_opt.
        if (self.freeze_distance_m > 0
                and self.prev_d is not None and self.prev_s is not None):
            prev_d_at = np.interp(s_av, self.prev_s, self.prev_d,
                                   left=d_opt[0], right=d_opt[-1])
            n_freeze = int(np.ceil(self.freeze_distance_m / delta_s))
            n_blend = int(np.ceil(self.freeze_blend_m / delta_s))
            for k in range(min(n_freeze + n_blend, n_knots)):
                if k < n_freeze:
                    d_opt[k] = prev_d_at[k]
                else:
                    t = (k - n_freeze) / max(n_blend, 1)
                    d_opt[k] = (1.0 - t) * prev_d_at[k] + t * d_opt[k]
            d_opt = np.clip(d_opt, d_lb, d_ub)

        #ego-obstacle collision check (physical overlap, not planned path)
        ego_collision = False
        for o in considered_obs:
            s_dist = (self.cur_s - o.s_start) % self.scaled_max_s
            s_len = (o.s_end - o.s_start) % self.scaled_max_s
            if s_dist <= s_len:  # ego is alongside obstacle in s
                obs_d = (o.d_left + o.d_right) / 2.0
                obs_hw = (o.d_left - o.d_right) / 2.0
                gap = abs(self.current_d - obs_d) - obs_hw - self.width_car / 2.0
                if gap < 0:
                    ego_collision = True
                    break
        self.collision_pub.publish(Bool(data=ego_collision))

        #dense interpolation (0.1m spacing)
        n_dense = max(len(s_av), int((end_av - start_av) / self.scaled_delta_s))
        s_dense = np.linspace(start_av, end_av, n_dense)
        d_dense = np.interp(s_dense, s_av, d_opt)

        s_wrap = np.mod(s_dense, self.scaled_max_s)
        xyz = self.converter.get_cartesian_3d(s_wrap, d_dense).T
        evasion_x = xyz[:, 0]
        evasion_y = xyz[:, 1]
        evasion_z = xyz[:, 2]
        evasion_s = s_wrap
        evasion_d = d_dense

        coords = np.column_stack((evasion_x, evasion_y))
        el_lengths = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        el_lengths = np.where(el_lengths < 1e-4, 1e-4, el_lengths)
        psi, kappa_path = tph.calc_head_curv_num.calc_head_curv_num(
            path=coords,
            el_lengths=el_lengths,
            is_closed=False,
        )
        psi = psi + np.pi / 2

        # ---- velocity profile -----------------------------------------------
        if info.get('v_opt') is not None:
            # NLP-optimized velocity → interpolate to dense grid
            v_profile = np.interp(s_dense, s_av, info['v_opt'])
        else:
            try:
                sw = self.scaled_wpnts_msg.wpnts
                s_ref = np.array([w.s_m for w in sw])
                v_ref = np.array([w.vx_mps for w in sw])
                mu_ref = np.array([getattr(w, 'mu_rad', 0.0) for w in sw])
                v_at_path = np.interp(s_wrap, s_ref, v_ref)
                mu_at_path = np.interp(s_wrap, s_ref, mu_ref)

                # velocity_mode dispatch (2_5d / nlp / 3d)
                if self.velocity_mode == '2_5d':
                    # vel_planner_25d: ggv + slope(elevation angle) mode
                    # slope=mu_at_path (track pitch angle rad)
                    # mu not passed → defaults to friction=1.0 (ggv mode)
                    v_profile = self.vp.profile(
                        kappa=kappa_path,
                        el_lengths=el_lengths,
                        v_start=max(self.cur_v, 0.1),
                        v_max=self.scaled_vmax,
                        slope=mu_at_path,
                    )
                elif self.velocity_mode == 'nlp':
                    # reduced NLP velocity optimization (GGManager + endogenous g_tilde)
                    if self.gg_manager is not None:
                        V_opt, ax_opt, nlp_ok = solve_velocity_nlp(
                            kappa=kappa_path,
                            el_lengths=el_lengths,
                            mu=mu_at_path,
                            gg=self.gg_manager,
                            v_start=max(self.cur_v, 0.5),
                            v_max=self.scaled_vmax,
                            v_init=v_at_path,
                        )
                        if nlp_ok:
                            v_profile = V_opt
                        else:
                            # NLP failed → fallback to 3d mode fwbw
                            ax_max, ay_max = self._ggv_lookup(
                                v_at_path, mu_at_path)
                            loc_gg = np.column_stack([ax_max, ay_max])
                            v_profile = self.vp.profile(
                                kappa=kappa_path,
                                el_lengths=el_lengths,
                                v_start=max(self.cur_v, 0.1),
                                v_max=self.scaled_vmax,
                                loc_gg=loc_gg,
                            )
                    else:
                        rospy.logwarn_throttle(5.0,
                            '[FastSQP] nlp mode but GGManager unavailable → fallback')
                        ax_max, ay_max = self._ggv_lookup(
                            v_at_path, mu_at_path)
                        loc_gg = np.column_stack([ax_max, ay_max])
                        v_profile = self.vp.profile(
                            kappa=kappa_path,
                            el_lengths=el_lengths,
                            v_start=max(self.cur_v, 0.1),
                            v_max=self.scaled_vmax,
                            loc_gg=loc_gg,
                        )
                elif self.velocity_mode == '3d':
                    # real C++ FBGA (libs/FBGA/GIGI_test_unicorn.exe)
                    used_fbga = False
                    if self._fbga_runner is not None:
                        s_path = np.cumsum(np.concatenate([[0.0], el_lengths]))
                        res = self._fbga_runner.solve_open(
                            s_path, kappa_path, mu_at_path, v_seed=v_at_path)
                        if res is not None:
                            v_profile = res[0]
                            used_fbga = True
                        else:
                            rospy.logwarn_throttle(1.0,
                                '[FastSQP] FBGA solve failed; fallback to loc_gg')
                    else:
                        rospy.logwarn_throttle(5.0,
                            '[FastSQP] velocity_mode=3d but FBGA runner unavailable; fallback')
                    if not used_fbga:
                        ax_max, ay_max = self._ggv_lookup(v_at_path, mu_at_path)
                        loc_gg = np.column_stack([ax_max, ay_max])
                        v_profile = self.vp.profile(
                            kappa=kappa_path,
                            el_lengths=el_lengths,
                            v_start=max(self.cur_v, 0.1),
                            v_max=self.scaled_vmax,
                            loc_gg=loc_gg,
                        )
                else:
                    # fallback: legacy loc_gg mode
                    ax_max, ay_max = self._ggv_lookup(v_at_path, mu_at_path)
                    loc_gg = np.column_stack([ax_max, ay_max])
                    v_profile = self.vp.profile(
                        kappa=kappa_path,
                        el_lengths=el_lengths,
                        v_start=max(self.cur_v, 0.1),
                        v_max=self.scaled_vmax,
                        loc_gg=loc_gg,
                    )
            except Exception as exc:   # noqa: BLE001
                rospy.logwarn_throttle(1.0,
                    '[FastSQP] velocity profile failed: %s — fallback to raceline v',
                    exc)
                v_profile = np.interp(s_dense, s_av,
                                      np.array([c.vx_mps for c in corr]))

        # ax profile (for Wpnt.ax_mps2)
        ax_profile = tph.calc_ax_profile.calc_ax_profile(
            vx_profile=v_profile,
            el_lengths=el_lengths,
            eq_length_output=True)

        # ---- abort checks ------------------------------------------------
        # GP samples for performance abort: gp_v at s_av
        gp_v_at_s = self._sample_opponent_vs(s_wrap)

        # Use first observed obstacle to test safety envelope
        obs0 = considered_obs[0]
        gp_d_at_obs, gp_dvar_at_obs = self._sample_opponent_d_with_var(obs0.s_center)

        abort_reason = self.abort.step(
            now=rospy.Time.now(),
            opp_obs_d=obs0.d_center,
            opp_obs_s=obs0.s_center,
            gp_d_at_s=gp_d_at_obs,
            gp_d_var_at_s=gp_dvar_at_obs,
            s_grid=s_dense,
            v_grid=v_profile,
            gp_v_at_s=gp_v_at_s,
        )

        # compute report t_ot / t_trail for debug
        ds = np.diff(s_dense)
        v_mid = 0.5 * (v_profile[:-1] + v_profile[1:])
        v_mid = np.maximum(v_mid, 1e-3)
        t_ot = float(np.sum(ds / v_mid))
        gp_v_mid = 0.5 * (gp_v_at_s[:-1] + gp_v_at_s[1:])
        gp_v_mid = np.maximum(gp_v_mid, 1e-3)
        t_trail = float(np.sum(ds / gp_v_mid))

        if abort_reason != AbortReason.NONE:
            rospy.loginfo_throttle(1.0,
                '[FastSQP] ABORT=%s t_ot=%.2f t_trail=%.2f',
                abort_reason.value, t_ot, t_trail)
            # still publish path even on abort (SM needs it for trailing)
            # fall through to publish below

        #log start-point gap for diagnosing "path starts behind ego"
        s0_gap = float(np.mod(evasion_s[0] - self.cur_s, self.scaled_max_s))
        if s0_gap > self.scaled_max_s / 2.0:
            s0_gap -= self.scaled_max_s    # signed: negative = behind ego
        rospy.loginfo_throttle(1.0,
            '[FastSQP] path start gap: evasion_s[0]-cur_s=%.3fm '
            '(start_av-cur_s=%.3f v=%.2f)',
            s0_gap, start_av - self.cur_s, self.cur_v)

        # ---- publish OTWpntArray ------------------------------------------
        msg = OTWpntArray(header=Header(stamp=rospy.Time.now(), frame_id='map'))
        wpnts = []
        for i in range(len(evasion_s)):
            w = Wpnt(
                id=i,
                s_m=float(evasion_s[i]),
                d_m=float(evasion_d[i]),
                x_m=float(evasion_x[i]),
                y_m=float(evasion_y[i]),
                z_m=float(evasion_z[i]),
                psi_rad=float(psi[i]),
                kappa_radpm=float(kappa_path[i]),
                vx_mps=float(v_profile[i] * self.v_scale),
                ax_mps2=float(ax_profile[i]) if i < len(ax_profile) else 0.0,
            )
            wpnts.append(w)
        msg.wpnts = wpnts
        msg.ot_side = side
        mean_d = float(np.mean(evasion_d))
        msg.ot_line = 'left' if mean_d > 0 else 'right'

        if not dry_run:
            self._pending_msg = msg  # main loop decides whether to publish
            if considered_obs:
                self.merger_pub.publish(Float32MultiArray(
                    data=[considered_obs[-1].s_end % self.scaled_max_s,
                          evasion_s[-1] % self.scaled_max_s]))
            # remember for next cycle's warm-start (in refined-knot domain)
            self.prev_d = d_opt.copy()
            self.prev_s = s_av.copy()
            self.prev_v = info.get('v_opt')
            self.last_ot_side = msg.ot_line
            self.fail_streak = 0

        # only visualize if msg will be published (skip cold start markers)
        if self._pending_msg is not None:
            self._visualize(evasion_s, evasion_d, evasion_x, evasion_y, v_profile,
                            dry_run=dry_run)

        return True, t_ot, t_trail, AbortReason.NONE, side

    # ---------------------------------------------------------------------
    #estimate opponent speed for prediction horizon limit
    def _estimate_opponent_speed(self) -> float:
        if (self.opponent_wpnts_vs is not None
                and self.opponent_wpnts_vs.size > 0
                and self.opponent_wpnts_sm is not None):
            s_mod = self.cur_s % self.scaled_max_s
            return float(np.interp(s_mod, self.opponent_wpnts_sm,
                                   self.opponent_wpnts_vs))
        return max(self.cur_v, 1.0)

    def _sample_opponent_vs(self, s_query: np.ndarray) -> np.ndarray:
        """Sample GP opponent longitudinal speed at s_query. Fallback: raceline vmax."""
        if (self.opponent_wpnts_sm is None or self.opponent_wpnts_sm.size == 0
                or self.opponent_wpnts_vs is None):
            return np.full_like(s_query, self.scaled_vmax or 1.0, dtype=float)
        s_max = self.opponent_wpnts_sm[-1] + 1e-6
        s_mod = np.mod(s_query, s_max)
        return np.interp(s_mod, self.opponent_wpnts_sm, self.opponent_wpnts_vs)

    def _sample_opponent_d_with_var(self, s_query: float):
        if (self.opponent_wpnts_sm is None or self.opponent_wpnts_sm.size == 0):
            return None, None
        s_max = self.opponent_wpnts_sm[-1] + 1e-6
        s_mod = s_query % s_max
        d = float(np.interp(s_mod, self.opponent_wpnts_sm, self.opponent_wpnts_d))
        dvar = float(np.interp(s_mod, self.opponent_wpnts_sm, self.opponent_wpnts_dvar))
        return d, dvar


if __name__ == '__main__':
    node = FastSQPPlannerNode()
    node.loop()
