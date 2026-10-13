#!/usr/bin/env python3
# Event-triggered local racing line mux node
# Normally pass-through global waypoints, compute local racing line when vehicle deviates

import os
import sys
import yaml
import numpy as np
import time
import rospy
import tf.transformations

from nav_msgs.msg import Odometry
from std_msgs.msg import String
from sensor_msgs.msg import Imu
from f110_msgs.msg import WpntArray, Wpnt

# Add paths for solver imports
_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_dir)  # for track3D_lite
_solver_src = os.path.join(_dir, '..', 'src')
sys.path.append(os.path.abspath(_solver_src))

# use track3D_lite (pandas-free) instead of track3D
from track3D_lite import Track3D
from ggManager import GGManager
from local_racing_line_planner import LocalRacinglinePlanner
from point_mass_model import export_point_mass_ode_model


class LocalRacelineMux:
    """
    Local racing line mux node
    - PASSTHROUGH: forward the global waypoints as is
    - ACTIVE: compute a local racing line with the solver and overwrite the global waypoints with it
    """

    PASSTHROUGH = "PASSTHROUGH"
    ACTIVE = "ACTIVE"

    def __init__(self):
        rospy.init_node('local_raceline_mux', anonymous=False)

        # -- Parameters --
        self.d_threshold = rospy.get_param('~d_threshold', 0.3)
        self.v_threshold = rospy.get_param('~v_threshold', 1.5)
        self.convergence_d_threshold = rospy.get_param('~convergence_d_threshold', 0.1)
        self.convergence_v_threshold = rospy.get_param('~convergence_v_threshold', 0.5)
        self.optimization_horizon = rospy.get_param('~optimization_horizon', 10.0)
        self.N_steps = rospy.get_param('~N_steps', 30)
        self.gg_mode = rospy.get_param('~gg_mode', 'diamond')
        self.safety_distance = rospy.get_param('~safety_distance', 0.05)
        self.is_sim = rospy.get_param('/sim', False)
        self.map_name = rospy.get_param('/map', '')
        # filenames as params (map name and file prefix may differ)
        self.smoothed_track_file = rospy.get_param('~smoothed_track_file', 'gazebo_wall_2_3d_smoothed.csv')

        # -- Paths --
        pkg_path = os.path.abspath(os.path.join(_dir, '..'))  # global_line/
        map_path = os.path.abspath(os.path.join(
            _dir, '..', '..', '..', '..', 'stack_master', 'maps', self.map_name))

        self.smoothed_track_path = os.path.join(map_path, self.smoothed_track_file)
        self.vehicle_params_path = os.path.join(
            pkg_path, 'data', 'vehicle_params', 'params_rc_car_10th.yml')
        self.gg_diagram_path = os.path.join(
            pkg_path, 'data', 'gg_diagrams', 'rc_car_10th', 'velocity_frame')

        # -- State variables --
        self.mode = self.PASSTHROUGH
        self.prev_solution = None
        self.gb_wpnts = None  # latest WpntArray from vel_scaler
        self.cur_s = 0.0
        self.cur_d = 0.0
        self.cur_vs = 0.0
        self.cur_yaw = 0.0
        self.cur_imu_ax = 0.0
        self.cur_imu_ay = 0.0
        self.solver_initialized = False

        # -- Load vehicle params --
        with open(self.vehicle_params_path, 'r') as f:
            self.params = yaml.safe_load(f)

        # -- Load global raceline reference (for deviation check) --
        self._load_global_raceline_ref()

        # -- Initialize solver (Track3D, GGManager, model, planner) --
        self._init_solver()

        # -- Publishers --
        self.wpnt_pub = rospy.Publisher(
            '/global_waypoints_scaled', WpntArray, queue_size=1)
        # online_waypoints — local racing line computed by the solver (for RViz)
        self.online_wpnt_pub = rospy.Publisher(
            '/online_waypoints', WpntArray, queue_size=1)
        self.status_pub = rospy.Publisher(
            '/local_raceline/status', String, queue_size=1)

        # -- Subscribers --
        rospy.Subscriber(
            '/global_waypoints_scaled_raw', WpntArray, self._gb_wpnts_cb)
        rospy.Subscriber(
            '/car_state/odom_frenet', Odometry, self._frenet_cb)
        rospy.Subscriber(
            '/car_state/odom', Odometry, self._odom_cb)

        if not self.is_sim:
            rospy.Subscriber('/ekf/imu/data', Imu, self._imu_cb)

        # -- Timer (10Hz) --
        self.timer = rospy.Timer(rospy.Duration(0.1), self._timer_cb)

        rospy.loginfo("[LocalRacelineMux] Initialized. Mode: PASSTHROUGH, sim=%s, map=%s",
                      self.is_sim, self.map_name)

    # =======================================================
    # Initialization helpers
    # =======================================================

    def _load_global_raceline_ref(self):
        """Build the arrays for the deviation check and the solver initial guess from global_waypoints.json"""
        import json
        json_path = os.path.join(
            os.path.abspath(os.path.join(
                _dir, '..', '..', '..', '..', 'stack_master', 'maps', self.map_name)),
            'global_waypoints.json')
        with open(json_path, 'r') as f:
            data = json.load(f)
        wpnts = data['global_traj_wpnts_iqp']['wpnts']
        self.ref_s = np.array([w['s_m'] for w in wpnts])
        self.ref_v = np.array([w['vx_mps'] for w in wpnts])
        self.ref_n = np.array([w['d_m'] for w in wpnts])

        # load s, n relative to the centerline (initial guess for a solver cold start)
        # stored by export_global_waypoints.py in a separate 'centerline_ref' section
        if 'centerline_ref' in data:
            self.ref_s_center = np.array(data['centerline_ref']['s_center_m'])
            self.ref_n_center = np.array(data['centerline_ref']['n_center_m'])
            rospy.loginfo("[LocalRacelineMux] Loaded centerline ref: n_center=[%.3f, %.3f]",
                          self.ref_n_center.min(), self.ref_n_center.max())
        else:
            # older JSON files — fall back when centerline_ref is missing
            self.ref_s_center = self.ref_s
            self.ref_n_center = np.zeros_like(self.ref_s)
            rospy.logwarn("[LocalRacelineMux] No centerline_ref in JSON, using fallback (n=0)")

        rospy.loginfo("[LocalRacelineMux] Loaded raceline ref from JSON: %d pts, s=[%.1f, %.1f]",
                      len(self.ref_s), self.ref_s[0], self.ref_s[-1])

    def _init_solver(self):
        """Initialize the solver (same pattern as sim_local_racing_line.py)"""
        rospy.loginfo("[LocalRacelineMux] Initializing solver (acados compile may take 10-30s)...")
        t0 = time.time()

        # Track3D
        self.track_handler = Track3D(path=self.smoothed_track_path)
        self.track_length = self.track_handler.s[-1]

        # GGManager
        self.gg_handler = GGManager(
            gg_path=self.gg_diagram_path,
            gg_margin=0.0
        )

        # Point mass ODE model
        self.model = export_point_mass_ode_model(
            vehicle_params=self.params['vehicle_params'],
            track_handler=self.track_handler,
            gg_handler=self.gg_handler,
            optimization_horizon=self.optimization_horizon,
            gg_mode=self.gg_mode
        )

        # stronger solver settings
        # - SQP_RTI 1 iter → 3 iter (better convergence, still real-time)
        # - w_slack_n 1.0 → 1e3 (much larger penalty on track bound violations)
        self.planner = LocalRacinglinePlanner(
            params=self.params,
            track_handler=self.track_handler,
            gg_handler=self.gg_handler,
            model=self.model,
            nlp_solver_type='SQP_RTI',
            optimization_horizon=self.optimization_horizon,
            gg_mode=self.gg_mode,
            N_steps=self.N_steps,
            sqp_max_iter=3,
            qp_max_iter=50,
            w_slack_n=1e3,
        )

        self.solver_initialized = True
        rospy.loginfo("[LocalRacelineMux] Solver initialized in %.1fs", time.time() - t0)

    # =======================================================
    # Callbacks
    # =======================================================

    def _gb_wpnts_cb(self, msg):
        """Global waypoints (from vel_scaler, remapped to _raw)"""
        self.gb_wpnts = msg

    def _frenet_cb(self, msg):
        """Frenet odometry: s, d, vs"""
        self.cur_s = msg.pose.pose.position.x
        self.cur_d = msg.pose.pose.position.y
        self.cur_vs = msg.twist.twist.linear.x

    def _odom_cb(self, msg):
        """Map-frame odometry: yaw for chi calculation"""
        q = msg.pose.pose.orientation
        _, _, yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.cur_yaw = yaw

    def _imu_cb(self, msg):
        """IMU acceleration (real car only)"""
        self.cur_imu_ax = msg.linear_acceleration.x
        self.cur_imu_ay = msg.linear_acceleration.y

    # =======================================================
    # Main timer loop (10Hz)
    # =======================================================

    def _timer_cb(self, event):
        if self.gb_wpnts is None or not self.solver_initialized:
            return

        # -- Deviation check --
        s = self.cur_s % self.track_length
        n_ref = np.interp(s, self.ref_s, self.ref_n, period=self.track_length)
        v_ref = np.interp(s, self.ref_s, self.ref_v, period=self.track_length)
        d_err = abs(self.cur_d - n_ref)
        v_err = abs(self.cur_vs - v_ref)

        # -- Mode transitions --
        if self.mode == self.PASSTHROUGH:
            if d_err > self.d_threshold or v_err > self.v_threshold:
                self.mode = self.ACTIVE
                rospy.logwarn("[LocalRacelineMux] -> ACTIVE (d_err=%.3f, v_err=%.3f)",
                              d_err, v_err)

        if self.mode == self.ACTIVE:
            self._run_active_mode(s, d_err, v_err)
        else:
            # Pass-through: publish global waypoints unchanged
            self.wpnt_pub.publish(self.gb_wpnts)

        # -- Status publish --
        self.status_pub.publish(String(data=self.mode))

    def _run_active_mode(self, s, d_err, v_err):
        """Call the solver, convert to WpntArray and check convergence"""
        # -- chi --
        theta_track = float(self.track_handler.theta_interpolator(s))
        chi = self._normalize_angle(self.cur_yaw - theta_track)

        # -- ax, ay --
        ax, ay = self._get_accelerations()

        # -- solver call --
        V = max(self.cur_vs, 0.5)  # minimum velocity for solver stability

        # on a cold start, build a fake prev_solution from the global raceline
        # without prev_solution the solver copies the current state as a constant initial guess
        # → off-track this is an infeasible guess → SQP_RTI fails to converge
        # so provide the global raceline's (s, n, V) relative to the centerline as the initial guess
        prev_sol = self.prev_solution
        if prev_sol is None:
            prev_sol = self._build_global_raceline_guess(self.cur_s)
            rospy.loginfo("[LocalRacelineMux] Cold start: using global raceline as initial guess")

        t0 = time.time()
        try:
            raceline = self.planner.calc_raceline(
                s=self.cur_s,
                V=V,
                n=self.cur_d,
                chi=chi,
                ax=ax,
                ay=ay,
                safety_distance=self.safety_distance,
                prev_solution=prev_sol,
                V_max=self.params['vehicle_params']['v_max']
            )
            self.prev_solution = raceline
            dt_ms = (time.time() - t0) * 1000
            # log epsilon_n (track bound slack) — 0 is normal, larger means leaving the track
            eps_n = raceline.get('epsilon_n', np.zeros(1))
            eps_n_max = float(np.max(eps_n))
            n_min, n_max = float(np.min(raceline['n'])), float(np.max(raceline['n']))
            rospy.loginfo_throttle(2.0,
                "[LocalRacelineMux] Solver: %.1fms, d_err=%.3f, v_err=%.3f | "
                "input: s=%.2f V=%.2f n=%.3f chi=%.3f ax=%.2f ay=%.2f | "
                "output: V=[%.1f,%.1f] n=[%.3f,%.3f] | "
                "eps_n_max=%.4f (0=OK)",
                dt_ms, d_err, v_err,
                self.cur_s, V, self.cur_d, chi, ax, ay,
                raceline['V'][0], raceline['V'][-1],
                n_min, n_max,
                eps_n_max)
        except Exception as e:
            import traceback
            rospy.logerr("[LocalRacelineMux] Solver failed: %s\n%s", str(e), traceback.format_exc())
            self.wpnt_pub.publish(self.gb_wpnts)
            return

        # -- convergence check (both the solver end and the current position are close to the global line) --
        tail_start = max(0, len(raceline['n']) - 10)
        tail_s = raceline['s'][tail_start:] % self.track_length
        tail_n = raceline['n'][tail_start:]
        tail_V = raceline['V'][tail_start:]
        tail_n_ref = np.interp(tail_s, self.ref_s, self.ref_n, period=self.track_length)
        tail_v_ref = np.interp(tail_s, self.ref_s, self.ref_v, period=self.track_length)

        max_n_err = np.max(np.abs(tail_n - tail_n_ref))
        max_v_err = np.max(np.abs(tail_V - tail_v_ref))

        if (max_n_err < self.convergence_d_threshold and
                max_v_err < self.convergence_v_threshold and
                d_err < self.convergence_d_threshold and
                v_err < self.convergence_v_threshold):
            self.mode = self.PASSTHROUGH
            self.prev_solution = None
            rospy.logwarn("[LocalRacelineMux] -> PASSTHROUGH (converged)")
            self.wpnt_pub.publish(self.gb_wpnts)
            return

        # -- solver output → WpntArray --
        modified = self._raceline_to_wpntarray(raceline)
        self.wpnt_pub.publish(modified)

        # online_waypoints — publish only the solver segment (for RViz)
        online = self._raceline_to_online_wpntarray(raceline)
        self.online_wpnt_pub.publish(online)

    # =======================================================
    # Solver output → WpntArray conversion
    # =======================================================

    def _raceline_to_wpntarray(self, raceline):
        """
        Interpolate the solver output onto the global waypoint grid to build a WpntArray:
        copy the global WpntArray and overwrite only the solver horizon segment.
        """
        wpnts = list(self.gb_wpnts.wpnts)
        num_wpnts = len(wpnts)
        if num_wpnts < 2:
            return self.gb_wpnts

        wpnt_dist = wpnts[1].s_m - wpnts[0].s_m

        # unwrap the solver output s (so it increases monotonically)
        # np.unwrap's period arg needs numpy>=1.21 — unwrap manually for compatibility
        solver_s = raceline['s'].copy()
        for i in range(1, len(solver_s)):
            diff = solver_s[i] - solver_s[i - 1]
            if diff < -self.track_length / 2.0:
                solver_s[i:] += self.track_length
            elif diff > self.track_length / 2.0:
                solver_s[i:] -= self.track_length

        # indices of the global waypoint grid covered by the solver segment
        s_start = solver_s[0]
        s_end = solver_s[-1]
        idx_start = int(round((s_start % self.track_length) / wpnt_dist)) % num_wpnts

        n_cover = int(round((s_end - s_start) / wpnt_dist))
        n_cover = min(n_cover, num_wpnts)  # never more than one lap

        for i in range(n_cover):
            idx = (idx_start + i) % num_wpnts
            s_target = s_start + i * wpnt_dist  # unwrapped s

            # interpolate from the solver output
            V_interp = float(np.interp(s_target, solver_s, raceline['V']))
            n_interp = float(np.interp(s_target, solver_s, raceline['n']))
            chi_interp = float(np.interp(s_target, solver_s, raceline['chi']))
            ax_interp = float(np.interp(s_target, solver_s, raceline['ax']))
            ay_interp = float(np.interp(s_target, solver_s, raceline['ay']))
            x_interp = float(np.interp(s_target, solver_s, raceline['x']))
            y_interp = float(np.interp(s_target, solver_s, raceline['y']))
            z_interp = float(np.interp(s_target, solver_s, raceline['z']))

            s_mod = s_target % self.track_length
            theta_track = float(self.track_handler.theta_interpolator(s_mod))

            w = Wpnt()
            w.id = idx
            w.s_m = wpnts[idx].s_m
            w.d_m = n_interp
            w.x_m = x_interp
            w.y_m = y_interp
            w.z_m = z_interp
            w.psi_rad = self._normalize_angle(theta_track + chi_interp)
            w.kappa_radpm = ay_interp / max(V_interp ** 2, 0.01)
            w.vx_mps = V_interp
            w.ax_mps2 = ax_interp

            # Track bounds adjusted by lateral offset
            w_tr_right = float(np.interp(
                s_mod, self.track_handler.s, self.track_handler.w_tr_right,
                period=self.track_length))
            w_tr_left = float(np.interp(
                s_mod, self.track_handler.s, self.track_handler.w_tr_left,
                period=self.track_length))
            w.d_right = abs(w_tr_right) - n_interp
            w.d_left = w_tr_left - n_interp

            # mu from track
            w.mu_rad = float(self.track_handler.mu_interpolator(s_mod))

            wpnts[idx] = w

        out = WpntArray()
        out.header.stamp = rospy.Time.now()
        out.wpnts = wpnts
        return out

    def _raceline_to_online_wpntarray(self, raceline):
        """Convert the solver output points directly to a WpntArray (for RViz)"""
        wpnts = []
        cartesian = self.track_handler.sn2cartesian(
            raceline['s'] % self.track_length, raceline['n'])
        for i in range(len(raceline['s'])):
            w = Wpnt()
            w.id = i
            w.s_m = float(raceline['s'][i] % self.track_length)
            w.d_m = float(raceline['n'][i])
            w.x_m = float(cartesian[i, 0])
            w.y_m = float(cartesian[i, 1])
            w.z_m = float(cartesian[i, 2])
            s_mod = float(raceline['s'][i] % self.track_length)
            theta_track = float(self.track_handler.theta_interpolator(s_mod))
            w.psi_rad = self._normalize_angle(theta_track + float(raceline['chi'][i]))
            w.vx_mps = float(raceline['V'][i])
            w.ax_mps2 = float(raceline['ax'][i])
            w.kappa_radpm = float(raceline['ay'][i]) / max(float(raceline['V'][i]) ** 2, 0.01)
            wpnts.append(w)
        out = WpntArray()
        out.header.stamp = rospy.Time.now()
        out.wpnts = wpnts
        return out

    # =======================================================
    # Helpers
    # =======================================================

    def _build_global_raceline_guess(self, cur_s):
        """Build a fake prev_solution from the global raceline (for a cold start).
        The solver's __gen_raceline interpolates the initial guess from prev_solution.
        Built from the global raceline's (s_center, n_center, V) over the horizon.
        """
        horizon = self.optimization_horizon
        N = self.N_steps
        s_array = np.linspace(cur_s, cur_s + horizon, N)
        s_mod = s_array % self.track_length

        # interpolate s, n, V relative to the centerline from the global raceline
        # ref_s_center is a centerline s and may not be uniform, so interpolate with a period
        V_array = np.interp(s_mod, self.ref_s_center, self.ref_v, period=self.track_length)
        n_array = np.interp(s_mod, self.ref_s_center, self.ref_n_center, period=self.track_length)

        return {
            's': s_array,
            'V': V_array,
            'n': n_array,
            'chi': np.zeros(N),
            'ax': np.zeros(N),
            'ay': V_array ** 2 * np.interp(
                s_mod, self.track_handler.s, self.track_handler.Omega_z,
                period=self.track_length),
            'jx': np.zeros(N),
            'jy': np.zeros(N),
        }

    def _get_accelerations(self):
        """Return ax, ay: prev_solution/0 in sim, the IMU on the car"""
        if not self.is_sim:
            return self.cur_imu_ax, self.cur_imu_ay

        # Sim: use previous solution if available
        if self.prev_solution is not None:
            return float(self.prev_solution['ax'][0]), float(self.prev_solution['ay'][0])
        return 0.0, 0.0

    @staticmethod
    def _normalize_angle(angle):
        """Wrap angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


if __name__ == '__main__':
    try:
        node = LocalRacelineMux()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
