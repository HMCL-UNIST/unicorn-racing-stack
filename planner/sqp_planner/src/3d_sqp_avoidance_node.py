#!/usr/bin/env python3
"""
3D SQP Avoidance — 3D version of the 2D SQP avoidance node
    - smart_static mode removed (unused in 3D).
    - Wrap-around utils applied (circular_s_dist, signed_s_dist) to avoid
      start/end jumps at the s wrap seam.
    - Track3DValidator checks the spline portion (warm-start fallback on
      failure).
    - z_m = spline_z(s) for 3D output.
    - Local curvature velocity: v = ref.vx * sqrt(|1 - d*kappa|).
    - SLSQP tuned for stability: ftol=1e-3, maxiter=50, warm start kept.
    - Original structure / objective / constraints / SLSQP retained; only
      bugs are fixed.
"""
import time
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from f110_msgs.msg import Wpnt, WpntArray, Obstacle, ObstacleArray, OTWpntArray, OpponentTrajectory, OppWpnt, BehaviorStrategy
from dynamic_reconfigure.msg import Config
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Float32MultiArray, Float32
from scipy.optimize import minimize
from frenet_converter.frenet_converter import FrenetConverter
from std_msgs.msg import Bool
from copy import deepcopy

from ccma import CCMA
import trajectory_planning_helpers as tph
from tf.transformations import euler_from_quaternion

from track_3d_validator import (
    Track3DValidator,
    circular_s_dist,
    signed_s_dist,
)


class SQPAvoidance3DNode:
    def __init__(self):
        rospy.init_node('3d_sqp_avoidance_node')
        self.rate = rospy.Rate(20)

        # Params
        self.frenet_state = Odometry()
        self.local_wpnts = None
        self.lookahead = 15
        self.past_avoidance_d = []

        # Scaled waypoints params
        self.scaled_wpnts = None  # shape (N, 2): [s, d]
        self.scaled_wpnts_msg = WpntArray()
        self.scaled_vmax = None
        self.scaled_max_idx = None
        self.scaled_max_s = None
        self.scaled_delta_s = None
        self.gb_wpnts_list = None  # for Track3DValidator

        # Updated waypoints params
        self.wpnts_updated = None
        self.max_s_updated = None
        self.max_idx_updated = None

        # Obstacles params
        self.obs = ObstacleArray()
        self.obs_perception = ObstacleArray()
        self.obs_predict = ObstacleArray()

        # Opponent waypoint params
        self.opponent_waypoints = OpponentTrajectory()
        self.max_opp_idx = None
        self.opponent_wpnts_sm = None

        # OT params
        self.last_ot_side = ""
        self.ot_section_check = False

        # Solver params
        self.min_radius = 0.55
        self.max_kappa = 1 / self.min_radius
        self.width_car = 0.30
        self.avoidance_resolution = 20
        self.back_to_raceline_before = 5
        self.back_to_raceline_after = 5
        self.obs_traj_tresh = 2

        # Dynamic solver params
        self.down_sampled_delta_s = None
        self.global_traj_kappas = None

        # ROS Parameters
        self.measure = rospy.get_param("/measure", False)

        # Dynamic reconf params
        self.avoid_static_obs = True

        self.converter = None
        self.global_waypoints = None
        self.track_validator = None

        # CCMA init
        self.ccma = CCMA(w_ma=10, w_cc=3)

        # Subscribers
        rospy.Subscriber("/tracking/obstacles", ObstacleArray, self.obs_perception_cb)
        rospy.Subscriber("/opponent_prediction/obstacles", ObstacleArray, self.obs_prediction_cb)
        rospy.Subscriber("/car_state/odom_frenet", Odometry, self.state_frenet_cb)
        rospy.Subscriber("/car_state/odom", Odometry, self.state_cartesian_cb)
        rospy.Subscriber("/global_waypoints_scaled", WpntArray, self.scaled_wpnts_cb)
        rospy.Subscriber("/behavior_strategy", BehaviorStrategy, self.behavior_cb)
        rospy.Subscriber("/dynamic_sqp_tuner_node/parameter_updates", Config, self.dyn_param_cb)
        rospy.Subscriber("/global_waypoints", WpntArray, self.gb_cb)
        rospy.Subscriber("/global_waypoints_updated", WpntArray, self.updated_wpnts_cb)
        rospy.Subscriber("/opponent_trajectory", OpponentTrajectory, self.opponent_trajectory_cb)
        rospy.Subscriber("/ot_section_check", Bool, self.ot_sections_check_cb)

        # Publishers
        self.mrks_pub = rospy.Publisher("/planner/avoidance/markers_sqp", MarkerArray, queue_size=10)
        self.evasion_pub = rospy.Publisher("/planner/avoidance/otwpnts", OTWpntArray, queue_size=10)
        self.merger_pub = rospy.Publisher("/planner/avoidance/merger", Float32MultiArray, queue_size=10)
        if self.measure:
            self.measure_pub = rospy.Publisher("/planner/pspliner_sqp/latency", Float32, queue_size=10)

        self.converter = self.initialize_converter()

    ### Callbacks ###
    def obs_perception_cb(self, data: ObstacleArray):
        self.obs_perception = data
        self.obs_perception.obstacles = [obs for obs in data.obstacles if obs.is_static == True]
        if self.avoid_static_obs == True:
            self.obs.header = data.header
            self.obs.obstacles = self.obs_perception.obstacles + self.obs_predict.obstacles

    def obs_prediction_cb(self, data: ObstacleArray):
        self.obs_predict = data
        self.obs = self.obs_predict
        if self.avoid_static_obs == True:
            self.obs.obstacles = self.obs.obstacles + self.obs_perception.obstacles

    def state_frenet_cb(self, data: Odometry):
        self.frenet_state = data
        quaternion = [data.pose.pose.orientation.x, data.pose.pose.orientation.y,
                      data.pose.pose.orientation.z, data.pose.pose.orientation.w]
        _, _, yaw = euler_from_quaternion(quaternion)
        self.cur_yaw = yaw

    def state_cartesian_cb(self, msg):
        self.cur_x = msg.pose.pose.position.x
        self.cur_y = msg.pose.pose.position.y
        self.cur_v = msg.twist.twist.linear.x

    def gb_cb(self, data: WpntArray):
        self.global_waypoints = np.array([[wpnt.x_m, wpnt.y_m, wpnt.z_m] for wpnt in data.wpnts])
        self.gb_wpnts_list = data.wpnts
        # Init Track3DValidator once converter is ready
        if self.track_validator is None and self.converter is not None:
            self.track_validator = Track3DValidator(
                self.converter, self.gb_wpnts_list, safety_margin=0.15
            )

    def scaled_wpnts_cb(self, data: WpntArray):
        self.scaled_wpnts = np.array([[wpnt.s_m, wpnt.d_m] for wpnt in data.wpnts])
        self.scaled_wpnts_msg = data
        v_max = np.max(np.array([wpnt.vx_mps for wpnt in data.wpnts]))
        if self.scaled_vmax != v_max:
            self.scaled_vmax = v_max
            # BUGFIX: use len(), not .id. The original used
            # data.wpnts[-1].id = N-1 as a modulus, which aliased the last
            # waypoint (N-1) onto index 0 — an off-by-one wrap error.
            self.scaled_max_idx = len(data.wpnts)
            self.scaled_max_s = data.wpnts[-1].s_m
            self.scaled_delta_s = data.wpnts[1].s_m - data.wpnts[0].s_m

    def updated_wpnts_cb(self, data: WpntArray):
        self.wpnts_updated = data.wpnts[:-1]
        self.max_s_updated = self.wpnts_updated[-1].s_m
        # BUGFIX: same off-by-one. Use the array length.
        self.max_idx_updated = len(self.wpnts_updated)

    def behavior_cb(self, data: BehaviorStrategy):
        self.local_wpnts = np.array([[wpnt.s_m, wpnt.d_m] for wpnt in data.local_wpnts])

    def opponent_trajectory_cb(self, data: OpponentTrajectory):
        self.opponent_waypoints = data.oppwpnts
        self.max_opp_idx = len(data.oppwpnts) - 1
        self.opponent_wpnts_sm = np.array([wpnt.s_m for wpnt in data.oppwpnts])

    def ot_sections_check_cb(self, data: Bool):
        self.ot_section_check = data.data

    def dyn_param_cb(self, params: Config):
        self.evasion_dist = rospy.get_param("dynamic_sqp_tuner_node/evasion_dist", 0.65)
        self.obs_traj_tresh = rospy.get_param("dynamic_sqp_tuner_node/obs_traj_tresh", 1.5)
        self.spline_bound_mindist = rospy.get_param("dynamic_sqp_tuner_node/spline_bound_mindist", 0.2)
        self.lookahead = rospy.get_param("dynamic_sqp_tuner_node/lookahead_dist", 15)
        self.avoidance_resolution = rospy.get_param("dynamic_sqp_tuner_node/avoidance_resolution", 20)
        self.back_to_raceline_before = rospy.get_param("dynamic_sqp_tuner_node/back_to_raceline_before", 5)
        self.back_to_raceline_after = rospy.get_param("dynamic_sqp_tuner_node/back_to_raceline_after", 5)
        self.avoid_static_obs = rospy.get_param("dynamic_sqp_tuner_node/avoid_static_obs", True)

        rospy.loginfo(
            f"[3D SQP] Dyn reconf: evasion_dist={self.evasion_dist}, obs_tresh={self.obs_traj_tresh}, "
            f"mindist={self.spline_bound_mindist}, lookahead={self.lookahead}, "
            f"res={self.avoidance_resolution}, back_before={self.back_to_raceline_before}, "
            f"back_after={self.back_to_raceline_after}, avoid_static={self.avoid_static_obs}"
        )

    def loop(self):
        rospy.loginfo("[3D SQP] Waiting for messages and services...")
        rospy.wait_for_message("/global_waypoints_scaled", WpntArray)
        rospy.wait_for_message("/car_state/odom", Odometry)
        rospy.wait_for_message("/car_state/odom_frenet", Odometry)
        rospy.wait_for_message("/dynamic_sqp_tuner_node/parameter_updates", Config)
        rospy.wait_for_message("/behavior_strategy", BehaviorStrategy)
        rospy.wait_for_message("/global_waypoints_updated", WpntArray)
        rospy.loginfo("[3D SQP] Ready!")

        while not rospy.is_shutdown():
            start_time = time.perf_counter()
            obs = deepcopy(self.obs)
            mrks = MarkerArray()
            frenet_state = self.frenet_state
            self.current_d = frenet_state.pose.pose.position.y
            self.cur_s = frenet_state.pose.pose.position.x

            # Pre-requisites ready? (callbacks may not have all fired yet on very first tick)
            ready = (
                self.scaled_wpnts is not None
                and self.wpnts_updated is not None
                and self.max_idx_updated is not None
                and self.scaled_max_s is not None
            )
            if not ready:
                self.rate.sleep()
                continue

            # Obstacle pre-processing — wrap-safe forward distance
            obs.obstacles = sorted(obs.obstacles, key=lambda o: o.s_start)
            considered_obs = []
            for ob in obs.obstacles:
                s_forward = signed_s_dist(self.cur_s, ob.s_start, self.scaled_max_s)
                # Only consider obstacles ahead of us within lookahead and
                # whose d_center is close to the raceline.
                # If /opponent_trajectory has not been received yet, skip
                # dynamic obstacles (avoid circular_s_dist(None)).
                if not (abs(ob.d_center) < self.obs_traj_tresh and 0 <= s_forward < self.lookahead):
                    continue
                if not ob.is_static and self.opponent_wpnts_sm is None:
                    # Dynamic obstacle but /opponent_trajectory not yet
                    # received — skip this cycle (or treat as static).
                    rospy.logwarn_throttle(
                        2.0, "[3D SQP] dynamic obstacle received but /opponent_trajectory not yet available — skipping"
                    )
                    continue
                considered_obs.append(ob)

            if len(considered_obs) > 0 and self.ot_section_check:
                evasion_x, evasion_y, evasion_s, evasion_d, evasion_v = self.sqp_solver(
                    considered_obs, frenet_state.pose.pose.position.x
                )
                if len(evasion_s) > 0:
                    self.merger_pub.publish(Float32MultiArray(
                        data=[considered_obs[-1].s_end % self.scaled_max_s,
                              evasion_s[-1] % self.scaled_max_s]
                    ))
            else:
                mrks = MarkerArray()
                del_mrk = Marker(header=rospy.Header(stamp=rospy.Time.now()))
                del_mrk.action = Marker.DELETEALL
                mrks.markers = [del_mrk]
                self.mrks_pub.publish(mrks)

            if self.measure:
                self.measure_pub.publish(Float32(data=time.perf_counter() - start_time))
            self.rate.sleep()

    def sqp_solver(self, considered_obs: list, cur_s: float):
        # Initial guess obstacle (ROC bounds)
        initial_guess_object = self.group_objects(considered_obs)

        # wrap-safe closest-index lookup on scaled_wpnts (s column only)
        initial_guess_object_start_idx = int(np.argmin(
            circular_s_dist(self.scaled_wpnts[:, 0], initial_guess_object.s_start, self.scaled_max_s)
        ))
        initial_guess_object_end_idx = int(np.argmin(
            circular_s_dist(self.scaled_wpnts[:, 0], initial_guess_object.s_end, self.scaled_max_s)
        ))

        gb_idxs = np.array(range(
            initial_guess_object_start_idx,
            initial_guess_object_start_idx + (initial_guess_object_end_idx - initial_guess_object_start_idx) % self.scaled_max_idx
        )) % self.scaled_max_idx

        if len(gb_idxs) < 20:
            gb_idxs = [int(initial_guess_object.s_center / self.scaled_delta_s + i) % self.scaled_max_idx for i in range(20)]

        side, initial_apex = self._more_space(initial_guess_object, self.scaled_wpnts_msg.wpnts, gb_idxs)
        self.desired_side = side
        kappas = np.array([self.scaled_wpnts_msg.wpnts[gb_idx].kappa_radpm for gb_idx in gb_idxs])
        max_kappa = np.max(np.abs(kappas))
        outside = "left" if np.sum(kappas) < 0 else "right"

        # Enlongate ROC if overtaking on outside
        if side == outside:
            for i in range(len(considered_obs)):
                considered_obs[i].s_end = considered_obs[i].s_end + (considered_obs[i].s_end - considered_obs[i].s_start) % self.max_s_updated * max_kappa * (self.width_car + self.evasion_dist)

        min_s_obs_start = self.scaled_max_s
        max_s_obs_end = 0
        for ob in considered_obs:
            if ob.s_start < min_s_obs_start:
                min_s_obs_start = ob.s_start
            if ob.s_end > max_s_obs_end:
                max_s_obs_end = ob.s_end

        start_avoidance = cur_s
        end_avoidance = max_s_obs_end + self.back_to_raceline_after

        s_avoidance = np.linspace(start_avoidance, end_avoidance, self.avoidance_resolution)
        self.down_sampled_delta_s = s_avoidance[1] - s_avoidance[0]

        # wrap-safe nearest scaled-waypoint index per s_avoidance
        s_avoid_mod = s_avoidance % self.scaled_max_s
        scaled_wpnts_indices = np.array([
            int(np.argmin(circular_s_dist(self.scaled_wpnts[:, 0], s, self.scaled_max_s)))
            for s in s_avoid_mod
        ])
        corresponding_scaled_wpnts = [self.scaled_wpnts_msg.wpnts[i] for i in scaled_wpnts_indices]
        bounds = np.array([
            (-wpnt.d_right + self.spline_bound_mindist, wpnt.d_left - self.spline_bound_mindist)
            for wpnt in corresponding_scaled_wpnts
        ])

        # numerical curvature at each s_avoidance point
        x_global_points = np.array([wpnt.x_m for wpnt in corresponding_scaled_wpnts])
        y_global_points = np.array([wpnt.y_m for wpnt in corresponding_scaled_wpnts])
        x_prime = np.diff(x_global_points)
        x_prime = np.where(x_prime == 0, 1e-6, x_prime)
        y_prime = np.diff(y_global_points)
        y_prime = np.where(y_prime == 0, 1e-6, y_prime)
        x_prime_prime = np.diff(x_prime)
        y_prime_prime = np.diff(y_prime)
        x_prime = x_prime[:-1]
        y_prime = y_prime[:-1]
        self.global_traj_kappas = (x_prime * y_prime_prime - y_prime * x_prime_prime) / ((x_prime ** 2 + y_prime ** 2) ** (3 / 2))

        # Obstacle indices / centerline / min distance on downsampled grid
        self.obs_downsampled_indices = np.array([])
        self.obs_downsampled_center_d = np.array([])
        self.obs_downsampled_min_dist = np.array([])

        for ob in considered_obs:
            # wrap-unwrap ob.s_end across lap boundary.
            # Obstacle publisher emits s_start/s_end modulo scaled_max_s,
            # so when the obstacle straddles s=0 the raw s_end < s_start.
            # argmin on monotonic s_avoidance then gives obs_idx_end <
            # obs_idx_start, and np.full(obs_idx_end - obs_idx_start + 1, ...)
            # below crashes with "negative dimensions are not allowed".
            # Unwrap s_end relative to s_start so downstream index math
            # stays monotonic; no effect when s_end >= s_start already.
            ob_s_start = ob.s_start
            ob_s_end = ob.s_end
            if ob_s_end < ob_s_start:
                ob_s_end += self.scaled_max_s
            obs_idx_start = np.abs(s_avoidance - ob_s_start).argmin()
            obs_idx_end = np.abs(s_avoidance - ob_s_end).argmin()

            if obs_idx_start < len(s_avoidance) - 2:
                if ob.is_static or obs_idx_end == obs_idx_start:
                    if obs_idx_end == obs_idx_start:
                        obs_idx_end = obs_idx_start + 1
                    self.obs_downsampled_indices = np.append(self.obs_downsampled_indices, np.arange(obs_idx_start, obs_idx_end + 1))
                    self.obs_downsampled_center_d = np.append(self.obs_downsampled_center_d, np.full(obs_idx_end - obs_idx_start + 1, (ob.d_left + ob.d_right) / 2))
                    self.obs_downsampled_min_dist = np.append(self.obs_downsampled_min_dist, np.full(obs_idx_end - obs_idx_start + 1, (ob.d_left - ob.d_right) / 2 + self.width_car + self.evasion_dist))
                else:
                    indices = np.arange(obs_idx_start, obs_idx_end + 1)
                    self.obs_downsampled_indices = np.append(self.obs_downsampled_indices, indices)
                    # wrap-safe nearest opponent waypoint per downsampled index
                    opp_wpnts_idx = [
                        int(np.argmin(circular_s_dist(
                            self.opponent_wpnts_sm,
                            s_avoidance[int(idx)] % self.scaled_max_s,
                            self.scaled_max_s
                        ))) for idx in indices
                    ]
                    d_opp_downsampled_array = np.array([self.opponent_waypoints[i].d_m for i in opp_wpnts_idx])
                    self.obs_downsampled_center_d = np.append(self.obs_downsampled_center_d, d_opp_downsampled_array)
                    self.obs_downsampled_min_dist = np.append(self.obs_downsampled_min_dist, np.full(obs_idx_end - obs_idx_start + 1, self.width_car + self.evasion_dist))
            else:
                rospy.loginfo_throttle(2.0, f"[3D SQP] obs end idx < start idx (n_obs={len(considered_obs)} start={ob.s_start:.2f} end={ob.s_end:.2f})")

        self.obs_downsampled_indices = self.obs_downsampled_indices.astype(int)

        # min radius based on speed
        clipped_speed = np.clip(self.frenet_state.twist.twist.linear.x, 1, a_max=None)
        radius_speed = min([clipped_speed, self.wpnts_updated[(scaled_wpnts_indices[0]) % self.max_idx_updated].vx_mps])
        self.min_radius = np.interp(radius_speed, [1, 6, 7], [0.2, 2, 4])
        self.max_kappa = 1 / self.min_radius

        # Initial guess — warm start if available
        if len(self.past_avoidance_d) == len(s_avoidance):
            initial_guess = self.past_avoidance_d
        else:
            initial_guess = np.full(len(s_avoidance), initial_apex)

        result = self.solve_sqp(initial_guess, bounds)

        if result.success:
            # Build global-density s-array and interpolate d
            n_global_avoidance_points = max(2, int((end_avoidance - start_avoidance) / self.scaled_delta_s))
            s_array = np.linspace(start_avoidance, end_avoidance, n_global_avoidance_points)
            evasion_d = np.interp(s_array, s_avoidance, result.x)
            evasion_s = np.mod(s_array, self.scaled_max_s)

            # Cartesian + CCMA smoothing
            resp = self.converter.get_cartesian(evasion_s, evasion_d).transpose()
            smoothed_xy_points = self.ccma.filter(resp)

            # BUGFIX: previously we re-projected the smoothed XY via
            # get_frenet, a 2D nearest projection that on 3D tracks with
            # XY overlap can flip a sample's s to a different layer.
            # s_array is already fixed (linspace) and CCMA only lightly
            # smooths XY without shifting samples along s, so we keep s
            # as-is and recompute d as a geometric normal projection —
            # same approach as the recovery / static fix.
            evasion_x = smoothed_xy_points[:, 0]
            evasion_y = smoothed_xy_points[:, 1]
            # evasion_s is already set above: np.mod(s_array, scaled_max_s)

            ref_x = np.asarray(self.converter.spline_x(evasion_s)).flatten()
            ref_y = np.asarray(self.converter.spline_y(evasion_s)).flatten()
            dx_ds = np.asarray(self.converter.spline_x(evasion_s, 1)).flatten()
            dy_ds = np.asarray(self.converter.spline_y(evasion_s, 1)).flatten()
            t_norm = np.sqrt(dx_ds * dx_ds + dy_ds * dy_ds) + 1e-9
            nx = -dy_ds / t_norm
            ny = dx_ds / t_norm
            evasion_d = (evasion_x - ref_x) * nx + (evasion_y - ref_y) * ny

            # Track3DValidator: reject if spline crosses track boundary
            if self.track_validator is not None:
                xy_check = np.column_stack([evasion_x, evasion_y])
                valid, first_invalid, fail_stage = self.track_validator.validate_trajectory(
                    xy_check, evasion_s, evasion_d
                )
                if not valid:
                    rospy.logwarn_throttle(
                        1.0,
                        f"[3D SQP] spline invalid @ idx={first_invalid} stage={fail_stage} — dropping evasion"
                    )
                    self.past_avoidance_d = []
                    empty_msg = OTWpntArray(header=rospy.Header(stamp=rospy.Time.now(), frame_id="map"))
                    self.evasion_pub.publish(empty_msg)
                    self.visualize_sqp([], [], [], [], [])
                    return [], [], [], [], []

            evasion_coords = np.column_stack((evasion_x, evasion_y))
            evasion_psi, evasion_kappa = tph.calc_head_curv_num.calc_head_curv_num(
                path=evasion_coords,
                el_lengths=0.1 * np.ones(len(evasion_coords) - 1),
                is_closed=False,
            )
            evasion_psi += np.pi / 2

            # Local curvature velocity: v = v_ref * sqrt(|1 - d*kappa|)
            ref_vx = np.array([wpnt.vx_mps for wpnt in corresponding_scaled_wpnts])
            ref_kappa = np.array([wpnt.kappa_radpm for wpnt in corresponding_scaled_wpnts])
            v_ds = np.interp(s_array, s_avoidance, ref_vx)
            k_ds = np.interp(s_array, s_avoidance, ref_kappa)
            evasion_v = v_ds * np.sqrt(np.maximum(np.abs(1.0 - evasion_d * k_ds), 1e-4))

            # z from spline_z(s) for 3D output
            try:
                evasion_z = self.converter.spline_z(evasion_s).astype(float)
            except Exception:
                evasion_z = np.zeros_like(evasion_s)

            evasion_wpnts_msg = OTWpntArray(header=rospy.Header(stamp=rospy.Time.now(), frame_id="map"))
            evasion_wpnts = [
                Wpnt(
                    id=i, s_m=float(s), d_m=float(d), x_m=float(x), y_m=float(y), z_m=float(z),
                    psi_rad=float(p), kappa_radpm=float(k), vx_mps=float(v),
                )
                for i, (x, y, z, s, d, p, k, v) in enumerate(
                    zip(evasion_x, evasion_y, evasion_z, evasion_s, evasion_d, evasion_psi, evasion_kappa, evasion_v)
                )
            ]
            evasion_wpnts_msg.wpnts = evasion_wpnts

            self.past_avoidance_d = result.x
            self.last_ot_side = "left" if np.mean(evasion_d) > 0 else "right"
        else:
            evasion_x, evasion_y, evasion_s, evasion_d, evasion_v = [], [], [], [], []
            evasion_wpnts_msg = OTWpntArray(header=rospy.Header(stamp=rospy.Time.now(), frame_id="map"))
            evasion_wpnts_msg.wpnts = []
            self.past_avoidance_d = []

        self.evasion_pub.publish(evasion_wpnts_msg)
        self.visualize_sqp(evasion_s, evasion_d, evasion_x, evasion_y, evasion_v)

        return evasion_x, evasion_y, evasion_s, evasion_d, evasion_v

    ### Optimal Trajectory Solver ###
    def objective_function(self, d):
        return np.sum(d ** 2) * 10 + np.sum(np.diff(np.diff(d)) ** 2) * 100 + (np.diff(d)[0] ** 2) * 1000

    ## Constraints ##
    def start_on_raceline_constraint(self, d):
        # d[0] pinned to current_d; last two forced to raceline → smooth return
        return np.array([0.02 - abs(d[0] - self.current_d), 0.02 - abs(d[-2]), 0.02 - abs(d[-1])])

    def psi_constraint(self, d):
        delta_s = self.down_sampled_delta_s
        e_psi = self.converter.get_e_psi(self.cur_x, self.cur_y, self.cur_yaw)
        desired_dd = np.tan(e_psi) * delta_s
        # BUGFIX: the original wrapped desired_dd in abs(), erasing its
        # sign. So with e_psi < 0 (vehicle heading right of the tangent)
        # it still forced d[1]-d[0] > 0, starting the path in the wrong
        # direction. Keep the sign.
        return np.array([0.02 - abs((d[1] - d[0]) - desired_dd)])

    def obstacle_constraint(self, d):
        distance_to_obstacle = np.abs(d[self.obs_downsampled_indices] - self.obs_downsampled_center_d)
        return distance_to_obstacle - self.obs_downsampled_min_dist

    def consecutive_points_constraint(self, d):
        points = d[self.obs_downsampled_indices]
        violations = []
        for i in range(len(points) - 1):
            if not ((points[i] > self.obs_downsampled_center_d[i] and points[i + 1] > self.obs_downsampled_center_d[i + 1]) or
                    (points[i] < self.obs_downsampled_center_d[i] and points[i + 1] < self.obs_downsampled_center_d[i + 1])):
                violations.append(-1)
            else:
                violations.append(1)
        return violations

    def turning_radius_constraint(self, d):
        # BUGFIX: the original treated only d''(s) as kappa and dropped
        # the raceline's kappa_ref. The actual path curvature is
        # approximately kappa_ref + d''(s); in corners kappa_ref already
        # consumes most of the friction budget, so bounding only d''(s)
        # lets the total curvature exceed the limit. global_traj_kappas
        # was computed but never read in the original — wire it in here.
        y_prime = np.diff(d)
        y_prime = np.where(y_prime == 0, 1e-6, y_prime)
        y_prime_prime = np.diff(y_prime)
        # y_prime_prime has shape (N-2,); global_traj_kappas is (N-2,) too.
        kappa_offset = y_prime_prime / (self.down_sampled_delta_s ** 2)
        kappa_total = self.global_traj_kappas + kappa_offset
        mu = 0.318
        g = 9.81
        kappa_limit = mu * g / ((self.frenet_state.twist.twist.linear.x + 1e-6) ** 2)
        return abs(kappa_limit) - abs(kappa_total)

    def first_point_constraint(self, d):
        return np.array([self.down_sampled_delta_s - abs(d[1] - d[0])])

    def side_consistency_constraint(self, d):
        # BUGFIX: the original applied the side constraint to all d,
        # which conflicts with d[0] = current_d when the car is on the
        # opposite side (making the solver infeasible). Apply it only
        # over the obstacle range indices.
        if len(self.obs_downsampled_indices) == 0:
            return np.array([1.0])
        d_obs = d[self.obs_downsampled_indices]
        if self.desired_side == "left":
            return d_obs
        elif self.desired_side == "right":
            return -d_obs
        else:
            return d_obs

    def solve_sqp(self, d_array, track_boundaries):
        result = minimize(
            self.objective_function, d_array, method='SLSQP', jac='10-point',
            bounds=track_boundaries,
            constraints=[
                {'type': 'eq', 'fun': self.start_on_raceline_constraint},
                {'type': 'eq', 'fun': self.psi_constraint},
                {'type': 'ineq', 'fun': self.obstacle_constraint},
                {'type': 'ineq', 'fun': self.consecutive_points_constraint},
                {'type': 'ineq', 'fun': self.turning_radius_constraint},
                {'type': 'ineq', 'fun': self.first_point_constraint},
                {'type': 'ineq', 'fun': self.side_consistency_constraint},
            ],
            options={'ftol': 1e-3, 'maxiter': 50, 'disp': False},
        )
        return result

    def group_objects(self, obstacles: list):
        initial_guess_object = obstacles[0]
        for ob in obstacles:
            if ob.d_left > initial_guess_object.d_left:
                initial_guess_object.d_left = ob.d_left
            if ob.d_right < initial_guess_object.d_right:
                initial_guess_object.d_right = ob.d_right
            if ob.s_start < initial_guess_object.s_start:
                initial_guess_object.s_start = ob.s_start
            if ob.s_end > initial_guess_object.s_end:
                initial_guess_object.s_end = ob.s_end
        initial_guess_object.s_center = (initial_guess_object.s_start + initial_guess_object.s_end) / 2
        return initial_guess_object

    def _more_space(self, obstacle: Obstacle, gb_wpnts, gb_idxs):
        left_boundary_mean = np.mean([gb_wpnts[gb_idx].d_left for gb_idx in gb_idxs])
        right_boundary_mean = np.mean([gb_wpnts[gb_idx].d_right for gb_idx in gb_idxs])
        left_gap = abs(left_boundary_mean - obstacle.d_left)
        right_gap = abs(right_boundary_mean + obstacle.d_right)
        min_space = self.evasion_dist + self.spline_bound_mindist

        if right_gap > min_space and left_gap < min_space:
            d_apex_right = obstacle.d_right - self.evasion_dist
            if d_apex_right > 0:
                d_apex_right = 0
            return "right", d_apex_right
        elif left_gap > min_space and right_gap < min_space:
            d_apex_left = obstacle.d_left + self.evasion_dist
            if d_apex_left < 0:
                d_apex_left = 0
            return "left", d_apex_left
        else:
            candidate_d_apex_left = obstacle.d_left + self.evasion_dist
            candidate_d_apex_right = obstacle.d_right - self.evasion_dist
            if abs(candidate_d_apex_left) <= abs(candidate_d_apex_right):
                if candidate_d_apex_left < 0:
                    candidate_d_apex_left = 0
                return "left", candidate_d_apex_left
            else:
                if candidate_d_apex_right > 0:
                    candidate_d_apex_right = 0
                return "right", candidate_d_apex_right

    ### Visualize SQP Rviz ###
    def visualize_sqp(self, evasion_s, evasion_d, evasion_x, evasion_y, evasion_v):
        mrks = MarkerArray()
        if len(evasion_s) == 0:
            del_mrk = Marker(header=rospy.Header(stamp=rospy.Time.now()))
            del_mrk.action = Marker.DELETEALL
            mrks.markers = [del_mrk]
            self.mrks_pub.publish(mrks)
            return

        # z from spline_z for 3D marker placement
        try:
            evasion_z = self.converter.spline_z(np.asarray(evasion_s)).astype(float)
        except Exception:
            evasion_z = np.zeros(len(evasion_s))

        for i in range(len(evasion_s)):
            mrk = Marker(header=rospy.Header(stamp=rospy.Time.now(), frame_id="map"))
            mrk.type = mrk.CYLINDER
            mrk.scale.x = 0.1
            mrk.scale.y = 0.1
            mrk.scale.z = evasion_v[i] / self.scaled_vmax
            mrk.color.a = 1.0
            mrk.color.g = 0.13
            mrk.color.r = 0.63
            mrk.color.b = 0.94
            mrk.id = i
            mrk.pose.position.x = evasion_x[i]
            mrk.pose.position.y = evasion_y[i]
            mrk.pose.position.z = float(evasion_z[i]) + evasion_v[i] / self.scaled_vmax / 2
            mrk.pose.orientation.w = 1
            mrks.markers.append(mrk)
        self.mrks_pub.publish(mrks)

    def initialize_converter(self):
        rospy.wait_for_message("/global_waypoints", WpntArray)
        converter = FrenetConverter(
            self.global_waypoints[:, 0], self.global_waypoints[:, 1], self.global_waypoints[:, 2]
        )
        rospy.loginfo("[3D SQP] initialized FrenetConverter object")
        # Build Track3DValidator now that converter + gb_wpnts_list ready
        if self.gb_wpnts_list is not None and self.track_validator is None:
            self.track_validator = Track3DValidator(
                converter, self.gb_wpnts_list, safety_margin=0.15
            )
        return converter


if __name__ == "__main__":
    node = SQPAvoidance3DNode()
    node.loop()
