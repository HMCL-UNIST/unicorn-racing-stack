#!/usr/bin/env python3
"""
3D Static Avoidance Node
    Based on: the 2D static avoidance node (structure)
    Ported from the 2D smart static avoidance node:
        - _decide_obstacle_strategy_gb_aware (direction decision)
        - Savitzky-Golay smoothing + GB transition blending
        - RVIZ debug markers (spline samples, more_space)
    Replaced: GridFilter → Track3DValidator (s-based boundary + d-bound)
    Added: 3D fields (z_m, mu_rad, d_right, d_left), marker throttle
"""
import copy
import time
from typing import List, Any, Tuple

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray
from scipy.interpolate import BPoly
from scipy.signal import savgol_filter
from dynamic_reconfigure.msg import Config
from f110_msgs.msg import (
    Obstacle, ObstacleArray, OTWpntArray, Wpnt, WpntArray, BehaviorStrategy,
)
from frenet_converter.frenet_converter import FrenetConverter
import tf.transformations as tf_trans
from track_3d_validator import Track3DValidator
import trajectory_planning_helpers as tph

# --- Smoothing parameters (from the 2D smart static avoidance node) ---
SMOOTH_OTWPNTS = True
SMOOTH_OTWPNTS_WINDOW = 51   # must be odd
SMOOTH_OTWPNTS_POLYORDER = 2
GB_BLEND_LEN = 40             # waypoints for quadratic easing back to GB

# --- Direction decision (from smart) ---
OPPOSITE_SPACE_THRESHOLD = 0.6  # meters


class StaticAvoidance3D:
    """
    3D static obstacle avoidance via spline generation.
    Uses Track3DValidator instead of GridFilter.
    """

    def __init__(self):
        self.name = "3d_static_avoidance_node"
        rospy.init_node(self.name)

        # State
        self.obs_in_interest = None
        self.gb_wpnts = WpntArray()
        self.gb_scaled_wpnts = WpntArray()
        self.gb_vmax = None
        self.gb_max_idx = None
        self.gb_max_s = None
        self.cur_s = 0.0
        self.cur_d = 0.0
        self.cur_vs = 0.0
        self.cur_x = 0.0
        self.cur_y = 0.0
        self.cur_yaw = 0.0
        self.waypoints = None
        self.frame_count = 0

        # Parameters
        self.from_bag = rospy.get_param("/from_bag", False)
        self.measuring = rospy.get_param("/measure", False)
        self.sampling_dist = rospy.get_param("/sampling_dist", 20.0)
        self.spline_scale = rospy.get_param("/spline_scale", 0.8)
        self.post_min_dist = rospy.get_param("/post_min_dist", 1.5)
        self.post_max_dist = rospy.get_param("/post_max_dist", 5.0)
        self.safety_margin = rospy.get_param("dyn_planner_tuner/static_avoidance/safety_margin", 0.15)

        self.evasion_dist = 0.65
        self.obs_traj_tresh = 0.3
        self.spline_bound_mindist = 0.2
        self.n_loc_wpnts = 80
        self.width_car = 0.30
        self.kd_obs_pred = 1.0
        self.fixed_pred_time = 0.15

        self.track_validator = None

        # Publishers (before subscribers — no callback race)
        self.mrks_pub = rospy.Publisher(
            "/planner/avoidance/markers", MarkerArray, queue_size=10)
        self.evasion_pub = rospy.Publisher(
            "/planner/avoidance/otwpnts", OTWpntArray, queue_size=10)
        self.closest_obs_pub = rospy.Publisher(
            "/planner/avoidance/considered_OBS", Marker, queue_size=10)
        self.pub_propagated = rospy.Publisher(
            "/planner/avoidance/propagated_obs", Marker, queue_size=10)
        self.debug_spline_pub = rospy.Publisher(
            "/planner/avoidance/temp_do_spline_markers", MarkerArray, queue_size=10)
        self.debug_space_pub = rospy.Publisher(
            "/planner/avoidance/more_space_debug", MarkerArray, queue_size=10)
        self.validity_pub = rospy.Publisher(
            "/planner/avoidance/3d_validity", MarkerArray, queue_size=10)
        if self.measuring:
            self.latency_pub = rospy.Publisher(
                "/planner/avoidance/latency", Float32, queue_size=10)

        # Init converter + validator BEFORE subscribers
        self.converter = self._init_converter()

        # Subscribers (safe now — converter exists)
        rospy.Subscriber("/behavior_strategy", BehaviorStrategy, self.behavior_cb)
        rospy.Subscriber("/car_state/odom_frenet", Odometry, self.state_frenet_cb)
        rospy.Subscriber("/car_state/odom", Odometry, self.state_cb)
        rospy.Subscriber("/global_waypoints", WpntArray, self.gb_cb)
        rospy.Subscriber("/global_waypoints_scaled", WpntArray, self.gb_scaled_cb)
        if not self.from_bag:
            rospy.Subscriber(
                "/dyn_planner_tuner/static_avoidance/parameter_updates", Config, self.dyn_param_cb)

        self.rate = rospy.Rate(20)  # same as original static avoidance

    # =========================================================================
    # INIT
    # =========================================================================

    def _init_converter(self) -> FrenetConverter:
        msg = rospy.wait_for_message("/global_waypoints", WpntArray)
        self.waypoints = np.array([[w.x_m, w.y_m, w.z_m] for w in msg.wpnts])
        converter = FrenetConverter(
            self.waypoints[:, 0], self.waypoints[:, 1], self.waypoints[:, 2])
        rospy.loginfo(f"[{self.name}] FrenetConverter initialized")
        return converter

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def behavior_cb(self, data: BehaviorStrategy):
        if len(data.overtaking_targets) != 0:
            self.obs_in_interest = data.overtaking_targets[0]
        else:
            self.obs_in_interest = None

    def state_frenet_cb(self, data: Odometry):
        self.cur_s = data.pose.pose.position.x
        self.cur_d = data.pose.pose.position.y
        self.cur_vs = data.twist.twist.linear.x

    def state_cb(self, data: Odometry):
        self.cur_x = data.pose.pose.position.x
        self.cur_y = data.pose.pose.position.y
        quat = data.pose.pose.orientation
        euler = tf_trans.euler_from_quaternion(
            [quat.x, quat.y, quat.z, quat.w])
        self.cur_yaw = euler[2]

    def gb_cb(self, data: WpntArray):
        self.waypoints = np.array(
            [[w.x_m, w.y_m, w.z_m] for w in data.wpnts])
        self.gb_wpnts = data
        if self.gb_vmax is None:
            self.gb_vmax = np.max([w.vx_mps for w in data.wpnts])
            self.gb_max_idx = data.wpnts[-1].id
            self.gb_max_s = data.wpnts[-1].s_m
        if self.track_validator is None:
            self.track_validator = Track3DValidator(
                converter=self.converter,
                gb_wpnts=data.wpnts,
                safety_margin=self.spline_bound_mindist,
            )
            rospy.loginfo(f"[{self.name}] Track3DValidator initialized")

    def gb_scaled_cb(self, data: WpntArray):
        self.gb_scaled_wpnts = data

    def dyn_param_cb(self, params: Config):
        self.evasion_dist = rospy.get_param("dyn_planner_tuner/static_avoidance/evasion_dist", 0.65)
        self.obs_traj_tresh = rospy.get_param("dyn_planner_tuner/static_avoidance/obs_traj_tresh", 0.3)
        self.spline_bound_mindist = rospy.get_param("dyn_planner_tuner/static_avoidance/spline_bound_mindist", 0.2)
        self.kd_obs_pred = rospy.get_param("dyn_planner_tuner/static_avoidance/kd_obs_pred", 1.0)
        self.fixed_pred_time = rospy.get_param("dyn_planner_tuner/static_avoidance/fixed_pred_time", 0.15)
        self.sampling_dist = rospy.get_param("dyn_planner_tuner/static_avoidance/post_sampling_dist", 20.0)
        self.spline_scale = rospy.get_param("dyn_planner_tuner/static_avoidance/spline_scale", 0.8)
        self.post_min_dist = rospy.get_param("dyn_planner_tuner/static_avoidance/post_min_dist", 1.5)
        self.post_max_dist = rospy.get_param("dyn_planner_tuner/static_avoidance/post_max_dist", 5.0)
        self.safety_margin = rospy.get_param("dyn_planner_tuner/static_avoidance/safety_margin", 0.15)
        if hasattr(self, 'track_validator') and self.track_validator is not None:
            self.track_validator.safety_margin = self.safety_margin

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    def loop(self):
        rospy.loginfo(f"[{self.name}] Waiting for messages...")
        rospy.wait_for_message("/global_waypoints_scaled", WpntArray)
        rospy.wait_for_message("/car_state/odom", Odometry)
        rospy.wait_for_message("/dyn_planner_tuner/static_avoidance/parameter_updates", Config)
        rospy.loginfo(f"[{self.name}] Ready! (20Hz)")

        while not rospy.is_shutdown():
            if self.measuring:
                start = time.perf_counter()
            self.frame_count += 1

            gb_scaled_wpnts = self.gb_scaled_wpnts.wpnts
            wpnts = OTWpntArray()
            mrks = MarkerArray()

            if self.obs_in_interest is not None:
                wpnts, mrks = self.do_spline(
                    obs=copy.deepcopy(self.obs_in_interest),
                    gb_wpnts=gb_scaled_wpnts)
            else:
                del_mrk = Marker()
                del_mrk.header.stamp = rospy.Time.now()
                del_mrk.action = Marker.DELETEALL
                mrks.markers.append(del_mrk)

            if self.measuring:
                end = time.perf_counter()
                self.latency_pub.publish(end - start)
            self.evasion_pub.publish(wpnts)
            self.mrks_pub.publish(mrks)
            self.rate.sleep()

    # =========================================================================
    # _more_space — ported from smart (gb-aware direction decision)
    # =========================================================================

    def _more_space(self, obstacle: Obstacle, gb_wpnts: List[Any],
                    obs_s_idx: int) -> Tuple[str, float]:
        """
        Determine evasion direction and d_apex.
        Uses _decide_obstacle_strategy_gb_aware from smart for consistent direction.
        """
        _, evasion_direction = self._decide_obstacle_strategy_gb_aware(
            obstacle.s_center, obstacle.d_center, obstacle.size / 2,
            gb_wpnts)

        if evasion_direction == "left":
            d_apex = obstacle.d_left + self.evasion_dist
            if d_apex < 0:
                d_apex = 0
        else:
            d_apex = obstacle.d_right - self.evasion_dist
            if d_apex > 0:
                d_apex = 0

        # Debug: publish more_space markers
        self._publish_space_debug(obstacle, gb_wpnts, obs_s_idx, evasion_direction)

        return evasion_direction, d_apex

    def _decide_obstacle_strategy_gb_aware(
            self, obs_s, obs_d, obs_radius, gb_wpnts) -> Tuple[str, str]:
        """
        GB-aware direction decision.
        BUGFIX: the original only compared gb_wp.d_left against a threshold
        and ignored the obstacle position. If an obstacle hugs the left wall
        while the raceline-to-left-wall distance is large, the real gap for
        passing can be near zero, so the planner would pick the narrow side
        and the Track3DValidator would reject the resulting spline.
        New logic computes the actual gap on each side (obstacle edge <->
        wall) and picks the side with more room.
        """
        if gb_wpnts is None or len(gb_wpnts) == 0:
            return ('right', 'left') if obs_d > 0 else ('left', 'right')

        wpnt_dist = gb_wpnts[1].s_m - gb_wpnts[0].s_m
        obs_s_idx = int(obs_s / wpnt_dist) % len(gb_wpnts)
        gb_wp = gb_wpnts[obs_s_idx]

        # Obstacle left/right edges in d (sign convention: +d = left)
        obs_d_left = obs_d + obs_radius
        obs_d_right = obs_d - obs_radius

        # Gap = obstacle edge to wall
        left_gap = gb_wp.d_left - obs_d_left          # positive if room on left
        right_gap = gb_wp.d_right + obs_d_right       # positive if room on right
        # (right wall at d = -gb_wp.d_right; gap = obs_d_right - (-gb_wp.d_right))

        min_space = self.width_car + self.evasion_dist

        left_ok = left_gap >= min_space
        right_ok = right_gap >= min_space

        if left_ok and not right_ok:
            return ('right', 'left')
        elif right_ok and not left_ok:
            return ('left', 'right')
        else:
            # Both sides acceptable or both tight: pick the wider gap
            if left_gap >= right_gap:
                return ('right', 'left')
            else:
                return ('left', 'right')

    def _publish_space_debug(self, obstacle, gb_wpnts, obs_s_idx, direction):
        """Publish left/right gap debug markers."""
        mrks = MarkerArray()
        gb_wp = gb_wpnts[obs_s_idx]

        for i, (label, d_val, r, g, b) in enumerate([
            ("left", gb_wp.d_left, 0.0, 1.0, 0.0),
            ("right", gb_wp.d_right, 1.0, 0.0, 0.0),
        ]):
            resp = self.converter.get_cartesian(
                np.array([obstacle.s_center]), np.array([d_val]))
            mrk = Marker()
            mrk.header.frame_id = "map"
            mrk.header.stamp = rospy.Time.now()
            mrk.type = Marker.SPHERE
            mrk.id = i
            mrk.scale.x = mrk.scale.y = mrk.scale.z = 0.15
            mrk.color.a = 1.0
            mrk.color.r, mrk.color.g, mrk.color.b = r, g, b
            mrk.pose.position.x = float(resp[0])
            mrk.pose.position.y = float(resp[1])
            z = float(self.converter.spline_z(np.array([obstacle.s_center])))
            mrk.pose.position.z = z
            mrk.pose.orientation.w = 1
            mrks.markers.append(mrk)

        self.debug_space_pub.publish(mrks)

    # =========================================================================
    # DO_SPLINE — 3D version with smart smoothing
    # =========================================================================

    def do_spline(self, obs: Obstacle, gb_wpnts) -> Tuple[OTWpntArray, MarkerArray]:
        """
        3D spline avoidance trajectory.
        - XY BPoly spline + z from spline_z(s)
        - Savitzky-Golay smoothing (from smart)
        - GB transition blending (quadratic easing)
        - Track3DValidator batch validation
        - Full Wpnt fields
        """
        mrks = MarkerArray()
        wpnts = OTWpntArray()
        wpnts.header.stamp = rospy.Time.now()
        wpnts.header.frame_id = "map"

        if not gb_wpnts or len(gb_wpnts) < 2:
            return wpnts, mrks

        wpnt_dist = gb_wpnts[1].s_m - gb_wpnts[0].s_m
        ref_max_idx = self.gb_max_idx
        if ref_max_idx is None or ref_max_idx <= 0:
            return wpnts, mrks

        if not obs.is_static:
            return wpnts, mrks

        # Distance check
        pre_dist = (obs.s_center - self.cur_s) % self.gb_max_s
        if pre_dist < 0.5 or pre_dist > self.gb_max_s / 2:
            return wpnts, mrks

        obs_s_idx = int(obs.s_center / wpnt_dist) % ref_max_idx

        # --- Direction & apex (smart gb-aware) ---
        more_space, d_apex = self._more_space(obs, gb_wpnts, obs_s_idx)

        # --- Control points along s ---
        s_list = [obs.s_center]
        d_list = [d_apex]
        post_dist = min(
            min(max(pre_dist, self.post_min_dist), self.post_max_dist),
            self.gb_max_s / 2)
        num_post_ref = int(post_dist // self.sampling_dist) + 1

        for i in range(num_post_ref):
            s_list.append(obs.s_center + post_dist * ((i + 1) / num_post_ref))
            d_list.append(d_apex * (1 - (i + 1) / num_post_ref))

        s_array = np.array(s_list) % self.gb_max_s
        d_array = np.array(d_list)
        s_idx = np.round(s_array / wpnt_dist).astype(int) % ref_max_idx

        # --- Frenet → XY control points ---
        resp = self.converter.get_cartesian(s_array, d_array)

        points = [[self.cur_x, self.cur_y]]
        tangents = [[np.cos(self.cur_yaw), np.sin(self.cur_yaw)]]
        for i in range(len(s_idx)):
            points.append(resp[:, i])
            tangents.append(np.array([
                np.cos(gb_wpnts[s_idx[i]].psi_rad),
                np.sin(gb_wpnts[s_idx[i]].psi_rad)]))

        tangents = np.dot(tangents, self.spline_scale * np.eye(2))
        points = np.asarray(points)
        nPoints, dim = points.shape

        # --- BPoly spline (XY) ---
        dp = np.linalg.norm(np.diff(points, axis=0), axis=1)
        d_cum = np.hstack([[0], np.cumsum(dp)])
        l = d_cum[-1]
        if l < 1e-3:
            return wpnts, mrks
        nSamples = max(int(l / wpnt_dist), 2)
        s_param = np.linspace(0, l, nSamples)

        spline_result = np.empty([nPoints, dim], dtype=object)
        for i, ref in enumerate(points):
            spline_result[i, :] = list(zip(ref, tangents[i]))

        samples = np.zeros([nSamples, dim])
        for i in range(dim):
            poly = BPoly.from_derivatives(d_cum, spline_result[:, i])
            samples[:, i] = poly(s_param)

        # --- Savitzky-Golay smoothing (from smart) ---
        if SMOOTH_OTWPNTS and len(samples) >= SMOOTH_OTWPNTS_WINDOW:
            start_pt = samples[0].copy()
            end_pt = samples[-1].copy()
            samples[:, 0] = savgol_filter(samples[:, 0], SMOOTH_OTWPNTS_WINDOW, SMOOTH_OTWPNTS_POLYORDER)
            samples[:, 1] = savgol_filter(samples[:, 1], SMOOTH_OTWPNTS_WINDOW, SMOOTH_OTWPNTS_POLYORDER)
            # BUGFIX: original only blended the end. The start just had
            # samples[0] = start_pt with no taper, so the first few samples
            # (savgol-shifted) created a visible kink at the very start.
            # Mirror the end-side blending at the start.
            blend_len = min(5, len(samples) - 1)
            # Start-side taper: samples[0] pinned, blend toward savgol output
            for bi in range(blend_len):
                idx = bi
                w = bi / blend_len  # 0 at idx=0 (pure start_pt), ->1 (savgol)
                samples[idx] = start_pt * (1 - w) + samples[idx] * w
            samples[0] = start_pt
            # End-side taper: savgol -> end_pt
            for bi in range(blend_len):
                idx = len(samples) - blend_len + bi
                w = bi / blend_len
                samples[idx] = samples[idx] * (1 - w) + end_pt * w
            samples[-1] = end_pt

        # --- Append additional waypoints from reference ---
        n_additional = 100
        xy_additional = np.array([
            (gb_wpnts[(s_idx[-1] + i + 1) % ref_max_idx].x_m,
             gb_wpnts[(s_idx[-1] + i + 1) % ref_max_idx].y_m)
            for i in range(n_additional)
        ])

        # --- GB transition blending (quadratic easing, from smart) ---
        if SMOOTH_OTWPNTS and len(samples) > 0 and len(xy_additional) > 0:
            blend_to_gb_len = min(GB_BLEND_LEN, len(samples) - 1)
            for bi in range(blend_to_gb_len):
                idx = len(samples) - blend_to_gb_len + bi
                t = (bi + 1) / (blend_to_gb_len + 1)
                w = t * t  # quadratic easing
                gb_idx_for_blend = (s_idx[-1] - blend_to_gb_len + bi + 1) % ref_max_idx
                target_pt = np.array([
                    gb_wpnts[gb_idx_for_blend].x_m,
                    gb_wpnts[gb_idx_for_blend].y_m])
                samples[idx] = samples[idx] * (1 - w) + target_pt * w

        # Record spline portion length BEFORE appending GB (for validator scope)
        spline_seg = np.linalg.norm(np.diff(samples, axis=0), axis=1)
        spline_arc_len = float(np.sum(spline_seg))

        samples_raw = np.vstack([samples, xy_additional])

        # --- Uniform arc-length resampling (kappa accuracy + state machine consistency) ---
        seg = np.linalg.norm(np.diff(samples_raw, axis=0), axis=1)
        arc = np.concatenate([[0], np.cumsum(seg)])
        total_len = float(arc[-1])
        if total_len < 1e-3:
            return wpnts, mrks
        n_uni = max(int(total_len / wpnt_dist) + 1, 2)
        arc_uni = np.linspace(0, total_len, n_uni)
        samples = np.column_stack([
            np.interp(arc_uni, arc, samples_raw[:, 0]),
            np.interp(arc_uni, arc, samples_raw[:, 1]),
        ])
        # Index in resampled array where spline ends (GB additions begin)
        n_spline_uni = int(np.searchsorted(arc_uni, spline_arc_len))

        # --- 3D-safe s, d (no 2D nearest projection) ---
        # BUGFIX: the original get_frenet(x, y) is a 2D nearest projection,
        # so on 3D tracks with XY overlap (bridges / elevated crossovers
        # where two layers share the same XY) a single sample's (s, d) pair
        # can flip to the wrong layer, pulling spline_z / gb_wpnt_i /
        # ref.vx_mps with it simultaneously. Observed and verified on the
        # recovery planner fix (commit e7d5157). The BPoly spline always
        # starts at (cur_x, cur_y), so:
        #   s: cur_s + cumulative arc-length (cur_s comes from the
        #      3D-aware C++ frenet_conversion)
        #   d: signed projection onto the raceline tangent-normal at that s
        s_arr = (float(self.cur_s) + arc_uni) % float(self.gb_max_s)

        ref_x = np.asarray(self.converter.spline_x(s_arr)).flatten()
        ref_y = np.asarray(self.converter.spline_y(s_arr)).flatten()
        dx_ds = np.asarray(self.converter.spline_x(s_arr, 1)).flatten()
        dy_ds = np.asarray(self.converter.spline_y(s_arr, 1)).flatten()
        t_norm = np.sqrt(dx_ds * dx_ds + dy_ds * dy_ds) + 1e-9
        nx = -dy_ds / t_norm
        ny = dx_ds / t_norm
        d_arr = (samples[:, 0] - ref_x) * nx + (samples[:, 1] - ref_y) * ny

        # z from reference path surface (uses s only, consistent with s_arr)
        z_arr = np.array(self.converter.spline_z(s_arr)).flatten()

        # --- Heading & curvature (uniform spacing → accurate kappa) ---
        psi_, kappa_ = tph.calc_head_curv_num.calc_head_curv_num(
            path=samples,
            el_lengths=(total_len / (n_uni - 1)) * np.ones(n_uni - 1),
            is_closed=False)

        # --- Track3DValidator — validate only spline portion (GB additions trusted) ---
        danger_flag = False
        first_invalid = -1
        fail_stage = 0  # 0=valid, 1=d-bound, 2=wall crossing
        if self.track_validator is not None and n_spline_uni >= 2:
            valid, fi, st = self.track_validator.validate_trajectory(
                samples[:n_spline_uni], s_arr[:n_spline_uni], d_arr[:n_spline_uni])
            if not valid:
                danger_flag = True
                first_invalid = fi
                fail_stage = st
                # # DEBUG: uncomment for detailed bounds violation logging
                # fi = first_invalid
                # gb_i = int((s_arr[fi] / wpnt_dist) % ref_max_idx)
                # gb_i = min(gb_i, len(gb_wpnts) - 1)
                # rospy.loginfo_throttle(2,
                #     f"[{self.name}] bounds violation idx={fi}: "
                #     f"d={d_arr[fi]:.3f}, d_left={gb_wpnts[gb_i].d_left:.3f}, "
                #     f"d_right={gb_wpnts[gb_i].d_right:.3f}, s={s_arr[fi]:.3f}")

        # --- Build waypoints + markers (ALWAYS — red=invalid, green=valid) ---
        for i in range(len(samples)):
            gb_wpnt_i = int((s_arr[i] / wpnt_dist) % ref_max_idx)
            gb_wpnt_i = min(gb_wpnt_i, len(gb_wpnts) - 1)
            ref = gb_wpnts[gb_wpnt_i]
            # Local curvature speed scaling (Frenet parallel-offset formula):
            # v_local / v_raceline = sqrt(|1 - d*kappa|)
            # NOTE: state machine update_velocity() overwrites this with full 2.5D vel profile.
            radius_ratio = max(1.0 - float(d_arr[i]) * ref.kappa_radpm, 0.1)
            vi = ref.vx_mps * np.sqrt(radius_ratio)
            is_invalid = danger_flag and i >= first_invalid

            if not is_invalid:
                wpnt = Wpnt()
                wpnt.id = i
                wpnt.x_m = samples[i, 0]
                wpnt.y_m = samples[i, 1]
                wpnt.z_m = float(z_arr[i])
                wpnt.s_m = float(s_arr[i])
                wpnt.d_m = float(d_arr[i])
                wpnt.psi_rad = psi_[i] + np.pi / 2
                wpnt.kappa_radpm = kappa_[i]
                wpnt.vx_mps = vi
                # d_right/d_left = vehicle-to-wall clearance (d_m compensated)
                # → controller computes safety_ratio = min(d_left, d_right) / (d_left + d_right)
                # TODO: decide where to use safety_ratio (speed scaling? recovery trigger? viz?)
                wpnt.d_right = ref.d_right + wpnt.d_m
                wpnt.d_left = ref.d_left - wpnt.d_m
                wpnt.mu_rad = ref.mu_rad
                wpnts.wpnts.append(wpnt)

            # Markers: green=valid, yellow=Stage1 (d-bound), red=Stage2 (wall crossing)
            mrk = Marker()
            mrk.header.frame_id = "map"
            mrk.header.stamp = rospy.Time.now()
            mrk.type = Marker.CYLINDER
            mrk.scale.x = mrk.scale.y = 0.1
            mrk.scale.z = max(vi / self.gb_vmax, 0.05) if self.gb_vmax else 0.1
            mrk.color.a = 1.0
            if is_invalid and fail_stage == 2:
                mrk.color.r = 1.0
                mrk.color.g = mrk.color.b = 0.0
            elif is_invalid and fail_stage == 1:
                mrk.color.r = 1.0
                mrk.color.g = 1.0
                mrk.color.b = 0.0
            else:
                mrk.color.g = 0.8
                mrk.color.r = 0.2
                mrk.color.b = 0.2
            mrk.id = i
            mrk.pose.position.x = samples[i, 0]
            mrk.pose.position.y = samples[i, 1]
            mrk.pose.position.z = float(z_arr[i])
            mrk.pose.orientation.w = 1
            mrks.markers.append(mrk)

        if danger_flag:
            wpnts.wpnts = []

        return wpnts, mrks


if __name__ == "__main__":
    node = StaticAvoidance3D()
    node.loop()
