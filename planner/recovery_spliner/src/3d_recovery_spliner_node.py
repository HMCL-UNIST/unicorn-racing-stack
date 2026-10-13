#!/usr/bin/env python3
"""
3D Recovery Spliner — 3D version of the 2D recovery spliner
    - GridFilter → Track3DValidator (s-based boundary + d-bound hybrid)
    - XY BPoly spline + z from spline_z(s) interpolation
    - Full Wpnt fields: z_m, mu_rad, d_right, d_left
    - 40Hz loop, recovery waypoints and markers published every cycle
    - get_frenet (2D vectorized) to keep the loop fast
    - no Smart/fixed path logic (not used in 3D)
"""
import time
from typing import List, Tuple

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray
from scipy.interpolate import BPoly
from dynamic_reconfigure.msg import Config
from f110_msgs.msg import Wpnt, WpntArray
from frenet_converter.frenet_converter import FrenetConverter
import tf.transformations as tf_trans
from track_3d_validator import Track3DValidator
import trajectory_planning_helpers as tph


class RecoverySpliner3D:
    """
    3D Recovery planner — generates smooth spline back to raceline.
    Replaces GridFilter with Track3DValidator for 3D track collision checking.
    """

    def __init__(self):
        self.name = "3d_recovery_spliner_node"
        rospy.init_node(self.name)

        # State
        self.gb_wpnts = WpntArray()
        self.gb_scaled_wpnts = WpntArray()
        self.gb_vmax = None
        self.gb_max_idx = None
        self.gb_max_s = None
        self.cur_s = 0
        self.cur_d = 0
        self.cur_vs = 0
        self.cur_x = 0.0
        self.cur_y = 0.0
        self.cur_yaw = 0.0
        self.waypoints = None
        self.inflection_points = np.array([])
        self.frame_count = 0

        # Parameters
        self.min_candidates_lookahead_n = rospy.get_param(
            "/recovery_planner_spline/min_candidates_lookahead_n", 20)
        self.num_kappas = rospy.get_param(
            "/recovery_planner_spline/num_kappas", 20)
        self.spline_scale = rospy.get_param(
            "/recovery_planner_spline/spline_scale", 0.8)
        self.safety_margin = rospy.get_param("/dyn_planner_tuner/recovery/safety_margin", 0.15)
        self.smooth_len = rospy.get_param(
            "/recovery_planner_spline/smooth_len", 1.0)
        self.n_loc_wpnts = rospy.get_param("/state_machine/n_loc_wpnts", 80)
        self.from_bag = rospy.get_param("/from_bag", False)
        self.measuring = rospy.get_param("/measure", False)

        self.track_validator = None

        # Publishers (before subscribers)
        self.mrks_pub = rospy.Publisher(
            "/planner/recovery/markers", MarkerArray, queue_size=10)
        self.recovery_wpnts_pub = rospy.Publisher(
            "/planner/recovery/wpnts", WpntArray, queue_size=10)
        self.recovery_lookahead_pub = rospy.Publisher(
            "/planner/recovery/lookahead_point", Marker, queue_size=10)
        self.validity_pub = rospy.Publisher(
            "/planner/recovery/3d_validity", MarkerArray, queue_size=10)

        if self.measuring:
            self.latency_pub = rospy.Publisher(
                "/planner/recovery/latency", Float32, queue_size=10)
            self.checkpoints_pub = rospy.Publisher(
                "/planner/recovery/checkpoints", MarkerArray, queue_size=10)

        # Init converter BEFORE subscribers
        self.converter = self._init_converter()

        # Subscribers (safe now — converter exists)
        rospy.Subscriber("/car_state/odom_frenet", Odometry, self.state_frenet_cb)
        rospy.Subscriber("/car_state/odom", Odometry, self.state_cb)
        rospy.Subscriber("/global_waypoints", WpntArray, self.gb_cb)
        rospy.Subscriber("/global_waypoints_scaled", WpntArray, self.gb_scaled_cb)
        if not self.from_bag:
            rospy.Subscriber(
                "/dyn_planner_tuner/recovery/parameter_updates", Config, self.dyn_param_cb)

        self.rate = rospy.Rate(40)  # same as original recovery

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

        # Inflection points (curvature sign changes)
        kappas = np.array([w.kappa_radpm for w in data.wpnts])
        self.inflection_points = np.where(np.diff(np.sign(kappas)) != 0)[0]

        #init Track3DValidator once gb_wpnts arrive
        if self.track_validator is None:
            self.track_validator = Track3DValidator(
                converter=self.converter,
                gb_wpnts=data.wpnts,
                safety_margin=self.safety_margin,
            )
            rospy.loginfo(f"[{self.name}] Track3DValidator initialized")

        if self.measuring:
            mrks = MarkerArray()
            for idx in self.inflection_points:
                mrk = Marker()
                mrk.header.frame_id = "map"
                mrk.header.stamp = rospy.Time.now()
                mrk.type = Marker.CYLINDER
                mrk.scale.x = mrk.scale.y = mrk.scale.z = 0.3
                mrk.color.a = 1.0
                mrk.color.b = mrk.color.r = 0.75
                mrk.id = idx
                mrk.pose.position.x = data.wpnts[idx].x_m
                mrk.pose.position.y = data.wpnts[idx].y_m
                #use actual z
                mrk.pose.position.z = data.wpnts[idx].z_m
                mrk.pose.orientation.w = 1
                mrks.markers.append(mrk)
            self.checkpoints_pub.publish(mrks)

    def gb_scaled_cb(self, data: WpntArray):
        self.gb_scaled_wpnts = data

    def dyn_param_cb(self, params: Config):
        self.min_candidates_lookahead_n = rospy.get_param(
            "/dyn_planner_tuner/recovery/min_candidates_lookahead_n", 20)
        self.num_kappas = rospy.get_param(
            "/dyn_planner_tuner/recovery/num_kappas", 20)
        self.spline_scale = rospy.get_param(
            "/dyn_planner_tuner/recovery/spline_scale", 0.8)
        self.safety_margin = rospy.get_param(
            "/dyn_planner_tuner/recovery/safety_margin", 0.15)
        self.smooth_len = rospy.get_param(
            "/dyn_planner_tuner/recovery/smooth_len", 1.0)
        if self.track_validator is not None:
            self.track_validator.safety_margin = self.safety_margin

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    def loop(self):
        rospy.loginfo(f"[{self.name}] Waiting for messages...")
        rospy.wait_for_message("/global_waypoints_scaled", WpntArray)
        rospy.wait_for_message("/car_state/odom", Odometry)
        rospy.wait_for_message("/car_state/odom_frenet", Odometry)
        rospy.wait_for_message("/dyn_planner_tuner/recovery/parameter_updates", Config)
        rospy.loginfo(f"[{self.name}] Ready! (40Hz)")

        while not rospy.is_shutdown():
            if self.measuring:
                start = time.perf_counter()

            self.frame_count += 1
            gb_wpnts = self.gb_scaled_wpnts.wpnts

            # Clear previous markers
            del_mrks = MarkerArray()
            del_mrk = Marker()
            del_mrk.header.stamp = rospy.Time.now()
            del_mrk.action = Marker.DELETEALL
            del_mrks.markers.append(del_mrk)
            self.mrks_pub.publish(del_mrks)

            wpnts, mrks = self.do_spline(gb_wpnts)

            # Publish every frame
            self.recovery_wpnts_pub.publish(wpnts)
            self.mrks_pub.publish(mrks)

            if self.measuring:
                end = time.perf_counter()
                self.latency_pub.publish(1.0 / max(end - start, 1e-6))

            self.rate.sleep()

    # =========================================================================
    # UTILS
    # =========================================================================

    def find_tangent_idx(self, xy_m, psi_rads):
        """Find waypoint whose heading best aligns with vehicle heading."""
        cur_x, cur_y = self.cur_x, self.cur_y
        smooth = (np.cos(self.cur_yaw) * self.smooth_len,
                  np.sin(self.cur_yaw) * self.smooth_len)

        dx = xy_m[:, 0] - (cur_x + smooth[0])
        dy = xy_m[:, 1] - (cur_y + smooth[1])
        norm = np.sqrt(dx**2 + dy**2)
        unit_vectors = np.vstack((dx / norm, dy / norm)).T

        psi_unit = np.vstack((np.cos(psi_rads), np.sin(psi_rads))).T
        cos_theta = np.clip(np.sum(unit_vectors * psi_unit, axis=1), -1.0, 1.0)
        angles = np.arccos(cos_theta)
        return np.argmin(angles)

    # =========================================================================
    # DO_SPLINE — 3D version
    # =========================================================================

    def do_spline(self, gb_wpnts) -> Tuple[WpntArray, MarkerArray]:
        """
        3D recovery spline generation.
        - XY BPoly spline + z from spline_z(s) interpolation
        - Track3DValidator batch validation (replaces GridFilter)
        - Full Wpnt fields: z_m, mu_rad, d_right, d_left
        """
        mrks = MarkerArray()
        wpnts = WpntArray()
        wpnts.header.stamp = rospy.Time.now()
        wpnts.header.frame_id = "map"

        if len(gb_wpnts) < 2:
            return wpnts, mrks

        wpnt_dist = gb_wpnts[1].s_m - gb_wpnts[0].s_m
        ref_max_idx = self.gb_max_idx
        if ref_max_idx is None or ref_max_idx <= 0:
            return wpnts, mrks

        cur_s = self.cur_s
        cur_d = self.cur_d
        cur_s_idx = int(cur_s / wpnt_dist)

        # --- Inflection point lookahead ---
        if len(self.inflection_points) != 0:
            infl_idx = np.searchsorted(self.inflection_points, cur_s_idx)
            next_infl = self.inflection_points[infl_idx % len(self.inflection_points)]
            candidate_len = (next_infl - cur_s_idx + ref_max_idx
                             if infl_idx == len(self.inflection_points)
                             else next_infl - cur_s_idx)
        else:
            candidate_len = ref_max_idx // 2

        candidate_len = max(candidate_len, self.min_candidates_lookahead_n)

        gb_idxs = [(cur_s_idx + i) % ref_max_idx for i in range(candidate_len)]
        max_avail = len(gb_wpnts) - 1
        gb_idxs = [min(idx, max_avail) for idx in gb_idxs]

        # --- Tangent alignment ---
        num_kappas_ = min(self.num_kappas, self.min_candidates_lookahead_n)
        kappas = np.array([gb_wpnts[i].kappa_radpm for i in gb_idxs[:num_kappas_]])

        xy_m = np.array([(gb_wpnts[i].x_m, gb_wpnts[i].y_m) for i in gb_idxs])
        psi_rads = np.array([gb_wpnts[i].psi_rad for i in gb_idxs])
        tangent_idx = self.find_tangent_idx(xy_m, psi_rads)

        # --- Lookahead debug marker ---
        if self.measuring:
            mrk = Marker()
            mrk.header.frame_id = "map"
            mrk.header.stamp = rospy.Time.now()
            mrk.type = Marker.SPHERE
            mrk.scale.x = mrk.scale.y = mrk.scale.z = 0.5
            mrk.color.a = 1.0
            mrk.color.b = 1.0
            mrk.color.g = 0.65
            mrk.pose.position.x = xy_m[tangent_idx, 0]
            mrk.pose.position.y = xy_m[tangent_idx, 1]
            #z from reference waypoint
            mrk.pose.position.z = gb_wpnts[gb_idxs[tangent_idx]].z_m
            mrk.pose.orientation.w = 1
            self.recovery_lookahead_pub.publish(mrk)

        # === XY BPoly spline (2D, same as original) ===
        points = [
            [self.cur_x, self.cur_y],
            [xy_m[tangent_idx, 0], xy_m[tangent_idx, 1]],
        ]
        tangents = [
            np.array([np.cos(self.cur_yaw), np.sin(self.cur_yaw)]),
            np.array([np.cos(psi_rads[tangent_idx]), np.sin(psi_rads[tangent_idx])]),
        ]
        tangents = np.dot(tangents, self.spline_scale * np.eye(2))
        points = np.asarray(points)
        nPoints, dim = points.shape

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

        samples_xy_raw = np.zeros([nSamples, dim])
        for i in range(dim):
            poly = BPoly.from_derivatives(d_cum, spline_result[:, i])
            samples_xy_raw[:, i] = poly(s_param)

        # === Append additional waypoints from reference path ===
        n_additional = 80
        xy_additional = np.array([
            (gb_wpnts[(tangent_idx + cur_s_idx + i + 1) % ref_max_idx].x_m,
             gb_wpnts[(tangent_idx + cur_s_idx + i + 1) % ref_max_idx].y_m)
            for i in range(n_additional)
        ])
        samples_xy_raw = np.vstack([samples_xy_raw, xy_additional])

        # === Uniform arc-length resampling (kappa accuracy + state machine consistency) ===
        # Record spline portion length BEFORE resampling (for validator scope)
        spline_arc_len = float(l)  # BPoly portion length
        seg = np.linalg.norm(np.diff(samples_xy_raw, axis=0), axis=1)
        arc = np.concatenate([[0], np.cumsum(seg)])
        total_len = float(arc[-1])
        if total_len < 1e-3:
            return wpnts, mrks
        n_uni = max(int(total_len / wpnt_dist) + 1, 2)
        arc_uni = np.linspace(0, total_len, n_uni)
        samples_xy = np.column_stack([
            np.interp(arc_uni, arc, samples_xy_raw[:, 0]),
            np.interp(arc_uni, arc, samples_xy_raw[:, 1]),
        ])
        # Index in resampled array where spline ends (GB additions begin)
        n_spline_uni = int(np.searchsorted(arc_uni, spline_arc_len))

        # === 3D-safe s, d computation (no 2D nearest projection) ===
        # BUGFIX: the original get_frenet(x, y) is a 2D nearest projection,
        # so on 3D tracks with XY overlap (bridges / elevated crossovers
        # where two layers share the same XY) a single sample's (s, d) pair
        # can flip to the wrong layer, pulling spline_z / gb_wpnt_i /
        # ref.vx_mps with it simultaneously.
        # Observed in logs: s 17 -> 17 -> 50 -> 17 (Δ~33m jump on one sample),
        # z 0.14 <-> 0.60, vx 4 -> 6 on a single cylinder.
        # The BPoly spline always starts at (cur_x, cur_y), so:
        #   s: cur_s + cumulative arc-length (cur_s comes from the 3D-aware
        #      C++ frenet_conversion)
        #   d: signed projection onto the raceline tangent-normal at that s
        # Both avoid the 2D nearest projection ambiguity entirely.
        s_arr = (float(self.cur_s) + arc_uni) % float(self.gb_max_s)

        ref_x = np.asarray(self.converter.spline_x(s_arr)).flatten()
        ref_y = np.asarray(self.converter.spline_y(s_arr)).flatten()
        dx_ds = np.asarray(self.converter.spline_x(s_arr, 1)).flatten()
        dy_ds = np.asarray(self.converter.spline_y(s_arr, 1)).flatten()
        t_norm = np.sqrt(dx_ds * dx_ds + dy_ds * dy_ds) + 1e-9
        # unit normal toward +d (left, 90° CCW from tangent) — matches get_cartesian convention
        nx = -dy_ds / t_norm
        ny = dx_ds / t_norm
        d_arr = (samples_xy[:, 0] - ref_x) * nx + (samples_xy[:, 1] - ref_y) * ny

        # z from reference path surface (uses s only, consistent with s_arr)
        z_arr = np.array(self.converter.spline_z(s_arr)).flatten()

        # === Heading & curvature (uniform spacing → accurate kappa) ===
        psi_, kappa_ = tph.calc_head_curv_num.calc_head_curv_num(
            path=samples_xy,
            el_lengths=(total_len / (n_uni - 1)) * np.ones(n_uni - 1),
            is_closed=False)

        # === Track3DValidator — validate only spline portion (GB additions are on raceline, trusted) ===
        danger_flag = False
        first_invalid = -1
        fail_stage = 0  # 0=valid, 1=d-bound, 2=wall crossing
        if self.track_validator is not None and n_spline_uni >= 2:
            valid, fi, st = self.track_validator.validate_trajectory(
                samples_xy[:n_spline_uni], s_arr[:n_spline_uni], d_arr[:n_spline_uni])
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

        # === Build waypoints + markers (ALWAYS — red for invalid, green for valid) ===
        for i in range(len(samples_xy)):
            gb_wpnt_i = int((s_arr[i] / wpnt_dist) % ref_max_idx)
            gb_wpnt_i = min(gb_wpnt_i, len(gb_wpnts) - 1)
            ref = gb_wpnts[gb_wpnt_i]
            # Local curvature speed scaling (Frenet parallel-offset formula):
            # kappa_local = kappa / (1 - d*kappa)  →  R_local = R_raceline * (1 - d*kappa)
            # v_local / v_raceline = sqrt(R_local / R_raceline) = sqrt(|1 - d*kappa|)
            # NOTE: state machine update_velocity() overwrites this with full 2.5D vel profile.
            radius_ratio = max(1.0 - float(d_arr[i]) * ref.kappa_radpm, 0.1)
            vi = ref.vx_mps * np.sqrt(radius_ratio)
            is_invalid = danger_flag and i >= first_invalid

            # Waypoints: only add valid points
            if not is_invalid:
                wpnt = Wpnt()
                wpnt.id = i
                wpnt.x_m = samples_xy[i, 0]
                wpnt.y_m = samples_xy[i, 1]
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

            # Markers: ALWAYS show — green=valid, yellow=Stage1 fail (d-bound), red=Stage2 fail (wall)
            mrk = Marker()
            mrk.header.frame_id = "map"
            mrk.header.stamp = rospy.Time.now()
            mrk.type = Marker.CYLINDER
            mrk.scale.x = mrk.scale.y = 0.1
            mrk.scale.z = max(vi / self.gb_vmax, 0.05) if self.gb_vmax else 0.1
            mrk.color.a = 1.0
            if is_invalid and fail_stage == 2:
                # Stage 2: wall crossing — red
                mrk.color.r = 1.0
                mrk.color.g = mrk.color.b = 0.0
            elif is_invalid and fail_stage == 1:
                # Stage 1: d-bound — yellow
                mrk.color.r = 1.0
                mrk.color.g = 1.0
                mrk.color.b = 0.0
            else:
                # valid — green
                mrk.color.g = 0.8
                mrk.color.r = 0.2
                mrk.color.b = 0.2
            mrk.id = i
            mrk.pose.position.x = samples_xy[i, 0]
            mrk.pose.position.y = samples_xy[i, 1]
            mrk.pose.position.z = float(z_arr[i])
            mrk.pose.orientation.w = 1
            mrks.markers.append(mrk)

        # If danger, still clear wpnts (state machine shouldn't use invalid trajectory)
        if danger_flag:
            wpnts.wpnts = []

        return wpnts, mrks


if __name__ == "__main__":
    node = RecoverySpliner3D()
    node.loop()
