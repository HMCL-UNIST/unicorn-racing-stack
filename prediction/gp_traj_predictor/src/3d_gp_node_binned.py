#!/usr/bin/env python3
# !!!!!! DO NOT MOVE THESE LINES !!!!!!
import os  # noqa
os.environ["OMP_NUM_THREADS"] = "1"  # noqa
# !!!!!! DO NOT MOVE THESE LINES !!!!!!
"""
3d_gp_node_binned.py  (multi-opponent + multiprocessing capable)
================================================================
Single-node replacement for the three-node learning chain:
  - 3d_opponent_trajectory.py        (half-lap accumulation)
  - gaussian_process_opp_traj.py     (GP fit + CCMA whole-lap)
  - predictor_opponent_trajectory.py (off-trajectory GP patch)

Data aggregation is switched from "half-lap accumulation" to bin-based
aggregation (inspired by ForzaETH/multiopponent-pspliner, IROS 2025):
the track arc length s is bucketed into fixed-size bins, and each bin
holds at most `bin_capacity` most-recent observations. This removes the
cold-start gap of the original pipeline and lets outliers get evicted.

Fit trigger is event-based: a GP fit only runs when `refit_new_bins`
new bins have been added since the previous fit.

Modes controlled by ~num_opponents:
  * num_opponents == 1 (default, single-opp path):
      First dynamic obstacle (is_static==False) is used, the same
      selection as 3d_opp_prediction.py.
  * num_opponents >= 2 (multi-opp):
      Each detected dynamic obstacle is matched to a stable slot via
      detection_matching. Each slot learns its own GP in a shared
      multiprocessing.Pool worker (worker count = ~pool_size). Outputs:
        - /opponent_trajectory        → closest opp
        - /perception/opp_trajectories → all opps
        - /opponent_traj_markerarray  → per-opp coloured markers

Note: ~num_opponents is a slot UPPER BOUND. The number of actually
processed opponents is dynamic in [0, num_opponents]: empty slots are
simply not dispatched. Slots are never auto-cleared — a new obs id
either fills a None slot or overwrites the least-recently-seen slot.

Inputs:
  /tracking/obstacles          f110_msgs/ObstacleArray
  /global_waypoints            f110_msgs/WpntArray     (one-shot at init)
  /car_state/odom_frenet       nav_msgs/Odometry       (multi-opp only)

Outputs:
  /opponent_trajectory         f110_msgs/OpponentTrajectory
  /perception/opp_trajectories f110_msgs/OpponentTrajectories (multi-opp)
  /opponent_traj_markerarray   visualization_msgs/MarkerArray
"""

import colorsys
import warnings
from multiprocessing import Pool

import numpy as np
import rospy
from f110_msgs.msg import (
    ObstacleArray,
    OpponentTrajectory,
    OpponentTrajectories,
    OppWpnt,
    WpntArray,
)
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from scipy.optimize import fmin_l_bfgs_b
from frenet_converter.frenet_converter import FrenetConverter
from ccma import CCMA


def _lbfgs_optimizer(obj_func, initial_theta, bounds):
    """sklearn GPR custom optimizer: plain L-BFGS-B without restarts."""
    solution, function_value, _ = fmin_l_bfgs_b(obj_func, initial_theta, bounds=bounds)
    return solution, function_value


# multi-opp inlined replacement for forza_helpers.id_to_rgb_color.
# Golden-ratio hue stepping gives well-spread per-opp colors.
def _id_to_rgb_color(track_id):
    if track_id is None:
        return (1.0, 1.0, 0.0)
    h = (int(track_id) * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.8, 0.9)
    return (float(r), float(g), float(b))


class BinnedGPNode:
    def __init__(self):
        # rospy.init_node is performed in __main__ (before Pool creation) so
        # multiprocessing.Pool fork() does not inherit ROS sockets.

        # ROS params (tunable without rebuild)
        self.BIN_SPACING = rospy.get_param('~bin_spacing', 0.1)
        self.BIN_CAPACITY = rospy.get_param('~bin_capacity', 2)
        self.COVERAGE_THRESHOLD = rospy.get_param('~coverage_threshold', 0.30)
        self.REFIT_NEW_BINS = rospy.get_param('~refit_new_bins', 5)
        self.USE_CCMA_WHOLE_LAP = rospy.get_param('~use_ccma_whole_lap', True)
        self.CCMA_COVERAGE = rospy.get_param('~ccma_coverage', 0.95)
        self.WRAP_PAD_M = rospy.get_param('~wrap_pad_m', 3.0)

        # multi-opp: num_opponents=1 runs the single-opponent path
        self.NUM_OPPONENTS = int(rospy.get_param('~num_opponents', 1))
        self.RECENT_SEEN_SEC = float(rospy.get_param('~recent_seen_sec', 0.3))
        self.POOL_SIZE = int(rospy.get_param('~pool_size', 2))

        # same kernels as gaussian_process_opp_traj.py to keep GP priors identical
        constant_kernel1_d = ConstantKernel(constant_value=0.5, constant_value_bounds=(1e-6, 1e3))
        constant_kernel2_d = ConstantKernel(constant_value=0.2, constant_value_bounds=(1e-6, 1e3))
        constant_kernel1_vs = ConstantKernel(constant_value=0.5, constant_value_bounds=(1e-6, 1e3))
        constant_kernel2_vs = ConstantKernel(constant_value=0.2, constant_value_bounds=(1e-6, 1e3))
        self.kernel_vs = constant_kernel1_vs * RBF(length_scale=1.0) + constant_kernel2_vs * WhiteKernel(noise_level=1)
        self.kernel_d = constant_kernel1_d * Matern(length_scale=1.0, nu=3 / 2) + constant_kernel2_d * WhiteKernel(noise_level=1)

        # State (single-opp path — always created so _init_oppwpnts works uniformly)
        self.bins = {}                # bin_idx -> {"s": [], "d": [], "vs": [], "vd": []}
        self.last_fit_bin_count = 0
        self.track_length = None
        self.converter = None
        self.global_wpnts = None
        self.ego_s_original = None
        self.max_velocity = None
        self.oppwpnts_list = None
        self.last_bin_count_logged = -1

        # multi-opp state (active only when NUM_OPPONENTS >= 2)
        N = self.NUM_OPPONENTS
        self.bins_multi = {k: {} for k in range(N)}
        self.last_fit_bin_count_m = {k: 0 for k in range(N)}
        self.predictions_multi = {k: None for k in range(N)}
        self.oppwpnts_lists_multi = {k: None for k in range(N)}
        self.is_processing = [False] * N
        self.detection_matching = {
            k: {
                "id": None,
                "dist": float("inf"),
                "last_seen": rospy.Time(0),
                "last_s": None,
                "cumulated_dist": 0.0,
            }
            for k in range(N)
        }
        self.collected_messages = []     # list of per-tick [Obstacle|None, ...]
        self.current_ego_s = 0.0
        self.tracked_obs = None
        self.pool = None                 # assigned by __main__ block
        self.boundaries = None           # set by _precompute_boundary_lut

        # Publishers
        self.opp_traj_gp_pub = rospy.Publisher('/opponent_trajectory', OpponentTrajectory, queue_size=10)
        self.opp_traj_marker_pub = rospy.Publisher('/opponent_traj_markerarray', MarkerArray, queue_size=10)
        # multi-opp — full trajectory set (populated only when NUM_OPPONENTS >= 2)
        self.opp_trajs_pub = rospy.Publisher('/perception/opp_trajectories', OpponentTrajectories, queue_size=1)

        # One-shot init from global waypoints
        rospy.loginfo("[GP-binned] waiting for /global_waypoints ...")
        glb = rospy.wait_for_message('/global_waypoints', WpntArray)
        self._init_from_waypoints(glb)

        # multi-opp — precompute s→(d_right, d_left) LUT for GP corridor clipping
        self._precompute_boundary_lut()
        if self.NUM_OPPONENTS >= 2:
            for k in range(self.NUM_OPPONENTS):
                self.oppwpnts_lists_multi[k] = self._init_oppwpnts()

        # Publish the sentinel trajectory once so downstream consumers get a
        # well-formed OpponentTrajectory message even before the first fit.
        self._publish_current(lap_count=0.0)

        # Subscribers AFTER init (and after Pool creation in __main__) so
        # fork() does not inherit ROS sockets and callbacks never see None state.
        rospy.Subscriber('/tracking/obstacles', ObstacleArray, self.tracker_cb, queue_size=10)
        # multi-opp — ego s position for forward s-gap assignment
        if self.NUM_OPPONENTS >= 2:
            rospy.Subscriber('/car_state/odom_frenet', Odometry, self.odom_frenet_cb, queue_size=10)

        rospy.loginfo(
            "[GP-binned] ready: bin_spacing=%.2fm bin_capacity=%d coverage_thr=%.0f%% refit_every=%d bins "
            "ccma_whole_lap=%s ccma_cov=%.2f track_len=%.2fm max_bins=%d num_opp=%d pool=%d",
            self.BIN_SPACING, self.BIN_CAPACITY, self.COVERAGE_THRESHOLD * 100.0,
            self.REFIT_NEW_BINS, self.USE_CCMA_WHOLE_LAP, self.CCMA_COVERAGE,
            self.track_length, self._max_bin_count(), self.NUM_OPPONENTS, self.POOL_SIZE,
        )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _init_from_waypoints(self, data: WpntArray):
        waypoints = np.array([[wp.x_m, wp.y_m, wp.z_m] for wp in data.wpnts])
        self.converter = FrenetConverter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2])
        self.track_length = data.wpnts[-1].s_m
        self.global_wpnts = data.wpnts[:-1]  # drop duplicated closing point
        self.ego_s_original = [wp.s_m for wp in self.global_wpnts]
        self.max_velocity = max(wp.vx_mps for wp in self.global_wpnts) if self.global_wpnts else 10.0
        if self.max_velocity <= 1e-3:
            self.max_velocity = 10.0
        self.oppwpnts_list = self._init_oppwpnts()

    def _max_bin_count(self):
        return int(self.track_length / self.BIN_SPACING) + 1

    # multi-opp — build s→(d_right, d_left) LUT from FrenetConverter's stored
    # track bounds so GP posterior d can be clipped inside the drivable corridor
    # without subscribing to /trackbounds/markers (FrenetConverter auto-loads it).
    def _precompute_boundary_lut(self):
        if not getattr(self.converter, "has_track_bounds", False):
            rospy.logwarn("[GP-binned] no track bounds available → boundary clipping disabled")
            self.boundaries = None
            return
        lb = np.asarray(self.converter.left_bounds)    # Nx3
        rb = np.asarray(self.converter.right_bounds)   # Nx3
        lb_f = self.converter.get_frenet(lb[:, 0], lb[:, 1])
        rb_f = self.converter.get_frenet(rb[:, 0], rb[:, 1])
        lb_s = np.asarray(lb_f[0]).flatten()
        lb_d = np.asarray(lb_f[1]).flatten()
        rb_s = np.asarray(rb_f[0]).flatten()
        rb_d = np.asarray(rb_f[1]).flatten()
        lb_order = np.argsort(lb_s)
        rb_order = np.argsort(rb_s)
        ego_s = np.array(self.ego_s_original)
        d_left = np.interp(ego_s, lb_s[lb_order], lb_d[lb_order])
        d_right = np.interp(ego_s, rb_s[rb_order], rb_d[rb_order])
        # col 0 = right (lower bound, usually negative), col 1 = left (upper bound, usually positive)
        self.boundaries = np.stack((d_right, d_left), axis=-1)

    # Sentinel initializer, same as make_initial_opponent_trajectory_msg in gaussian_process_opp_traj.py. Keeps the raceline-vx * 0.9 fallback and is_observed=False for unobserved cells so downstream viz in 3d_opp_prediction.py renders unchanged.
    def _init_oppwpnts(self):
        xy = self.converter.get_cartesian(self.ego_s_original, [0.0] * len(self.ego_s_original))
        wpnts = []
        for i, s in enumerate(self.ego_s_original):
            w = OppWpnt()
            w.x_m = float(xy[0][i])
            w.y_m = float(xy[1][i])
            w.s_m = float(s)
            w.d_m = 0.0
            w.proj_vs_mps = float(self.global_wpnts[i].vx_mps * 0.9)
            w.vd_mps = 0.0
            w.d_var = 0.0
            w.vs_var = 0.0
            w.is_observed = False
            wpnts.append(w)
        return wpnts

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------
    def tracker_cb(self, msg: ObstacleArray):
        # num_opponents==1 -> single-opponent path, otherwise multi-opp path
        if self.NUM_OPPONENTS == 1:
            self._tracker_cb_single(msg)
        else:
            # Multi-opp: just cache; actual matching + dispatch runs in main loop.
            self.tracked_obs = msg

    # multi-opp — ego s comes from /car_state/odom_frenet (pose.position.x)
    def odom_frenet_cb(self, msg: Odometry):
        self.current_ego_s = float(msg.pose.pose.position.x)

    def _tracker_cb_single(self, msg: ObstacleArray):
        # Single-opponent: pick first dynamic obstacle. Same gate as
        # 3d_opp_prediction.py so the two nodes see the same target.
        obs = None
        for o in msg.obstacles:
            if not o.is_static:
                obs = o
                break
        if obs is None:
            return

        # Wrap s into [0, track_length); tracker may publish slightly > L.
        s = obs.s_center % self.track_length
        bin_idx = int(s // self.BIN_SPACING)

        b = self.bins.setdefault(bin_idx, {"s": [], "d": [], "vs": [], "vd": []})
        b["s"].append(s)
        b["d"].append(float(obs.d_center))
        b["vs"].append(float(obs.vs))
        b["vd"].append(float(obs.vd))
        # Evict oldest beyond capacity (same index across all four lists).
        while len(b["s"]) > self.BIN_CAPACITY:
            for k in ("s", "d", "vs", "vd"):
                b[k].pop(0)

        self._maybe_fit()

    def _maybe_fit(self):
        n_bins = len(self.bins)
        coverage = n_bins / float(self._max_bin_count())

        # Gate 1: wait until enough of the track has been observed.
        if coverage < self.COVERAGE_THRESHOLD:
            if n_bins != self.last_bin_count_logged and n_bins % 10 == 0:
                rospy.loginfo("[GP-binned] accumulating bins: %d / %.0f%% coverage",
                              n_bins, coverage * 100.0)
                self.last_bin_count_logged = n_bins
            return

        # Gate 2: only re-fit once enough new bins have appeared. First fit
        # fires as soon as the coverage threshold is crossed.
        if self.last_fit_bin_count > 0 and (n_bins - self.last_fit_bin_count) < self.REFIT_NEW_BINS:
            return

        self._fit_and_publish(coverage)
        self.last_fit_bin_count = n_bins

    # ------------------------------------------------------------------
    # Fit + publish (single-opp path, preserved bit-identical)
    # ------------------------------------------------------------------
    def _fit_and_publish(self, coverage: float):
        train_s, train_d, train_vs, train_vd = self._flatten_bins_with_wrap(self.WRAP_PAD_M)
        n_train = train_s.shape[0]
        if n_train < 3:
            return  # not enough samples; defensive early-return

        s_train_col = train_s.reshape(-1, 1)
        ego_s_arr = np.array(self.ego_s_original).reshape(-1, 1)

        # GP fits (d and vs) — same kernels/optimizer as the original pipeline.
        gpr_vs = GaussianProcessRegressor(kernel=self.kernel_vs, optimizer=_lbfgs_optimizer)
        gpr_vs.fit(s_train_col, train_vs.reshape(-1, 1))
        vs_pred, sigma_vs = gpr_vs.predict(ego_s_arr, return_std=True)

        gpr_d = GaussianProcessRegressor(kernel=self.kernel_d, optimizer=_lbfgs_optimizer)
        gpr_d.fit(s_train_col, train_d.reshape(-1, 1))
        d_pred, sigma_d = gpr_d.predict(ego_s_arr, return_std=True)

        # Flatten column outputs from sklearn to 1-D for downstream use.
        vs_pred = np.asarray(vs_pred).flatten()
        sigma_vs = np.asarray(sigma_vs).flatten()
        d_pred = np.asarray(d_pred).flatten()
        sigma_d = np.asarray(sigma_d).flatten()

        # Optional CCMA hybrid: once track coverage is (almost) complete, smooth
        # d in Cartesian to reduce corner "in-curving" artefacts of the GP.
        ccma_used = False
        if self.USE_CCMA_WHOLE_LAP and coverage >= self.CCMA_COVERAGE:
            d_pred = self._apply_ccma(self.ego_s_original, d_pred)
            ccma_used = True

        # vd is not GP-fit (matches the original pipeline: vd is just
        # interpolated from raw observations).
        sort_idx = np.argsort(train_s)
        vd_pred = np.interp(self.ego_s_original, train_s[sort_idx], train_vd[sort_idx])

        # Write posterior into oppwpnts_list and publish.
        xy = self.converter.get_cartesian(self.ego_s_original, d_pred.tolist())
        for i, s in enumerate(self.ego_s_original):
            bin_idx = int(s // self.BIN_SPACING)
            w = self.oppwpnts_list[i]
            w.x_m = float(xy[0][i])
            w.y_m = float(xy[1][i])
            w.d_m = float(d_pred[i])
            w.proj_vs_mps = float(vs_pred[i])
            w.vd_mps = float(vd_pred[i])
            # When CCMA replaces d, its variance is undefined → keep the GP d
            # sigma so downstream uncertainty-aware consumers still see a value.
            w.d_var = float(sigma_d[i])
            w.vs_var = float(sigma_vs[i])
            w.is_observed = bin_idx in self.bins

        # Synthetic lap_count keeps the downstream ≥1 gate satisfied as soon
        # as the first fit publishes. Caps at 2.0 (same semantics as "one
        # full lap accumulated" in the original pipeline).
        lap_count = max(1.0, min(2.0, 2.0 * coverage))
        self._publish_current(lap_count=lap_count)

        rospy.loginfo(
            "[GP-binned] fit n_train=%d bins=%d coverage=%.1f%% lap_count=%.2f ccma=%s",
            n_train, len(self.bins), coverage * 100.0, lap_count, ccma_used,
        )

    def _flatten_bins_with_wrap(self, wrap_pad_m: float):
        """
        Flatten bin contents into 1-D train arrays, then prepend a copy of the
        last `wrap_pad_m` metres of data with `s -= track_length` and append
        the first `wrap_pad_m` metres with `s += track_length`. This mirrors
        the wrap-around padding pattern from the original
        gaussian_process_opp_traj.py (L261-307) so the GP sees a continuous
        curve across the s=0 seam.
        """
        ss, dd, vss, vdd = [], [], [], []
        for b in self.bins.values():
            ss.extend(b["s"])
            dd.extend(b["d"])
            vss.extend(b["vs"])
            vdd.extend(b["vd"])
        if not ss:
            return np.array([]), np.array([]), np.array([]), np.array([])

        ss = np.array(ss)
        dd = np.array(dd)
        vss = np.array(vss)
        vdd = np.array(vdd)

        near_end_mask = ss > (self.track_length - wrap_pad_m)
        near_start_mask = ss < wrap_pad_m

        pre_s = ss[near_end_mask] - self.track_length
        app_s = ss[near_start_mask] + self.track_length

        out_s = np.concatenate([pre_s, ss, app_s])
        out_d = np.concatenate([dd[near_end_mask], dd, dd[near_start_mask]])
        out_vs = np.concatenate([vss[near_end_mask], vss, vss[near_start_mask]])
        out_vd = np.concatenate([vdd[near_end_mask], vdd, vdd[near_start_mask]])
        return out_s, out_d, out_vs, out_vd

    # CCMA xy-space smoothing adapted from gaussian_process_opp_traj.py. Same window sizes (w_ma=5, w_cc=3) so shape of smoothing matches the whole-lap pipeline; sigma_d from the GP is kept as-is for downstream variance consumers.
    def _apply_ccma(self, ego_s, d_pred):
        s_list = list(ego_s)
        d_list = [float(v) for v in d_pred]
        # Prepend last point, append first point — stretches the ring so the
        # symmetric CCMA kernel behaves cleanly around the s=0 seam.
        s_list.insert(0, s_list[-1])
        d_list.insert(0, d_list[-1])
        s_list.append(s_list[1])
        d_list.append(d_list[1])

        noisy_xy = self.converter.get_cartesian(s_list, d_list).transpose()
        ccma = CCMA(w_ma=5, w_cc=3)
        smoothed_xy = ccma.filter(noisy_xy)

        smoothed_sd = self.converter.get_frenet(smoothed_xy[:, 0], smoothed_xy[:, 1])
        sm_s = np.asarray(smoothed_sd[0]).flatten()
        sm_d = np.asarray(smoothed_sd[1]).flatten()
        order = np.argsort(sm_s)
        return np.interp(ego_s, sm_s[order], sm_d[order])

    # ------------------------------------------------------------------
    # Publish helpers (single-opp path)
    # ------------------------------------------------------------------
    def _publish_current(self, lap_count: float):
        msg = OpponentTrajectory()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "opponent_trajectory"
        msg.lap_count = float(lap_count)
        msg.opp_is_on_trajectory = True  # continuous bin re-fit → always "on trajectory"
        msg.oppwpnts = self.oppwpnts_list
        self.opp_traj_gp_pub.publish(msg)

        self.opp_traj_marker_pub.publish(self._visualize())

    # Marker rendering in the same style as _visualize_opponent_wpnts in
    # gaussian_process_opp_traj.py, placed on the 3D track surface so the
    # cylinders do not sit on z=0.
    def _visualize(self):
        arr = MarkerArray()
        max_vel = self.max_velocity if self.max_velocity > 1e-3 else 10.0
        # Recompute x, y and z together from (s, d) via get_cartesian_3d.
        # Taking x, y from the waypoint and z from spline_z(s) separately can
        # mix layers where the track crosses itself (same x, y, different z),
        # which tangles the trajectory markers.
        s_arr = np.array([w.s_m for w in self.oppwpnts_list])
        d_arr = np.array([w.d_m for w in self.oppwpnts_list])
        try:
            xyz = np.asarray(self.converter.get_cartesian_3d(s_arr, d_arr))
            x_arr, y_arr, z_arr = xyz[0].flatten(), xyz[1].flatten(), xyz[2].flatten()
        except Exception:
            x_arr = np.array([w.x_m for w in self.oppwpnts_list])
            y_arr = np.array([w.y_m for w in self.oppwpnts_list])
            z_arr = np.zeros(len(s_arr))

        for i, w in enumerate(self.oppwpnts_list):
            is_sentinel = not w.is_observed
            marker_height = w.proj_vs_mps / max_vel
            m = Marker(header=rospy.Header(frame_id="map"), id=i, type=Marker.CYLINDER)
            m.pose.position.x = float(x_arr[i])
            m.pose.position.y = float(y_arr[i])
            m.pose.position.z = float(z_arr[i]) + marker_height / 2.0
            m.pose.orientation.w = 1.0
            m.scale.x = min(max(5.0 * w.d_var, 0.07), 0.7)
            m.scale.y = min(max(5.0 * w.d_var, 0.07), 0.7)
            m.scale.z = marker_height
            m.color.a = 1.0
            if is_sentinel:
                m.color.r, m.color.g, m.color.b = 1.0, 0.5, 0.0  # orange: unobserved fallback
            else:
                m.color.r, m.color.g, m.color.b = 1.0, 1.0, 0.0  # yellow: GP posterior
            arr.markers.append(m)
        return arr

    # ==================================================================
    # multi-opp path (num_opponents >= 2)
    # ==================================================================

    def _assign_obstacle_to_trajectory(self):
        """Assigns detected obstacles to stable slots
        based on (a) matching obs.id to an existing slot id, (b) filling
        unoccupied slots, (c) overwriting the least-recently-seen slot if full."""
        if self.tracked_obs is None:
            return
        dyn_obs = [o for o in self.tracked_obs.obstacles if not o.is_static]
        if not dyn_obs:
            return
        L = self.track_length
        # forward s-gap from ego (closest first)
        sorted_obs = sorted(
            dyn_obs, key=lambda o: (o.s_center - self.current_ego_s) % L
        )
        N = self.NUM_OPPONENTS
        new_list = [None] * N
        unmatched = []

        # Pass 1: reuse slot if id matches
        for obs in sorted_obs:
            ids = [self.detection_matching[k]["id"] for k in range(N)]
            if obs.id is not None and obs.id in ids:
                k = ids.index(obs.id)
                dist = (obs.s_center - self.current_ego_s) % L
                self.detection_matching[k]["dist"] = dist
                self.detection_matching[k]["last_seen"] = rospy.Time.now()
                new_list[k] = obs
            else:
                unmatched.append(obs)

        # Pass 2: fill vacancies (prefer never-assigned slot, else stalest empty slot)
        for obs in unmatched:
            if None not in new_list:
                break  # all N slots are filled this tick; drop excess obs
            empty_slots = [k for k in range(N) if new_list[k] is None]
            empty_never = [k for k in empty_slots if self.detection_matching[k]["id"] is None]
            if empty_never:
                k = empty_never[0]
            else:
                ages = [self.detection_matching[k]["last_seen"].to_sec()
                        for k in empty_slots]
                k = empty_slots[int(np.argmin(ages))]

            dist = (obs.s_center - self.current_ego_s) % L
            # Reset slot's learned bins when the id changes (new opponent there)
            if self.detection_matching[k]["id"] != obs.id:
                self.bins_multi[k] = {}
                self.last_fit_bin_count_m[k] = 0
                self.detection_matching[k]["cumulated_dist"] = 0.0
            self.detection_matching[k].update({
                "dist": dist,
                "id": obs.id,
                "last_s": obs.s_center,
                "last_seen": rospy.Time.now(),
            })
            new_list[k] = obs

        # cumulated_dist bookkeeping (reference L216-227)
        for k in range(N):
            obs = new_list[k]
            last_s = self.detection_matching[k]["last_s"]
            if obs is None or last_s is None:
                continue
            if obs.s_center > last_s:
                self.detection_matching[k]["cumulated_dist"] += obs.s_center - last_s
                self.detection_matching[k]["last_s"] = obs.s_center
            elif last_s - obs.s_center > 0.5 * L:  # wrap
                self.detection_matching[k]["cumulated_dist"] += L + obs.s_center - last_s
                self.detection_matching[k]["last_s"] = obs.s_center

        self.collected_messages.append(new_list)
        if not any(self.is_processing):
            self._trigger_processing()

    def _trigger_processing(self):
        if any(self.is_processing) or not self.collected_messages:
            return
        msgs = list(self.collected_messages)
        del self.collected_messages[:]

        per_opp = {k: [] for k in range(self.NUM_OPPONENTS)}
        for tick in msgs:
            for k, obs in enumerate(tick):
                if obs is not None:
                    per_opp[k].append(obs)

        active_slots = [k for k in range(self.NUM_OPPONENTS) if per_opp[k]]
        if not active_slots:
            return

        # Only active slots are "processing"; empty slots stay idle.
        self.is_processing = [k in active_slots for k in range(self.NUM_OPPONENTS)]

        for k in active_slots:
            # Reduce Obstacle msgs to picklable (s, d, vs, vd) tuples to avoid
            # ROS msg serialization overhead across process boundaries.
            obs_records = [
                (
                    float(o.s_center) % float(self.track_length),
                    float(o.d_center),
                    float(o.vs),
                    float(o.vd),
                )
                for o in per_opp[k]
            ]
            self.pool.apply_async(
                BinnedGPNode._process_opponent,
                args=(obs_records, k),
                kwds=dict(
                    binned_data=self.bins_multi[k],
                    bin_spacing=self.BIN_SPACING,
                    bin_capacity=self.BIN_CAPACITY,
                    max_s=self.track_length,
                    wrap_pad_m=self.WRAP_PAD_M,
                    ego_s_grid=list(self.ego_s_original),
                    kernel_d=self.kernel_d,
                    kernel_vs=self.kernel_vs,
                    last_fit_bin_count=self.last_fit_bin_count_m[k],
                    refit_new_bins=self.REFIT_NEW_BINS,
                    coverage_threshold=self.COVERAGE_THRESHOLD,
                    use_ccma=self.USE_CCMA_WHOLE_LAP,
                    ccma_coverage=self.CCMA_COVERAGE,
                    max_bin_count=self._max_bin_count(),
                    boundaries=self.boundaries,
                    converter_xy=(
                        np.asarray(self.converter.waypoints_x),
                        np.asarray(self.converter.waypoints_y),
                        np.asarray(self.converter.waypoints_z),
                    ),
                ),
                callback=self._on_processing_done,
                error_callback=self._on_processing_error,
            )

    def _on_processing_done(self, result):
        if not result:
            return
        k = result["opponent_id"]
        self.bins_multi[k] = result["binned_data"]
        if result["predictions"] is not None:
            self.predictions_multi[k] = result["predictions"]
            self.last_fit_bin_count_m[k] = result["bin_count_at_fit"]
        self.is_processing[k] = False

        if not any(self.is_processing):
            self._publish_predictions_multi()

    def _on_processing_error(self, err):
        rospy.logerr("[GP-binned-multi] worker error: %s", err)
        self.is_processing = [False] * self.NUM_OPPONENTS

    # ---------------- static workers (must be pickleable) ----------------

    @staticmethod
    def _bin_data_static(obs_records, bin_spacing, bin_capacity, binned_data):
        """Returns a deep-copied
        dict with the new observations merged in (capacity-evicted)."""
        b = {k: {"s": list(v["s"]), "d": list(v["d"]),
                 "vs": list(v["vs"]), "vd": list(v["vd"])}
             for k, v in binned_data.items()}
        for s, d, vs_, vd_ in obs_records:
            bin_idx = int(s // bin_spacing)
            slot = b.setdefault(bin_idx, {"s": [], "d": [], "vs": [], "vd": []})
            slot["s"].append(s)
            slot["d"].append(d)
            slot["vs"].append(vs_)
            slot["vd"].append(vd_)
            while len(slot["s"]) > bin_capacity:
                for kk in ("s", "d", "vs", "vd"):
                    slot[kk].pop(0)
        return b

    @staticmethod
    def _flatten_bins_with_wrap_static(bins, max_s, wrap_pad_m):
        ss, dd, vss, vdd = [], [], [], []
        for b in bins.values():
            ss.extend(b["s"])
            dd.extend(b["d"])
            vss.extend(b["vs"])
            vdd.extend(b["vd"])
        if not ss:
            return (np.array([]),) * 4
        ss = np.array(ss); dd = np.array(dd); vss = np.array(vss); vdd = np.array(vdd)
        near_end_mask = ss > (max_s - wrap_pad_m)
        near_start_mask = ss < wrap_pad_m
        pre_s = ss[near_end_mask] - max_s
        app_s = ss[near_start_mask] + max_s
        out_s = np.concatenate([pre_s, ss, app_s])
        out_d = np.concatenate([dd[near_end_mask], dd, dd[near_start_mask]])
        out_vs = np.concatenate([vss[near_end_mask], vss, vss[near_start_mask]])
        out_vd = np.concatenate([vdd[near_end_mask], vdd, vdd[near_start_mask]])
        return out_s, out_d, out_vs, out_vd

    @staticmethod
    def _ccma_static(ego_s, d_pred, converter_xy):
        """Worker-local CCMA smoothing. Rebuilds a FrenetConverter once per
        fork so self (un-picklable ROS node) is not needed."""
        x, y, z = converter_xy
        conv = FrenetConverter(np.asarray(x), np.asarray(y), np.asarray(z))
        s_list = list(ego_s)
        d_list = [float(v) for v in d_pred]
        s_list.insert(0, s_list[-1])
        d_list.insert(0, d_list[-1])
        s_list.append(s_list[1])
        d_list.append(d_list[1])
        noisy_xy = conv.get_cartesian(s_list, d_list).transpose()
        ccma = CCMA(w_ma=5, w_cc=3)
        smoothed_xy = ccma.filter(noisy_xy)
        smoothed_sd = conv.get_frenet(smoothed_xy[:, 0], smoothed_xy[:, 1])
        sm_s = np.asarray(smoothed_sd[0]).flatten()
        sm_d = np.asarray(smoothed_sd[1]).flatten()
        order = np.argsort(sm_s)
        return np.interp(ego_s, sm_s[order], sm_d[order])

    @staticmethod
    def _process_opponent(
        obs_records, opponent_id, *,
        binned_data, bin_spacing, bin_capacity, max_s, wrap_pad_m,
        ego_s_grid, kernel_d, kernel_vs,
        last_fit_bin_count, refit_new_bins, coverage_threshold,
        use_ccma, ccma_coverage, max_bin_count, boundaries, converter_xy,
    ):
        """Worker entry point: update bins → decide whether to fit → if fit,
        GP predict, clip to corridor, CCMA smooth, return predictions dict."""
        try:
            new_binned = BinnedGPNode._bin_data_static(
                obs_records, bin_spacing, bin_capacity, binned_data
            )
            n_bins = len(new_binned)
            coverage = n_bins / float(max_bin_count)

            predictions = None
            fit_gate_pass = coverage >= coverage_threshold and not (
                last_fit_bin_count > 0 and (n_bins - last_fit_bin_count) < refit_new_bins
            )
            if fit_gate_pass:
                train_s, train_d, train_vs, train_vd = (
                    BinnedGPNode._flatten_bins_with_wrap_static(
                        new_binned, max_s, wrap_pad_m
                    )
                )
                if train_s.shape[0] >= 3:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ConvergenceWarning)
                        s_col = train_s.reshape(-1, 1)
                        ego_s_arr = np.array(ego_s_grid).reshape(-1, 1)

                        gpr_vs = GaussianProcessRegressor(
                            kernel=kernel_vs, optimizer=_lbfgs_optimizer
                        )
                        gpr_vs.fit(s_col, train_vs.reshape(-1, 1))
                        vs_pred, sigma_vs = gpr_vs.predict(ego_s_arr, return_std=True)

                        gpr_d = GaussianProcessRegressor(
                            kernel=kernel_d, optimizer=_lbfgs_optimizer
                        )
                        gpr_d.fit(s_col, train_d.reshape(-1, 1))
                        d_pred, sigma_d = gpr_d.predict(ego_s_arr, return_std=True)

                    vs_pred = np.asarray(vs_pred).flatten()
                    sigma_vs = np.asarray(sigma_vs).flatten()
                    d_pred = np.asarray(d_pred).flatten()
                    sigma_d = np.asarray(sigma_d).flatten()

                    # Clip to drivable corridor (col 0 = right, col 1 = left)
                    if boundaries is not None:
                        d_pred = np.clip(d_pred, boundaries[:, 0], boundaries[:, 1])

                    # Whole-lap CCMA smoothing once coverage is close to 100%
                    if use_ccma and coverage >= ccma_coverage:
                        d_pred = BinnedGPNode._ccma_static(ego_s_grid, d_pred, converter_xy)

                    # vd is not GP-fit (interp from raw observations — same as single-opp path)
                    sort_idx = np.argsort(train_s)
                    vd_pred = np.interp(
                        ego_s_grid, train_s[sort_idx], train_vd[sort_idx]
                    )

                    predictions = dict(
                        s=np.asarray(ego_s_grid),
                        d_pred=d_pred,
                        sigma_d=sigma_d,
                        vs_pred=vs_pred,
                        sigma_vs=sigma_vs,
                        vd_pred=vd_pred,
                        coverage=coverage,
                        n_train=train_s.shape[0],
                    )

            return dict(
                opponent_id=opponent_id,
                predictions=predictions,
                binned_data=new_binned,
                bin_count_at_fit=n_bins,
            )
        except Exception as e:
            # Return partial result so is_processing[k] can reset
            return dict(
                opponent_id=opponent_id,
                predictions=None,
                binned_data=binned_data,
                bin_count_at_fit=last_fit_bin_count,
                error=str(e),
            )

    # ---------------- multi-opp publish ----------------

    def _publish_predictions_multi(self):
        """Build per-opp OppWpnt list + OpponentTrajectory, then publish:
           - /perception/opp_trajectories  (all opps)
           - /opponent_trajectory          (closest recent opp)
           - /opponent_traj_markerarray    (combined, per-opp coloured)"""
        N = self.NUM_OPPONENTS
        trajs_msg = OpponentTrajectories()
        markers = MarkerArray()
        marker_id = 0
        max_vel = self.max_velocity if self.max_velocity > 1e-3 else 10.0
        ego_s_arr = np.array(self.ego_s_original)
        try:
            z_surface_arr = np.asarray(self.converter.spline_z(ego_s_arr)).flatten()
        except Exception:
            z_surface_arr = np.zeros(len(ego_s_arr))

        per_opp_traj = {}  # k -> OpponentTrajectory (only for fitted opps)

        for k in range(N):
            pred = self.predictions_multi[k]
            if pred is None:
                continue

            d_pred = pred["d_pred"]
            xy = self.converter.get_cartesian(self.ego_s_original, d_pred.tolist())
            wpnts = self.oppwpnts_lists_multi[k]
            for i, s in enumerate(self.ego_s_original):
                bin_idx = int(s // self.BIN_SPACING)
                w = wpnts[i]
                w.x_m = float(xy[0][i])
                w.y_m = float(xy[1][i])
                w.d_m = float(pred["d_pred"][i])
                w.proj_vs_mps = float(pred["vs_pred"][i])
                w.vd_mps = float(pred["vd_pred"][i])
                w.d_var = float(pred["sigma_d"][i])
                w.vs_var = float(pred["sigma_vs"][i])
                w.is_observed = bin_idx in self.bins_multi[k]

            traj = OpponentTrajectory()
            traj.header.stamp = rospy.Time.now()
            traj.header.frame_id = "opponent_trajectory"
            traj.lap_count = float(max(1.0, min(2.0, 2.0 * pred["coverage"])))
            traj.opp_is_on_trajectory = True
            traj.oppwpnts = list(wpnts)
            per_opp_traj[k] = traj
            trajs_msg.trajectories.append(traj)

            # Per-opp marker (colour by tracker id)
            color = _id_to_rgb_color(self.detection_matching[k]["id"])
            for i, w in enumerate(wpnts):
                marker_height = w.proj_vs_mps / max_vel
                m = Marker(
                    header=rospy.Header(frame_id="map"),
                    id=marker_id,
                    type=Marker.CYLINDER,
                )
                marker_id += 1
                m.pose.position.x = w.x_m
                m.pose.position.y = w.y_m
                m.pose.position.z = float(z_surface_arr[i]) + marker_height / 2.0
                m.pose.orientation.w = 1.0
                m.scale.x = min(max(5.0 * w.d_var, 0.07), 0.7)
                m.scale.y = min(max(5.0 * w.d_var, 0.07), 0.7)
                m.scale.z = marker_height
                m.color.a = 1.0
                if not w.is_observed:
                    m.color.r, m.color.g, m.color.b = 1.0, 0.5, 0.0  # orange: unobserved fallback
                else:
                    m.color.r, m.color.g, m.color.b = color
                markers.markers.append(m)

        self.opp_trajs_pub.publish(trajs_msg)
        self.opp_traj_marker_pub.publish(markers)

        # Backwards-compat: closest recent opp → /opponent_trajectory (reference L533-547).
        now = rospy.Time.now()
        recent = [
            k for k in per_opp_traj
            if (now - self.detection_matching[k]["last_seen"]).to_sec()
            < self.RECENT_SEEN_SEC
        ]
        if recent:
            closest = min(recent, key=lambda kk: self.detection_matching[kk]["dist"])
            self.opp_traj_gp_pub.publish(per_opp_traj[closest])
        elif per_opp_traj:
            k0 = sorted(per_opp_traj.keys())[0]
            self.opp_traj_gp_pub.publish(per_opp_traj[k0])


    # ==================================================================
    # Main loop
    # ==================================================================

    def main(self):
        if self.NUM_OPPONENTS == 1:
            rospy.spin()
            return
        loop_rate = rospy.Rate(40)
        while not rospy.is_shutdown():
            self._assign_obstacle_to_trajectory()
            try:
                loop_rate.sleep()
            except rospy.ROSInterruptException:
                return


if __name__ == '__main__':
    # Init ROS before Pool so rospy.get_param works; Pool must be created
    # BEFORE any ROS subscriber (i.e. before BinnedGPNode.__init__'s subs)
    # so fork() does not inherit ROS sockets.
    rospy.init_node('gp_binned_trajectory', anonymous=False)
    _num_opp = int(rospy.get_param('~num_opponents', 1))
    _pool_size = int(rospy.get_param('~pool_size', 2))
    try:
        if _num_opp >= 2:
            with Pool(processes=max(1, _pool_size)) as pool:
                node = BinnedGPNode()
                node.pool = pool
                node.main()
        else:
            node = BinnedGPNode()
            node.main()
    except rospy.ROSInterruptException:
        pass
