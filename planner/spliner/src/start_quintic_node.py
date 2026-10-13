#!/usr/bin/env python3
# standalone test node for start path generation.
# Frenet-frame quintic d(s) with corrected heading-to-slope conversion
# and G2 boundary at the convergence point. RViz-only, does NOT publish
# into the live planning pipeline (everything sits under /start_planner_v3/*).
#
# Inputs : /global_waypoints (f110_msgs/WpntArray), /car_state/odom (Odometry)
# Outputs:
#   /start_planner_v3/path     f110_msgs/WpntArray   sampled path (x, y, s, d, kappa, psi)
#   /start_planner_v3/markers  MarkerArray           LINE_STRIP + ego/target spheres + status text
#   /start_planner_v3/debug    std_msgs/String       per-tick metrics (L, max_kappa, margins, feasible)
#
# Math :
#   d(s_0) = d_0
#   d'(s_0) = (1 - k_ref(s_0) * d_0) * tan(psi_rel_0)   # Frenet heading-to-slope
#   d''(s_0) = 0
#   d(s*) = 0, d'(s*) = 0, d''(s*) = 0                  # G2 convergence
#   Cartesian curvature uses full formula so feasibility is honest in curves.
import math
import os
import json
import yaml
import rospy
import rospkg
import numpy as np
import tf.transformations as tft

from nav_msgs.msg import Odometry
from std_msgs.msg import String, Bool
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped
from scipy.interpolate import BPoly
from dynamic_reconfigure.server import Server
import dynamic_reconfigure.client

from f110_msgs.msg import WpntArray, Wpnt, OTWpntArray
from frenet_converter.frenet_converter import FrenetConverter

from spliner.cfg import start_quinticConfig

# keys persisted to per-car yaml. Trigger-style bools (snapshot, clear_points,
# abort_start, save_params, load_params, auto_commit_on_snapshot) intentionally excluded
# — they are momentary commands, not state. auto_commit_on_snapshot is a behavior toggle
# but trivial; keep it out of yaml to avoid load resurrecting an unexpected default.
_PERSISTED_PARAM_KEYS = [
    "path_mode",
    "target_delta_s_m", "bound_margin_m", "ay_max_mps2", "assume_speed_mps",
    "n_samples", "min_delta_s_m", "zero_start_slope",
    "use_straight_prefix", "straight_prefix_m",
    "spline_scale", "points_tail_n",
    "points_use_straight_prefix", "points_straight_prefix_m",
]


def _wrap_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class StartQuinticNode:
    def __init__(self):
        rospy.init_node("start_planner_v3", anonymous=False)
        self.name = "start_planner_v3"

        # Raceline state (populated on first /global_waypoints).
        self.converter = None
        self.gb_s = None
        self.gb_x = None
        self.gb_y = None
        self.gb_z = None
        self.gb_d_left = None
        self.gb_d_right = None
        self.gb_max_s = None
        self.gb_wpnt_dist = None

        # Ego pose.
        self.cur_x = None
        self.cur_y = None
        self.cur_yaw = None

        # Dyn-reconfigure params (defaults match start_quintic.cfg).
        self.path_mode = 0                       # 0=QUINTIC, 1=VIAPOINTS
        self.target_delta_s_m = 5.0
        self.bound_margin_m = 0.15
        self.ay_max_mps2 = 4.0
        self.assume_speed_mps = 3.0
        self.n_samples = 80
        self.min_delta_s_m = 1.0
        self.zero_start_slope = False
        self.use_straight_prefix = False
        self.straight_prefix_m = 1.0
        self.spline_scale = 0.8
        self.points_tail_n = 40
        self.points_use_straight_prefix = False
        self.points_straight_prefix_m = 1.0
        # One-shot pending flag set by rqt 'snapshot_to_state_machine' trigger. The next
        # _tick after a flip publishes once on /planner/start_wpnts then clears the flag.
        # state_machine buffers the snapshot indefinitely; /save_start_traj commits it.
        self._pending_snapshot = False

        # POINTS state: each entry is (x, y, yaw_rad) captured from RViz 2D Nav Goal.
        self.points = []

        # I/O.
        rospy.Subscriber("/global_waypoints", WpntArray, self.gb_cb, queue_size=1)
        rospy.Subscriber("/car_state/odom", Odometry, self.odom_cb, queue_size=1)
        # RViz "2D Nav Goal" → each click appends a point (POINTS mode only).
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.point_cb, queue_size=10)
        self.path_pub = rospy.Publisher("/start_planner_v3/path", WpntArray, queue_size=1)
        self.mrk_pub = rospy.Publisher("/start_planner_v3/markers", MarkerArray, queue_size=1)
        self.dbg_pub = rospy.Publisher("/start_planner_v3/debug", String, queue_size=1)
        # OTWpntArray on the same topic the previous start planner used, so the
        # 3d_state_machine_node's start_wpnts_cb picks it up unmodified. Only emits
        # when publish_to_state_machine is True (rqt toggle).
        self.sm_pub = rospy.Publisher("/planner/start_wpnts", OTWpntArray, queue_size=1)
        # auto-commit + abort publishers. /save_start_traj is the same topic the
        # dynamic_statemachine save_start_traj toggle drives; we add a second publisher
        # so the rqt toggle there remains as a manual fallback. /abort_start_traj is new.
        self.save_pub = rospy.Publisher("/save_start_traj", Bool, queue_size=1)
        self.abort_pub = rospy.Publisher("/abort_start_traj", Bool, queue_size=1)
        self.auto_commit_on_snapshot = True

        # resolve per-car yaml path. racecar_version is the standard rosparam used
        # across the stack (launch files set it from $(env CAR_NAME)). Fall back to env
        # for standalone runs without launch, then to 'default' so the node never errors.
        car = rospy.get_param("~racecar_version", None)
        if not car:
            car = rospy.get_param("/racecar_version", None)
        if not car:
            car = os.getenv("CAR_NAME", None)
        if not car:
            car = "default"
            rospy.logwarn(f"[{self.name}] no racecar_version / CAR_NAME found, using '{car}'")
        self._car_name = car
        try:
            cfg_root = rospkg.RosPack().get_path("stack_master")
        except rospkg.ResourceNotFound:
            cfg_root = os.path.expanduser("~/catkin_ws/src/race_stack/stack_master")
            rospy.logwarn(f"[{self.name}] stack_master not found via rospack, falling back to {cfg_root}")
        self._yaml_path = os.path.join(cfg_root, "config", car, "start_quintic.yaml")
        rospy.loginfo(f"[{self.name}] yaml param store: {self._yaml_path}")

        # Server must be created before dyn_client (client needs the service up).
        self._dyn_server = Server(start_quinticConfig, self.dyn_cb)

        # self-client for load_params — pushes loaded values back through dyn
        # service so dyn_cb runs and self.* attributes stay in sync with rqt sliders.
        self._dyn_client = dynamic_reconfigure.client.Client(self.name, timeout=5.0)

        # auto-load on startup. If yaml exists, push it. Otherwise dump the
        # current .cfg defaults so the file exists for next time.
        if rospy.get_param("~auto_load_on_startup", True):
            if os.path.isfile(self._yaml_path):
                self._load_params_from_yaml()
            else:
                rospy.loginfo(f"[{self.name}] no yaml at {self._yaml_path}, creating with .cfg defaults")
                self._save_params_to_yaml()

        self.rate = rospy.Rate(10.0)

    # yaml persistence helpers -------------------
    def _current_params_dict(self):
        """Read current self.* values for the persisted keys."""
        return {k: getattr(self, k) for k in _PERSISTED_PARAM_KEYS}

    def _save_params_to_yaml(self):
        try:
            os.makedirs(os.path.dirname(self._yaml_path), exist_ok=True)
            data = self._current_params_dict()
            with open(self._yaml_path, "w") as f:
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
            rospy.loginfo(f"[{self.name}] params saved to {self._yaml_path}")
        except Exception as e:
            rospy.logwarn(f"[{self.name}] save_params failed: {e}")

    def _load_params_from_yaml(self):
        try:
            with open(self._yaml_path, "r") as f:
                data = yaml.safe_load(f) or {}
            update = {k: data[k] for k in _PERSISTED_PARAM_KEYS if k in data}
            if not update:
                rospy.logwarn(f"[{self.name}] yaml at {self._yaml_path} had no known keys")
                return
            self._dyn_client.update_configuration(update)
            rospy.loginfo(f"[{self.name}] params loaded from {self._yaml_path} ({len(update)} keys)")
        except Exception as e:
            rospy.logwarn(f"[{self.name}] load_params failed: {e}")

    # ------------------- Callbacks -------------------
    def gb_cb(self, msg: WpntArray):
        if not msg.wpnts:
            return
        x = np.array([w.x_m for w in msg.wpnts])
        y = np.array([w.y_m for w in msg.wpnts])
        z = np.array([w.z_m for w in msg.wpnts])
        d_left = np.array([w.d_left for w in msg.wpnts])
        d_right = np.array([w.d_right for w in msg.wpnts])

        # Build the FrenetConverter into a LOCAL var first; only commit to self.*
        # at the end. self.converter is assigned LAST so the main loop's
        # 'if self.converter is None' check acts as a real ready-gate — when it
        # passes, gb_max_s / gb_s / gb_wpnt_dist are guaranteed already set.
        # (Race observed before fix: gb_cb finished setting self.converter and
        # logged 'FrenetConverter built', then GIL yielded to the main thread
        # *before* gb_max_s assignment ran, so np.mod(s, None) crashed.)
        if self.converter is None or len(self.converter.waypoints_x) != len(x):
            converter = FrenetConverter(x, y, z)
            rospy.loginfo(f"[{self.name}] FrenetConverter built: {len(x)} wpnts, "
                          f"raceline_length={converter.raceline_length:.2f}m, "
                          f"d_left range=[{d_left.min():.2f}, {d_left.max():.2f}]m, "
                          f"d_right range=[{d_right.min():.2f}, {d_right.max():.2f}]m")
        else:
            converter = self.converter

        # Stage all derived fields, then publish self.converter as the gate.
        self.gb_d_left = d_left
        self.gb_d_right = d_right
        self.gb_s = converter.waypoints_s
        self.gb_x, self.gb_y, self.gb_z = x, y, z
        self.gb_max_s = float(converter.raceline_length)
        self.gb_wpnt_dist = float(converter.waypoints_distance_m)
        self.converter = converter   # LAST: ready gate for main loop

    def odom_cb(self, msg: Odometry):
        first = self.cur_x is None
        self.cur_x = msg.pose.pose.position.x
        self.cur_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.cur_yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        if first:
            rospy.loginfo(f"[{self.name}] first odom received: "
                          f"(x={self.cur_x:.2f}, y={self.cur_y:.2f}, yaw_deg={math.degrees(self.cur_yaw):.1f})")

    def dyn_cb(self, config, _level):
        self.path_mode = int(config.path_mode)
        self.target_delta_s_m = config.target_delta_s_m
        self.bound_margin_m = config.bound_margin_m
        self.ay_max_mps2 = config.ay_max_mps2
        self.assume_speed_mps = config.assume_speed_mps
        self.n_samples = int(config.n_samples)
        self.min_delta_s_m = config.min_delta_s_m
        self.zero_start_slope = config.zero_start_slope
        self.use_straight_prefix = config.use_straight_prefix
        self.straight_prefix_m = config.straight_prefix_m
        self.spline_scale = config.spline_scale
        self.points_tail_n = int(config.points_tail_n)
        self.points_use_straight_prefix = config.points_use_straight_prefix
        self.points_straight_prefix_m = config.points_straight_prefix_m
        # Trigger-style toggle: arm the pending flag, auto-reset the rqt checkbox.
        if config.snapshot_to_state_machine:
            self._pending_snapshot = True
            config.snapshot_to_state_machine = False
        # Trigger-style toggle: clear the points list and auto-reset the bool.
        if config.clear_points:
            rospy.loginfo(f"[{self.name}] cleared {len(self.points)} point(s)")
            self.points = []
            config.clear_points = False
        # new triggers — auto-commit setting, abort, save/load yaml.
        self.auto_commit_on_snapshot = config.auto_commit_on_snapshot
        if config.abort_start:
            self.abort_pub.publish(Bool(data=True))
            rospy.logwarn(f"[{self.name}] abort_start published")
            config.abort_start = False
        if config.save_params:
            self._save_params_to_yaml()
            config.save_params = False
        if config.load_params:
            # update_configuration would re-enter dyn_cb; defer one tick is overkill —
            # the recursive call just runs through this block again with all triggers
            # already False, which is safe.
            self._load_params_from_yaml()
            config.load_params = False
        return config

    def point_cb(self, msg: PoseStamped):
        # only consume RViz "2D Nav Goal" clicks in VIAPOINTS mode.
        # Other nodes share /move_base_simple/goal; ignore it unless we are
        # actively in POINTS mode so we don't accumulate foreign goals.
        if self.path_mode != 1:
            return
        q = msg.pose.orientation
        yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        self.points.append((float(msg.pose.position.x),
                            float(msg.pose.position.y),
                            float(yaw)))
        rospy.loginfo(f"[{self.name}] point #{len(self.points)} added: "
                      f"({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}, "
                      f"yaw={math.degrees(yaw):+.1f}°)")

    # ------------------- Raceline helpers (spline-based, smooth) -------------------
    def _psi_kappa_at(self, s_arr: np.ndarray):
        """Return (psi, kappa) on the raceline spline at given s values (wrap-safe)."""
        s_mod = np.mod(s_arr, self.gb_max_s)
        dx = self.converter.spline_x(s_mod, 1)
        dy = self.converter.spline_y(s_mod, 1)
        ddx = self.converter.spline_x(s_mod, 2)
        ddy = self.converter.spline_y(s_mod, 2)
        psi = np.arctan2(dy, dx)
        denom = (dx * dx + dy * dy) ** 1.5
        kappa = np.where(denom > 1e-9, (dx * ddy - dy * ddx) / denom, 0.0)
        return psi, kappa

    def _bounds_at(self, s_arr: np.ndarray):
        """Linear-interp (d_left, d_right) at given s values (wrap by mod)."""
        s_mod = np.mod(s_arr, self.gb_max_s)
        d_left = np.interp(s_mod, self.gb_s, self.gb_d_left)
        d_right = np.interp(s_mod, self.gb_s, self.gb_d_right)
        return d_left, d_right

    # ------------------- Quintic solve -------------------
    @staticmethod
    def _quintic_coeffs(d0, a0, b0):
        """
        Closed-form solution for d(tau) = c0 + c1*tau + ... + c5*tau^5
        with tau = (s - s0) / L, given:
            d(0)=d0, d'_s(0)=a0, d''_s(0)=b0,
            d(1)=0, d'_s(1)=0,   d''_s(1)=0
        where a0, b0 are slopes w.r.t. physical s (not tau). Caller supplies
        L externally; c1, c2 must be scaled by L, L^2 respectively when computing.
        """
        # The closed form is derived for c0=d0, c1, c2 already pre-scaled (i.e. in tau).
        # See node header comment for derivation.
        c3 = -10.0 * d0 - 6.0 * a0 - 3.0 * b0
        c4 = 15.0 * d0 + 8.0 * a0 + 3.0 * b0
        c5 = -6.0 * d0 - 3.0 * a0 - 1.0 * b0
        return d0, a0, b0, c3, c4, c5

    @staticmethod
    def _eval_quintic(coeffs, tau):
        c0, c1, c2, c3, c4, c5 = coeffs
        t1, t2, t3, t4, t5 = tau, tau ** 2, tau ** 3, tau ** 4, tau ** 5
        d = c0 + c1 * t1 + c2 * t2 + c3 * t3 + c4 * t4 + c5 * t5
        d_dtau = c1 + 2 * c2 * t1 + 3 * c3 * t2 + 4 * c4 * t3 + 5 * c5 * t4
        d_ddtau = 2 * c2 + 6 * c3 * t1 + 12 * c4 * t2 + 20 * c5 * t3
        return d, d_dtau, d_ddtau

    # ------------------- Cartesian curvature in Frenet frame -------------------
    @staticmethod
    def _cart_kappa(d, d_p, d_pp, k_ref):
        """Full curvature of a path given by d(s) on top of a curved reference."""
        one_minus_kd = 1.0 - k_ref * d
        num = (k_ref * one_minus_kd ** 2
               + d_pp * one_minus_kd
               + 2.0 * k_ref * d_p ** 2)
        den = (one_minus_kd ** 2 + d_p ** 2) ** 1.5
        return np.where(den > 1e-9, num / den, 0.0)

    # ------------------- Main loop -------------------
    def loop(self):
        rospy.loginfo(f"[{self.name}] waiting for /global_waypoints and /car_state/odom...")
        while not rospy.is_shutdown():
            if self.converter is None or self.cur_x is None:
                self.rate.sleep()
                continue
            self._tick()
            self.rate.sleep()

    def _tick(self):
        # Dispatch by mode. Quintic uses target_delta_s + heading-aware Frenet d(s);
        # viapoints uses RViz 2D Nav Goal clicks + BPoly through ego/clicks/GB.
        if self.path_mode == 0:
            self._tick_quintic()
        else:
            self._tick_points()

    def _tick_quintic(self):
        # 1. Ego in Frenet.
        s_arr, d_arr = self.converter.get_frenet(np.array([self.cur_x]), np.array([self.cur_y]))
        s_0 = float(s_arr[0])
        d_0 = float(d_arr[0])

        # 2. Optional straight prefix along ego heading (Cartesian).
        #    Heading stays constant along the line; only s/d move on the Frenet manifold.
        L_pre = self.straight_prefix_m if self.use_straight_prefix else 0.0
        if L_pre > 1e-3:
            step = max(self.gb_wpnt_dist, 0.05)
            M = max(2, int(round(L_pre / step)) + 1)
            u = np.linspace(0.0, 1.0, M)
            pre_x = self.cur_x + u * L_pre * math.cos(self.cur_yaw)
            pre_y = self.cur_y + u * L_pre * math.sin(self.cur_yaw)
            pre_s_raw, pre_d = self.converter.get_frenet(pre_x, pre_y)
            # Use the prefix endpoint as the quintic start state.
            s_q0 = float(pre_s_raw[-1])
            d_q0 = float(pre_d[-1])
            has_pre = True
        else:
            pre_x = pre_y = pre_s_raw = pre_d = np.array([])
            s_q0 = s_0
            d_q0 = d_0
            has_pre = False

        # 3. Raceline geometry at quintic-start s_q0 (heading unchanged along straight prefix).
        psi_ref_q0, kappa_ref_q0 = self._psi_kappa_at(np.array([s_q0]))
        psi_rel_q0 = _wrap_pi(self.cur_yaw - float(psi_ref_q0[0]))

        # 4. Boundary conditions for quintic in physical s.
        if self.zero_start_slope:
            a0_phys = 0.0
        else:
            # Clamp tan to avoid blowup when ego heading is wildly off; quintic can't fix that anyway.
            psi_rel_clamped = float(np.clip(psi_rel_q0, -math.radians(75.0), math.radians(75.0)))
            a0_phys = (1.0 - float(kappa_ref_q0[0]) * d_q0) * math.tan(psi_rel_clamped)
        b0_phys = 0.0

        # 5. Convergence horizon, measured from quintic-start s_q0 (after the prefix).
        #    target_delta_s_m is RELATIVE to ego's current s_0, so the slider is independent
        #    of where on the track the car sits (no wrap-around surprises).
        s_target_raw = (s_0 + float(self.target_delta_s_m)) % self.gb_max_s
        L_total = float(self.target_delta_s_m)                   # requested ego→target arc
        L_pre_arc = (s_q0 - s_0) % self.gb_max_s if has_pre else 0.0
        L = (s_target_raw - s_q0) % self.gb_max_s
        status = "ok"
        # Guard: prefix endpoint must sit between ego and target along the forward arc.
        # If not, the mod wraps L into a full lap. Happens when straight_prefix_m
        # is >= target_delta_s_m, or when ego heading drives the prefix backward in s.
        if has_pre and L_pre_arc >= L_total:
            status = (f"prefix overshoots target: L_pre_arc={L_pre_arc:.2f}m "
                      f">= target_delta_s={L_total:.2f}m  "
                      f"(reduce straight_prefix_m or raise target_delta_s_m)")
            self._publish_status_only(s_0, d_0, s_target_raw, L, status)
            return
        if L < self.min_delta_s_m:
            status = (f"L={L:.2f}m < min_delta_s={self.min_delta_s_m:.2f}m"
                      + (f" (after prefix={L_pre:.2f}m)" if has_pre else ""))
            self._publish_status_only(s_0, d_0, s_target_raw, L, status)
            return

        # 6. Quintic in normalized tau ∈ [0, 1]. Scale slopes to tau-domain.
        c0 = d_q0
        c1 = a0_phys * L
        c2 = b0_phys * (L * L) / 2.0
        coeffs = self._quintic_coeffs(c0, c1, c2)

        # 7. Sample quintic.
        N = max(2, int(self.n_samples))
        tau = np.linspace(0.0, 1.0, N)
        d_tau, dprime_tau, dppr_tau = self._eval_quintic(coeffs, tau)
        d_q = d_tau
        dprime_s = dprime_tau / L                     # d'(s)
        dpp_s = dppr_tau / (L * L)                    # d''(s)
        s_q = s_q0 + tau * L                          # absolute s (may exceed max_s; mod when querying)

        # 8. Cartesian curvature on the quintic part (straight prefix has κ=0).
        _, kappa_ref_q = self._psi_kappa_at(s_q)
        kappa_cart_q = self._cart_kappa(d_q, dprime_s, dpp_s, kappa_ref_q)

        # 9. Build combined sample arrays (prefix + quintic, drop the duplicated junction sample).
        if has_pre:
            s_all = np.concatenate([pre_s_raw[:-1], s_q])
            d_all = np.concatenate([pre_d[:-1], d_q])
            kappa_all = np.concatenate([np.zeros(len(pre_s_raw) - 1), kappa_cart_q])
        else:
            s_all = s_q
            d_all = d_q
            kappa_all = kappa_cart_q

        # 10. Feasibility — curvature (quintic part) + bounds (whole path).
        kappa_grip = self.ay_max_mps2 / max(self.assume_speed_mps, 0.1) ** 2
        max_abs_k = float(np.max(np.abs(kappa_cart_q)))
        d_left_all, d_right_all = self._bounds_at(s_all)
        left_margin = d_left_all - d_all - self.bound_margin_m
        right_margin = d_all + d_right_all - self.bound_margin_m
        min_left = float(np.min(left_margin))
        min_right = float(np.min(right_margin))
        kappa_ok = max_abs_k <= kappa_grip
        bounds_ok = (min_left > 0.0) and (min_right > 0.0)
        feasible = kappa_ok and bounds_ok
        if not feasible:
            status = (f"kappa_ok={kappa_ok} (max|k|={max_abs_k:.3f}/limit={kappa_grip:.3f}), "
                      f"bounds_ok={bounds_ok} (min_left={min_left:.3f}, min_right={min_right:.3f})")

        # 11. Cartesian samples for viz / WpntArray.
        #     Straight prefix: use the literal cartesian samples (avoids Frenet round-trip jitter).
        #     Quintic: Frenet → Cartesian via converter.
        xy_q = self.converter.get_cartesian(np.mod(s_q, self.gb_max_s), d_q)
        x_q, y_q = xy_q[0], xy_q[1]
        if has_pre:
            x_all = np.concatenate([pre_x[:-1], x_q])
            y_all = np.concatenate([pre_y[:-1], y_q])
        else:
            x_all = x_q
            y_all = y_q
        psi_path = np.arctan2(np.gradient(y_all), np.gradient(x_all))

        # 12. Publish.
        self._publish_path(s_all, d_all, x_all, y_all, psi_path, kappa_all)
        self._publish_markers(x_all, y_all, s_target_raw, feasible,
                              L, max_abs_k, kappa_grip, min_left, min_right,
                              kappa_ok=kappa_ok, bounds_ok=bounds_ok,
                              prefix_len=(len(pre_x) - 1) if has_pre else 0)
        self._publish_debug(s_0, d_0, math.degrees(psi_rel_q0), float(kappa_ref_q0[0]),
                            a0_phys, L_pre, L, max_abs_k, kappa_grip,
                            min_left, min_right, feasible, status)
        if self._pending_snapshot:
            self._publish_to_state_machine(s_all, d_all, x_all, y_all, psi_path, kappa_all)
            self._pending_snapshot = False
            rospy.loginfo(f"[{self.name}] snapshot sent to state_machine candidate")
            # auto-commit — fire /save_start_traj in the same tick so the
            # state_machine pending-flag is set right after start_wpnts_cb stores
            # the candidate. Race-safe on the state_machine side: both subs land in
            # the same callback queue and the main loop drains the flag next tick.
            if self.auto_commit_on_snapshot:
                self.save_pub.publish(Bool(data=True))
                rospy.loginfo(f"[{self.name}] auto-commit: /save_start_traj published")

    # ------------------- POINTS mode -------------------
    def _tick_points(self):
        """Ported from the previous start planner (Cartesian BPoly).
        Path = optional straight prefix → BPoly through ego + click points →
               snap last click to nearest GB wpnt → optional GB tail.
        Independent straight prefix toggle/length from QUINTIC mode.
        Feasibility checked with the same kappa + bounds rules as quintic.
        """
        if len(self.points) == 0:
            self._publish_status_only(0.0, 0.0, 0.0, 0.0,
                                       "POINTS: click RViz '2D Nav Goal' to add points")
            return

        # 1. Optional straight prefix along ego heading (Cartesian samples).
        L_pre = self.points_straight_prefix_m if self.points_use_straight_prefix else 0.0
        if L_pre > 1e-3:
            step = max(self.gb_wpnt_dist, 0.05)
            M = max(2, int(round(L_pre / step)) + 1)
            u = np.linspace(0.0, 1.0, M)
            pre_x = self.cur_x + u * L_pre * math.cos(self.cur_yaw)
            pre_y = self.cur_y + u * L_pre * math.sin(self.cur_yaw)
            spline_start_x = float(pre_x[-1])
            spline_start_y = float(pre_y[-1])
            has_pre = True
        else:
            pre_x = pre_y = np.array([])
            spline_start_x = self.cur_x
            spline_start_y = self.cur_y
            has_pre = False

        # 2. Build (point, tangent) sequence for BPoly: start + all-but-last clicks + GB-snap final.
        bpts = [[spline_start_x, spline_start_y]]
        btgs = [[math.cos(self.cur_yaw), math.sin(self.cur_yaw)]]
        for vx, vy, vyaw in self.points[:-1]:
            bpts.append([vx, vy])
            btgs.append([math.cos(vyaw), math.sin(vyaw)])

        # Snap last click to nearest GB wpnt, terminate with raceline tangent.
        last_vx, last_vy, _ = self.points[-1]
        s_last_arr, _ = self.converter.get_frenet(np.array([last_vx]), np.array([last_vy]))
        s_last_mod = float(s_last_arr[0]) % self.gb_max_s
        last_idx = int(round(s_last_mod / self.gb_wpnt_dist)) % len(self.gb_x)
        psi_last_arr, _ = self._psi_kappa_at(np.array([s_last_mod]))
        psi_last = float(psi_last_arr[0])
        bpts.append([float(self.gb_x[last_idx]), float(self.gb_y[last_idx])])
        btgs.append([math.cos(psi_last), math.sin(psi_last)])

        bpts = np.asarray(bpts)
        btgs = np.asarray(btgs) * self.spline_scale
        if len(bpts) < 2:
            self._publish_status_only(0.0, 0.0, s_last_mod, 0.0, "POINTS: need ≥1 click")
            return

        # 3. Cumulative chord length parametrization for BPoly.
        dp = np.linalg.norm(np.diff(bpts, axis=0), axis=1)
        d_param = np.hstack([[0.0], np.cumsum(dp)])
        spline_len = float(d_param[-1])
        if spline_len < 1e-2:
            self._publish_status_only(0.0, 0.0, s_last_mod, 0.0,
                                       "POINTS: total chord length ~0")
            return

        step = max(self.gb_wpnt_dist, 0.05)
        n_samples = max(2, int(spline_len / step))
        s_param = np.linspace(0.0, spline_len, n_samples)

        xy = np.zeros((n_samples, 2))
        for i in range(2):
            spline_pairs = [(bpts[k, i], btgs[k, i]) for k in range(len(bpts))]
            poly = BPoly.from_derivatives(d_param, spline_pairs)
            xy[:, i] = poly(s_param)
        x_main, y_main = xy[:, 0], xy[:, 1]

        # 4. Combine prefix + BPoly + GB tail (drop the prefix endpoint duplicate at the junction).
        if self.points_tail_n > 0:
            idxs = [(last_idx + k + 1) % len(self.gb_x) for k in range(self.points_tail_n)]
            tail_x = np.array([self.gb_x[k] for k in idxs])
            tail_y = np.array([self.gb_y[k] for k in idxs])
        else:
            tail_x = tail_y = np.array([])

        chunks_x = []
        chunks_y = []
        if has_pre:
            chunks_x.append(pre_x[:-1]); chunks_y.append(pre_y[:-1])
        chunks_x.append(x_main); chunks_y.append(y_main)
        if len(tail_x) > 0:
            chunks_x.append(tail_x); chunks_y.append(tail_y)
        x_all = np.concatenate(chunks_x)
        y_all = np.concatenate(chunks_y)
        prefix_len = (len(pre_x) - 1) if has_pre else 0
        total_len = float(spline_len + L_pre)

        # 5. Cartesian curvature via numerical derivatives (prefix part has ~0).
        dx = np.gradient(x_all)
        dy = np.gradient(y_all)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        chord_len = np.hypot(dx, dy)
        kappa_all = np.where(chord_len > 1e-9,
                             (dx * ddy - dy * ddx) / np.maximum(chord_len, 1e-9) ** 3,
                             0.0)

        # 6. Frenet (s, d) per sample → boundary check across the whole path.
        s_raw, d_all = self.converter.get_frenet(x_all, y_all)
        s_all = np.mod(s_raw, self.gb_max_s)
        d_left_all, d_right_all = self._bounds_at(s_all)
        left_margin = d_left_all - d_all - self.bound_margin_m
        right_margin = d_all + d_right_all - self.bound_margin_m
        min_left = float(np.min(left_margin))
        min_right = float(np.min(right_margin))

        kappa_grip = self.ay_max_mps2 / max(self.assume_speed_mps, 0.1) ** 2
        max_abs_k = float(np.max(np.abs(kappa_all)))
        kappa_ok = max_abs_k <= kappa_grip
        bounds_ok = (min_left > 0.0) and (min_right > 0.0)
        feasible = kappa_ok and bounds_ok
        status = "ok" if feasible else (
            f"kappa_ok={kappa_ok} (max|k|={max_abs_k:.3f}/limit={kappa_grip:.3f}), "
            f"bounds_ok={bounds_ok} (min_left={min_left:.3f}, min_right={min_right:.3f})")

        psi_path = np.arctan2(np.gradient(y_all), np.gradient(x_all))
        self._publish_path(s_all, d_all, x_all, y_all, psi_path, kappa_all)
        self._publish_markers(x_all, y_all, s_last_mod, feasible,
                              total_len, max_abs_k, kappa_grip, min_left, min_right,
                              kappa_ok=kappa_ok, bounds_ok=bounds_ok,
                              prefix_len=prefix_len)
        self._publish_points_markers()
        self._publish_debug_points(total_len, L_pre, len(self.points), s_last_mod,
                                    max_abs_k, kappa_grip, min_left, min_right,
                                    feasible, status)
        if self._pending_snapshot:
            self._publish_to_state_machine(s_all, d_all, x_all, y_all, psi_path, kappa_all)
            self._pending_snapshot = False
            rospy.loginfo(f"[{self.name}] snapshot sent to state_machine candidate")
            # auto-commit (POINTS mode) — same flow as QUINTIC.
            if self.auto_commit_on_snapshot:
                self.save_pub.publish(Bool(data=True))
                rospy.loginfo(f"[{self.name}] auto-commit: /save_start_traj published")

    # ------------------- Publish helpers -------------------
    def _publish_path(self, s, d, x, y, psi, kappa):
        msg = WpntArray()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        for i in range(len(s)):
            w = Wpnt()
            w.id = i
            w.s_m = float(s[i] % self.gb_max_s)
            w.d_m = float(d[i])
            w.x_m = float(x[i])
            w.y_m = float(y[i])
            w.psi_rad = float(psi[i])
            w.kappa_radpm = float(kappa[i])
            msg.wpnts.append(w)
        self.path_pub.publish(msg)

    def _publish_to_state_machine(self, s, d, x, y, psi, kappa):
        """Emit the same path as OTWpntArray on /planner/start_wpnts.
        state_machine.start_wpnts_cb stores this as cur_start_wpnts_candidate; the
        rqt 'save_start_traj' toggle later applies the curvature-based vel profile
        and forces cur_state = START. vx_mps is left 0 here — update_velocity()
        overwrites it.
        """
        msg = OTWpntArray()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        for i in range(len(s)):
            w = Wpnt()
            w.id = i
            w.s_m = float(s[i] % self.gb_max_s)
            w.d_m = float(d[i])
            w.x_m = float(x[i])
            w.y_m = float(y[i])
            w.psi_rad = float(psi[i])
            w.kappa_radpm = float(kappa[i])
            msg.wpnts.append(w)
        self.sm_pub.publish(msg)

    def _publish_markers(self, x, y, s_target, feasible,
                         L, max_abs_k, kappa_grip, min_left, min_right,
                         kappa_ok=True, bounds_ok=True,
                         prefix_len=0):
        ma = MarkerArray()
        # Clear previous frame.
        clr = Marker()
        clr.header.frame_id = "map"
        clr.header.stamp = rospy.Time.now()
        clr.action = Marker.DELETEALL
        ma.markers.append(clr)

        # Path as sphere-per-sample (waypoint-like viz, mirrors how GB raceline is rendered).
        # Color: green if feasible, red if not.
        quintic_start = max(0, prefix_len)
        sph = Marker()
        sph.header.frame_id = "map"
        sph.header.stamp = rospy.Time.now()
        sph.ns = "path_quintic"
        sph.id = 0
        sph.type = Marker.SPHERE_LIST
        sph.action = Marker.ADD
        sph.scale.x = sph.scale.y = sph.scale.z = 0.10
        sph.color.a = 1.0
        if feasible:
            sph.color.r, sph.color.g, sph.color.b = 0.0, 1.0, 0.2
        else:
            sph.color.r, sph.color.g, sph.color.b = 1.0, 0.2, 0.2
        sph.pose.orientation.w = 1.0
        for xi, yi in zip(x[quintic_start:], y[quintic_start:]):
            p = Point()
            p.x, p.y, p.z = float(xi), float(yi), 0.05
            sph.points.append(p)
        ma.markers.append(sph)

        # Straight prefix waypoints (cyan) — only when prefix_len > 0. Include junction point.
        if prefix_len > 0:
            pre_sph = Marker()
            pre_sph.header.frame_id = "map"
            pre_sph.header.stamp = rospy.Time.now()
            pre_sph.ns = "path_prefix"
            pre_sph.id = 4
            pre_sph.type = Marker.SPHERE_LIST
            pre_sph.action = Marker.ADD
            pre_sph.scale.x = pre_sph.scale.y = pre_sph.scale.z = 0.11
            pre_sph.color.a = 1.0
            pre_sph.color.r, pre_sph.color.g, pre_sph.color.b = 0.0, 0.8, 1.0
            pre_sph.pose.orientation.w = 1.0
            for xi, yi in zip(x[: prefix_len + 1], y[: prefix_len + 1]):
                p = Point()
                p.x, p.y, p.z = float(xi), float(yi), 0.05
                pre_sph.points.append(p)
            ma.markers.append(pre_sph)

        # Ego sphere.
        ego = Marker()
        ego.header.frame_id = "map"
        ego.header.stamp = rospy.Time.now()
        ego.ns = "ego"
        ego.id = 1
        ego.type = Marker.SPHERE
        ego.scale.x = ego.scale.y = ego.scale.z = 0.18
        ego.color.a = 1.0
        ego.color.r, ego.color.g, ego.color.b = 0.2, 0.6, 1.0
        ego.pose.position.x = float(self.cur_x)
        ego.pose.position.y = float(self.cur_y)
        ego.pose.position.z = 0.05
        ego.pose.orientation.w = 1.0
        ma.markers.append(ego)

        # Target sphere on raceline.
        tgt_xy = self.converter.get_cartesian(np.array([s_target]), np.array([0.0]))
        tgt = Marker()
        tgt.header.frame_id = "map"
        tgt.header.stamp = rospy.Time.now()
        tgt.ns = "target"
        tgt.id = 2
        tgt.type = Marker.SPHERE
        tgt.scale.x = tgt.scale.y = tgt.scale.z = 0.25
        tgt.color.a = 1.0
        tgt.color.r, tgt.color.g, tgt.color.b = 1.0, 0.8, 0.0
        tgt.pose.position.x = float(tgt_xy[0][0])
        tgt.pose.position.y = float(tgt_xy[1][0])
        tgt.pose.position.z = 0.05
        tgt.pose.orientation.w = 1.0
        ma.markers.append(tgt)

        # Multi-line status text — explicit per-check FAIL/OK so user can read why.
        kappa_tag = "OK  " if kappa_ok else "FAIL"
        left_tag = "OK  " if (min_left > 0.0) else "FAIL"
        right_tag = "OK  " if (min_right > 0.0) else "FAIL"
        header = "FEASIBLE" if feasible else "INFEASIBLE"
        lines = [
            f"[{header}]",
            f"L={L:.2f} m",
            f"kappa  [{kappa_tag}]  max|k|={max_abs_k:.3f} / limit={kappa_grip:.3f}",
            f"left   [{left_tag}]  margin={min_left:+.3f} m",
            f"right  [{right_tag}]  margin={min_right:+.3f} m",
        ]
        txt = Marker()
        txt.header.frame_id = "map"
        txt.header.stamp = rospy.Time.now()
        txt.ns = "status"
        txt.id = 3
        txt.type = Marker.TEXT_VIEW_FACING
        txt.scale.z = 0.25
        txt.color.a = 1.0
        if feasible:
            txt.color.r, txt.color.g, txt.color.b = 0.6, 1.0, 0.6
        else:
            txt.color.r, txt.color.g, txt.color.b = 1.0, 0.5, 0.5
        txt.pose.position.x = float(self.cur_x)
        txt.pose.position.y = float(self.cur_y)
        txt.pose.position.z = 1.2
        txt.text = "\n".join(lines)
        ma.markers.append(txt)

        self.mrk_pub.publish(ma)

    def _publish_status_only(self, s_0, d_0, s_target, L, status):
        ma = MarkerArray()
        clr = Marker()
        clr.header.frame_id = "map"
        clr.header.stamp = rospy.Time.now()
        clr.action = Marker.DELETEALL
        ma.markers.append(clr)
        txt = Marker()
        txt.header.frame_id = "map"
        txt.header.stamp = rospy.Time.now()
        txt.ns = "status"
        txt.id = 0
        txt.type = Marker.TEXT_VIEW_FACING
        txt.scale.z = 0.30
        txt.color.a = 1.0
        txt.color.r, txt.color.g, txt.color.b = 1.0, 0.5, 0.0
        txt.pose.position.x = float(self.cur_x)
        txt.pose.position.y = float(self.cur_y)
        txt.pose.position.z = 0.8
        txt.text = f"SKIP: {status}"
        ma.markers.append(txt)
        self.mrk_pub.publish(ma)
        self.dbg_pub.publish(String(data=json.dumps({
            "skip": status, "s0": s_0, "d0": d_0, "s_target": s_target, "L": L,
        })))

    def _publish_debug(self, s0, d0, psi_rel_q0_deg, kappa_ref_q0, a0_phys,
                       L_pre, L, max_abs_k, kappa_grip, min_left, min_right, feasible, status):
        payload = {
            "mode": "QUINTIC",
            "s0": round(s0, 3),
            "d0": round(d0, 3),
            "L_pre": round(L_pre, 3),
            "psi_rel_q0_deg": round(psi_rel_q0_deg, 2),
            "kappa_ref_q0": round(kappa_ref_q0, 4),
            "a0_phys": round(a0_phys, 4),
            "L": round(L, 3),
            "max_abs_kappa": round(max_abs_k, 4),
            "kappa_grip": round(kappa_grip, 4),
            "min_left_margin": round(min_left, 3),
            "min_right_margin": round(min_right, 3),
            "feasible": bool(feasible),
            "status": status,
        }
        self.dbg_pub.publish(String(data=json.dumps(payload)))

    def _publish_debug_points(self, total_len, L_pre, n_points, s_snap,
                              max_abs_k, kappa_grip, min_left, min_right, feasible, status):
        payload = {
            "mode": "POINTS",
            "n_points": int(n_points),
            "L_pre": round(L_pre, 3),
            "s_snap": round(s_snap, 3),
            "total_len": round(total_len, 3),
            "max_abs_kappa": round(max_abs_k, 4),
            "kappa_grip": round(kappa_grip, 4),
            "min_left_margin": round(min_left, 3),
            "min_right_margin": round(min_right, 3),
            "feasible": bool(feasible),
            "status": status,
        }
        self.dbg_pub.publish(String(data=json.dumps(payload)))

    def _publish_points_markers(self):
        """Draw an orange sphere + numeric label for each captured click point.
        Renders only in POINTS mode; appended to the same MarkerArray topic so
        RViz config stays the same.
        """
        ma = MarkerArray()
        for i, (vx, vy, _) in enumerate(self.points):
            sph = Marker()
            sph.header.frame_id = "map"
            sph.header.stamp = rospy.Time.now()
            sph.ns = "click_points"
            sph.id = i * 2
            sph.type = Marker.SPHERE
            sph.scale.x = sph.scale.y = sph.scale.z = 0.20
            sph.color.a = 1.0
            sph.color.r, sph.color.g, sph.color.b = 1.0, 0.5, 0.0
            sph.pose.position.x = vx
            sph.pose.position.y = vy
            sph.pose.position.z = 0.10
            sph.pose.orientation.w = 1.0
            ma.markers.append(sph)
            lbl = Marker()
            lbl.header.frame_id = "map"
            lbl.header.stamp = rospy.Time.now()
            lbl.ns = "click_points"
            lbl.id = i * 2 + 1
            lbl.type = Marker.TEXT_VIEW_FACING
            lbl.scale.z = 0.22
            lbl.color.a = 1.0
            lbl.color.r, lbl.color.g, lbl.color.b = 1.0, 0.7, 0.2
            lbl.pose.position.x = vx
            lbl.pose.position.y = vy
            lbl.pose.position.z = 0.45
            lbl.text = f"#{i + 1}"
            ma.markers.append(lbl)
        if ma.markers:
            self.mrk_pub.publish(ma)


if __name__ == "__main__":
    node = StartQuinticNode()
    try:
        node.loop()
    except rospy.ROSInterruptException:
        pass
