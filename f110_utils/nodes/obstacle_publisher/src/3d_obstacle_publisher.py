#!/usr/bin/env python3

# 3D port of obstacle_publisher.py
# - Glob2FrenetArr now requires (x, y, z); pass wpnt.z_m
# - Frenet2GlobArr now returns z; fill Obstacle.z_m and OppWpnt height
# - Marker visualization uses obstacle.z_m so it sits on the track surface
# - Node name / topics unchanged so it can drop-in replace the 2D node
#
# mimic real tracker output
# (1) Detect gate: only publish when obstacle is within [s_back, s_fwd] of ego
# and |dd| < d_max (Frenet, with s-wrap).
# (2) Coast: once lost, keep publishing last posterior for `coast_duration` s
# (freeze mode), matching ttl_dynamic=40 / rate_tracking=40Hz = 1.0s.
# (3) Cartesian xyz noise injection → re-derive (s,d) via glob2frenet so
# msg fields stay consistent.
# (4) Linear KF on state x=[s, d, vs, vd] with const-velocity model;
# posterior is what we publish, so downstream sees a smoothed tracker-
# like output rather than raw noisy measurements.

import rospy
import numpy as np
import copy
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Float64MultiArray
from f110_msgs.msg import ObstacleArray, Obstacle, WpntArray, Wpnt, OpponentTrajectory, OppWpnt
from visualization_msgs.msg import (
    Marker, MarkerArray,
    InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback,
)
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler
from frenet_conversion.srv import Glob2FrenetArr, Frenet2GlobArr
from nav_msgs.msg import Odometry


class ObstaclePublisher3D:
    """3D-aware dynamic obstacle publisher.

    Same behavior as obstacle_publisher.py but exercises the updated 3D
    Frenet service signatures and propagates z_m through the obstacle/marker
    pipeline so visualization sits on the actual track surface.
    """
    def __init__(self):
        looprate = 50
        self.rate = rospy.Rate(looprate)
        self.looptime = 1/looprate

        self.dynamic_obstacle = self.init_dynamic_obstacle()
        self.obj_len = 0.5

        # Parameters (use ~private namespace via param server, mirroring 2D node)
        self.speed_scaler = rospy.get_param("obstacle_publisher/speed_scaler", 1)
        self.constant = rospy.get_param("obstacle_publisher/constant_speed", False)

        self.waypoints_type = rospy.get_param("/obstacle_publisher/trajectory", "min_curv")
        if self.waypoints_type == "min_curv":
            self.waypoints_topic = "/global_waypoints"
        elif self.waypoints_type == "shortest_path":
            self.waypoints_topic = "/global_waypoints/shortest_path"
        elif self.waypoints_type == "centerline":
            self.waypoints_topic = "/centerline_waypoints"
        elif self.waypoints_type == "updated":
            self.waypoints_topic = "/global_waypoints_updated"
            print("Using updated waypoints")
        elif self.waypoints_type == "min_time":
            raise NotImplementedError(
                "LTO Trajectory is not currently implemented. Choose another trajectory type."
            )
        else:
            raise ValueError(
                f"Waypoints of type {self.waypoints_type} are not supported."
            )

        self.starting_s = rospy.get_param("/obstacle_publisher/start_s", 0)

        # d-oscillation (mechanism ported from obstacle_publisher_grid.py)
        # Mode "temporal": phase advances by 2π·f·dt → constant time period
        # Mode "spatial":  phase locked to s → constant spatial wavelength (λ=max_s/n)
        # Use long temporal periods OR spatial mode to get a "uniform overshoot line"
        self.d_oscillate = rospy.get_param("/obstacle_publisher/d_oscillate", False)
        self.osc_mode = rospy.get_param("/obstacle_publisher/osc_mode", "spatial")
        self.osc_n_waves_per_lap = rospy.get_param("/obstacle_publisher/osc_n_waves_per_lap", 1.0)
        self.osc_frequency = rospy.get_param("/obstacle_publisher/osc_frequency", 0.1)
        self.osc_amplitude = rospy.get_param("/obstacle_publisher/osc_amplitude", 0.4)
        self.osc_phase0 = rospy.get_param("/obstacle_publisher/osc_phase0", 0.0)
        # temporal-mode state
        self._osc_phase = self.osc_phase0

        # detect gate / coast / noise / KF parameters
        # Detect gate (ego-relative, Frenet, with s-wrap)
        self.detect_s_back = rospy.get_param("/obstacle_publisher/detect_s_back", -5.0)
        self.detect_s_fwd  = rospy.get_param("/obstacle_publisher/detect_s_fwd",  10.0)
        self.detect_d_max  = rospy.get_param("/obstacle_publisher/detect_d_max",  10.0)
        # Coast (matches ttl_dynamic=40 / rate_tracking=40Hz = 1.0s by default)
        self.coast_duration = rospy.get_param("/obstacle_publisher/coast_duration", 1.0)
        # Measurement noise (cartesian xyz); re-derived s,d via glob2frenet
        self.noise_enable   = rospy.get_param("/obstacle_publisher/noise_enable", True)
        self.noise_sigma_xy = rospy.get_param("/obstacle_publisher/noise_sigma_xy", 0.05)
        self.noise_sigma_z  = rospy.get_param("/obstacle_publisher/noise_sigma_z",  0.02)
        # Linear KF on [s, d, vs, vd] (const-velocity); Q/R diag
        self.kf_enable      = rospy.get_param("/obstacle_publisher/kf_enable", True)
        self.kf_q_s   = rospy.get_param("/obstacle_publisher/kf_q_s",   1e-4)
        self.kf_q_d   = rospy.get_param("/obstacle_publisher/kf_q_d",   1e-4)
        self.kf_q_vs  = rospy.get_param("/obstacle_publisher/kf_q_vs",  0.5)
        self.kf_q_vd  = rospy.get_param("/obstacle_publisher/kf_q_vd",  0.5)
        # Measurement noise in (s,d); default to rough xy sigma (curvature-dependent)
        self.kf_r_s   = rospy.get_param("/obstacle_publisher/kf_r_s",   (self.noise_sigma_xy) ** 2)
        self.kf_r_d   = rospy.get_param("/obstacle_publisher/kf_r_d",   (self.noise_sigma_xy) ** 2)

        # KF state (lazy init on first visible tick)
        self._kf_x = None            # np.array shape (4,)
        self._kf_P = None            # np.array shape (4,4)
        self._last_seen_t = None     # rospy.Time when obstacle last passed detect gate
        self._last_posterior = None  # last Obstacle msg content for coast freeze
        self._rng = np.random.default_rng()

        rospy.Subscriber("/car_state/odom_frenet", Odometry, self.odom_cb)
        self.car_odom = Odometry()
        self._ego_s = 0.0
        self._ego_d = 0.0

        # pending respawn s set by interactive-marker "Spawn" menu;
        # consumed at the top of each tick before s advances.
        self._pending_respawn_s = None

        # RViz Interactive Marker for spawn/delete buttons
        # Single interactive marker in "map" frame, MOVE_PLANE on xy + right-click menu:
        #   "Spawn dynamic obstacle here" → reads marker xyz, glob2frenet, sets pending
        #                                    respawn s and re-activates publishing
        #   "Delete dynamic obstacle"     → deactivates dyn-obs publishing on both
        #                                    /tracking/obstacles and /tracking/obstacles_truth
        self._dyn_obs_active = True
        self._marker_xyz = None  # (x, y, z) latest marker pose; None until first feedback
        self._marker_seeded = False  # marker is INSERTED into the server lazily
        self._marker_server = InteractiveMarkerServer("dynamic_obstacle_control")
        self._menu_handler = MenuHandler()
        self._menu_id_spawn  = self._menu_handler.insert(
            "Spawn dynamic obstacle here", callback=self._marker_menu_cb)
        self._menu_id_delete = self._menu_handler.insert(
            "Delete dynamic obstacle", callback=self._marker_menu_cb)
        # Don't insert the marker here — only after first tick gives us a real
        # obstacle xyz. Inserting at (0,0,0) and later setPose'ing was racing
        # with RViz right-click → menu and crashing the GUI.

        self.obstacle_pub = rospy.Publisher("/tracking/obstacles", ObstacleArray, queue_size=10)
        # always-on clean ground-truth channel for collision_detector
        # `/tracking/obstacles` is gated/noisy (planner-facing). Collision must use truth.
        self.obstacle_truth_pub = rospy.Publisher("/tracking/obstacles_truth", ObstacleArray, queue_size=10)
        self.obstacle_mrk_pub = rospy.Publisher("/dummy_obstacle_markers", MarkerArray, queue_size=10)
        self.opponent_traj_pub = rospy.Publisher("/opponent_waypoints", OpponentTrajectory, queue_size=10)

        rospy.wait_for_service("convert_glob2frenet_service")
        # BUGFIX: persistent=True reuses one TCP connection across calls.
        # Without it rospy opens/closes a socket per call -> with a non-loopback
        # ROS_MASTER_URI the kernel can't recycle the (local_ip, port) tuple
        # (TIME_WAIT ~60s) and ephemeral ports for that destination exhaust
        # after sustained high-rate calls, killing the node with
        # OSError 99 / ServiceException("unable to contact master").
        self.glob2frenet = rospy.ServiceProxy(
            "convert_glob2frenetarr_service", Glob2FrenetArr, persistent=True)
        self.frenet2glob = rospy.ServiceProxy(
            "convert_frenet2globarr_service", Frenet2GlobArr, persistent=True)
        self.mincurv_wpnts = None

        # absorbed gazebo static obstacle publisher (was separate launch)
        # Subscribes to gazebo static obstacle topics and appends them to the
        # single /tracking/obstacles stream so we don't fight a dual publisher.
        # If topics never publish (non-gazebo env), behavior reduces to dummy-only.
        self.enable_gz_static = rospy.get_param(
            "/obstacle_publisher/enable_gazebo_static_obs", True)
        self.gz_poses_topic = rospy.get_param(
            "/obstacle_publisher/gz_in_poses_topic", "/gazebo/static_obstacles/poses")
        self.gz_radii_topic = rospy.get_param(
            "/obstacle_publisher/gz_in_radii_topic", "/gazebo/static_obstacles/radii")
        self.gz_markers_topic = rospy.get_param(
            "/obstacle_publisher/gz_out_markers_topic", "/gazebo_static_obstacle_markers")
        self.gz_id_offset = rospy.get_param(
            "/obstacle_publisher/gz_id_offset", 100)
        # s-gate for static obstacles on /tracking/obstacles (planner-facing).
        # Truth channel ignores this gate (collision_detector sees everything).
        self.gz_static_s_back = rospy.get_param(
            "/obstacle_publisher/gz_static_s_back", -2.0)
        self.gz_static_s_fwd  = rospy.get_param(
            "/obstacle_publisher/gz_static_s_fwd",  15.0)

        self._gz_poses = None     # list of geometry_msgs/Pose
        self._gz_radii = None     # list of float
        self._gz_cached = []      # list of Obstacle, recomputed on message change
        self._gz_cache_key = None
        self.gz_marker_pub = None
        if self.enable_gz_static:
            rospy.Subscriber(self.gz_poses_topic, PoseArray, self._gz_poses_cb, queue_size=1)
            rospy.Subscriber(self.gz_radii_topic, Float64MultiArray, self._gz_radii_cb, queue_size=1)
            self.gz_marker_pub = rospy.Publisher(self.gz_markers_topic, MarkerArray, queue_size=10)

    def init_dynamic_obstacle(self) -> Obstacle:
        dynamic_obstacle = Obstacle()
        dynamic_obstacle.id = 1
        dynamic_obstacle.d_right = -0.1
        dynamic_obstacle.d_left = 0.1
        dynamic_obstacle.is_actually_a_gap = False
        return dynamic_obstacle

    ### CALLBACKS ###
    def wpnts_cb(self, data: WpntArray):
        # Last wpnt is a duplicate of the first (closes the loop), with s_m equal
        # to the actual lap length L. Use it for max_s before dropping; otherwise
        # max_s would underestimate L by one waypoint spacing and frenet service
        # output near s≈L would slip past every `% max_s` wrap downstream.
        max_s = data.wpnts[-1].s_m
        wpnts = data.wpnts[:-1]
        return wpnts, max_s

    def odom_cb(self, data: Odometry):
        self.car_odom = data
        # cache ego s/d for detect gate
        # Convention in this stack: odom_frenet.pose.pose.position.x = s, .y = d
        self._ego_s = data.pose.pose.position.x
        self._ego_d = data.pose.pose.position.y

    # Interactive marker setup + callbacks (spawn/delete buttons)
    def _make_visual_marker(self):
        """Cyan sphere — both the visual body AND the click target for menu/drag."""
        m = Marker()
        m.type = Marker.SPHERE
        m.scale.x = 0.6
        m.scale.y = 0.6
        m.scale.z = 0.6
        m.color.r = 0.0
        m.color.g = 0.9
        m.color.b = 1.0
        m.color.a = 0.85
        return m

    def _build_int_marker(self, x, y, z):
        """Build an InteractiveMarker rooted at (x, y, z), RViz basic_controls
        style: one MOVE_PLANE control that *contains* the visual sphere. Menu
        is attached at marker level so right-click anywhere opens it."""
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "map"
        int_marker.name = "dynamic_obstacle_ctrl"
        int_marker.description = "Dynamic Obstacle"
        # Auto-drawn drag-handle size (square indicator + arrows). 1.2 keeps it
        # noticeable but not so big it covers half the track.
        int_marker.scale = 1.2
        int_marker.pose.position.x = float(x)
        int_marker.pose.position.y = float(y)
        int_marker.pose.position.z = float(z)
        int_marker.pose.orientation.w = 1.0

        # Single combined control: MOVE_PLANE on xy AND houses the visual sphere.
        # Orientation rotates default control x-axis (= plane normal) onto +z so
        # the drag plane is xy. always_visible=True keeps the sphere on screen
        # even when not hovered.
        ctrl = InteractiveMarkerControl()
        ctrl.name = "move_xy_body"
        ctrl.interaction_mode = InteractiveMarkerControl.MOVE_PLANE
        ctrl.orientation.w = 0.7071068
        ctrl.orientation.x = 0.0
        ctrl.orientation.y = 0.7071068
        ctrl.orientation.z = 0.0
        ctrl.always_visible = True
        ctrl.markers.append(self._make_visual_marker())
        int_marker.controls.append(ctrl)
        return int_marker

    def _marker_pose_cb(self, feedback: InteractiveMarkerFeedback):
        """POSE_UPDATE while user drags the marker; cache xyz for the spawn button."""
        if feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            p = feedback.pose.position
            self._marker_xyz = (float(p.x), float(p.y), float(p.z))

    def _marker_menu_cb(self, feedback: InteractiveMarkerFeedback):
        """Menu select handler: dispatch on entry id."""
        if feedback.event_type != InteractiveMarkerFeedback.MENU_SELECT:
            return
        if feedback.menu_entry_id == self._menu_id_spawn:
            # Prefer the marker's current pose (dragged target). Fall back to feedback.
            if self._marker_xyz is None:
                p = feedback.pose.position
                self._marker_xyz = (float(p.x), float(p.y), float(p.z))
            x, y, z = self._marker_xyz
            try:
                resp = self.glob2frenet([x], [y], [z])
                s_new = float(resp.s[0])
            except Exception as e:
                rospy.logwarn(f"[3d_obstacle_publisher] menu spawn glob2frenet failed: {e}")
                return
            self._pending_respawn_s = s_new
            self._dyn_obs_active = True
            # KF reset so coast/posterior won't bridge the teleport
            self._kf_x = None
            self._kf_P = None
            self._last_posterior = None
            self._last_seen_t = None
            rospy.loginfo(f"[3d_obstacle_publisher] menu Spawn at s={s_new:.2f}m "
                          f"(xy=({x:.2f}, {y:.2f}))")
        elif feedback.menu_entry_id == self._menu_id_delete:
            self._dyn_obs_active = False
            self._kf_x = None
            self._kf_P = None
            self._last_posterior = None
            self._last_seen_t = None
            rospy.loginfo("[3d_obstacle_publisher] menu Delete: dyn obstacle deactivated")

    def _seed_marker_pose_once(self):
        """One-shot: build & insert the interactive marker at the obstacle's
        first real xyz. After this the marker is *never* moved by the node —
        only by user drag. Per-tick setPose+applyChanges races with RViz's
        right-click menu and crashes the GUI; this design avoids that entirely.
        Live obstacle position is visible via /dummy_obstacle_markers."""
        if self._marker_seeded:
            return
        x = float(self.dynamic_obstacle.x_m)
        y = float(self.dynamic_obstacle.y_m)
        z = float(self.dynamic_obstacle.z_m)
        int_marker = self._build_int_marker(x, y, z)
        self._marker_server.insert(int_marker, self._marker_pose_cb)
        self._menu_handler.apply(self._marker_server, int_marker.name)
        self._marker_server.applyChanges()
        self._marker_xyz = (x, y, z)
        self._marker_seeded = True

    # gazebo static obstacle callbacks + cache build
    def _gz_poses_cb(self, msg: PoseArray):
        self._gz_poses = list(msg.poses)

    def _gz_radii_cb(self, msg: Float64MultiArray):
        self._gz_radii = list(msg.data)

    def _build_gz_static_obstacles(self, max_s):
        """Convert cached gazebo poses/radii to a list of Obstacle (static).

        Rebuilds only when the input arrays change (pose count or content id).
        frenet service is called once per change to avoid per-tick cost.
        """
        if (self._gz_poses is None or self._gz_radii is None
                or len(self._gz_poses) == 0 or len(self._gz_radii) == 0):
            return []

        n = min(len(self._gz_poses), len(self._gz_radii))
        # cheap change-detect: (n, first/last xy, first radius)
        p0 = self._gz_poses[0].position
        pN = self._gz_poses[n - 1].position
        key = (n, round(p0.x, 4), round(p0.y, 4),
               round(pN.x, 4), round(pN.y, 4),
               round(float(self._gz_radii[0]), 4))
        if key == self._gz_cache_key and self._gz_cached:
            return self._gz_cached

        xs = [float(self._gz_poses[i].position.x) for i in range(n)]
        ys = [float(self._gz_poses[i].position.y) for i in range(n)]
        zs = [float(self._gz_poses[i].position.z) for i in range(n)]
        try:
            resp = self.glob2frenet(xs, ys, zs)
        except Exception as e:
            rospy.logwarn_throttle(2.0,
                f"[3d_obstacle_publisher] gz static frenet convert failed: {e}")
            return self._gz_cached  # fall back to previous

        obstacles = []
        for i in range(n):
            radius = float(self._gz_radii[i])
            s_c = float(resp.s[i])
            d_c = float(resp.d[i])
            obs = Obstacle()
            obs.id = self.gz_id_offset + i
            obs.x_m = xs[i]
            obs.y_m = ys[i]
            obs.z_m = zs[i]
            # Wrap s_center to [0, max_s) for parity with s_start/s_end.
            obs.s_center = s_c % max_s
            obs.d_center = d_c
            obs.s_start = (s_c - radius) % max_s
            obs.s_end = (s_c + radius) % max_s
            obs.d_right = d_c - radius
            obs.d_left = d_c + radius
            obs.size = radius * 2
            obs.vs = 0.0
            obs.vd = 0.0
            obs.is_static = True
            obs.is_visible = True
            obs.is_actually_a_gap = False
            obs.sector_id = -1
            obs.in_static_obs_sector = False
            obstacles.append(obs)

        self._gz_cached = obstacles
        self._gz_cache_key = key
        rospy.loginfo_throttle(5.0,
            f"[3d_obstacle_publisher] gz static obstacles active: n={n}")
        return obstacles

    def _gz_in_s_gate(self, s_obs, s_ego, L):
        """Planner-channel gate for static obstacles (s-only, ego-relative, s-wrap)."""
        ds = self._wrap_to_half(s_obs - s_ego, L)
        return self.gz_static_s_back <= ds <= self.gz_static_s_fwd

    def _publish_gz_static_markers(self, obstacles):
        if self.gz_marker_pub is None:
            return
        marker_msg = MarkerArray()
        now = rospy.Time.now()
        for obs in obstacles:
            r = obs.size / 2.0 if obs.size > 0 else 0.2
            mrk = Marker()
            mrk.header.frame_id = "map"
            mrk.header.stamp = now
            mrk.ns = "gazebo_static_obs"
            mrk.id = obs.id
            mrk.type = Marker.CYLINDER
            mrk.action = Marker.ADD
            mrk.pose.position.x = obs.x_m
            mrk.pose.position.y = obs.y_m
            mrk.pose.position.z = obs.z_m
            mrk.pose.orientation.w = 1.0
            mrk.scale.x = r * 2
            mrk.scale.y = r * 2
            mrk.scale.z = 0.3
            mrk.color.r = 1.0
            mrk.color.g = 0.3
            mrk.color.b = 0.0
            mrk.color.a = 0.8
            mrk.lifetime = rospy.Duration(0.5)
            marker_msg.markers.append(mrk)
        self.gz_marker_pub.publish(marker_msg)

    ### HELPERS ###
    def publish_obstacle_cartesian(self, obstacles):
        """Visualizes obstacles in cartesian frame (3D-aware)."""
        obs_markers = MarkerArray()
        for obs in obstacles:
            obs_marker = Marker(header=rospy.Header(frame_id="map"), id=obs.id, type=Marker.SPHERE)
            obs_marker.scale.x = 0.5
            obs_marker.scale.y = 0.5
            obs_marker.scale.z = 0.5
            obs_marker.color.a = 0.5
            obs_marker.color.b = 0.5
            obs_marker.color.r = 0.5

            obs_marker.pose.position.x = obs.x_m
            obs_marker.pose.position.y = obs.y_m
            # use z_m for 3D marker visualization
            obs_marker.pose.position.z = obs.z_m
            obs_marker.pose.orientation.w = 1
            obs_markers.markers.append(obs_marker)

        self.obstacle_mrk_pub.publish(obs_markers)

    def shutdown(self):
        rospy.loginfo("BEEP BOOP DUMMY OD SHUTDOWN")
        self.obstacle_pub.publish(ObstacleArray())
        self.obstacle_truth_pub.publish(ObstacleArray())

    # helpers for detect gate / KF
    def _wrap_to_half(self, ds, L):
        """Wrap ds into [-L/2, +L/2] for closed-loop track."""
        return (ds + L / 2.0) % L - L / 2.0

    def _in_detect_gate(self, s_obs, d_obs, s_ego, d_ego, L):
        ds = self._wrap_to_half(s_obs - s_ego, L)
        if not (self.detect_s_back <= ds <= self.detect_s_fwd):
            return False
        if abs(d_obs - d_ego) > self.detect_d_max:
            return False
        return True

    def _kf_reset(self, s0, d0, vs0, vd0):
        self._kf_x = np.array([s0, d0, vs0, vd0], dtype=float)
        # init covariance: trust position somewhat, velocity less
        self._kf_P = np.diag([0.01, 0.01, 1.0, 1.0])

    def _kf_predict(self, dt):
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1,  0],
                      [0, 0, 0,  1]], dtype=float)
        Q = np.diag([self.kf_q_s, self.kf_q_d, self.kf_q_vs, self.kf_q_vd])
        self._kf_x = F @ self._kf_x
        self._kf_P = F @ self._kf_P @ F.T + Q

    def _kf_update(self, s_meas, d_meas, L):
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]], dtype=float)
        R = np.diag([self.kf_r_s, self.kf_r_d])
        # innovation in s uses wrap to avoid jumps across s=0/L boundary
        s_pred = self._kf_x[0]
        y_s = self._wrap_to_half(s_meas - s_pred, L)
        y_d = d_meas - self._kf_x[1]
        y = np.array([y_s, y_d])
        S = H @ self._kf_P @ H.T + R
        K = self._kf_P @ H.T @ np.linalg.inv(S)
        self._kf_x = self._kf_x + K @ y
        # keep s within [0, L)
        self._kf_x[0] = self._kf_x[0] % L
        I4 = np.eye(4)
        self._kf_P = (I4 - K @ H) @ self._kf_P

    ### MAIN ###
    def ros_loop(self):
        rospy.loginfo("3D Dummy Obstacle Publisher waiting for waypoints...")
        rospy.wait_for_service("convert_frenet2globarr_service")
        rospy.wait_for_service("convert_glob2frenetarr_service")

        if self.waypoints_type == "updated":
            global_wpnts_msg = rospy.wait_for_message("/global_waypoints_updated", WpntArray)
        else:
            global_wpnts_msg = rospy.wait_for_message("/global_waypoints", WpntArray)
        global_wpnts, max_s = self.wpnts_cb(data=global_wpnts_msg)
        s_array = np.array([wpnt.s_m for wpnt in global_wpnts])

        if self.constant:
            for i in range(len(global_wpnts)):
                global_wpnts[i].vx_mps = 1 * self.speed_scaler
        else:
            for i in range(len(global_wpnts)):
                global_wpnts[i].vx_mps = global_wpnts[i].vx_mps*self.speed_scaler

        opponent_wpnts_msg = rospy.wait_for_message(self.waypoints_topic, WpntArray)
        opponent_wpnts_list, _ = self.wpnts_cb(data=opponent_wpnts_msg)

        # pass z_m for 3D Glob2FrenetArr
        opponent_xy = self.glob2frenet(
            [wpnt.x_m for wpnt in opponent_wpnts_list],
            [wpnt.y_m for wpnt in opponent_wpnts_list],
            [wpnt.z_m for wpnt in opponent_wpnts_list],
        )
        opponent_s = opponent_xy.s
        opponent_d = opponent_xy.d
        sorted_indices = sorted(range(len(opponent_s)), key=lambda i: opponent_s[i])
        opponent_s_sorted = [opponent_s[i] for i in sorted_indices]
        opponent_d_sorted = [opponent_d[i] for i in sorted_indices]
        resampeld_opponent_d = np.interp(s_array, opponent_s_sorted, opponent_d_sorted)
        resampeld_opponent_vs = [wpnt.vx_mps for wpnt in global_wpnts]
        resampled_opponent_xy = self.frenet2glob(s_array, resampeld_opponent_d)

        self.opponent_wpnts = OpponentTrajectory()
        for i in range(len(s_array)):
            wpnt = OppWpnt()
            wpnt.x_m = resampled_opponent_xy.x[i]
            wpnt.y_m = resampled_opponent_xy.y[i]
            # OppWpnt has no z_m field — z is consumed downstream via Obstacle.z_m only
            wpnt.proj_vs_mps = resampeld_opponent_vs[i]
            wpnt.s_m = s_array[i]
            wpnt.d_m = resampeld_opponent_d[i]
            self.opponent_wpnts.oppwpnts.append(wpnt)

        rospy.sleep(0.1)

        self.dynamic_obstacle.s_center = self.starting_s

        opponent_s_array = np.array([wpnt.s_m for wpnt in self.opponent_wpnts.oppwpnts])
        rospy.loginfo("3D Dummy Obstacle Publisher ready.")

        counter = 0
        while not rospy.is_shutdown():
            obstacle_msg = ObstacleArray()
            obstacle_msg.header.stamp = rospy.Time.now()
            obstacle_msg.header.frame_id = "frenet"

            # tick-level gazebo static obstacle list (cached by content key).
            # static_obs_list   → raw (truth channel, always published)
            # static_obs_gated  → s-gate filtered (planner channel /tracking/obstacles)
            static_obs_list = (
                self._build_gz_static_obstacles(max_s)
                if self.enable_gz_static else []
            )
            if static_obs_list:
                static_obs_gated = [
                    o for o in static_obs_list
                    if self._gz_in_s_gate(o.s_center, self._ego_s, max_s)
                ]
            else:
                static_obs_gated = []

            # apply pending nav-goal respawn before advancing s
            if self._pending_respawn_s is not None:
                self.dynamic_obstacle.s_center = self._pending_respawn_s % max_s
                self._osc_phase = self.osc_phase0  # reset the temporal-mode phase
                self._pending_respawn_s = None

            s = self.dynamic_obstacle.s_center
            approx_idx = np.abs(opponent_s_array - s).argmin()

            self.dyn_obstacle_speed = self.opponent_wpnts.oppwpnts[approx_idx].proj_vs_mps
            self.dynamic_obstacle.s_center = (self.dynamic_obstacle.s_center + self.dyn_obstacle_speed * self.looptime) % max_s
            self.dynamic_obstacle.s_start = (self.dynamic_obstacle.s_center - self.obj_len/2) % max_s
            self.dynamic_obstacle.s_end = (self.dynamic_obstacle.s_center + self.obj_len/2) % max_s
            self.dynamic_obstacle.d_center = self.opponent_wpnts.oppwpnts[approx_idx].d_m

            # add oscillation on top of baseline d
            if self.d_oscillate:
                if self.osc_mode == "spatial":
                    phase = 2.0 * np.pi * self.osc_n_waves_per_lap * (s / max_s) + self.osc_phase0
                else:  # temporal
                    self._osc_phase += 2.0 * np.pi * self.osc_frequency * self.looptime
                    phase = self._osc_phase
                self.dynamic_obstacle.d_center += self.osc_amplitude * np.sin(phase)

            size = 0.4
            self.dynamic_obstacle.size = size
            self.dynamic_obstacle.d_right = self.dynamic_obstacle.d_center - size/2
            self.dynamic_obstacle.d_left = self.dynamic_obstacle.d_center + size/2

            self.dynamic_obstacle.vs = self.dyn_obstacle_speed
            self.dynamic_obstacle.vd = 0.0
            self.dynamic_obstacle.is_static = False
            self.dynamic_obstacle.is_visible = True
            resp = self.frenet2glob([self.dynamic_obstacle.s_center], [self.dynamic_obstacle.d_center])
            self.dynamic_obstacle.x_m = resp.x[0]
            self.dynamic_obstacle.y_m = resp.y[0]
            # fill z_m from 3D Frenet2GlobArr response
            self.dynamic_obstacle.z_m = resp.z[0] if hasattr(resp, 'z') and len(resp.z) > 0 else 0.0

            # build TRUTH msg (always-on, clean) for collision_detector
            truth_msg = ObstacleArray()
            truth_msg.header.stamp = obstacle_msg.header.stamp
            truth_msg.header.frame_id = obstacle_msg.header.frame_id
            # interactive-marker delete button gates dyn obstacle out of truth too
            if self._dyn_obs_active:
                truth_obs = copy.deepcopy(self.dynamic_obstacle)
                truth_msg.obstacles.append(truth_obs)
            else:
                truth_obs = None
            # merge gz static obstacles into truth (always visible, no noise)
            if static_obs_list:
                truth_msg.obstacles.extend(copy.deepcopy(static_obs_list))
                self._publish_gz_static_markers(static_obs_list)
            self.obstacle_truth_pub.publish(truth_msg)
            # marker uses truth (dummy only) so RViz always shows ground-truth position
            if truth_obs is not None:
                self.publish_obstacle_cartesian([truth_obs])
            else:
                self.publish_obstacle_cartesian([])  # clear stale dummy marker

            # seed interactive-marker handle at obstacle's initial xyz (one-shot only).
            # Continuous sync was racing with RViz's right-click menu handling and
            # freezing the GUI. The live obstacle position is already visible via
            # /dummy_obstacle_markers; the interactive marker is just a drag handle.
            self._seed_marker_pose_once()

            # build NOISY+gated+KF msg for /tracking/obstacles
            s_true = self.dynamic_obstacle.s_center
            d_true = self.dynamic_obstacle.d_center
            x_true = self.dynamic_obstacle.x_m
            y_true = self.dynamic_obstacle.y_m
            z_true = self.dynamic_obstacle.z_m

            # delete button forces visible=False, dropping dyn obs from
            # /tracking/obstacles (static-only stream still flows below)
            if not self._dyn_obs_active:
                visible = False
            else:
                visible = self._in_detect_gate(s_true, d_true, self._ego_s, self._ego_d, max_s)

            if visible:
                # Measurement: cartesian xyz noise → glob2frenet to keep (s,d) consistent
                if self.noise_enable:
                    xn = x_true + self._rng.normal(0.0, self.noise_sigma_xy)
                    yn = y_true + self._rng.normal(0.0, self.noise_sigma_xy)
                    zn = z_true + self._rng.normal(0.0, self.noise_sigma_z)
                    meas = self.glob2frenet([xn], [yn], [zn])
                    s_meas = float(meas.s[0])
                    d_meas = float(meas.d[0])
                else:
                    xn, yn, zn = x_true, y_true, z_true
                    s_meas, d_meas = s_true, d_true

                if self.kf_enable:
                    if self._kf_x is None:
                        self._kf_reset(s_meas, d_meas, self.dyn_obstacle_speed, 0.0)
                    else:
                        self._kf_predict(self.looptime)
                        self._kf_update(s_meas, d_meas, max_s)
                    s_pub, d_pub = float(self._kf_x[0]) % max_s, float(self._kf_x[1])
                    vs_pub, vd_pub = float(self._kf_x[2]), float(self._kf_x[3])
                    # re-derive xyz from posterior (s,d) so msg fields stay consistent
                    rp = self.frenet2glob([s_pub], [d_pub])
                    x_pub, y_pub = float(rp.x[0]), float(rp.y[0])
                    z_pub = float(rp.z[0]) if hasattr(rp, 'z') and len(rp.z) > 0 else 0.0
                else:
                    s_pub, d_pub = s_meas, d_meas
                    vs_pub, vd_pub = self.dyn_obstacle_speed, 0.0
                    x_pub, y_pub, z_pub = xn, yn, zn

                pub_obs = copy.deepcopy(self.dynamic_obstacle)
                pub_obs.s_center = s_pub
                pub_obs.d_center = d_pub
                pub_obs.s_start = (s_pub - self.obj_len / 2) % max_s
                pub_obs.s_end   = (s_pub + self.obj_len / 2) % max_s
                pub_obs.d_right = d_pub - pub_obs.size / 2
                pub_obs.d_left  = d_pub + pub_obs.size / 2
                pub_obs.x_m, pub_obs.y_m, pub_obs.z_m = x_pub, y_pub, z_pub
                pub_obs.vs = vs_pub
                pub_obs.vd = vd_pub
                pub_obs.is_visible = True

                obstacle_msg.obstacles.append(pub_obs)
                # merge s-gated gz static obstacles into /tracking/obstacles
                if static_obs_gated:
                    obstacle_msg.obstacles.extend(copy.deepcopy(static_obs_gated))
                self.obstacle_pub.publish(obstacle_msg)

                self._last_posterior = pub_obs
                self._last_seen_t = rospy.Time.now()
            else:
                now = rospy.Time.now()
                if (self._last_seen_t is not None
                        and self._last_posterior is not None
                        and (now - self._last_seen_t).to_sec() < self.coast_duration):
                    # coast: freeze last posterior, flag not-visible
                    coast_obs = copy.deepcopy(self._last_posterior)
                    coast_obs.is_visible = False
                    obstacle_msg.obstacles.append(coast_obs)
                    # s-gated static obstacles stay in stream during coast
                    if static_obs_gated:
                        obstacle_msg.obstacles.extend(copy.deepcopy(static_obs_gated))
                    self.obstacle_pub.publish(obstacle_msg)
                else:
                    # drop: publish empty, reset KF so next visibility re-inits
                    self._kf_x = None
                    self._kf_P = None
                    self._last_posterior = None
                    self._last_seen_t = None
                    # s-gated static still published even when dummy dropped
                    if static_obs_gated:
                        obstacle_msg.obstacles.extend(copy.deepcopy(static_obs_gated))
                    self.obstacle_pub.publish(obstacle_msg)  # empty (or static-only)

            counter = counter + 1

            if counter > 25:
                opponent_traj_msg = OpponentTrajectory(header=rospy.Header(frame_id="map", stamp=rospy.Time.now()), lap_count=2)
                opponent_traj_msg.oppwpnts = self.opponent_wpnts.oppwpnts
                self.opponent_traj_pub.publish(opponent_traj_msg)
                counter = 0

            self.rate.sleep()


if __name__ == "__main__":
    rospy.init_node("obstacle_publisher", anonymous=False, log_level=rospy.INFO)
    obstacle_publisher = ObstaclePublisher3D()
    rospy.on_shutdown(obstacle_publisher.shutdown)
    obstacle_publisher.ros_loop()
