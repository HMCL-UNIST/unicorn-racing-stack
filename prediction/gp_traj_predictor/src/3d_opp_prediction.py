#!/usr/bin/env python3
"""
3D Opponent Predictor — minimal port of the 2D opponent predictor
  - Smart static mode completely removed (not used in 3D)
  - glob2frenet/frenet2glob services kept (C++ side is 3D-aware)
  - Structure & logic preserved as-is from original
  - Debug logs cleaned
  - TODO: single opponent only (multi-vehicle support planned)
"""
import rospy
import numpy as np
import copy
import time

from std_msgs.msg import Bool, Float32, String
from nav_msgs.msg import Odometry
from f110_msgs.msg import ObstacleArray, WpntArray, Wpnt, Obstacle, OpponentTrajectory, OppWpnt, Prediction, PredictionArray
from visualization_msgs.msg import Marker, MarkerArray
from frenet_conversion.srv import Glob2FrenetArr, Frenet2GlobArr
from dynamic_reconfigure.msg import Config

from std_srvs.srv import SetBool, SetBoolResponse
from tf.transformations import euler_from_quaternion

from track_3d_validator import circular_s_dist, signed_s_dist

OPP_TRAJ_USE_THRESHOLD = 0.35


class OppTrajPredictor:
    def __init__(self):
        rospy.init_node('opponent_propagation_predictor', anonymous=True)

        # ROS Parameters
        self.opponent_traj_topic = '/opponent_trajectory'
        # BUGFIX: persistent=True reuses one TCP connection across calls.
        # Without it rospy opens/closes a socket per call -> with a non-loopback
        # ROS_MASTER_URI the kernel can't recycle the (local_ip, port) tuple
        # (TIME_WAIT ~60s) and ephemeral ports exhaust after sustained
        # high-rate calls, killing the node with OSError 99 /
        # ServiceException("unable to contact master").
        self.glob2frenet = rospy.ServiceProxy(
            "convert_glob2frenetarr_service", Glob2FrenetArr, persistent=True)
        self.frenet2glob = rospy.ServiceProxy(
            "convert_frenet2globarr_service", Frenet2GlobArr, persistent=True)
        self.loop_rate = 10  # Hz

        self.USE_V_D = rospy.get_param('~use_vd_prediction', True)
        rospy.loginfo(f"[Opp. Pred.] USE_V_D prediction: {self.USE_V_D}")

        # Publishers
        self.marker_pub_beginn = rospy.Publisher("/opponent_predict/beginn", Marker, queue_size=10)
        self.marker_pub_end = rospy.Publisher("/opponent_predict/end", Marker, queue_size=10)
        self.prediction_obs_pub = rospy.Publisher("/opponent_prediction/obstacles", ObstacleArray, queue_size=10)
        self.prediction_obs_pred_pub = rospy.Publisher("/opponent_prediction/obstacles_pred", PredictionArray, queue_size=10)
        self.force_trailing_pub = rospy.Publisher("/opponent_prediction/force_trailing", Bool, queue_size=10)

        self.opp_traj_gp_pub = rospy.Publisher('/opponent_trajectory', OpponentTrajectory, queue_size=10)
        self.opp_traj_marker_pub = rospy.Publisher('/opponent_traj_markerarray', MarkerArray, queue_size=10)
        self.opp_marker_pub = rospy.Publisher('/opponent_prediction_markerarray', MarkerArray, queue_size=10)

        # Callback data
        self.opponent_pos = ObstacleArray()
        self.car_odom = Odometry()
        self.wpnts_opponent = list()
        self.wpnts_updated = list()
        self.state = String()

        self.speed_offset = 0

        # Subscribers
        rospy.Subscriber("/tracking/obstacles", ObstacleArray, self.opponent_state_cb)
        rospy.Subscriber("/car_state/odom_frenet", Odometry, self.odom_cb)
        rospy.Subscriber(self.opponent_traj_topic, OpponentTrajectory, self.opponent_trajectory_cb)
        rospy.Subscriber('/global_waypoints_scaled', WpntArray, self.wpnts_updated_cb)
        rospy.Subscriber("/centerline_waypoints", WpntArray, self.center_wpnts_cb)
        rospy.Subscriber("/state_machine", String, self.state_cb)
        rospy.Subscriber("/dynamic_prediction_tuner_node/parameter_updates", Config, self.dyn_param_cb)

        # Service
        rospy.Service('/init_opp_trajectory', SetBool, self.init_opp_bool)

        # Simulation parameters
        self.time_steps = 200
        self.dt = 0.02
        self.save_distance_front = 0.6
        self.save_distance_back = 0.4
        self.max_v = 10
        self.min_v = 0
        self.max_a = 5.5
        self.min_a = 5
        self.max_expire_counter = 10

        self.obstacle_half_width = rospy.get_param('~obstacle_half_width', 0.3)
        rospy.loginfo(f"[Opp. Pred.] Obstacle half width: {self.obstacle_half_width}m")

        self.expire_counter = 0

        # Visualization
        self.marker_beginn = self.marker_init(a=0.5, r=0.63, g=0.13, b=0.94, id=0)
        self.marker_end = self.marker_init(a=0.5, r=0.63, g=0.13, b=0.94, id=1)

        # Opponent
        self.opponent_lap_count = None

        # Stanley params
        self.k = 0.5
        self.Kp = 1.0
        self.L = 0.33
        self.max_steer = np.radians(30.0)

        self.init_fixed_wpnts = False
        self.fixed_wpnts = []

    # Service function
    def init_opp_bool(self, req):
        if req.data:
            rospy.loginfo("Received request: ON")
            success = True
            message = "Feature turned ON"
            self.global_to_opptraj(self.wpnts_updated)
        else:
            rospy.loginfo("Received request: OFF")
            success = True
            message = "Feature turned OFF"
        return SetBoolResponse(success, message)

    def global_to_opptraj(self, wptlist: list):
        # global Wpnt carries z_m, so pass it through to keep OppWpnt.z_m consistent.
        wpnts_opponent = [OppWpnt(s_m=wp.s_m, d_m=wp.d_m, x_m=wp.x_m, y_m=wp.y_m, z_m=wp.z_m, proj_vs_mps=wp.vx_mps) for wp in wptlist]
        opp_traj_gp_msg = self.make_opponent_trajectory_msg(wpnts_opponent)
        opp_traj_marker_array = self.visualize_opponent_wpnts(wpnts_opponent)
        self.opp_traj_gp_pub.publish(opp_traj_gp_msg)
        self.opp_traj_marker_pub.publish(opp_traj_marker_array)

    def make_opponent_trajectory_msg(self, oppwpnts_list: list):
        opponent_trajectory_msg = OpponentTrajectory()
        opponent_trajectory_msg.header.stamp = rospy.Time.now()
        opponent_trajectory_msg.oppwpnts = oppwpnts_list
        return opponent_trajectory_msg

    def visualize_opponent_wpnts(self, oppwpnts_list: list):
        opp_traj_marker_array = MarkerArray()
        # Match the normalization used by gaussian_process_opp_traj /
        # predictor_opponent_trajectory so all three viz nodes produce
        # consistent cylinder heights on /opponent_traj_markerarray.
        max_vel = max((wp.vx_mps for wp in self.wpnts_updated), default=10.0)
        if max_vel <= 1e-3:
            max_vel = 10.0
        # OppWpnt carries z_m already filled on the correct layer by the publisher,
        # so the visualization uses wp.x_m/y_m/z_m directly (no frenet2glob service call).
        for i in range(len(oppwpnts_list)):
            # Orange marks unobserved regions (proj_vs_mps is the
            # raceline-vx * 0.9 fallback, not a GP posterior).
            is_sentinel = not oppwpnts_list[i].is_observed
            marker_height = oppwpnts_list[i].proj_vs_mps / max_vel
            marker = Marker(header=rospy.Header(frame_id="map"), id=i, type=Marker.CYLINDER)
            marker.pose.position.x = oppwpnts_list[i].x_m
            marker.pose.position.y = oppwpnts_list[i].y_m
            marker.pose.position.z = oppwpnts_list[i].z_m + marker_height / 2
            marker.pose.orientation.w = 1.0
            marker.scale.x = min(max(5 * oppwpnts_list[i].d_var, 0.07), 0.7)
            marker.scale.y = min(max(5 * oppwpnts_list[i].d_var, 0.07), 0.7)
            marker.scale.z = marker_height
            if is_sentinel:
                marker.color.a = 1.0
                marker.color.r = 1.0
                marker.color.g = 0.5
                marker.color.b = 0.0
            elif oppwpnts_list[i].vs_var == 69:
                marker.color.a = 0.8
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            else:
                marker.color.a = 1.0
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            opp_traj_marker_array.markers.append(marker)
        return opp_traj_marker_array

    ### CALLBACKS ###
    # TODO: Only sees the first dynamic obstacle as opponent (multi-vehicle not supported yet)
    def opponent_state_cb(self, data: ObstacleArray):
        self.opponent_pos.header = data.header
        is_dynamic = False
        if len(data.obstacles) > 0:
            for obs in data.obstacles:
                if obs.is_static == False:
                    self.opponent_pos.obstacles = [obs]
                    is_dynamic = True
                    break
        if is_dynamic == False:
            self.opponent_pos.obstacles = []

    def odom_cb(self, data: Odometry):
        self.car_odom = data

    def opponent_trajectory_cb(self, data: OpponentTrajectory):
        self.wpnts_opponent = data.oppwpnts
        self.max_s_opponent = self.wpnts_opponent[-1].s_m
        self.opponent_lap_count = data.lap_count

    def wpnts_updated_cb(self, data: WpntArray):
        self.wpnts_updated = data.wpnts[:-1]
        self.max_s_updated = self.wpnts_updated[-1].s_m

    def center_wpnts_cb(self, data: WpntArray):
        self.center_wpnts_msg = data
        self.center_wpnts_max_s = data.wpnts[-1].s_m
        self.center_wpnts_max_idx = data.wpnts[-1].id

        if not self.init_fixed_wpnts:
            self.fixed_wpnts = [[wp.x_m, wp.y_m] for wp in data.wpnts]
            self.cx, self.cy = zip(*self.fixed_wpnts)
            self.cyaw = [wp.psi_rad for wp in data.wpnts]
            self.init_fixed_wpnts = True

    def state_cb(self, data: String):
        self.state = data.data

    # centralize force_trailing publish with START-phase guard.
    # Branches 1/2 (VD blended, linear-to-center) emit True because the
    # underlying prediction is a stop-gap when GP raceline is not yet
    # trustworthy. But during START the controller is still settling and
    # forcing trailing kills the launch dash. State machine has its own
    # _check_free_frenet that catches a real path violation, so it is safe
    # to suppress the force flag while state == "START".
    def _publish_force_trailing(self, value: bool):
        current_state = self.state if isinstance(self.state, str) else ""
        if current_state == "START":
            self.force_trailing_pub.publish(False)
        else:
            self.force_trailing_pub.publish(value)

    def dyn_param_cb(self, params: Config):
        self.time_steps = rospy.get_param("dynamic_prediction_tuner_node/n_time_steps", 200)
        self.dt = rospy.get_param("dynamic_prediction_tuner_node/dt", 0.02)
        self.save_distance_front = rospy.get_param("dynamic_prediction_tuner_node/save_distance_front", 0.6)
        self.max_expire_counter = rospy.get_param("dynamic_prediction_tuner_node/max_expire_counter", 10)
        self.speed_offset = rospy.get_param("dynamic_prediction_tuner_node/speed_offset", 0)
        print(
            f"[Opp. Pred.] Dynamic reconf triggered new params:\n"
            f" N time stepts: {self.time_steps}, \n"
            f" dt: {self.dt} [s], \n"
            f" save_distance_front: {self.save_distance_front} [m], \n"
            f" max_expire_counter: {self.max_expire_counter}"
        )

    ### HELPER FUNCTIONS ###
    def marker_init(self, a=1, r=1, g=0, b=0, id=0):
        marker = Marker(header=rospy.Header(stamp=rospy.Time.now(), frame_id="map"), id=id, type=Marker.SPHERE)
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.4
        marker.scale.y = 0.4
        marker.scale.z = 0.4
        marker.color.a = a
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        return marker

    def delete_all(self) -> None:
        empty_marker = Marker(header=rospy.Header(stamp=rospy.Time.now(), frame_id="map"), id=0)
        empty_marker.action = Marker.DELETE
        self.marker_pub_beginn.publish(empty_marker)
        empty_marker.id = 1
        self.marker_pub_end.publish(empty_marker)

        empty_obs_arr = ObstacleArray(header=rospy.Header(stamp=rospy.Time.now(), frame_id="map"))
        self.prediction_obs_pub.publish(empty_obs_arr)

    ### MAIN LOOP ###
    def loop(self):
        rate = rospy.Rate(self.loop_rate)

        rospy.loginfo("[Opp. Pred.] Opponent Predictor wating...")
        rospy.wait_for_message("/global_waypoints", WpntArray)
        rospy.wait_for_message("/global_waypoints_scaled", WpntArray)
        rospy.loginfo("[Opp. Pred.] Scaled waypoints recived!")
        rospy.wait_for_message(self.opponent_traj_topic, OpponentTrajectory)
        rospy.loginfo("[Opp. Pred.] Opponent waypoints recived!")
        rospy.wait_for_message("/tracking/obstacles", ObstacleArray)
        rospy.loginfo("[Opp. Pred.] Obstacles reveived!")
        rospy.loginfo("[Opp. Pred.] Opponent Predictor ready!")

        while not rospy.is_shutdown():

            opponent_pos_copy = copy.deepcopy(self.opponent_pos)

            prediction_obs_pred_arr = PredictionArray()

            if len(opponent_pos_copy.obstacles) != 0:
                # All calculations in GB Frenet
                current_ego_s = self.car_odom.pose.pose.position.x
                current_opponent_s = opponent_pos_copy.obstacles[0].s_center

                # Handle wrap around
                if current_ego_s > self.max_s_updated * (2/3) and current_opponent_s < self.max_s_updated * (1/3):
                    current_opponent_s += self.max_s_updated

                current_opponent_d = opponent_pos_copy.obstacles[0].d_center

                # Centerline (GB) convergence target — wrap-safe argmin
                s_points_center_array = np.array([wpnt.s_m for wpnt in self.center_wpnts_msg.wpnts])
                approx_opponent_center_indx = int(np.argmin(
                    circular_s_dist(s_points_center_array, current_opponent_s, self.center_wpnts_max_s)))
                opponent_approx_center_d = self.center_wpnts_msg.wpnts[approx_opponent_center_indx].d_m

                current_opponent_v = opponent_pos_copy.obstacles[0].vs

                has_opponent_raceline = len(self.wpnts_opponent) > 1 and self.opponent_lap_count is not None

                if has_opponent_raceline:
                    approx_s_points_global_array = np.array([wpnt.s_m for wpnt in self.wpnts_opponent])
                    opponent_approx_indx = int(np.argmin(
                        circular_s_dist(approx_s_points_global_array, current_opponent_s, self.max_s_opponent)))
                    opponent_approx_raceline_d = self.wpnts_opponent[opponent_approx_indx].d_m
                else:
                    approx_s_points_global_array = np.array([])
                    opponent_approx_raceline_d = current_opponent_d

                start = time.process_time()

                # Common: beginn/end detection (used by all branches for
                # consistent /opponent_predict/beginn + /end markers)
                opp_size = opponent_pos_copy.obstacles[0].size
                fwd_dist = signed_s_dist(current_ego_s, current_opponent_s, self.max_s_updated)
                do_beginn = abs(fwd_dist) < self.save_distance_front
                initial_opponent_s = current_opponent_s  # save before any branch modifies it

                # Choose prediction method
                use_vd_method = self.USE_V_D and self.opponent_lap_count >= 1 and abs(current_opponent_d - opponent_approx_raceline_d) > OPP_TRAJ_USE_THRESHOLD
                use_original_method = (not self.USE_V_D or self.opponent_lap_count < 1) and (abs(current_opponent_d - opponent_approx_raceline_d) > OPP_TRAJ_USE_THRESHOLD or self.opponent_lap_count < 1)

                # ===== VD blended method =====
                if use_vd_method:
                    self._publish_force_trailing(True)  # START-guarded publish

                    obstacle_list = []
                    prediction_list = []
                    opp_marker_array = MarkerArray()

                    current_opponent_vd = opponent_pos_copy.obstacles[0].vd

                    vd_decay = 0.6
                    vd_blend_weight = 0.3

                    has_opponent_trajectory = len(self.wpnts_opponent) > 1 and hasattr(self, 'max_s_opponent')

                    for i in range(self.time_steps):
                        future_s = current_opponent_s + i * current_opponent_v * self.dt

                        raceline_vd = 0.0
                        if has_opponent_trajectory:
                            future_opponent_idx = int(np.argmin(
                                circular_s_dist(approx_s_points_global_array,
                                                future_s % self.max_s_opponent,
                                                self.max_s_opponent)))
                            next_opponent_idx = (future_opponent_idx + 1) % len(self.wpnts_opponent)

                            raceline_d_current = self.wpnts_opponent[future_opponent_idx].d_m
                            raceline_d_next = self.wpnts_opponent[next_opponent_idx].d_m
                            raceline_vs = self.wpnts_opponent[future_opponent_idx].proj_vs_mps

                            if raceline_vs > 0.1:
                                s_step = self.wpnts_opponent[next_opponent_idx].s_m - self.wpnts_opponent[future_opponent_idx].s_m
                                if s_step < 0:  # wrap-around
                                    s_step += self.max_s_opponent
                                raceline_vd = (raceline_d_next - raceline_d_current) / (s_step / raceline_vs) if s_step > 0 else 0.0

                        decayed_current_vd = current_opponent_vd * (vd_decay ** i)
                        blended_vd = vd_blend_weight * decayed_current_vd + (1 - vd_blend_weight) * raceline_vd

                        lateral_displacement = 0.0
                        for j in range(i + 1):
                            step_vd = vd_blend_weight * current_opponent_vd * (vd_decay ** j) + (1 - vd_blend_weight) * raceline_vd
                            lateral_displacement += step_vd * self.dt

                        predicted_d = current_opponent_d + lateral_displacement

                        obs = Obstacle()
                        obs.id = i
                        obs.s_center = future_s % self.max_s_updated
                        obs.s_start = (future_s - opp_size / 2) % self.max_s_updated
                        obs.s_end = (future_s + opp_size / 2) % self.max_s_updated
                        obs.d_center = predicted_d
                        obs.d_left = obs.d_center + self.obstacle_half_width
                        obs.d_right = obs.d_center - self.obstacle_half_width
                        obs.size = opp_size
                        obs.vs = current_opponent_v
                        obs.vd = blended_vd
                        obs.is_actually_a_gap = False
                        obs.is_static = False
                        obstacle_list.append(obs)

                        pds = Prediction()
                        pds.id = i
                        pds.pred_s = obs.s_center
                        pds.pred_d = obs.d_center
                        prediction_list.append(pds)

                        marker = Marker()
                        marker.header.stamp = rospy.Time.now()
                        marker.header.frame_id = "map"
                        marker.id = i
                        marker.type = Marker.CYLINDER
                        marker.action = Marker.ADD
                        marker.pose.orientation.w = 1.0

                        pos = self.frenet2glob([obs.s_center % self.max_s_updated], [obs.d_center])
                        marker.pose.position.x = pos.x[0]
                        marker.pose.position.y = pos.y[0]
                        # Frenet2GlobArr returns track-surface z; lift by half the
                        # cylinder height (scale.z = 0.15 below) so the cylinder base
                        # sits exactly on the surface instead of half-buried.
                        marker.pose.position.z = (pos.z[0] if hasattr(pos, 'z') and len(pos.z) > 0 else 0.0) + 0.075

                        marker.scale.x = 0.15
                        marker.scale.y = 0.15
                        marker.scale.z = 0.15
                        marker.color.a = 0.8
                        marker.color.r = 0.0
                        marker.color.g = 1.0
                        marker.color.b = 0.0

                        opp_marker_array.markers.append(marker)

                    prediction_obs_arr = ObstacleArray(header=rospy.Header(stamp=rospy.Time.now(), frame_id="map"), obstacles=obstacle_list)
                    self.prediction_obs_pub.publish(prediction_obs_arr)

                    prediction_obs_pred_arr = PredictionArray(header=rospy.Header(stamp=rospy.Time.now(), frame_id="map"), id=opponent_pos_copy.obstacles[0].id, predictions=prediction_list)
                    self.prediction_obs_pred_pub.publish(prediction_obs_pred_arr)

                    self.opp_marker_pub.publish(opp_marker_array)

                    self.expire_counter = 0

                # ===== Linear interpolation to center =====
                elif use_original_method:
                    self._publish_force_trailing(True)  # START-guarded publish

                    obstacle_list = []
                    prediction_list = []
                    opp_marker_array = MarkerArray()

                    for i in range(self.time_steps):
                        w = i / (self.time_steps - 1)

                        interpolated_d = (1 - w) * current_opponent_d + w * opponent_approx_center_d

                        future_s_linear = current_opponent_s + i * current_opponent_v * self.dt
                        obs = Obstacle()
                        obs.id = i
                        obs.s_center = future_s_linear % self.max_s_updated
                        obs.s_start = (future_s_linear - opp_size / 2) % self.max_s_updated
                        obs.s_end = (future_s_linear + opp_size / 2) % self.max_s_updated
                        obs.d_center = interpolated_d
                        obs.d_left = obs.d_center + self.obstacle_half_width
                        obs.d_right = obs.d_center - self.obstacle_half_width
                        obs.size = opp_size
                        obs.vs = current_opponent_v
                        obs.vd = 0
                        obs.is_actually_a_gap = False
                        obs.is_static = False
                        obstacle_list.append(obs)

                        pds = Prediction()
                        pds.id = i
                        pds.pred_s = obs.s_center
                        pds.pred_d = obs.d_center
                        prediction_list.append(pds)

                        marker = Marker()
                        marker.header.stamp = rospy.Time.now()
                        marker.header.frame_id = "map"
                        marker.id = i
                        marker.type = Marker.CYLINDER
                        marker.action = Marker.ADD
                        marker.pose.orientation.w = 1.0

                        pos = self.frenet2glob([obs.s_center % self.max_s_updated], [obs.d_center])
                        marker.pose.position.x = pos.x[0]
                        marker.pose.position.y = pos.y[0]
                        # Frenet2GlobArr returns track-surface z; lift by half the
                        # cylinder height (scale.z = 0.15 below) so the cylinder base
                        # sits exactly on the surface instead of half-buried.
                        marker.pose.position.z = (pos.z[0] if hasattr(pos, 'z') and len(pos.z) > 0 else 0.0) + 0.075

                        marker.scale.x = 0.15
                        marker.scale.y = 0.15
                        marker.scale.z = 0.15
                        marker.color.a = 0.8
                        marker.color.r = 1.0
                        marker.color.g = 0.0
                        marker.color.b = 0.0

                        opp_marker_array.markers.append(marker)

                    prediction_obs_arr = ObstacleArray(header=rospy.Header(stamp=rospy.Time.now(), frame_id="map"), obstacles=obstacle_list)
                    self.prediction_obs_pub.publish(prediction_obs_arr)

                    prediction_obs_pred_arr = PredictionArray(header=rospy.Header(stamp=rospy.Time.now(), frame_id="map"), id=opponent_pos_copy.obstacles[0].id, predictions=prediction_list)
                    self.prediction_obs_pred_pub.publish(prediction_obs_pred_arr)

                    self.opp_marker_pub.publish(opp_marker_array)

                    self.expire_counter = 0

                # ===== Follow learned trajectory =====
                else:
                    self._publish_force_trailing(False)  # START-guarded publish (passes through)

                    obstacle_list = []
                    prediction_list = []
                    opp_marker_array = MarkerArray()

                    for i in range(self.time_steps):
                        # Lookup at current prediction position BEFORE advance.
                        # This aligns prediction[0] with beginn_s (VD/linear do
                        # the same: obs.s_center = current + i*v*dt, i=0 → exact).
                        # The old code advanced THEN set obs.s_start to the
                        # already-shifted value, causing prediction markers to
                        # drift ahead of the begin sphere by 1-2 steps.
                        opponent_approx_indx = int(np.argmin(
                            circular_s_dist(approx_s_points_global_array,
                                            current_opponent_s % self.max_s_opponent,
                                            self.max_s_opponent)))
                        opponent_speed = self.wpnts_opponent[opponent_approx_indx].proj_vs_mps
                        opponent_d = self.wpnts_opponent[opponent_approx_indx].d_m

                        size = opponent_pos_copy.obstacles[0].size
                        obs = Obstacle()
                        obs.id = i
                        obs.s_center = current_opponent_s % self.max_s_updated
                        obs.s_start = (current_opponent_s - size / 2) % self.max_s_updated
                        obs.s_end = (current_opponent_s + size / 2) % self.max_s_updated
                        obs.d_center = opponent_d
                        obs.d_left = opponent_d + self.obstacle_half_width
                        obs.d_right = opponent_d - self.obstacle_half_width
                        obs.size = size
                        obs.vs = opponent_speed
                        obs.vd = 0
                        obs.is_actually_a_gap = False
                        obs.is_static = False
                        obstacle_list.append(obs)

                        pds = Prediction()
                        pds.id = i
                        pds.pred_s = current_opponent_s
                        pds.pred_d = opponent_d
                        prediction_list.append(pds)

                        # Advance AFTER obs creation for next iteration.
                        current_opponent_s += opponent_speed * self.dt

                        marker = Marker()
                        marker.header.stamp = rospy.Time.now()
                        marker.header.frame_id = "map"
                        marker.id = i
                        marker.type = Marker.CYLINDER
                        marker.action = Marker.ADD
                        marker.pose.orientation.w = 1.0

                        pos = self.frenet2glob([obs.s_center % self.max_s_updated], [obs.d_center])
                        marker.pose.position.x = pos.x[0]
                        marker.pose.position.y = pos.y[0]
                        # Frenet2GlobArr returns track-surface z; lift by half the
                        # cylinder height (scale.z = 0.15 below) so the cylinder base
                        # sits exactly on the surface instead of half-buried.
                        marker.pose.position.z = (pos.z[0] if hasattr(pos, 'z') and len(pos.z) > 0 else 0.0) + 0.075

                        marker.scale.x = 0.15
                        marker.scale.y = 0.15
                        marker.scale.z = 0.15
                        marker.color.a = 0.8
                        marker.color.r = 0.0
                        marker.color.g = 1.0
                        marker.color.b = 0.0

                        opp_marker_array.markers.append(marker)

                    prediction_obs_arr = ObstacleArray(header=rospy.Header(stamp=rospy.Time.now(), frame_id="map"), obstacles=obstacle_list)
                    self.prediction_obs_pub.publish(prediction_obs_arr)

                    prediction_obs_pred_arr = PredictionArray(header=rospy.Header(stamp=rospy.Time.now(), frame_id="map"), id=opponent_pos_copy.obstacles[0].id, predictions=prediction_list)
                    self.prediction_obs_pred_pub.publish(prediction_obs_pred_arr)

                    self.opp_marker_pub.publish(opp_marker_array)

                    self.expire_counter = 0

                # ===== Common: beginn/end markers (all branches) =====
                # Previously only trajectory-follow updated these, so they
                # froze when VD/linear was active → visible lag between
                # prediction markers and begin/end spheres.
                if do_beginn and len(obstacle_list) > 0:
                    beginn_s = initial_opponent_s % self.max_s_updated
                    beginn_d = current_opponent_d
                    end_s = obstacle_list[-1].s_center % self.max_s_updated
                    end_d = obstacle_list[-1].d_center

                    position_beginn = self.frenet2glob([beginn_s], [beginn_d])
                    self.marker_beginn.pose.position.x = position_beginn.x[0]
                    self.marker_beginn.pose.position.y = position_beginn.y[0]
                    surface_z_b = position_beginn.z[0] if hasattr(position_beginn, 'z') and len(position_beginn.z) > 0 else 0.0
                    self.marker_beginn.pose.position.z = surface_z_b + self.marker_beginn.scale.z / 2
                    self.marker_pub_beginn.publish(self.marker_beginn)

                    position_end = self.frenet2glob([end_s], [end_d])
                    self.marker_end.pose.position.x = position_end.x[0]
                    self.marker_end.pose.position.y = position_end.y[0]
                    surface_z_e = position_end.z[0] if hasattr(position_end, 'z') and len(position_end.z) > 0 else 0.0
                    self.marker_end.pose.position.z = surface_z_e + self.marker_end.scale.z / 2
                    self.marker_pub_end.publish(self.marker_end)

            self.prediction_obs_pred_pub.publish(prediction_obs_pred_arr)

            self.expire_counter += 1
            if self.expire_counter >= self.max_expire_counter:
                self.expire_counter = self.max_expire_counter
                self.delete_all()

        rate.sleep()


if __name__ == '__main__':
    node = OppTrajPredictor()
    try:
        node.loop()
    except rospy.ServiceException as e:
        # roscore shutting down mid-loop -> service calls fail; exit quietly.
        if rospy.is_shutdown():
            rospy.loginfo(f"[Opp. Pred.] shutdown during service call: {e}")
        else:
            raise
    except rospy.ROSInterruptException:
        pass
