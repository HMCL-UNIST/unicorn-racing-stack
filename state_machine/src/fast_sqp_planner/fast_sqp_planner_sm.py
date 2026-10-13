#!/usr/bin/env python3
import time
import numpy as np
import os
import json
from typing import List

import rospy
from rospkg import RosPack
import tf
from dynamic_reconfigure.msg import Config
from f110_msgs.msg import ObstacleArray, OTWpntArray, WpntArray, Wpnt, BehaviorStrategy, PredictionArray
from nav_msgs.msg import Odometry
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from std_msgs.msg import String, Float32, Float32MultiArray, Bool
from visualization_msgs.msg import Marker, MarkerArray
# 3D: use 2.5D velocity planner with slope/track_3d_params/grip_scale_exp
from vel_planner_25d.vel_planner import calc_vel_profile

import trajectory_planning_helpers as tph
import configparser

DEBUG_STATE_TRANSITION = True

try:
    # if we are in the car, vesc msgs are built and we read them
    from vesc_msgs.msg import VescStateStamped
except:
    pass

#add state_machine src to path for states.py / states_types.py
import rospkg as _rospkg
import sys as _sys
_sm_src = os.path.join(_rospkg.RosPack().get_path('state_machine'), 'src')
if _sm_src not in _sys.path:
    _sys.path.insert(0, _sm_src)

# import transitions from same source directory as this file
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in _sys.path:
    _sys.path.insert(0, _this_dir)
import fast_sqp_planner_transitions as state_transitions
import states
from states_types import StateType

class WaypointData:
    def __init__(self, planner_name, is_closed):
        self.name =planner_name
        self.node_name = "/dyn_planners_statemachine/" + self.name
        self.list = []
        self.array = None
        self.stamp = None
        self.is_init = False
        self.is_gb_track_wpnts = False
        self.is_ot_wpnts = False
        self.closest_target = None
        self.closest_gap = None
        self.is_closed = is_closed
        self.vel_planner_safety_factor = 1.0
        self.dyn_sub = rospy.Subscriber(self.node_name + "/parameter_updates", Config, self.dyn_param_cb)
        self.update_param()

    def dyn_param_cb(self, config):
        self.update_param()

    def update_param(self):
        self.min_horizon = rospy.get_param(self.node_name + "/min_horizon")
        self.max_horizon = rospy.get_param(self.node_name + "/max_horizon")
        self.lateral_width_m = rospy.get_param(self.node_name + "/lateral_width_m")
        self.free_scaling_reference_distance_m = rospy.get_param(self.node_name + "/free_scaling_reference_distance_m")
        self.latest_threshold = rospy.get_param(self.node_name + "/latest_threshold")
        self.on_spline_front_horizon_thres_m = rospy.get_param(self.node_name + "/on_spline_front_horizon_thres_m")
        self.on_spline_min_dist_thres_m = rospy.get_param(self.node_name + "/on_spline_min_dist_thres_m")
        self.hyst_timer_sec = rospy.get_param(self.node_name + "/hyst_timer_sec")
        self.killing_timer_sec = rospy.get_param(self.node_name + "/killing_timer_sec")

    def initialize_traj(self, wpnt):
        if len(wpnt.wpnts) != 0:
            self.stamp = wpnt.header.stamp
            self.list = wpnt.wpnts
            self.array = np.array([[wpnt.x_m, wpnt.y_m, wpnt.s_m, wpnt.d_m] for wpnt in wpnt.wpnts])
            self.is_init = True

class StateMachine:
    """
    This state machine ideally should subscribe to topics and calculate flags/conditions.
    State transistions and state behaviors are described in `state_transistions.py` and `states.py`
    """

    def __init__(self, name) -> None:
        self.name = name
        self.rate_hz = rospy.get_param("state_machine/rate")  # rate of planner in hertz
        self.n_loc_wpnts = rospy.get_param("state_machine/n_loc_wpnts")  # number of local waypoints published
        self.local_wpnts = WpntArray()
        self.waypoints_dist = 0.1  # [m]

        self.measuring = rospy.get_param("/measure", default=False)

        # get initial dynamic parameters
        self.racecar_version = rospy.get_param("/racecar_version")
        self.sectors_params = rospy.get_param("/map_params")
        self.timetrials_only = rospy.get_param("state_machine/timetrials_only", False)
        self.n_sectors = self.sectors_params["n_sectors"]
        # only ftg zones
        self.only_ftg_zones = []
        self.ftg_counter = 0

        self.cur_s = 0.0
        self.cur_d = 0.0
        self.cur_vs = 0.0

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

        # 3D: vel planner parameters
        # (1) Init from vel_planner.yaml (same source as global_velocity_planner_3d)
        # (2) Subscribe to /global_velplanner_3d/parameter_updates for rqt real-time sync
        import yaml as _yaml
        self._h_cog = self.pars["veh_params"].get("cog_z", 0.074)
        self._slope_correction = 1.0
        self._slope_brake_margin = 0.0
        self._slope_brake_vmax = 5.0
        self._grip_scale_exp = 0.7

        _vel_yaml_path = os.path.join(
            RosPack().get_path('stack_master'), 'config', self.racecar_version, 'vel_planner.yaml')
        try:
            with open(_vel_yaml_path) as _f:
                _vp = _yaml.safe_load(_f)
            self._apply_vel_planner_params(_vp)
            rospy.loginfo(f"[StateMachine3D] vel_planner.yaml loaded")
        except Exception as _e:
            rospy.logwarn(f"[StateMachine3D] vel_planner.yaml not found ({_e}), using defaults")

        # Subscribe to rqt dynamic_reconfigure from global_velplanner_3d node
        rospy.Subscriber("/global_velplanner_3d/parameter_updates", Config,
                         self._vel_planner_3d_param_cb)

        # overtaking variables
        self.ot_sectors_params = rospy.get_param("/ot_map_params")
        self.n_ot_sectors = self.ot_sectors_params["n_sectors"]
        self.overtake_wpnts = None
        self.overtake_zones = []
        self.ot_begin_margin = 0.5
        self.cur_volt = 11.69  # default value for sim
        self.volt_threshold = rospy.get_param("state_machine/volt_threshold", default=10)
        self.static_overtaking_mode = False
        # Planner parameters
        self.ot_planner = rospy.get_param("state_machine/ot_planner", default="predictive_spliner")

        # waypoint variables
        self.cur_id_ot = 1
        self.max_speed = -1  # max speed in global waypoints for visualising
        self.max_s = 0
        self.current_position = None
        self.gb_wpnts = None
        self.gb_max_idx = None
        self.wpnt_dist = self.waypoints_dist
        self.num_glb_wpnts = 0  # number of waypoints on global trajectory
        self.num_ot_points = 0
        self.previous_index = 0
        self.gb_ego_width_m = rospy.get_param("state_machine/gb_ego_width_m")
        self.lateral_width_gb_m = rospy.get_param("state_machine/lateral_width_gb_m", 0.3)  # [m] DYNIAMIC PARAMETER
        self.gb_horizon_m = rospy.get_param("state_machine/gb_horizon_m")
        self.interest_horizon_m = rospy.get_param("state_machine/interest_horizon_m", 20.0)

        self.last_recovery_update_time = None
        self.cur_gb_wpnts = WaypointData('global_tracking', True)
        self.cur_recovery_wpnts = WaypointData('recovery_planner', False)
        self.cur_avoidance_wpnts = WaypointData('dynamic_avoidance_planner', False)
        self.cur_static_avoidance_wpnts = WaypointData('static_avoidance_planner', False)
        self.cur_start_wpnts = WaypointData('start_planner', False)

        self.cur_avoidance_wpnts.is_ot_wpnts = True
        self.cur_static_avoidance_wpnts.is_ot_wpnts = True
        self.cur_gb_wpnts.is_gb_track_wpnts = True
        self.cur_recovery_wpnts.vel_planner_safety_factor = 0.5

        self.ot_closest_target = None
        self.ot_closest_gap = None

        self.behavior_strategy = BehaviorStrategy()
        # mincurv spline
        self.mincurv_spline_x = None
        self.mincurv_spline_y = None
        # ot spline
        self.ot_spline_x = None
        self.ot_spline_y = None
        self.ot_spline_d = None
        self.recompute_ot_spline = True

        # obstacle avoidance variables
        self.obstacles = []
        self.obstacles_in_interest = []
        self.cur_obstacles_in_interest = []
        self.obstacles_perception = []
        self.obstacles_prediction_id = None
        self.obstacles_prediction = []
        self.ego_prediction = []
        self.obstacle_was_here = True
        self.side_by_side_threshold = 0.6
        self.merger = None
        self.force_trailing = False
        self.use_force_trailing = not rospy.get_param("state_machine/use_force_trailing", False)

        # spliner variables
        self.splini_ttl = rospy.get_param("state_machine/splini_ttl", 2.0) if self.ot_planner == "spliner" else rospy.get_param("state_machine/pred_splini_ttl", 0.2)
        self.splini_ttl_counter = int(self.splini_ttl * self.rate_hz)  # convert seconds to counters
        self.avoidance_wpnts = None
        self.static_avoidance_wpnts = None
        self.start_wpnts = None
        self.start_wpnts_array = None
        self.last_valid_avoidance_wpnts = None
        self.last_valid_avoidance_array = None
        self.last_valid_static_avoidance_wpnts = None
        #direct fast_sqp_planner path storage (bypasses WaypointData)
        self._ot_fast_sqp_wpnts = None
        #collision detector closest obstacle dist
        self._opponent_dist = 100.0
        self._opponent_dist_threshold = rospy.get_param("state_machine/opponent_dist_threshold", 0.8)
        self.ot_collision_ttc_s = rospy.get_param("state_machine/ot_collision_ttc_s", 0.5)
        self.ot_engage_ttc_s = rospy.get_param("state_machine/ot_engage_ttc_s", 1.0)
        self.ot_engage_min_m = rospy.get_param("state_machine/ot_engage_min_m", 5.0)
        # trailing trigger params — lateral margin + s-gap based
        self.ot_margin_safe_m = rospy.get_param("state_machine/ot_margin_safe_m", 0.15)
        self.ot_blocking_s_gap_m = rospy.get_param("state_machine/ot_blocking_s_gap_m", 0.5)

        self.overtaking_horizon_m = rospy.get_param("state_machine/overtaking_horizon_m", 6.9)
        self.lateral_width_ot_m = rospy.get_param("state_machine/lateral_width_ot_m", 0.3)  # [m] DYNIAMIC PARAMETER
        self.splini_hyst_timer_sec = rospy.get_param("state_machine/splini_hyst_timer_sec", 0.75)
        self.emergency_break_horizon = rospy.get_param("state_machine/emergency_break_horizon", 1.1)
        self.emergency_break_d = 0.12  # [m]

        # Graph Based Variables
        self.graph_based_wpts = None
        self.gb_wpnts_arr = None
        #Frenet Variables
        self.frenet_wpnts = WpntArray()
        # Parameters
        self.track_length = rospy.get_param("/global_republisher/track_length")
        # FTG params
        self.ftg_speed_mps = rospy.get_param("state_machine/ftg_speed_mps", 1.0) # [mps] DYNIAMIC PARAMETER
        self.ftg_timer_sec = rospy.get_param("state_machine/ftg_timer_sec", 3.0) # [s] DYNIAMIC PARAMETER
        self.ftg_disabled = not rospy.get_param("state_machine/ftg_active", False)

        # Force GBTRACK state
        self.force_gbtrack_state = rospy.get_param("state_machine/force_GBTRACK", False)

        self.overtaking_ttl_sec = rospy.get_param("state_machine/overtaking_ttl_sec", 3.0)
        self.overtaking_ttl_count = 0
        self.overtaking_ttl_count_threshold = int(self.overtaking_ttl_sec * self.rate_hz)

        self.save_start_traj = False
        self.cur_start_wpnts_candidate = OTWpntArray()
        self.need_start_traj = False
        # visualization variables
        self.first_visualization = True
        self.x_viz = 0
        self.y_viz = 0

        # STATES
        self.cur_state = StateType.GB_TRACK
        self.local_wpnts_src = StateType.GB_TRACK
        self.static_avoid = False

        self.fail_trailing = False

        self.states = {
            StateType.GB_TRACK: states.GlobalTracking,
            StateType.OVERTAKE: states.Overtaking,
            StateType.FTGONLY: states.FTGOnly,
            StateType.START: states.START,
        }
        self.state_transitions = {
            StateType.GB_TRACK: state_transitions.GlobalTrackingTransition,
            StateType.TRAILING: state_transitions.TrailingTransition,
            StateType.OVERTAKE: state_transitions.OvertakingTransition,
            StateType.FTGONLY: state_transitions.FTGOnlyTransition,
            StateType.START: state_transitions.StartTransition,
        }

        # SUBSCRIPTIONS
        self.opponent = ObstacleArray()

        rospy.Subscriber("/car_state/odom", Odometry, self.odom_cb)
        rospy.wait_for_message("/car_state/odom", Odometry)
        rospy.Subscriber("/global_waypoints_scaled", WpntArray, self.glb_wpnts_cb)  # from velocity scaler
        rospy.Subscriber("/global_waypoints/overtaking", WpntArray, self.overtake_cb)
        # wait for global trajectory
        rospy.wait_for_message("/global_waypoints_scaled", WpntArray)
        rospy.wait_for_message("/global_waypoints/overtaking", WpntArray)
        rospy.Subscriber("/car_state/odom_frenet", Odometry, self.frenet_pose_cb)
        rospy.wait_for_message("/car_state/odom_frenet", Odometry)
        rospy.Subscriber("/global_waypoints", WpntArray, self.glb_wpnts_og_cb)  # from og wpnts

        # dynamic parameters subscriber
        rospy.Subscriber("/dyn_statemachine/parameter_updates", Config, self.dyn_param_cb)
        rospy.Subscriber("/dyn_sector_tuner/speed/parameter_updates", Config, self.sector_dyn_param_cb)
        rospy.Subscriber("/dyn_sector_tuner/overtake/parameter_updates", Config, self.ot_dyn_param_cb)
        rospy.Subscriber("/tracking/obstacles", ObstacleArray, self.obstacle_perception_cb)
        rospy.Subscriber("/opponent_prediction/obstacles_pred", PredictionArray, self.obstacle_prediction_cb)
        #direct fast_sqp_planner path subscription (single topic only)
        rospy.Subscriber("/planner/avoidance/otwpnts", OTWpntArray, self._ot_fast_sqp_cb)
        #collision detector dist for trailing trigger
        rospy.Subscriber("/opponent_dist", Float32, self._opponent_dist_cb)
        if self.ot_planner == "spliner" or self.ot_planner == "predictive_spliner":
            pass
            if self.ot_planner == "predictive_spliner":
                rospy.Subscriber("/planner/avoidance/static_otwpnts", OTWpntArray, self.static_avoidance_cb)
        if self.ot_planner == "predictive_spliner":
            rospy.Subscriber("/planner/avoidance/merger", Float32MultiArray, self.merger_cb)
            rospy.Subscriber("collision_prediction/force_trailing", Bool, self.force_trailing_cb)
            rospy.Subscriber("planner/avoidance/fail_trailing", Bool, self.fail_trailing_cb)
        if not rospy.get_param("/sim"):
            rospy.Subscriber("/vesc/sensors/core", VescStateStamped, self.vesc_state_cb) # for reading battery voltage

        rospy.Subscriber("/planner/start_wpnts", OTWpntArray, self.start_wpnts_cb)

        # PUBLICATIONS
        self.behavior_strategy_pub = rospy.Publisher("behavior_strategy", BehaviorStrategy, queue_size=1)
        self.trailing_marker_pub = rospy.Publisher("/state_machine/trailing_target", Marker, queue_size=10)
        self.overtaking_marker_pub = rospy.Publisher("/state_machine/overtaking_target", Marker, queue_size=10)
        self.obstacles_in_interest_marker_pub = rospy.Publisher("/state_machine/obstacles_in_interest", MarkerArray, queue_size=10)

        self.loc_wpnt_pub = rospy.Publisher("local_waypoints", WpntArray, queue_size=1)
        self.vis_loc_wpnt_pub = rospy.Publisher("local_waypoints/markers", MarkerArray, queue_size=10)
        self.vis_loc_vel_pub = rospy.Publisher("local_waypoints/vel_markers", MarkerArray, queue_size=10)
        self.state_pub = rospy.Publisher("state_machine", String, queue_size=1)
        self.state_mrk = rospy.Publisher("/state_marker", Marker, queue_size=10)
        self.state_wpnts_src_marker = rospy.Publisher("/state_wpnts_src_marker", Marker, queue_size=10)
        self.emergency_pub = rospy.Publisher("/emergency_marker", Marker, queue_size=5) # for low voltage
        self.ot_section_check_pub = rospy.Publisher("/ot_section_check", Bool, queue_size=1)
        if self.measuring:
            self.latency_pub = rospy.Publisher("/state_machine/latency", Float32, queue_size=10)

        rospy.Subscriber("/save_start_traj", Bool, self.save_start_traj_cb)

        # MAIN LOOP
        self.loop()

    def on_shutdown(self):
        rospy.loginfo(f"[{self.name}] Shutting down state machine")

    #############
    # CALLBACKS #
    #############
    def save_start_traj_cb(self, msg):
        if len(self.cur_start_wpnts_candidate.wpnts) !=0:
            self.update_velocity(self.cur_start_wpnts_candidate, self.cur_start_wpnts.vel_planner_safety_factor)
            self.cur_start_wpnts.initialize_traj(self.cur_start_wpnts_candidate)
            self.cur_state = StateType.START

    def vesc_state_cb(self, data):
        """vesc state callback, reads the voltage"""
        self.cur_volt = data.state.voltage_input

    #store fast_sqp_planner path directly as List[Wpnt]
    def _ot_fast_sqp_cb(self, data: OTWpntArray):
        if len(data.wpnts) > 0:
            self._ot_fast_sqp_wpnts = list(data.wpnts)
        elif self.cur_state != StateType.TRAILING:
            self._ot_fast_sqp_wpnts = None

    def _opponent_dist_cb(self, msg):
        self._opponent_dist = msg.data

    def static_avoidance_cb(self, data: OTWpntArray):
        """static splini waypoints"""
        if len(data.wpnts) !=0:
            self.update_velocity(data, self.cur_static_avoidance_wpnts.vel_planner_safety_factor)
        self.static_avoidance_wpnts = data

    def start_wpnts_cb(self, data: OTWpntArray):
        """start waypoints candidate"""
        if len(data.wpnts) !=0:
            self.cur_start_wpnts_candidate = data

    def overtake_cb(self, data):
        self.overtake_wpnts = data.wpnts
        self.num_ot_points = len(self.overtake_wpnts)

        # compute the OT spline when new spline
        if self.recompute_ot_spline and self.num_ot_points != 0:
            self.ot_splinification()
            self.recompute_ot_spline = False

    def glb_wpnts_cb(self, data: WpntArray):
        data.wpnts = data.wpnts[:-1] # exclude last point (because last point == first point)
        self.gb_wpnts = data
        self.num_glb_wpnts = len(data.wpnts)

        self.n_loc_wpnts = min(self.n_loc_wpnts, int(self.num_glb_wpnts/2))

        self.max_s = data.wpnts[-1].s_m
        # Get spacing between wpnts for rough approximations
        self.wpnt_dist = data.wpnts[1].s_m - data.wpnts[0].s_m
        self.waypoints_dist = self.wpnt_dist
        self.gb_max_idx = data.wpnts[-1].id
        if self.ot_planner == "graph_based":
            self.gb_wpnts_arr = np.array([
                [w.s_m, w.d_m, w.x_m, w.y_m, w.d_right, w.d_left, w.psi_rad,
                w.kappa_radpm, w.vx_mps, w.ax_mps2] for w in data.wpnts
            ])

    def glb_wpnts_og_cb(self, data):
        if self.max_speed == -1:
            self.max_speed = max([wpnt.vx_mps for wpnt in data.wpnts])
        else:
            pass

    def obstacle_perception_cb(self, data):
        """Use tracking data directly without filtering."""
        if not self.timetrials_only:
            self.obstacles_perception = data.obstacles[:]
            self.obstacles = data.obstacles
            self.obstacles_in_interest = data.obstacles[:]

    def obstacle_prediction_cb(self, data):
        if len(data.predictions) != 0:
            self.obstacles_prediction_id = data.id
            self.obstacles_prediction = data.predictions
        else:
            self.obstacles_prediction = []

    def frenet_pose_cb(self, data: Odometry):
        self.cur_s = data.pose.pose.position.x
        self.cur_d = data.pose.pose.position.y
        self.cur_vs = data.twist.twist.linear.x
        if self.num_ot_points != 0:
            self.cur_id_ot = int(self._find_nearest_ot_s())

    def odom_cb(self, data):
        """
        Callback function of /tracked_pose subscriber.

        Parameters
        ----------
        data
            Data received from /tracked_pose topic
        """
        x = data.pose.pose.position.x
        y = data.pose.pose.position.y
        theta = tf.transformations.euler_from_quaternion(
            [data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w]
        )[2]

        self.current_position = [x, y, theta]

    def dyn_param_cb(self, params: Config):
        """
        Notices the change in the State Machine parameters and sets
        """
        self.lateral_width_gb_m = rospy.get_param("dyn_statemachine/lateral_width_gb_m", 0.75)
        self.lateral_width_ot_m = rospy.get_param("dyn_statemachine/lateral_width_ot_m", 0.3)
        self.splini_ttl = rospy.get_param("dyn_statemachine/splini_ttl") if self.ot_planner == "spliner" else rospy.get_param("dyn_statemachine/pred_splini_ttl")
        self.splini_ttl_counter = int(self.splini_ttl * self.rate_hz)  # convert seconds to counter
        self.splini_hyst_timer_sec = rospy.get_param("dyn_statemachine/splini_hyst_timer_sec", 0.75)
        self.emergency_break_horizon = rospy.get_param("dyn_statemachine/emergency_break_horizon", 1.1)
        self.ftg_speed_mps = rospy.get_param("dyn_statemachine/ftg_speed_mps", 1.0)
        self.ftg_timer_sec = rospy.get_param("dyn_statemachine/ftg_timer_sec", 3.0)

        self.overtaking_ttl_sec = rospy.get_param("dyn_statemachine/overtaking_ttl_sec", 3.0)
        self.overtaking_ttl_count_threshold = int(self.overtaking_ttl_sec * self.rate_hz)

        self.ftg_disabled = not rospy.get_param("dyn_statemachine/ftg_active", False)
        self.force_gbtrack_state = rospy.get_param("dyn_statemachine/force_GBTRACK", False)
        self.use_force_trailing = rospy.get_param("dyn_statemachine/use_force_trailing", False)

        if self.force_gbtrack_state:
            rospy.logwarn(f"[{self.name}] GBTRACK state force activated!!!")

        rospy.logdebug(
            "[{}] Received new parameters for state machine: lateral_width_gb_m: {}, "
            "lateral_width_ot_m: {}, splini_ttl: {}, splini_hyst_timer_sec: {}, ftg_speed_mps: {}, "
            "ftg_timer_sec: {}, GBTRACK_force: {}".format(
                self.name,
                self.lateral_width_gb_m,
                self.lateral_width_ot_m,
                self.splini_ttl,
                self.splini_hyst_timer_sec,
                self.ftg_speed_mps,
                self.ftg_timer_sec,
                self.force_gbtrack_state
            )
        )

    def sector_dyn_param_cb(self, params: Config):
        """
        Notices the change in the parameters and sets no/only ftg zones
        """
        # reset ftg zones
        self.only_ftg_zones = []
        # update ftg zones

        for i in range(self.n_sectors):
            self.sectors_params[f"Sector{i}"]["only_FTG"] = params.bools[2 * i + 1].value
            if self.sectors_params[f"Sector{i}"]["only_FTG"]:
                self.only_ftg_zones.append(
                    [self.sectors_params[f"Sector{i}"]["start"], self.sectors_params[f"Sector{i}"]["end"]]
                )

    def ot_dyn_param_cb(self, params: Config):
        """
        Notices the change in the parameters and sets overtaking zones
        """
        # reset overtake zones
        self.overtake_zones = []
        # update overtake zones
        try:
            for i in range(self.n_ot_sectors):
                self.ot_sectors_params[f"Overtaking_sector{i}"]["ot_flag"] = params.bools[i + 1].value
                # add start and end index of the sector
                if self.ot_sectors_params[f"Overtaking_sector{i}"]["ot_flag"]:
                    self.overtake_zones.append(
                        [
                            self.ot_sectors_params[f"Overtaking_sector{i}"]["start"],
                            self.ot_sectors_params[f"Overtaking_sector{i}"]["end"] + 1,
                        ]
                    )
        except IndexError as e:
            raise IndexError(f"[State Machine] Error in overtaking sector numbers. \nTry switching map with the script in stack_master/scripts and re-source in every terminal. \nError thrown: {e}")

        self.ot_begin_margin = params.doubles[2].value  # Choose the dyn ot param value
        rospy.logwarn(f"[{self.name}] Using OT beginning { self.ot_begin_margin}[m] from param: {params.doubles[2].name}"        )
        # Spline new OT if they exist already
        self.recompute_ot_spline = True

    def merger_cb(self, data):
        self.merger = data.data

    def force_trailing_cb(self, data):
        if self.use_force_trailing:
            self.force_trailing = data.data
        else:
            self.force_trailing = False

    def fail_trailing_cb(self, data):
        self.fail_trailing = data.data

    ######################################
    # ATTRIBUTES/CONDITIONS CALCULATIONS #
    ######################################
    """ For consistency, all conditions should be calculated in this section, and should all have the same signature:
    def _check_condition(self) -> bool:
    ...
    """

    def _check_only_ftg_zone(self) -> bool:
        ftg_only = False
        if len(self.only_ftg_zones) != 0:
            for sector in self.only_ftg_zones:
                if sector[0] <= self.cur_s / self.waypoints_dist <= sector[1]:
                    ftg_only = True
                    break
        return ftg_only

    def _check_close_to_raceline(self, threshold_m=None) -> bool:
        if threshold_m is None:
            return np.abs(self.cur_d) < self.gb_ego_width_m  # [m]
        else:
            return np.abs(self.cur_d) < threshold_m  # [m]

    def _get_adaptive_close_threshold(self) -> float:
        """Speed-adaptive close-to-raceline threshold.
        - At launch / struggling (v_cur << v_target): tight -> recovery triggers easily
        - Normal driving (v_cur ~ v_target): loose -> recovery rarely triggers
        """
        tight = rospy.get_param("state_machine/close_threshold_tight", 0.2)
        loose = rospy.get_param("state_machine/close_threshold_loose", 0.5)
        speed_ratio_thres = rospy.get_param("state_machine/close_speed_ratio_thres", 0.5)
        try:
            cur_s_idx = int(self.cur_s / self.waypoints_dist) % self.num_glb_wpnts
            v_target = self.cur_gb_wpnts.list[cur_s_idx].vx_mps
            if v_target < 0.05:
                return loose
            if self.cur_vs < v_target * speed_ratio_thres:
                return tight
            return loose
        except Exception:
            return loose

    def _check_close_to_raceline_heading(self, threshold_deg=None) -> bool:
        cloest_wpnt_idx = int(self.cur_s / self.waypoints_dist)%self.num_glb_wpnts
        cloest_wpnt_psi = self.cur_gb_wpnts.list[cloest_wpnt_idx].psi_rad
        if threshold_deg is None:
            return np.abs(self.current_position[2] - cloest_wpnt_psi) < np.deg2rad(20)
        else:
            return np.abs(self.cur_d) < np.deg2rad(threshold_deg)

    def _check_ot_sector(self) -> bool:
        for sector in self.overtake_zones:
            if sector[0] <= self.cur_s / self.waypoints_dist <= sector[1]:
                self.ot_section_check_pub.publish(True)
                return True
        self.ot_section_check_pub.publish(False)
        return False

    def _check_getting_closer(self, threshold_m=7.0) -> bool:
        """Check if we are getting closer to obstacle in interest."""
        if len(self.obstacles_in_interest) == 0:
            return False

        obs = self.obstacles_in_interest[0]

        # Close enough: check velocity difference
        velocity_ok = self.cur_vs - obs.vs > -0.5
        return velocity_ok

    def _check_enemy_in_front(self) -> bool:
        # If we are in time trial only mode -> return free overtake i.e. GB_FREE True
        horizon = self.gb_horizon_m  # Horizon in front of cur_s [m]
        for obs in self.obstacles:
            gap = (obs.s_start - self.cur_s) % self.track_length
            if gap < horizon:
                return True
        return False

    def _check_ot_path_collision(self) -> bool:
        """Returns True if any obstacle overlaps with _ot_fast_sqp_wpnts (trailing needed)."""
        if self._ot_fast_sqp_wpnts is None or len(self._ot_fast_sqp_wpnts) == 0:
            return False
        ot_s = np.array([w.s_m for w in self._ot_fast_sqp_wpnts])
        ot_d = np.array([w.d_m for w in self._ot_fast_sqp_wpnts])
        half_width = 0.15  # half car width
        # speed-proportional collision horizon: v*0.5s, clamped [2.0, gb_horizon_m]
        ot_collision_horizon = np.clip(self.cur_vs * self.ot_collision_ttc_s, 2.0, self.gb_horizon_m)
        for obs in self.obstacles:
            if obs.is_static:
                continue
            gap = (obs.s_center - self.cur_s) % self.track_length
            if gap > ot_collision_horizon:
                continue
            # find closest ot waypoint by s
            s_dists = np.abs((ot_s - obs.s_center + self.track_length / 2) % self.track_length
                             - self.track_length / 2)
            idx = int(np.argmin(s_dists))
            path_d = ot_d[idx]
            obs_d = obs.d_center
            # GP uncertainty: expand obstacle width by d_var
            d_var = getattr(obs, 'd_var', 0.0)
            sigma = np.sqrt(max(d_var, 0.0))
            obs_half = obs.size / 2.0 + 2.0 * sigma  # 2-sigma envelope
            if abs(path_d - obs_d) < obs_half + half_width:
                return True
        return False

    # min lateral margin between OT path and dynamic obstacles
    def _get_ot_path_min_margin(self) -> float:
        """Returns min lateral margin [m]. +inf if no obstacles or no OT path."""
        if self._ot_fast_sqp_wpnts is None or len(self._ot_fast_sqp_wpnts) == 0:
            return float('inf')
        ot_s = np.array([w.s_m for w in self._ot_fast_sqp_wpnts])
        ot_d = np.array([w.d_m for w in self._ot_fast_sqp_wpnts])
        half_width = 0.15
        ot_collision_horizon = np.clip(
            self.cur_vs * self.ot_collision_ttc_s, 2.0, self.gb_horizon_m)
        min_margin = float('inf')
        for obs in self.obstacles:
            if obs.is_static:
                continue
            gap = (obs.s_center - self.cur_s) % self.track_length
            if gap > ot_collision_horizon:
                continue
            s_dists = np.abs(
                (ot_s - obs.s_center + self.track_length / 2) % self.track_length
                - self.track_length / 2)
            idx = int(np.argmin(s_dists))
            path_d = ot_d[idx]
            obs_d = obs.d_center
            d_var = getattr(obs, 'd_var', 0.0)
            sigma = np.sqrt(max(d_var, 0.0))
            obs_half = obs.size / 2.0 + 2.0 * sigma
            margin = abs(path_d - obs_d) - obs_half - half_width
            min_margin = min(min_margin, margin)
        return min_margin

    def _check_ot_path_excessive_kappa(self) -> bool:
        """Returns True if near-term OT path has curvature too high to follow safely."""
        if self._ot_fast_sqp_wpnts is None or len(self._ot_fast_sqp_wpnts) < 3:
            return False
        # only check near-term: v * 1.0s ahead, min 3m
        speed = max(self.cur_vs, 1.0)
        check_horizon = max(speed * 1.0, 3.0)
        s_arr = np.array([w.s_m for w in self._ot_fast_sqp_wpnts])
        d_arr = np.array([w.d_m for w in self._ot_fast_sqp_wpnts])
        s_rel = (s_arr - self.cur_s) % self.track_length
        near_mask = s_rel < check_horizon
        if np.sum(near_mask) < 3:
            return False
        d_near = d_arr[near_mask]
        s_near = s_arr[near_mask]
        ds = np.diff(s_near)
        ds = np.where(ds < 0.01, 0.01, ds)
        dd = np.diff(d_near)
        ddd = np.diff(dd)
        if len(ddd) == 0:
            return False
        kappa = ddd / (ds[:-1] * ds[1:])
        max_kappa = float(np.max(np.abs(kappa)))
        kappa_limit = float(np.interp(speed, [1, 3, 5, 7], [3.0, 0.8, 0.5, 0.3]))
        return max_kappa > kappa_limit

    ##################################################################
    def _check_latest_wpnts(self, src_wpnts, wpnts_data: WaypointData):
        if src_wpnts is None or len(src_wpnts.wpnts) == 0:
            return False

        time_diff = (rospy.Time.now() - src_wpnts.header.stamp).to_sec()
        if time_diff > wpnts_data.latest_threshold:
            return False

        wpnts_data.initialize_traj(src_wpnts)
        on_spline = self._check_on_spline(wpnts_data)
        return on_spline

    def _check_ftg(self) -> bool:
        # If we have been standing still for 3 seconds inside TRAILING -> FTG
        threshold = self.ftg_timer_sec * self.rate_hz
        if self.ftg_disabled:
            return False
        else:
            if self.cur_state == StateType.TRAILING and self.cur_vs < self.ftg_speed_mps:
                self.ftg_counter += 1
                rospy.logwarn(f"[{self.name}] FTG counter: {self.ftg_counter}/{threshold}")
            else:
                self.ftg_counter = 0

            if self.ftg_counter > threshold:
                return True
            else:
                return False

    def _check_on_spline(self, wpnt_data) -> bool:
        if wpnt_data.is_init:
            gap = (wpnt_data.list[-1].s_m - self.cur_s) % self.track_length
            min_dist = np.min(np.linalg.norm(wpnt_data.array[:, 0:2] - self.current_position[:2], axis=1))

            gap_ok = gap > wpnt_data.on_spline_front_horizon_thres_m
            dist_ok = min_dist < wpnt_data.on_spline_min_dist_thres_m

            if gap_ok and dist_ok:
                return True
        return False

    def _check_free_frenet(self, wpnts_data) -> bool:
        is_free = True
        closest_obs = None
        min_gap = 2.0
        # Slightly different for spliner
        min_horizon = wpnts_data.min_horizon
        max_horizon = wpnts_data.max_horizon
        is_gb_track_wpnts = wpnts_data.is_gb_track_wpnts
        is_ot_wpnts = wpnts_data.is_ot_wpnts

        free_scaling_reference_distance_m = wpnts_data.free_scaling_reference_distance_m
        lateral_width_m = wpnts_data.lateral_width_m

        obstacles = self.cur_obstacles_in_interest
        obstacle_predictions = self.obstacles_prediction
        safety_factor_sec = 0.5

        if wpnts_data.is_init:
            max_gap = (wpnts_data.array[-1,2] - self.cur_s) % self.max_s
            for obs in obstacles:
                obs_s = obs.s_center
                # Wrapping madness to check if infront
                gap = (obs_s - self.cur_s) % self.max_s
                relative_vs = self.cur_vs - obs.vs
                clip_vs  = max(relative_vs, 0.5)
                ttc = (gap - self.pars["veh_params"]["length"]) / clip_vs
                tt0 = (gap + 0.3 * self.pars["veh_params"]["length"]) / clip_vs

                if obs.is_static:

                    if not wpnts_data.is_closed and gap > max_gap:   # Closed Wpnts is Short!!
                        is_free = False
                        if closest_obs is None or min_gap > gap:
                            closest_obs = obs
                            min_gap = gap

                    elif gap < max_horizon: # main
                        obs_d = obs.d_center
                        # Get d wrt to mincurv from the overtaking line
                        ot_d = 0
                        if not is_gb_track_wpnts:
                            avoid_wpnt_idx = np.argmin(abs(wpnts_data.array[:,2] - obs_s))
                            ot_d = wpnts_data.list[avoid_wpnt_idx].d_m
                        min_dist = abs(ot_d - obs_d)

                        free_dist = min_dist - obs.size/2 - self.gb_ego_width_m /2

                        scaling_factor = np.clip(gap / free_scaling_reference_distance_m, 0.0, 1.0)
                        if free_dist < lateral_width_m * scaling_factor:
                            is_free = False
                            rospy.loginfo("[State Machine] FREE False, obs dist to ot lane: {} m".format(free_dist))
                            if closest_obs is None or min_gap > gap:
                                closest_obs = obs
                                min_gap = gap
                else:
                    if len(obstacle_predictions) != 0 and self.obstacles_prediction_id == obs.id:

                        start_idx = 0
                        end_idx = len(obstacle_predictions)

                        if is_ot_wpnts:
                            if ttc > 0:
                                start_idx = min(int(ttc / 0.05), len(obstacle_predictions))
                            if tt0 > 0:
                                end_idx = min(int(tt0 / 0.05), len(obstacle_predictions))

                        for obs_pred in obstacle_predictions[start_idx:end_idx]:
                            wpnt_idx = np.argmin(abs(wpnts_data.array[:,2] - obs_pred.pred_s))
                            wpnt_d = wpnts_data.list[wpnt_idx].d_m
                            min_dist = abs(wpnt_d - obs_pred.pred_d)
                            free_dist = min_dist - obs.size/2 - self.gb_ego_width_m/2
                            scaling_factor = np.clip(gap / free_scaling_reference_distance_m, 0.0, 1.0)
                            if is_ot_wpnts:
                                rospy.logwarn(f"free_dist: {free_dist}, lateral_width_m: {lateral_width_m}, scaling_factor: {scaling_factor}, obs.size: {obs.size}, wpnt_d:{wpnt_d}, obs_pred.pred_d: {obs_pred.pred_d} " )
                            if free_dist < lateral_width_m * scaling_factor:
                                is_free = False
                                if closest_obs is None or min_gap > gap:
                                    closest_obs = obs
                                    min_gap = gap

                    else:
                        if not wpnts_data.is_closed and gap > max_gap:
                            is_free = False
                            if closest_obs is None or min_gap > gap:
                                closest_obs = obs
                                min_gap = gap
                        elif gap < max_horizon:
                            ot_d = 0
                            if not is_gb_track_wpnts:
                                avoid_wpnt_idx = np.argmin(abs(wpnts_data.array[:,2] - obs.s_center))
                                ot_d = wpnts_data.list[avoid_wpnt_idx].d_m
                            min_dist = abs(ot_d - obs.d_center)

                            free_dist = min_dist - obs.size/2 - self.gb_ego_width_m/2

                            scaling_factor = np.clip(gap / free_scaling_reference_distance_m, 0.0, 1.0)
                            if free_dist < lateral_width_m * scaling_factor:
                                is_free = False
                                if closest_obs is None or min_gap > gap:
                                    closest_obs = obs
                                    min_gap = gap
        else:
            is_free = True

        wpnts_data.closest_target = closest_obs
        wpnts_data.closest_gap = min_gap
        return is_free

    def _check_free_cartesian(self, wpnts_data) -> bool:
        is_free = True
        closest_obs = None
        min_gap = None
        # Slightly different for spliner
        min_horizon = wpnts_data.min_horizon
        max_horizon = wpnts_data.max_horizon
        free_scaling_reference_distance_m = wpnts_data.free_scaling_reference_distance_m
        lateral_width_m = wpnts_data.lateral_width_m

        obstacles = self.cur_obstacles_in_interest
        if wpnts_data.is_init:
            for obs in obstacles:
                if True:
                    obs_s = obs.s_center
                    # Wrapping madness to check if infront
                    gap = (obs_s - self.cur_s) % self.max_s

                    if gap < max_horizon or min_horizon < (gap - self.max_s):
                        dists = np.linalg.norm(wpnts_data.array[:,0:2] - np.array([obs.x_m, obs.y_m]), axis=1)
                        min_dist = np.min(dists)

                        free_dist = min_dist - obs.size/2 - self.gb_ego_width_m /2

                        scaling_factor = np.clip(gap / free_scaling_reference_distance_m, 0.0, 1.0)

                        if free_dist < lateral_width_m * scaling_factor:
                            is_free = False
                            if closest_obs is None or min_gap > gap:
                                closest_obs = obs
                                min_gap = gap
                            rospy.loginfo(f"[{self.name}] RECOVERY_FREE False, obs dist to recovery lane: {min_dist} m")
                else:
                    pass
        else:
            is_free = True
        wpnts_data.closest_target = closest_obs
        wpnts_data.closest_gap = min_gap
        return is_free

    def _check_availability(self, wpnts, wpnts_data) -> bool:
        if wpnts is None or wpnts_data.stamp is None:
            return False

        if (rospy.Time.now() - wpnts_data.stamp).to_sec() > wpnts_data.killing_timer_sec:
            wpnts_data.is_init = False
            if self._check_latest_wpnts(wpnts, wpnts_data):
                return True
            else:
                return False

        if (rospy.Time.now() - wpnts_data.stamp).to_sec() > wpnts_data.hyst_timer_sec:
            if self._check_latest_wpnts(wpnts, wpnts_data):
                return True

        if not self._check_on_spline(wpnts_data):
            if self._check_latest_wpnts(wpnts, wpnts_data):
                return True
            else:
                return False

        return True

    def _check_sustainability(self, src_wpnts, wpnts_data) -> bool:
        if (
            self._check_availability(src_wpnts, wpnts_data)
            and self._check_free_frenet(wpnts_data)
        ):
            return True

        return False

    def _check_overtaking_mode(self) -> bool:
        ot_sector_check = self._check_ot_sector()
        closer_check = self._check_getting_closer(threshold_m = 10.0)
        latest_check = self._check_latest_wpnts(self.avoidance_wpnts, self.cur_avoidance_wpnts)
        free_check = self._check_free_frenet(self.cur_avoidance_wpnts)

        if ot_sector_check and closer_check and latest_check and free_check:
            self.static_overtaking_mode = False
            return True
        else:
            return False

    def _check_static_overtaking_mode(self) -> bool:
        vs_check = self.cur_vs < 5.0
        closer_check = self._check_getting_closer(threshold_m = 7.0)
        latest_check = self._check_latest_wpnts(self.static_avoidance_wpnts, self.cur_static_avoidance_wpnts)
        free_check = self._check_free_frenet(self.cur_static_avoidance_wpnts)

        if vs_check and closer_check and latest_check and free_check:
            self.static_overtaking_mode = True
            return True
        else:
            return False

    def _check_overtaking_mode_sustainability(self) -> bool:
        if self.static_overtaking_mode:
            if (
                self._check_availability(self.static_avoidance_wpnts, self.cur_static_avoidance_wpnts)
                and self._check_free_frenet(self.cur_static_avoidance_wpnts)
            ):
                return True
        else:
            if True:
                if self._check_availability(self.avoidance_wpnts, self.cur_avoidance_wpnts):
                    rospy.logwarn("AVAILABLE")
                    if self._check_free_frenet(self.cur_avoidance_wpnts):
                        return True

        return False

    ################
    # HELPER FUNCS #
    ################
    def _apply_vel_planner_params(self, params: dict):
        """
        Apply vel planner params from yaml or rqt dynamic_reconfigure.
        Updates ggv, ax_max tables, and 3D-specific params in-place.
        Same logic as global_velocity_planner_3d._apply_params_to_ggv.
        """
        self._slope_correction = params.get('slope_correction', self._slope_correction)
        self._slope_brake_margin = params.get('slope_brake_margin', self._slope_brake_margin)
        self._slope_brake_vmax = params.get('slope_brake_vmax', self._slope_brake_vmax)
        self._grip_scale_exp = params.get('grip_scale_exp', self._grip_scale_exp)

        # ggv / ax_max table overrides
        if 'a_x_max' in params and params['a_x_max'] is not None:
            self.ggv[:, 1] = params['a_x_max']
        if 'a_y_max' in params and params['a_y_max'] is not None:
            self.ggv[:, 2] = params['a_y_max']
        if 'ax_max_motor' in params and params['ax_max_motor'] is not None:
            self.ax_max_machines[:, 1] = params['ax_max_motor']
        if 'ax_max_brake' in params and params['ax_max_brake'] is not None:
            self.b_ax_max_machines[:, 1] = params['ax_max_brake']
        if 'v_max' in params and params['v_max'] is not None:
            self.pars["veh_params"]["v_max"] = params['v_max']
        if 'dyn_model_exp' in params and params['dyn_model_exp'] is not None:
            self.pars["vel_calc_opts"]["dyn_model_exp"] = params['dyn_model_exp']

    def _vel_planner_3d_param_cb(self, msg: Config):
        """
        Subscribe to /global_velplanner_3d/parameter_updates (dynamic_reconfigure).
        When user edits rqt vel_planner -> this fires -> state machine tables updated in real-time.
        """
        params = {}
        for p in msg.doubles:
            params[p.name] = p.value
        for p in msg.bools:
            params[p.name] = p.value

        # Skip save/load triggers
        if params.get('save_config') or params.get('save_csv') or params.get('load_yaml'):
            return

        self._apply_vel_planner_params(params)
        rospy.loginfo_throttle(2.0,
            f"[StateMachine3D] vel_planner rqt sync: grip_scale_exp={self._grip_scale_exp:.2f}, "
            f"a_y_max={self.ggv[0,2]:.1f}, v_max={self.pars['veh_params']['v_max']:.1f}")

    def _build_track_3d_params(self, wpnts):
        """
        Build slope and track_3d_params from planner waypoints' mu_rad/s_m/kappa.
        Same computation as global_velocity_planner_3d.py.
        """
        n = len(wpnts)
        mu = np.array([wp.mu_rad for wp in wpnts])
        kappa = np.array([wp.kappa_radpm for wp in wpnts])
        s = np.array([wp.s_m for wp in wpnts])

        # Flat track -> skip 3D params (pure 2D behavior)
        if np.all(np.abs(mu) < 1e-8):
            return mu, None

        slope = mu

        # Unwrap s across track wraparound (e.g., 84, 85, 0.5, 1.5 -> 84, 85, 86.3, 87.3)
        # max_s from converter (raceline_length)
        max_s = self.gb_wpnts.wpnts[-1].s_m + (
            self.gb_wpnts.wpnts[-1].s_m - self.gb_wpnts.wpnts[-2].s_m)
        ds = np.diff(s)
        # If a step is largely negative (>= half track), it's a wraparound: add max_s
        ds = np.where(ds < -max_s / 2, ds + max_s, ds)
        ds = np.where(ds > max_s / 2, ds - max_s, ds)
        ds = np.maximum(ds, 1e-6)  # avoid zero-length intervals
        s_for_grad = np.concatenate([[0], np.cumsum(ds)])

        dmu_ds = np.gradient(slope, s_for_grad)

        phi = np.zeros(n)  # no banking

        # Angular rates via Euler->body Jacobian
        omega_x = -np.sin(mu) * kappa
        omega_y = np.cos(phi) * dmu_ds + np.cos(mu) * np.sin(phi) * kappa
        omega_z = -np.sin(phi) * dmu_ds + np.cos(mu) * np.cos(phi) * kappa
        track_3d_params = {
            'mu': mu,
            'phi': phi,
            'omega_x': omega_x,
            'omega_y': omega_y,
            'omega_z': omega_z,
            'd_omega_x': np.gradient(omega_x, s_for_grad),
            'd_omega_y': np.gradient(omega_y, s_for_grad),
            'd_omega_z': np.gradient(omega_z, s_for_grad),
            'dmu_ds': dmu_ds,
            'h': self._h_cog,
            'slope_correction': self._slope_correction,
            'slope_brake_margin': self._slope_brake_margin,
            'slope_brake_vmax': self._slope_brake_vmax,
        }
        return slope, track_3d_params

    def update_velocity(self, wpnts_msg, safety_factor=1.0):
        """3D: uses vel_planner_25d with slope, track_3d_params, grip_scale_exp.
        Passes s_global for correct friction sector matching."""
        wpnts = wpnts_msg.wpnts
        kappa = np.array([wp.kappa_radpm for wp in wpnts])
        el_lengths = np.array([
            np.linalg.norm([
                wpnts[i+1].x_m - wpnts[i].x_m,
                wpnts[i+1].y_m - wpnts[i].y_m
            ])
            for i in range(len(wpnts)-1)
        ])

        glb_start_idx = int(wpnts_msg.wpnts[-1].s_m / self.wpnt_dist)
        v_end = self.gb_wpnts.wpnts[glb_start_idx % len(self.gb_wpnts.wpnts)].vx_mps

        ax_max_machines_sf = self.ax_max_machines.copy()
        b_ax_max_machines_sf = self.b_ax_max_machines.copy()
        ax_max_machines_sf[:, 1] *= safety_factor
        b_ax_max_machines_sf[:, 1] *= safety_factor

        # 3D: build slope and track_3d_params from waypoint mu_rad
        slope, track_3d_params = self._build_track_3d_params(wpnts)

        # Global s from waypoints for friction sector matching
        s_global = np.array([wp.s_m for wp in wpnts])

        vx_profile = calc_vel_profile(
            ax_max_machines=ax_max_machines_sf,
            kappa=kappa,
            el_lengths=el_lengths,
            closed=False,
            drag_coeff=self.pars["veh_params"]["dragcoeff"],
            m_veh=self.pars["veh_params"]["mass"],
            b_ax_max_machines=b_ax_max_machines_sf,
            ggv=self.ggv,
            v_max=self.pars["veh_params"]["v_max"],
            filt_window=self.pars["vel_calc_opts"]["vel_profile_conv_filt_window"],
            dyn_model_exp=self.pars["vel_calc_opts"]["dyn_model_exp"],
            v_start=self.cur_vs,
            v_end=v_end,
            slope=slope,
            track_3d_params=track_3d_params,
            grip_scale_exp=self._grip_scale_exp,
            s_global=s_global,
        )

        for i in range(len(vx_profile)):
            wpnts_msg.wpnts[i].vx_mps = vx_profile[i]

        ax_profile = tph.calc_ax_profile.calc_ax_profile(vx_profile=vx_profile,
                                                            el_lengths=el_lengths,
                                                            eq_length_output=False)

        for i in range(len(ax_profile)):
            wpnts_msg.wpnts[i].ax_mps2 = ax_profile[i]
        wpnts[len(ax_profile)].ax_mps2 = ax_profile[-1]

    def mincurv_splinification(self):
        coords = np.empty((len(self.cur_gb_wpnts.list), 4))
        for i, wpnt in enumerate(self.cur_gb_wpnts.list):
            coords[i, 0] = wpnt.s_m
            coords[i, 1] = wpnt.x_m
            coords[i, 2] = wpnt.y_m
            coords[i, 3] = wpnt.vx_mps

        self.mincurv_spline_x = Spline(coords[:, 0], coords[:, 1])
        self.mincurv_spline_y = Spline(coords[:, 0], coords[:, 2])
        self.mincurv_spline_v = Spline(coords[:, 0], coords[:, 3])
        rospy.loginfo(f"[{self.name}] Splinified Min Curve")

    def ot_splinification(self):
        coords = np.empty((len(self.overtake_wpnts), 5))
        for i, wpnt in enumerate(self.overtake_wpnts):
            coords[i, 0] = wpnt.s_m
            coords[i, 1] = wpnt.x_m
            coords[i, 2] = wpnt.y_m
            coords[i, 3] = wpnt.d_m
            coords[i, 4] = wpnt.vx_mps

        # Sort s_m to start splining at 0
        coords = coords[coords[:, 0].argsort()]
        self.ot_spline_x = Spline(coords[:, 0], coords[:, 1])
        self.ot_spline_y = Spline(coords[:, 0], coords[:, 2])
        self.ot_spline_d = Spline(coords[:, 0], coords[:, 3])
        self.ot_spline_v = Spline(coords[:, 0], coords[:, 4])
        rospy.loginfo(f"[{self.name}] Splinified Overtaking Curve")

    def _find_nearest_ot_s(self) -> float:
        half_search_dim = 5

        # create indices
        idxs = [
            i % self.num_ot_points for i in range(self.cur_id_ot - half_search_dim, self.cur_id_ot + half_search_dim)
        ]
        ses = np.array([self.overtake_wpnts[i].s_m for i in idxs])

        dists = np.abs(self.cur_s - ses)
        chose_id = np.argmin(dists)
        s_ot = idxs[chose_id]
        s_ot %= self.num_ot_points

        return s_ot

    def get_splini_wpts(self) -> WpntArray:
        """Obtain the waypoints by fusing those obtained by spliner with the
        global ones.
        """
        # Handle wrapping
        wpnts = None
        if self.static_overtaking_mode:
            wpnts = self.cur_static_avoidance_wpnts
        else:
            wpnts = self.cur_avoidance_wpnts

        if wpnts is None or not hasattr(wpnts, 'array') or wpnts.array is None or len(wpnts.array) == 0:
            rospy.logwarn_throttle(1.0, "[state_machine] Spliner waypoints invalid, using GB fallback")
            s = int(self.cur_s / self.waypoints_dist + 0.5)
            return [self.cur_gb_wpnts.list[(s + i) % self.num_glb_wpnts] for i in range(self.n_loc_wpnts)]

        diff = np.linalg.norm(wpnts.array[:, 0:2] - self.current_position[:2], axis=1)
        min_idx = np.argmin(diff)
        avoidance_wpnts = wpnts.list[min_idx:min_idx + self.n_loc_wpnts]

        if len(avoidance_wpnts) < self.n_loc_wpnts:
            # Use helper function to get extra waypoints (Smart or GB based on flag)
            last_s_m = wpnts.list[-1].s_m
            num_needed = self.n_loc_wpnts - len(avoidance_wpnts)
            extra_wpnts = self.get_extra_waypoints(last_s_m, num_needed)
            avoidance_wpnts.extend(extra_wpnts)
        return avoidance_wpnts

    def get_extra_waypoints(self, last_s_m: float, num_needed: int) -> List[Wpnt]:
        """Get extra GB waypoints to fill shortage at end of planner path."""
        gb_start_idx = int(last_s_m / self.wpnt_dist) + 1
        extra = [self.cur_gb_wpnts.list[(gb_start_idx + i) % len(self.cur_gb_wpnts.list)]
                 for i in range(num_needed)]
        return extra

    def get_start_wpts(self) -> WpntArray:
        """Obtain the waypoints by fusing those obtained by spliner with the
        global ones.
        """
        # Handle wrapping
        if self.cur_start_wpnts.is_init:
            diff = np.linalg.norm(self.cur_start_wpnts.array[:, 0:2] - self.current_position[:2], axis=1)
            min_idx = np.argmin(diff)
            start_wpnts = self.cur_start_wpnts.list[min_idx:min_idx + self.n_loc_wpnts]

            if len(start_wpnts) < self.n_loc_wpnts:
                glb_start_idx = int(self.cur_start_wpnts.list[-1].s_m / self.wpnt_dist) + 1
                extra_wpnts = [self.cur_gb_wpnts.list[(glb_start_idx + i) % len(self.cur_gb_wpnts.list)]
                            for i in range(self.n_loc_wpnts - len(start_wpnts))]

                start_wpnts.extend(extra_wpnts)
            return start_wpnts

        else:
            rospy.logwarn(f"[{self.name}] No valid avoidance waypoints, passing global waypoints")
            pass

    #######
    # VIZ #
    #######

    def _pub_local_wpnts(self, wpts):
        mrks = MarkerArray()
        del_mrk = Marker()
        del_mrk.header.stamp = rospy.Time.now()
        del_mrk.action = Marker.DELETEALL
        mrks.markers.append(del_mrk)
        self.vis_loc_wpnt_pub.publish(mrks)

        loc_markers = MarkerArray()
        loc_wpnts = WpntArray()
        loc_wpnts.wpnts = wpts
        loc_wpnts.header.stamp = rospy.Time.now()
        loc_wpnts.header.frame_id = "map"

        # Original: sphere markers with speed color
        vx_vals = [wpnt.vx_mps for wpnt in loc_wpnts.wpnts]
        vx_min = min(vx_vals) if vx_vals else 0.0
        vx_max = max(vx_vals) if vx_vals else 1.0
        for i, wpnt in enumerate(loc_wpnts.wpnts):
            mrk = Marker()
            mrk.header.frame_id = "map"
            mrk.type = mrk.SPHERE
            mrk.scale.x = 0.15
            mrk.scale.y = 0.15
            mrk.scale.z = 0.15
            mrk.color.a = 1.0
            t = (wpnt.vx_mps - vx_min) / (vx_max - vx_min) if vx_max > vx_min else 0.5
            mrk.color.r = max(0.0, min(1.0, 1.0 - 2.0 * (t - 0.5)))
            mrk.color.g = max(0.0, min(1.0, 2.0 * t))
            mrk.color.b = 0.0

            mrk.id = i
            mrk.pose.position.x = wpnt.x_m
            mrk.pose.position.y = wpnt.y_m
            mrk.pose.position.z = wpnt.z_m
            mrk.pose.orientation.w = 1
            loc_markers.markers.append(mrk)

        self.loc_wpnt_pub.publish(loc_wpnts)
        self.vis_loc_wpnt_pub.publish(loc_markers)

        # 3D: velocity cylinder markers (z = velocity, same scale as global)
        VEL_SCALE = 0.1317
        vel_markers = MarkerArray()
        for i, wpnt in enumerate(loc_wpnts.wpnts):
            mrk = Marker()
            mrk.header.frame_id = "map"
            mrk.header.stamp = rospy.Time.now()
            mrk.type = Marker.CYLINDER
            mrk.id = i
            mrk.scale.x = 0.1
            mrk.scale.y = 0.1
            height = max(wpnt.vx_mps * VEL_SCALE, 0.02)
            mrk.scale.z = height
            mrk.color.a = 0.7
            t = (wpnt.vx_mps - vx_min) / (vx_max - vx_min) if vx_max > vx_min else 0.5
            mrk.color.r = max(0.0, min(1.0, 1.0 - 2.0 * (t - 0.5)))
            mrk.color.g = max(0.0, min(1.0, 2.0 * t))
            mrk.color.b = 0.0
            mrk.pose.position.x = wpnt.x_m
            mrk.pose.position.y = wpnt.y_m
            mrk.pose.position.z = wpnt.z_m + height * 0.5
            mrk.pose.orientation.w = 1
            vel_markers.markers.append(mrk)
        self.vis_loc_vel_pub.publish(vel_markers)

    def visualize_state(self, state: str):
        """
        Function that visualizes the state of the car by displaying a colored cube in RVIZ.

        Parameters
        ----------
        action
            Current state of the car to be displayed
        """
        if self.first_visualization:
            self.first_visualization = False
            x0 = self.cur_gb_wpnts.list[0].x_m
            y0 = self.cur_gb_wpnts.list[0].y_m
            x1 = self.cur_gb_wpnts.list[1].x_m
            y1 = self.cur_gb_wpnts.list[1].y_m
            # compute normal vector of 125% length of trackboundary but to the left of the trajectory
            xy_norm = (
                -np.array([y1 - y0, x0 - x1]) / np.linalg.norm([y1 - y0, x0 - x1]) * 1.25 * self.cur_gb_wpnts.list[0].d_left
            )

            self.x_viz = x0 + xy_norm[0]
            self.y_viz = y0 + xy_norm[1]

        mrk = Marker()
        mrk.type = mrk.SPHERE
        mrk.id = 1
        mrk.header.frame_id = "map"
        mrk.header.stamp = rospy.Time.now()
        mrk.color.a = 1.0
        mrk.pose.position.x = self.x_viz
        mrk.pose.position.y = self.y_viz
        mrk.pose.position.z = 0
        mrk.pose.orientation.w = 1
        mrk.scale.x = 1
        mrk.scale.y = 1
        mrk.scale.z = 1

        # Set color and log info based on the state of the car
        if state == "GB_TRACK":
            mrk.color.b = 1.0
        elif state == "OVERTAKE":
            mrk.color.r = 1.0
            mrk.color.g = 0.0
            mrk.color.b = 0.0
        elif state == "TRAILING":
            mrk.color.r = 1.0
            mrk.color.g = 1.0
            mrk.color.b = 0.0
        elif state == "FTGONLY":
            mrk.color.r = 1.0
            mrk.color.g = 1.0
            mrk.color.b = 1.0
        else:
            mrk.color.r = 1.0
            mrk.color.g = 1.0
            mrk.color.b = 1.0
        self.state_mrk.publish(mrk)

        # Publish waypoint source + battery voltage text marker
        # Get waypoint source string (from transition result)
        wpnt_src_str = str(self.local_wpnts_src).replace("StateType.", "")

        # Format battery voltage
        voltage_str = f"{self.cur_volt:.1f}V" if hasattr(self, 'cur_volt') else "~V"

        # Center both lines to the same width (use longer string as reference)
        max_len = max(len(wpnt_src_str), len(voltage_str))
        # Add extra padding for better centering
        padding = 4
        wpnt_centered = wpnt_src_str.center(max_len + padding)
        voltage_centered = voltage_str.center(max_len + padding)

        # Text marker (black text, smaller) - waypoint source + battery voltage
        text_mrk = Marker()
        text_mrk.type = Marker.TEXT_VIEW_FACING
        text_mrk.id = 2  # Different ID from sphere marker
        text_mrk.header.frame_id = "map"
        text_mrk.header.stamp = rospy.Time.now()
        text_mrk.pose.position.x = self.x_viz
        text_mrk.pose.position.y = self.y_viz
        text_mrk.pose.position.z = 1.5  # Above the sphere
        text_mrk.pose.orientation.w = 1
        text_mrk.scale.z = 0.2  # Smaller text height
        text_mrk.color.r = 0.0  # Black text
        text_mrk.color.g = 0.0
        text_mrk.color.b = 0.0
        text_mrk.color.a = 1.0
        text_mrk.text = f"{wpnt_centered}\n {voltage_centered}"  # Both centered

        self.state_wpnts_src_marker.publish(text_mrk)

    def publish_not_ready_marker(self):
        """Publishes a text marker that warn the user that the car is not ready to run"""
        mrk = Marker()
        mrk.type = mrk.TEXT_VIEW_FACING
        mrk.id = 1
        mrk.header.frame_id = "map"
        mrk.header.stamp = rospy.Time.now()
        mrk.color.a = 1.0
        mrk.color.r = 1.0
        mrk.color.g = 0.0
        mrk.color.b = 0.0
        mrk.pose.position.x = np.mean(
            [wpnt.x_m for wpnt in self.cur_gb_wpnts.list]
        )  # publish in the center of the track, to avoid not seeing it
        mrk.pose.position.y = np.mean([wpnt.y_m for wpnt in self.cur_gb_wpnts.list])
        mrk.pose.position.z = 1.0
        mrk.pose.orientation.w = 1
        mrk.scale.x = 4.69
        mrk.scale.y = 4.69
        mrk.scale.z = 4.69
        mrk.text = "BATTERY TOO LOW!!!"
        self.emergency_pub.publish(mrk)

    def update_waypoints(self):
        if not self.cur_gb_wpnts.is_init:
            self.cur_gb_wpnts.initialize_traj(self.gb_wpnts)
        else:
            self.cur_gb_wpnts.list = self.gb_wpnts.wpnts

        self.cur_obstacles_in_interest = self.obstacles_in_interest
        return

    def get_overtaking_target(self):
        if self.cur_gb_wpnts.closest_target is not None:
            return [self.cur_gb_wpnts.closest_target]
        if self.cur_recovery_wpnts.closest_target is not None:
            return [self.cur_recovery_wpnts.closest_target]
        return []

    def get_traling_target(self):
        if self.local_wpnts_src == StateType.GB_TRACK and self.cur_gb_wpnts.closest_target is not None:
            return [self.cur_gb_wpnts.closest_target]
        elif self.local_wpnts_src == StateType.OVERTAKE and self.ot_closest_target is not None:
            return [self.ot_closest_target]
        else:
            return []

    def get_farthest_target(self, local_wpnts_src):
        if local_wpnts_src == StateType.GB_TRACK and self.cur_gb_wpnts.closest_target is not None:
            closest_target = self.cur_gb_wpnts.closest_target
            closest_gap = self.cur_gb_wpnts.closest_gap
            if self.cur_avoidance_wpnts.closest_target is not None and closest_gap <= self.cur_avoidance_wpnts.closest_gap:
                closest_gap = self.cur_avoidance_wpnts.closest_gap
                closest_target = self.cur_avoidance_wpnts.closest_target
                local_wpnts_src = StateType.OVERTAKE
            if self.cur_static_avoidance_wpnts.closest_target is not None and closest_gap < self.cur_static_avoidance_wpnts.closest_gap:
                closest_gap = self.cur_static_avoidance_wpnts.closest_gap
                closest_target = self.cur_static_avoidance_wpnts.closest_target
                local_wpnts_src = StateType.OVERTAKE
            if self.cur_start_wpnts.closest_target is not None and closest_gap < self.cur_start_wpnts.closest_gap:
                closest_gap = self.cur_start_wpnts.closest_gap
                closest_target = self.cur_start_wpnts.closest_target
                local_wpnts_src = StateType.START
            return [closest_target], local_wpnts_src

        # OVERTAKE wpnts_src: use closest obstacle directly as trailing target
        if local_wpnts_src == StateType.OVERTAKE and len(self.cur_obstacles_in_interest) > 0:
            closest = min(self.cur_obstacles_in_interest,
                          key=lambda o: (o.s_center - self.cur_s) % self.track_length)
            return [closest], local_wpnts_src

        return [], local_wpnts_src

    def check_ot_cloest_target(self):
        if self.cur_gb_wpnts.closest_target is not None and self.ot_closest_target is not None and self.local_wpnts_src == StateType.GB_TRACK:
            if self.ot_closest_gap > self.cur_gb_wpnts.closest_gap:
                self.local_wpnts_src = StateType.OVERTAKE

    #############
    # MAIN LOOP #
    #############
    def loop(self):
        """Main loop of the state machine. It is called at a fixed rate by the
        ROS node.
        """
        # do state transition (unless we want to force it into GB_TRACK via dynamic reconfigure)
        if self.measuring:
            start = time.perf_counter()

        self.update_waypoints()
        self.ot_closest_target = None
        need_vel_planner = False

        self.cur_gb_wpnts.closest_target = None
        self.cur_recovery_wpnts.closest_target = None
        self.cur_avoidance_wpnts.closest_target = None
        self.cur_static_avoidance_wpnts.closest_target = None
        self.cur_start_wpnts.closest_target = None

        # safety check
        if self.cur_volt < self.volt_threshold:
            rospy.logerr_throttle_identical(1, f"[{self.name}] VOLTS TOO LOW, STOP THE CAR")
            # publishes a marker that warn the user that the car is not ready to run
            self.publish_not_ready_marker()

        prev_state = self.cur_state
        prev_wpnts_src = self.local_wpnts_src

        if self.force_gbtrack_state:
            self.cur_state = StateType.GB_TRACK
            self.local_wpnts_src = StateType.GB_TRACK
        elif self._check_only_ftg_zone():
            self.cur_state = StateType.FTGONLY
            self.local_wpnts_src = StateType.FTGONLY
            rospy.logwarn(f"[{self.name}] FTGONLY sector !!!")
        else:
            self.cur_state, self.local_wpnts_src = self.state_transitions[self.cur_state](self)

        if DEBUG_STATE_TRANSITION:
            if prev_state != self.cur_state or prev_wpnts_src != self.local_wpnts_src:
                rospy.logwarn(f"[STATE CHANGE] {prev_state.name} -> {self.cur_state.name} | wpnts: {prev_wpnts_src.name} -> {self.local_wpnts_src.name}")

        if self.local_wpnts_src == StateType.OVERTAKE and prev_wpnts_src != StateType.OVERTAKE:
            self._ot_fast_sqp_wpnts = None

        if self.cur_state == StateType.TRAILING:
            self.check_ot_cloest_target()
            self.behavior_strategy.trailing_targets, self.local_wpnts_src = self.get_farthest_target(self.local_wpnts_src)
        else:
            self.behavior_strategy.trailing_targets = []

        self.behavior_strategy.overtaking_targets = self.get_overtaking_target()

        # local_wpnts: OT topic when wpnts_src=OVERTAKE, else GB/FTG/etc
        if self.local_wpnts_src == StateType.OVERTAKE and self._ot_fast_sqp_wpnts is not None:
            ot_list = list(self._ot_fast_sqp_wpnts)
            s_arr = np.array([w.s_m for w in ot_list])
            dists = np.abs((s_arr - self.cur_s + self.track_length / 2) % self.track_length
                           - self.track_length / 2)
            start_idx = int(np.argmin(dists))
            ot_local = list(ot_list[start_idx:])
            last_s = ot_local[-1].s_m if ot_local else self.cur_s
            gb_start = int(last_s / self.waypoints_dist + 1) % self.num_glb_wpnts
            while len(ot_local) < self.n_loc_wpnts:
                ot_local.append(self.cur_gb_wpnts.list[gb_start % self.num_glb_wpnts])
                gb_start += 1
            local_wpnts = ot_local[:self.n_loc_wpnts]
        elif self.local_wpnts_src == StateType.OVERTAKE and self._ot_fast_sqp_wpnts is None:
            # OT path not available → fallback to GB
            local_wpnts = self.states[StateType.GB_TRACK](self)
        elif self.local_wpnts_src in self.states:
            local_wpnts = self.states[self.local_wpnts_src](self)
        else:
            local_wpnts = self.states[StateType.GB_TRACK](self)

        need_vel_planner = False
        # get the proper local waypoints based on the new state
        self.behavior_strategy.header.stamp = rospy.Time.now()
        self.behavior_strategy.local_wpnts = local_wpnts
        self.behavior_strategy.state = self.cur_state.value
        self.behavior_strategy.need_vel_planner = need_vel_planner

        self.behavior_strategy_pub.publish(self.behavior_strategy)

        self.state_pub.publish(self.cur_state.value)
        self.visualize_state(state=self.cur_state.value)

        self._pub_local_wpnts(local_wpnts)
        # Clear FTG counter if not in TRAILING state
        if self.cur_state != StateType.TRAILING:
            self.ftg_counter = 0

        overtaking_target_mrk = Marker()
        if len(self.behavior_strategy.overtaking_targets) != 0:
            overtaking_target_mrk.header.frame_id = "map"
            overtaking_target_mrk.type = Marker.SPHERE
            overtaking_target_mrk.scale.x = 0.5
            overtaking_target_mrk.scale.y = 0.5
            overtaking_target_mrk.scale.z = 0.5
            overtaking_target_mrk.color.a = 1.0
            overtaking_target_mrk.color.b = 1.0
            overtaking_target_mrk.pose.position.x = self.behavior_strategy.overtaking_targets[0].x_m
            overtaking_target_mrk.pose.position.y = self.behavior_strategy.overtaking_targets[0].y_m
            overtaking_target_mrk.pose.orientation.w = 1
        else:
            overtaking_target_mrk.action = Marker.DELETEALL
        self.overtaking_marker_pub.publish(overtaking_target_mrk)

        trailing_target_mrk = Marker()
        if len(self.behavior_strategy.trailing_targets) != 0:
            trailing_target_mrk.header.frame_id = "map"
            trailing_target_mrk.type = Marker.SPHERE
            trailing_target_mrk.scale.x = 0.5
            trailing_target_mrk.scale.y = 0.5
            trailing_target_mrk.scale.z = 0.5
            trailing_target_mrk.color.a = 1.0
            trailing_target_mrk.color.g = 1.0
            trailing_target_mrk.pose.position.x = self.behavior_strategy.trailing_targets[0].x_m
            trailing_target_mrk.pose.position.y = self.behavior_strategy.trailing_targets[0].y_m
            trailing_target_mrk.pose.orientation.w = 1
        else:
            trailing_target_mrk.action = Marker.DELETEALL
        self.trailing_marker_pub.publish(trailing_target_mrk)

        if self.measuring:
            end = time.perf_counter()
            self.latency_pub.publish(1/(end - start))

if __name__ == "__main__":
    name = "state_machine"
    rospy.init_node(name, anonymous=False, log_level=rospy.WARN)

    # init and run state machine
    state_machine = StateMachine(name)

    rospy.on_shutdown(state_machine.on_shutdown)

    loop_rate = rospy.Rate(state_machine.rate_hz)
    while not rospy.is_shutdown():
        state_machine.loop()
        loop_rate.sleep()
