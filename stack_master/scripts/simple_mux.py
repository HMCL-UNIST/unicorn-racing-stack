#!/usr/bin/env python3

import csv
import datetime as _dt
import json
import math
import os
import threading
import rospkg
import rospy
import yaml
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty, Float64, String
from sensor_msgs.msg import Joy
from ackermann_msgs.msg import AckermannDriveStamped
from vesc_msgs.msg import VescStateStamped
from copy import deepcopy
# dynamic_reconfigure for live launch tuning via rqt_reconfigure
from dynamic_reconfigure.server import Server as DynRecServer
from stack_master.cfg import SimpleMuxConfig

# per-session log root (inside the repository, so it is visible on the host when the repo is mounted into docker)
_REC_LOG_ROOT = os.path.expanduser("~/catkin_ws/src/race_stack/launch_logs")
_REC_PLOT_HORIZON_SEC = 2.5  # window length rendered in PNG. CSV is full session.
_REC_NAN = float('nan')
class SimpleMuxNode:

    def __init__(self):
        """
        Initialize the node, subscribe to topics, create publishers and set up member variables.
        """

        # Initialize the node
        self.name = "simple_mux"
        # anonymous=False so the node keeps its launch-assigned name
        # (/vesc/simple_mux). With anonymous=True the random suffix
        # (/vesc/simple_mux_<pid>_<time>) moved the dynamic_reconfigure server to a
        # per-run namespace, so rqt_reconfigure never showed it under the vesc group.
        # Only one mux runs per launch, so a fixed name is safe.
        rospy.init_node(self.name, anonymous=False)

        self.out_topic  = rospy.get_param("/vesc/out_topic", "low_level/ackermann_cmd_mux/output")
        self.in_topic  = rospy.get_param("/vesc/in_topic", "high_level/ackermann_cmd_mux/input/nav_1")
        self.joy_topic  = rospy.get_param("/vesc/joy_topic", "/joy")
        self.rate_hz = rospy.get_param("/vesc/rate_hz", 50.0)
        self.max_speed = rospy.get_param("/vesc/joy_max_speed", 4.0)
        self.max_steer = rospy.get_param("/vesc/joy_max_steer", 0.4)
        self.joy_freshness_threshold = rospy.get_param("/vesc/joy_freshness_threshold", 1.0)


        vesc_servo_min = rospy.get_param("/vesc/vesc_driver/servo_min", 1.0)
        vesc_servo_max = rospy.get_param("/vesc/vesc_driver/servo_max", 1.0)
        steering_angle_to_servo_offset = rospy.get_param("/vesc/steering_angle_to_servo_offset", 1.0)
        steering_angle_to_servo_gain = rospy.get_param("/vesc/steering_angle_to_servo_gain", 1.0)

        servo_max_rad = (vesc_servo_max - steering_angle_to_servo_offset) / steering_angle_to_servo_gain
        servo_min_rad = (vesc_servo_min - steering_angle_to_servo_offset) / steering_angle_to_servo_gain

        self.servo_max_abs = min(abs(servo_max_rad), abs(servo_min_rad))

        self.current_host = None
        self.release_time = None  # timestamp when buttons released
        self.autodrive_latched = False  # RB edge-triggered latch for autodrive
        self.rb_prev = 0  # previous RB state for edge detection

        self.human_drive = None
        self.autodrive = None
        self.zero_msg = AckermannDriveStamped()
        self.zero_msg.header.stamp = rospy.Time.now()
        self.zero_msg.drive.steering_angle = 0
        self.zero_msg.drive.speed = 0
        self.cur_v = 0
        self.prev_del_v = 0
        self.vel_planner = 0

        # launch control parameters (overridable via rosparam, live-tunable via dyn_reconfigure)
        # Two profile modes share the same arm/fire/abort logic:
        #   CONSTANT (mode=0): flat target_I = launch_current_A  (v1 single-knob)
        #   CURVE    (mode=1): 4-phase exponential profile (v2, see _launch_i_cmd)
        self.launch_arm_button = int(rospy.get_param("~launch_arm_button", 0))   # A button on most pads
        self.launch_t_total = float(rospy.get_param("~launch_t_total", 1.0))     # launch active duration [s]
        self.launch_current_A = float(rospy.get_param("~launch_current_A", 15.0))  # [CONSTANT mode] flat current
        self.launch_safety_floor_margin = float(rospy.get_param("~launch_safety_floor_margin", 0.1))  # m/s; speed >= measured + this after launch hand-off
        # v2 curve-shape params (CURVE mode). Defaults follow exp-rise design (weight transfer tau).
        self.launch_profile_mode = int(rospy.get_param("~launch_profile_mode", 1))   # 0=CONSTANT, 1=CURVE
        self.launch_I_back = float(rospy.get_param("~launch_I_back", 30.0))         # phase 1 end [A]
        self.launch_I_peak = float(rospy.get_param("~launch_I_peak", 75.0))         # phase 2/3 peak [A]
        self.launch_t_back = float(rospy.get_param("~launch_t_back", 0.05))         # phase 1 end [s]
        self.launch_t_rise_end = float(rospy.get_param("~launch_t_rise_end", 0.30)) # phase 2 end [s]
        self.launch_tau = float(rospy.get_param("~launch_tau", 0.08))               # exp rise time const [s]
        # cpp jerk==512 path constants (custom_ackermann_to_vesc.cpp:130-131). Mirror them here for inverse mapping.
        self.a2c_gain = float(rospy.get_param("/vesc/acceleration_to_current_gain", 10.0))
        self.a2c_baseline = 10.0
        self.v_ff_coeff = 0.9
        self.v_ff_deadzone = 0.3

        # launch state
        self.launch_armed = False
        self.launch_active = False
        self.launch_t0 = None
        self.launch_post_done = False  # one-shot: first tick after DONE applies safety floor
        self.a_prev = 0  # arm-button edge tracking
        self.lb_prev = 0  # LB edge tracking (for explicit launch abort logging)

        # yaml path for save/load triggers. Broad except so init never dies
        # because of yaml path quirks — save/load just disables itself.
        self.launch_yaml_path = None
        try:
            pkg_path = rospkg.RosPack().get_path("stack_master")
            car_name = rospy.get_param('/racecar_version', os.environ.get('CAR_NAME', ''))
            if car_name:
                self.launch_yaml_path = os.path.join(
                    pkg_path, 'config', car_name, 'devices', 'launch_control.yaml')
        except Exception as e:
            rospy.logwarn("[simple_mux] launch_yaml_path resolve failed: %s", e)
        if self.launch_yaml_path is None:
            rospy.logwarn("[simple_mux] launch_yaml_path unresolved — save/load disabled. "
                          "Set /racecar_version or CAR_NAME and respawn.")
        else:
            rospy.loginfo("[simple_mux] launch_yaml_path = %s", self.launch_yaml_path)

        # Subscribe to the topics
        rospy.Subscriber(self.in_topic, AckermannDriveStamped, self.drive_callback)
        rospy.Subscriber("/vesc/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/vel_planner", Float64, self.planner_callback)

        # Do not use filtered velocity
        # rospy.Subscriber("/car_state/odom", Odometry, self.odom_callback)

        rospy.Subscriber(self.joy_topic, Joy, self.joy_callback)
        # external abort trigger (e.g. test_control_publisher duration cutoff)
        # Same effect as LB rising: drop autodrive latch, reset launch state, force release_time so timer
        # broadcasts 1s of zero ackermann then idles.
        rospy.Subscriber("/launch_controller/abort", Empty, self._abort_cb, queue_size=1)

        # optional state_machine subscription. simple_mux must run standalone
        # (without 3d_headtohead.launch), so the topic may never publish — in that
        # case state stays None and launch-override behaves as before. When
        # state_machine is up and emits TRAILING, the launch current-injection is
        # force-aborted so the controller's speed command reaches VESC.
        self.state_topic = rospy.get_param("~state_topic", "/state_machine")
        # how long /state_machine can be silent before _active_state() returns
        # None (= launch gate opens). Long enough to absorb brief publisher hiccups in
        # mid-race so the gate does not flap; short enough that "really dead"
        # state_machine eventually frees standalone/test rigs.
        self.state_stale_sec = float(rospy.get_param("~state_stale_sec", 5.0))
        self._cur_state = None       # last string received, e.g. 'TRAILING'
        self._state_stamp = None     # rospy.Time of last reception; None == never received
        self._trailing_abort_logged = False  # log-once per trailing-abort event
        rospy.Subscriber(self.state_topic, String, self._state_cb, queue_size=1)

        # ===== Recording (CSV/PNG) — START / launch_active windows
        # Master toggle (default False). When False, all recording paths are no-ops
        # and the node behaves byte-for-byte like the pre-recording revision.
        self.enable_recording = False
        # VESC telemetry mirrored for CSV (filled by new subscribers; NaN when unseen)
        self._rec_I_motor = _REC_NAN
        self._rec_I_input = _REC_NAN
        self._rec_duty = _REC_NAN
        self._rec_erpm = _REC_NAN
        self._rec_V_input = _REC_NAN
        self._rec_temp_pcb = _REC_NAN
        self._rec_vesc_cmd_current = _REC_NAN
        self._rec_car_state_v = _REC_NAN
        # ERPM <-> m/s conversion constants (used for target_erpm columns)
        self._s2e_gain = float(rospy.get_param("/vesc/speed_to_erpm_gain", 1875.0))
        self._s2e_offset = float(rospy.get_param("/vesc/speed_to_erpm_offset", 0.0))
        # Last controller speed intent (captured pre-override each tick when autodrive is fresh)
        self._rec_ctrl_speed_cmd = _REC_NAN
        # Edge tracking for trigger composition
        self._rec_trig_active_prev = False
        # Session bookkeeping (lazy: dir created on first enable_recording True)
        self._rec_session_dir = None
        self._rec_csv_dir = None
        self._rec_plot_dir = None
        self._rec_seq = 0
        self._rec_csv_file = None
        self._rec_csv_writer = None
        self._rec_csv_path = None
        self._rec_trigger_tags = set()  # collected over session lifetime
        self._rec_lock = threading.Lock()
        # Subscribers for recording telemetry. Always alive — callbacks just write
        # member vars; if enable_recording is False they are simply unused. This
        # keeps subscribe set static regardless of toggle state.
        rospy.Subscriber("/vesc/sensors/core", VescStateStamped,
                         self._rec_core_cb, queue_size=10)
        rospy.Subscriber("/vesc/commands/motor/current", Float64,
                         self._rec_vesc_cmd_current_cb, queue_size=10)
        rospy.Subscriber("/car_state/odom", Odometry,
                         self._rec_car_state_cb, queue_size=10)

        self.drive_pub = rospy.Publisher(self.out_topic, AckermannDriveStamped, queue_size=10)
        self.current_pub = rospy.Publisher("/vesc/commands/motor/current", Float64, queue_size=10)
        self.launch_debug_pub = rospy.Publisher("/launch_controller/debug", String, queue_size=10)

        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self.timer_callback)

        # dynamic_reconfigure server for live launch tuning (rqt_reconfigure → /simple_mux)
        self.dyn_srv = DynRecServer(SimpleMuxConfig, self._dyn_cb)

        rospy.loginfo("[simple_mux] launch ctrl ready: mode=%d arm_btn=%d t_total=%.2fs I_const=%.1fA "
                      "I_back=%.1fA I_peak=%.1fA t_back=%.3f t_rise_end=%.3f tau=%.3f",
                      self.launch_profile_mode, self.launch_arm_button, self.launch_t_total,
                      self.launch_current_A, self.launch_I_back, self.launch_I_peak,
                      self.launch_t_back, self.launch_t_rise_end, self.launch_tau)

    # dynamic_reconfigure callback. Refuses to update mid-launch to avoid mid-flight value jumps.
    def _dyn_cb(self, config, level):
        if self.launch_active:
            rospy.logwarn("[simple_mux] dynamic update ignored: launch active. Wait for DONE.")
            # Echo current values back so rqt shows actual state
            config.launch_profile_mode = self.launch_profile_mode
            config.launch_current_A = self.launch_current_A
            config.launch_I_back = self.launch_I_back
            config.launch_I_peak = self.launch_I_peak
            config.launch_t_back = self.launch_t_back
            config.launch_t_rise_end = self.launch_t_rise_end
            config.launch_tau = self.launch_tau
            config.launch_t_total = self.launch_t_total
            config.launch_safety_floor_margin = self.launch_safety_floor_margin
            config.enable_recording = self.enable_recording
            config.save_params = False
            config.load_params = False
            return config

        # save/load triggers — checkbox flips in rqt_reconfigure act as one-shot
        # buttons. Reset to False so the next tick clears the checkmark in rqt UI.
        if config.save_params:
            self._save_launch_yaml(config)
            config.save_params = False
        if config.load_params:
            config.load_params = False
            loaded = self._load_launch_yaml()
            if loaded is not None:
                config.update(loaded)

        self.launch_profile_mode = int(config.launch_profile_mode)
        self.launch_current_A = float(config.launch_current_A)
        self.launch_I_back = float(config.launch_I_back)
        self.launch_I_peak = float(config.launch_I_peak)
        self.launch_t_back = float(config.launch_t_back)
        self.launch_t_rise_end = float(config.launch_t_rise_end)
        self.launch_tau = float(config.launch_tau)
        self.launch_t_total = float(config.launch_t_total)
        self.launch_safety_floor_margin = float(config.launch_safety_floor_margin)
        # sanity: t_back <= t_rise_end <= t_total. clamp silently — slider order during drags causes
        # transient violations otherwise.
        if self.launch_t_back > self.launch_t_rise_end:
            self.launch_t_rise_end = self.launch_t_back
            config.launch_t_rise_end = self.launch_t_rise_end
        if self.launch_t_rise_end > self.launch_t_total:
            self.launch_t_total = self.launch_t_rise_end
            config.launch_t_total = self.launch_t_total
        rospy.loginfo("[simple_mux] dyn updated: mode=%d I_const=%.1fA I_back=%.1fA I_peak=%.1fA "
                      "t_back=%.3f t_rise_end=%.3f tau=%.3f t_total=%.2f floor=%.2f",
                      self.launch_profile_mode, self.launch_current_A, self.launch_I_back,
                      self.launch_I_peak, self.launch_t_back, self.launch_t_rise_end,
                      self.launch_tau, self.launch_t_total, self.launch_safety_floor_margin)
        # recording master toggle. Lazy session dir on first True.
        new_rec = bool(config.enable_recording)
        if new_rec and not self.enable_recording:
            self._rec_ensure_session_dir()
        self.enable_recording = new_rec
        return config

    # keys we round-trip through yaml. Subset of SimpleMux.cfg (save/load triggers themselves excluded).
    _LAUNCH_YAML_KEYS = (
        'launch_profile_mode',
        'launch_current_A',
        'launch_I_back',
        'launch_I_peak',
        'launch_t_back',
        'launch_t_rise_end',
        'launch_tau',
        'launch_t_total',
        'launch_safety_floor_margin',
    )

    # dump current dyn config to launch_control.yaml (same path launch reads).
    def _save_launch_yaml(self, config):
        if not self.launch_yaml_path:
            rospy.logerr("[simple_mux] save_params: yaml path unresolved, abort")
            return
        try:
            os.makedirs(os.path.dirname(self.launch_yaml_path), exist_ok=True)
            out = {}
            for k in self._LAUNCH_YAML_KEYS:
                v = getattr(config, k)
                # Coerce so PyYAML emits clean scalars (no numpy / dyn_reconfigure types).
                if isinstance(v, bool):
                    out[k] = bool(v)
                elif isinstance(v, int) and not isinstance(v, bool):
                    out[k] = int(v)
                else:
                    out[k] = float(v)
            with open(self.launch_yaml_path, 'w') as f:
                yaml.dump(out, f, default_flow_style=False, sort_keys=False)
            rospy.loginfo("[simple_mux] saved -> %s", self.launch_yaml_path)
        except (OSError, yaml.YAMLError) as e:
            rospy.logerr("[simple_mux] save failed: %s", e)

    # read launch_control.yaml and return a dict of only the known launch keys.
    # Returns None on error (so caller leaves config untouched).
    def _load_launch_yaml(self):
        if not self.launch_yaml_path:
            rospy.logerr("[simple_mux] load_params: yaml path unresolved, abort")
            return None
        try:
            with open(self.launch_yaml_path, 'r') as f:
                data = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as e:
            rospy.logerr("[simple_mux] load failed: %s", e)
            return None
        if not isinstance(data, dict):
            rospy.logerr("[simple_mux] load failed: yaml root not a mapping")
            return None
        out = {k: data[k] for k in self._LAUNCH_YAML_KEYS if k in data}
        rospy.loginfo("[simple_mux] loaded %d params from %s", len(out), self.launch_yaml_path)
        return out

    # i_cmd(t) — the launch profile itself.
    # CONSTANT mode = flat I_const (v1 behavior).
    # CURVE mode = 4-phase exp profile:
    #   P1 [0, t_back]            : linear 0 -> I_back   (backlash absorb)
    #   P2 [t_back, t_rise_end]   : I_back + (I_peak-I_back)*(1-exp(-(t-t_back)/tau))  (weight transfer)
    #   P3 [t_rise_end, t_total]  : I_peak                                              (traction plateau)
    # Phase id is returned alongside for debug/visualization.
    def _launch_i_cmd(self, t):
        if self.launch_profile_mode == 0:
            return (self.launch_current_A if 0.0 <= t < self.launch_t_total else 0.0), \
                   (1 if 0.0 <= t < self.launch_t_total else 0)
        # CURVE
        if t < 0.0:
            return 0.0, 0
        if t < self.launch_t_back:
            tb = max(self.launch_t_back, 1e-6)
            return self.launch_I_back * (t / tb), 1
        if t < self.launch_t_rise_end:
            tau = max(self.launch_tau, 1e-6)
            return self.launch_I_back + (self.launch_I_peak - self.launch_I_back) * \
                   (1.0 - math.exp(-(t - self.launch_t_back) / tau)), 2
        if t < self.launch_t_total:
            return self.launch_I_peak, 3
        return 0.0, 0

    def check_uptodate(self, drive_msg):
        # return True
        if drive_msg is None:
            return False

        if abs(drive_msg.header.stamp.to_sec() - rospy.Time.now().to_sec()) < self.joy_freshness_threshold:
            return True
        else:
            return False

    def clip_servo(self, in_drive_msg):
        drive_msg = AckermannDriveStamped()
        drive_msg = deepcopy(in_drive_msg)

        if drive_msg.drive.steering_angle > 0 and drive_msg.drive.steering_angle > self.servo_max_abs:
            drive_msg.drive.steering_angle = self.servo_max_abs
        elif drive_msg.drive.steering_angle < 0 and drive_msg.drive.steering_angle < - self.servo_max_abs:
            drive_msg.drive.steering_angle = -self.servo_max_abs

        return drive_msg

    # invert cpp's i_cmd = a2c_gain*accel + 0.9*v + 10 to make cpp publish exactly target_I.
    def _accel_for_target_current(self, target_I):
        v_ff = self.v_ff_coeff * self.cur_v if self.cur_v > self.v_ff_deadzone else 0.0
        return (target_I - self.a2c_baseline - v_ff) / max(self.a2c_gain, 1e-6)

    # apply launch override on top of an autodrive ackermann.
    # Steering is NEVER overridden. Only speed/jerk/acceleration get touched.
    # During launch active: jerk=512 with inverse-mapped accel so cpp emits exactly launch_current_A.
    # First tick after DONE: pass-through but enforce safety floor (speed >= measured + small margin)
    # to defend against the edge case where controller's first speed cmd happens to be below measured.
    def _apply_launch_override(self, drive_msg):
        # safety gate. If state_machine reports TRAILING, abort launch and
        # fall through so the controller's speed command reaches VESC. No-op when
        # state_machine is absent or state is fresh-but-non-TRAILING.
        self._enforce_launch_state_gate()
        if self.launch_active and self.launch_t0 is not None:
            t = (rospy.Time.now() - self.launch_t0).to_sec()
            if t >= self.launch_t_total:
                # DONE: hand off to controller, ENABLE one-shot safety floor for next tick(s)
                self.launch_active = False
                self.launch_t0 = None
                self.launch_post_done = True
                self._launch_done_until = rospy.Time.now() + rospy.Duration(0.10)  # 100ms safety window
                self._publish_launch_debug(t, "DONE", drive_msg, target_I=0.0)
                # fall through to safety-floor block below
            else:
                target_I, phase = self._launch_i_cmd(t)
                accel = self._accel_for_target_current(target_I)
                drive_msg.drive.acceleration = accel
                drive_msg.drive.jerk = 512
                drive_msg.drive.speed = 0  # cpp ignores speed when jerk==512
                self._publish_launch_debug(t, "ACTIVE", drive_msg, target_I=target_I, phase=phase)
                return drive_msg

        # Post-launch safety window: enforce speed floor so VESC speed PID gets positive error
        if self.launch_post_done:
            if rospy.Time.now() < self._launch_done_until:
                drive_msg.drive.speed = max(drive_msg.drive.speed, self.cur_v + self.launch_safety_floor_margin)
                self._publish_launch_debug(0.0, "POST", drive_msg, target_I=0.0)
            else:
                self.launch_post_done = False
        return drive_msg

    def _publish_launch_debug(self, t, mode, drive_msg, target_I=0.0, phase=0):
        d = {
            "t": round(float(t), 3),
            "armed": bool(self.launch_armed),
            "active": bool(self.launch_active),
            "post_done": bool(self.launch_post_done),
            "mode": mode,
            "profile": int(self.launch_profile_mode),
            "phase": int(phase),
            "jerk": int(drive_msg.drive.jerk),
            "speed_cmd": round(float(drive_msg.drive.speed), 3),
            "accel_cmd": round(float(drive_msg.drive.acceleration), 4),
            "target_I_A": round(float(target_I), 2),
            "measured_v": round(float(self.cur_v), 3),
            "sm_state": self._active_state() or "NONE",
        }
        self.launch_debug_pub.publish(String(data=json.dumps(d)))

    # always-on heartbeat for armed/active state, even when no autodrive ackermann is flowing.
    # Without this, /launch_controller/debug only ticks during launch active, leaving any subscriber's
    # cached `armed` value stale until the first launch fire.
    def _publish_idle_state(self):
        d = {
            "t": 0.0,
            "armed": bool(self.launch_armed),
            "active": bool(self.launch_active),
            "post_done": bool(self.launch_post_done),
            "mode": "IDLE",
            "profile": int(self.launch_profile_mode),
            "phase": 0,
            "jerk": 0,
            "speed_cmd": 0.0,
            "accel_cmd": 0.0,
            "target_I_A": 0.0,
            "measured_v": round(float(self.cur_v), 3),
            "sm_state": self._active_state() or "NONE",
        }
        self.launch_debug_pub.publish(String(data=json.dumps(d)))

    def timer_callback(self, event):
        # state safety abort runs at 50Hz from the timer, NOT only from
        # _apply_launch_override (which fires only when an autodrive ackermann is
        # flowing). This closes the gap where armed=True survives because no
        # autodrive is yet streaming (e.g. armed set before headtohead, or during
        # humandrive). With this here, the moment state becomes non-None and
        # non-START, armed/active are wiped in the very next tick.
        self._enforce_launch_state_gate()

        # always publish armed/active heartbeat at 50Hz so test_publisher / debug subs never go stale.
        # _apply_launch_override path will publish a richer message during launch ACTIVE/POST.
        if not self.launch_active and not self.launch_post_done:
            self._publish_idle_state()

        # reset per-tick controller intent. Refilled below ONLY when autodrive is fresh,
        # so during humandrive or stale-autodrive ticks it stays NaN in any CSV row written here.
        self._rec_ctrl_speed_cmd = _REC_NAN
        # Final mux output speed (cpp input) — what actually went on the wire. Set when we publish.
        mux_speed_out = _REC_NAN
        mux_jerk_out = 0
        mux_accel_out = 0.0

        # when no button is held, publish zero for 1s then stop
        if self.current_host is None:
            if self.release_time is not None and (rospy.Time.now() - self.release_time).to_sec() < 1.0:
                self.zero_msg.header.stamp = rospy.Time.now()
                self.drive_pub.publish(self.zero_msg)
                mux_speed_out = 0.0
            self._rec_update(mux_speed_out, mux_jerk_out, mux_accel_out)
            return
        if self.current_host == "autodrive" and self.check_uptodate(self.autodrive):
            drive_msg = self.clip_servo(self.autodrive)
            # was *= 1.1 (hidden understeer band-aid); set to 1.0 to expose true linkage,
            # nonlinear servo cal (steering_servo_poly_coeffs) will absorb the linkage shape.
            drive_msg.drive.steering_angle *= 1.0
            # capture controller intent BEFORE launch override clobbers it. NaN if autodrive stale.
            self._rec_ctrl_speed_cmd = float(drive_msg.drive.speed)
            # launch override (only when armed+launched). pass-through otherwise.
            drive_msg = self._apply_launch_override(drive_msg)
            self.drive_pub.publish(drive_msg)
            mux_speed_out = float(drive_msg.drive.speed)
            mux_jerk_out = int(drive_msg.drive.jerk)
            mux_accel_out = float(drive_msg.drive.acceleration)
        elif self.current_host == "humandrive" and self.check_uptodate(self.human_drive):
            drive_msg = self.clip_servo(self.human_drive)
            # if drive_msg.drive.speed > 0 and self.cur_v < 3.0:
            #     # current_msg = Float64()
            #     # current_msg.data = 50.0
            #     # self.current_pub.publish(current_msg)
            #     # rospy.logwarn(f"joy_command : {drive_msg.drive.speed}" )

            #     drive_msg.drive.speed = 6.0
            #     # self.drive_pub.publish(drive_msg)
            # else:
            self.drive_pub.publish(drive_msg)
            mux_speed_out = float(drive_msg.drive.speed)
            mux_jerk_out = int(drive_msg.drive.jerk)
            mux_accel_out = float(drive_msg.drive.acceleration)
            # self.human_drive = None
        # else:
        #     self.drive_pub.publish(self.zero_msg)
        # recording hook (no-op when enable_recording is False).
        self._rec_update(mux_speed_out, mux_jerk_out, mux_accel_out)
    # state_machine subscriber. Stores raw string + timestamp.
    def _state_cb(self, msg):
        self._cur_state = msg.data
        self._state_stamp = rospy.Time.now()

    # returns active state string or None.
    # None when: (a) state_machine never published (standalone simple_mux), or
    # (b) last message older than state_stale_sec (state_machine died mid-run).
    def _active_state(self):
        if self._state_stamp is None:
            return None
        if (rospy.Time.now() - self._state_stamp).to_sec() > self.state_stale_sec:
            return None
        return self._cur_state

    # ELRS-noise safety gate. Launch current may ONLY be issued when:
    #   (a) state_machine never published (standalone simple_mux / test rigs), or
    #   (b) state_machine is alive and reports START.
    # Any other live state (GB_TRACK, TRAILING, OVERTAKE, RECOVERY, FTGONLY,
    # SMART_STATIC, ATTACK, ...) means we are mid-race, and a current injection
    # there can only be the result of a corrupted ELRS packet flipping A/RB. Used
    # to gate BOTH armed (A toggle) and active (RB fire) transitions, and to
    # force-abort any armed/active that survives a state change away from START.
    def _launch_allowed_by_state(self):
        s = self._active_state()
        return s is None or s == 'START'

    # if state forbids launch, kill any in-flight or armed launch. Called every
    # tick from both timer_callback (50Hz, runs even without autodrive ackermann) and
    # _apply_launch_override (in-flight path). Returns True if a launch was just
    # aborted so the override path bails out and falls through to controller speed
    # command. State whitelist (allowed): None or 'START'. Anything else, including
    # TRAILING, GB_TRACK, OVERTAKE, RECOVERY, FTGONLY, SMART_STATIC, ATTACK, ...,
    # forces armed=False and active=False.
    def _enforce_launch_state_gate(self):
        if self._launch_allowed_by_state():
            self._trailing_abort_logged = False
            return False
        if not (self.launch_active or self.launch_armed):
            return False
        if not self._trailing_abort_logged:
            rospy.logwarn("[launch] ABORT(state=%s) armed=%s active=%s -> speed control",
                          self._active_state(), self.launch_armed, self.launch_active)
            self._trailing_abort_logged = True
        self.launch_armed = False
        self.launch_active = False
        self.launch_t0 = None
        self.launch_post_done = False
        return True

    # equivalent of LB press from external client.
    def _abort_cb(self, _msg):
        rospy.logwarn("[launch] ABORT(external) armed=%s active=%s -> humandrive idle",
                      self.launch_armed, self.launch_active)
        # Reset all launch state
        self.launch_armed = False
        self.launch_active = False
        self.launch_t0 = None
        self.launch_post_done = False
        # Drop autodrive latch and switch to humandrive zero so motor coasts immediately
        self.autodrive_latched = False
        self.human_drive = self.zero_msg
        if self.current_host is not None:
            self.release_time = rospy.Time.now()
        self.current_host = None  # timer_callback's release_time path will broadcast 1s of zero ack

    def planner_callback(self, msg):
        self.vel_planner = msg.data

    def odom_callback(self, msg):
        self.cur_v = msg.twist.twist.linear.x

    def joy_callback(self, msg):
        # prev_host = deepcopy(self.current_host)
        use_human_drive = msg.buttons[4]
        rb_pressed = msg.buttons[5]

        # RB rising edge → latch autodrive
        rb_edge_rising = rb_pressed and not self.rb_prev
        self.rb_prev = rb_pressed

        # LB rising edge → explicitly abort any in-flight launch (defensive; humandrive already overrides via mux)
        lb_rising = use_human_drive and not self.lb_prev
        self.lb_prev = use_human_drive
        if lb_rising and (self.launch_active or self.launch_armed):
            rospy.logwarn("[launch] ABORT(LB) armed=%s active=%s -> reset both", self.launch_armed, self.launch_active)
            self.launch_armed = False
            self.launch_active = False
            self.launch_t0 = None
            self.launch_post_done = False

        # arm button (default A=button[0]) — toggle armed only when not currently launching
        if len(msg.buttons) > self.launch_arm_button:
            a_pressed = msg.buttons[self.launch_arm_button]
            a_rising = a_pressed and not self.a_prev
            self.a_prev = a_pressed
            if a_rising and not self.launch_active:
                # state safety gate — block ARMING entirely when the car is
                # mid-race (state != None and != START). Disarming (T→F) is always
                # allowed so the user / safety paths can clear an unwanted armed.
                if not self.launch_armed and not self._launch_allowed_by_state():
                    rospy.logwarn("[launch] A_TOGGLE BLOCKED: state=%s (only None/START allowed)",
                                  self._active_state())
                else:
                    old = self.launch_armed
                    self.launch_armed = not self.launch_armed
                    # full button snapshot to identify cross-channel triggers
                    btn_snap = ",".join(str(b) for b in msg.buttons[:8])
                    rospy.loginfo("[launch] A_TOGGLE %s->%s (a_pressed=%d a_prev_was=%d, btns0-7=[%s])",
                                  old, self.launch_armed, int(a_pressed), int(not a_pressed) ^ 0, btn_snap)

        if use_human_drive:
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = rospy.Time.now()
            drive_msg.drive.steering_angle = msg.axes[3] * self.max_steer
            drive_msg.drive.speed = msg.axes[1] * self.max_speed


            # drive_msg.drive.jerk = 512.0
            # del_v = drive_msg.drive.speed -self.cur_v
            # drive_msg.drive.acceleration = del_v * 4.0 + (del_v - self.prev_del_v) * 0.1

            self.human_drive = drive_msg
            self.current_host = "humandrive"
            # LB takeover clears autodrive latch; need RB re-press to resume
            self.autodrive_latched = False
        elif rb_edge_rising:
            # RB pressed → latch autodrive ON
            self.autodrive_latched = True
            self.current_host = "autodrive"
            # trace whether RB rising sees armed=True (fire) or False (no-fire)
            btn_snap = ",".join(str(b) for b in msg.buttons[:8])
            rospy.loginfo("[launch] RB_RISING armed=%s active=%s (btns0-7=[%s])",
                          self.launch_armed, self.launch_active, btn_snap)
            # if armed, fire launch one-shot. armed -> False, active -> True at this exact tick.
            # Note: no state gate here on purpose — armed transition is already gated
            # by _launch_allowed_by_state in the A-toggle block above, so anything
            # that survived to RB rising was permitted then. The per-tick abort in
            # _enforce_launch_state_gate still catches state changes during the
            # in-flight window.
            if self.launch_armed and not self.launch_active:
                self.launch_armed = False
                self.launch_active = True
                self.launch_t0 = rospy.Time.now()
                rospy.logwarn("[launch] FIRED at t=%.3f, current=%.1fA for %.2fs (one-shot armed->False)",
                              self.launch_t0.to_sec(), self.launch_current_A, self.launch_t_total)
        elif self.autodrive_latched:
            # latch held → stay in autodrive even when RB released
            self.current_host = "autodrive"
        else:
            # no active control → zero out for 1s then idle
            if self.current_host is not None:
                self.release_time = rospy.Time.now()
            self.human_drive = self.zero_msg
            self.current_host = None

    # def drive_callback(self, msg):
    #     self.autodrive = msg


    def drive_callback(self, msg):
        drive_msg = AckermannDriveStamped()
        drive_msg = msg
        # drive_msg.drive.speed = self.vel_planner
        # drive_msg.drive.jerk = 512.0
        # del_v = drive_msg.drive.speed -self.cur_v
        # drive_msg.drive.acceleration = del_v * 4.0 + (del_v - self.prev_del_v) * 0.1
        # if drive_msg.drive.acceleration >0:
        #     drive_msg.drive.steering_angle *= (1+drive_msg.drive.acceleration*0.4)
        # else:
        #     drive_msg.drive.steering_angle *= (1+drive_msg.drive.acceleration*0.2)

        # self.prev_del_v = del_v

        self.autodrive = drive_msg

    # ===== Recording (CSV + PNG)
    # All recording paths are gated on self.enable_recording. When False the
    # callbacks below still fill member vars (cheap), but _rec_update returns
    # immediately so no trigger evaluation, file I/O, or render happens.
    # Triggers: launch_active|post_done (electrical, current-mode debug)
    #        ∪ sm_state == "START"     (mechanical, start-profile debug).
    # Rising edge of union opens CSV, falling edge closes and renders 2 PNGs.

    def _rec_core_cb(self, msg):
        try:
            self._rec_I_motor = float(msg.state.current_motor)
            self._rec_I_input = float(msg.state.current_input)
            self._rec_duty = float(msg.state.duty_cycle)
            self._rec_erpm = float(msg.state.speed)
            self._rec_V_input = float(msg.state.voltage_input)
            self._rec_temp_pcb = float(msg.state.temperature_pcb)
        except AttributeError:
            pass

    def _rec_vesc_cmd_current_cb(self, msg):
        # cpp publishes here only on jerk==512 branch. Latches at last value otherwise.
        self._rec_vesc_cmd_current = float(msg.data)

    def _rec_car_state_cb(self, msg):
        # Fused velocity from state_estimation. Absent on bench rig — stays NaN.
        self._rec_car_state_v = float(msg.twist.twist.linear.x)

    def _rec_ensure_session_dir(self):
        if self._rec_session_dir is not None:
            return
        try:
            session_ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._rec_session_dir = os.path.join(_REC_LOG_ROOT, "session_{}".format(session_ts))
            self._rec_csv_dir = os.path.join(self._rec_session_dir, "csv")
            self._rec_plot_dir = os.path.join(self._rec_session_dir, "plot")
            os.makedirs(self._rec_csv_dir, exist_ok=True)
            os.makedirs(self._rec_plot_dir, exist_ok=True)
            # meta.yaml: ERPM conversion + key launch params snapshot. Written once per session.
            meta_path = os.path.join(self._rec_session_dir, "meta.yaml")
            meta = {
                "session_ts": session_ts,
                "racecar_version": rospy.get_param('/racecar_version',
                                                   os.environ.get('CAR_NAME', '')),
                "s2e_gain": self._s2e_gain,
                "s2e_offset": self._s2e_offset,
                "a2c_gain": self.a2c_gain,
                "a2c_baseline": self.a2c_baseline,
                "v_ff_coeff": self.v_ff_coeff,
                "v_ff_deadzone": self.v_ff_deadzone,
                "rate_hz": self.rate_hz,
            }
            with open(meta_path, "w") as f:
                yaml.dump(meta, f, default_flow_style=False, sort_keys=False)
            rospy.loginfo("[simple_mux] rec session dir: %s", self._rec_session_dir)
        except OSError as e:
            rospy.logwarn("[simple_mux] rec session dir create failed: %s — recording disabled", e)
            self._rec_session_dir = None
            self._rec_csv_dir = None
            self._rec_plot_dir = None

    def _rec_trigger_active(self):
        """Union of trigger conditions. Returns (active, set_of_active_tags)."""
        tags = set()
        if self.launch_active or self.launch_post_done:
            tags.add("LAUNCH")
        sm = self._active_state()
        if sm == "START":
            tags.add("START")
        return (len(tags) > 0), tags

    def _rec_update(self, mux_speed_out, mux_jerk_out, mux_accel_out):
        if not self.enable_recording and self._rec_csv_writer is None:
            return  # fully off + no in-progress session: skip everything
        cur_active, cur_tags = self._rec_trigger_active()
        # Edge handling
        if cur_active and not self._rec_trig_active_prev:
            if self.enable_recording:
                self._rec_open_csv()
        if cur_active:
            self._rec_trigger_tags |= cur_tags
            self._rec_write_row(mux_speed_out, mux_jerk_out, mux_accel_out)
        if (not cur_active) and self._rec_trig_active_prev:
            self._rec_close_csv_and_render()
        self._rec_trig_active_prev = cur_active

    def _rec_open_csv(self):
        if self._rec_csv_dir is None:
            self._rec_ensure_session_dir()
            if self._rec_csv_dir is None:
                return
        with self._rec_lock:
            self._rec_seq += 1
            ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._rec_trigger_tags = set()
            self._rec_csv_path = os.path.join(
                self._rec_csv_dir, "rec_{}_seq{:03d}.csv".format(ts, self._rec_seq))
            try:
                self._rec_csv_file = open(self._rec_csv_path, "w", newline="")
                self._rec_csv_writer = csv.writer(self._rec_csv_file)
                self._rec_csv_writer.writerow([
                    "wall_t", "launch_t", "sm_state",
                    "launch_active", "launch_armed", "launch_post_done",
                    "launch_mode", "launch_phase", "launch_profile",
                    "ctrl_speed_cmd",
                    "mux_jerk", "mux_speed_cmd", "mux_accel_cmd", "mux_target_I_A",
                    "vesc_cmd_current_A",
                    "I_motor", "I_input", "duty",
                    "erpm", "V_input", "temperature_pcb",
                    "cur_v_vesc", "car_state_v",
                ])
                self._rec_csv_file.flush()
                rospy.loginfo("[simple_mux] rec CSV opened: %s", self._rec_csv_path)
            except OSError as e:
                rospy.logwarn("[simple_mux] rec CSV open failed: %s", e)
                self._rec_csv_file = None
                self._rec_csv_writer = None

    def _rec_compute_launch_meta(self):
        """Return (launch_t, mode, phase) reflecting the CURRENT override state."""
        if self.launch_active and self.launch_t0 is not None:
            t = (rospy.Time.now() - self.launch_t0).to_sec()
            if t >= self.launch_t_total:
                return t, "DONE", 0
            _, phase = self._launch_i_cmd(t)
            return t, "ACTIVE", phase
        if self.launch_post_done:
            return 0.0, "POST", 0
        return 0.0, "IDLE", 0

    def _rec_compute_target_I(self):
        if self.launch_active and self.launch_t0 is not None:
            t = (rospy.Time.now() - self.launch_t0).to_sec()
            if 0.0 <= t < self.launch_t_total:
                target_I, _ = self._launch_i_cmd(t)
                return target_I
        return 0.0

    def _rec_write_row(self, mux_speed_out, mux_jerk_out, mux_accel_out):
        if self._rec_csv_writer is None:
            return
        try:
            launch_t, mode, phase = self._rec_compute_launch_meta()
            target_I = self._rec_compute_target_I()
            sm = self._active_state() or "NONE"
            with self._rec_lock:
                if self._rec_csv_writer is None:
                    return
                self._rec_csv_writer.writerow([
                    "{:.4f}".format(rospy.Time.now().to_sec()),
                    "{:.4f}".format(launch_t),
                    sm,
                    int(self.launch_active),
                    int(self.launch_armed),
                    int(self.launch_post_done),
                    mode,
                    int(phase),
                    int(self.launch_profile_mode),
                    "{:.4f}".format(self._rec_ctrl_speed_cmd),
                    int(mux_jerk_out),
                    "{:.4f}".format(mux_speed_out),
                    "{:.4f}".format(mux_accel_out),
                    "{:.2f}".format(target_I),
                    "{:.3f}".format(self._rec_vesc_cmd_current),
                    "{:.3f}".format(self._rec_I_motor),
                    "{:.3f}".format(self._rec_I_input),
                    "{:.4f}".format(self._rec_duty),
                    "{:.0f}".format(self._rec_erpm),
                    "{:.2f}".format(self._rec_V_input),
                    "{:.1f}".format(self._rec_temp_pcb),
                    "{:.4f}".format(self.cur_v),
                    "{:.4f}".format(self._rec_car_state_v),
                ])
        except Exception as e:
            rospy.logwarn_throttle(1.0, "[simple_mux] rec write error: %s", e)

    def _rec_close_csv_and_render(self):
        with self._rec_lock:
            if self._rec_csv_file is None:
                return
            path = self._rec_csv_path
            try:
                self._rec_csv_file.flush()
                self._rec_csv_file.close()
                rospy.loginfo("[simple_mux] rec CSV closed: %s", path)
            except Exception as e:
                rospy.logwarn("[simple_mux] rec CSV close error: %s", e)
            self._rec_csv_file = None
            self._rec_csv_writer = None
            tags = sorted(self._rec_trigger_tags)
            self._rec_trigger_tags = set()
        # PNG render off-thread so 50Hz tick isn't blocked
        threading.Thread(target=self._rec_render_pngs,
                         args=(path, tags), daemon=True).start()

    def _rec_render_pngs(self, csv_path, tags):
        if csv_path is None or self._rec_plot_dir is None:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import csv as _csv
        except Exception as e:
            rospy.logwarn("[simple_mux] matplotlib import failed: %s", e)
            return
        # Load CSV
        try:
            rows = []
            with open(csv_path, "r") as f:
                reader = _csv.DictReader(f)
                for r in reader:
                    rows.append(r)
        except OSError as e:
            rospy.logwarn("[simple_mux] rec CSV read failed: %s", e)
            return
        if len(rows) < 2:
            rospy.logwarn("[simple_mux] rec rows < 2, skipping PNG: %s", csv_path)
            return

        def _f(r, k):
            try:
                return float(r.get(k, "nan"))
            except (TypeError, ValueError):
                return _REC_NAN

        t0 = _f(rows[0], "wall_t")
        t_rel = []
        target_I, vesc_cmd_I, I_act, I_in, duty = [], [], [], [], []
        ctrl_v, mux_v, cur_v_vesc, car_v = [], [], [], []
        erpm = []
        ctrl_target_erpm, mux_target_erpm = [], []
        mode_marks, sm_marks = [], []
        for r in rows:
            tr = _f(r, "wall_t") - t0
            if tr > _REC_PLOT_HORIZON_SEC:
                break
            t_rel.append(tr)
            target_I.append(_f(r, "mux_target_I_A"))
            vesc_cmd_I.append(_f(r, "vesc_cmd_current_A"))
            I_act.append(_f(r, "I_motor"))
            I_in.append(_f(r, "I_input"))
            duty.append(_f(r, "duty"))
            ctrl_v_val = _f(r, "ctrl_speed_cmd")
            mux_v_val = _f(r, "mux_speed_cmd")
            ctrl_v.append(ctrl_v_val)
            mux_v.append(mux_v_val)
            cur_v_vesc.append(_f(r, "cur_v_vesc"))
            car_v.append(_f(r, "car_state_v"))
            erpm.append(_f(r, "erpm"))
            ctrl_target_erpm.append(ctrl_v_val * self._s2e_gain + self._s2e_offset
                                    if not math.isnan(ctrl_v_val) else _REC_NAN)
            mux_target_erpm.append(mux_v_val * self._s2e_gain + self._s2e_offset
                                   if not math.isnan(mux_v_val) else _REC_NAN)
            mode_marks.append(r.get("launch_mode", "IDLE"))
            sm_marks.append(r.get("sm_state", "NONE"))

        if len(t_rel) < 2:
            rospy.logwarn("[simple_mux] rec < 2 rows in plot horizon, skipping PNG")
            return

        # Phase shading: ACTIVE/POST/DONE (from launch_mode) + START (from sm_state).
        # Compute contiguous spans for each color.
        def _spans(marks, valid_set):
            out = []
            cur = None
            start_t = None
            for tr, m in zip(t_rel, marks):
                if m in valid_set and cur is None:
                    cur = m
                    start_t = tr
                elif (m not in valid_set or m != cur) and cur is not None:
                    out.append((cur, start_t, tr))
                    cur = None
                    if m in valid_set:
                        cur = m
                        start_t = tr
            if cur is not None:
                out.append((cur, start_t, t_rel[-1]))
            return out

        launch_spans = _spans(mode_marks, {"ACTIVE", "POST", "DONE"})
        start_spans = _spans(sm_marks, {"START"})

        def _shade(ax):
            colors_launch = {"ACTIVE": "#ffeeaa", "POST": "#cceeff", "DONE": "#eeeeee"}
            for m, s, e in launch_spans:
                ax.axvspan(s, e, alpha=0.35, color=colors_launch.get(m, "#ffeeaa"), zorder=0)
            for _, s, e in start_spans:
                ax.axvspan(s, e, alpha=0.25, color="#bbeeaa", zorder=0)

        tag_str = "+".join(tags) if tags else "NONE"
        title_base = "{}  [trig={}]".format(os.path.basename(csv_path), tag_str)
        base_no_ext = os.path.splitext(os.path.basename(csv_path))[0]

        # ----- electrical PNG: Currents / ERPM / Duty -----
        try:
            fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
            axes[0].plot(t_rel, target_I, label="mux_target_I (intent)",
                         linestyle="--", color="black")
            axes[0].plot(t_rel, vesc_cmd_I, label="vesc_cmd_current (cpp->driver)",
                         color="blue", linewidth=1.0, alpha=0.7)
            axes[0].plot(t_rel, I_act, label="I_motor (actual)",
                         color="red", linewidth=1.5)
            axes[0].plot(t_rel, I_in, label="I_input (battery)",
                         color="orange", alpha=0.6)
            axes[0].set_ylabel("Current [A]")
            axes[0].set_title(title_base + "  — electrical")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(loc="upper right", fontsize=8)
            _shade(axes[0])

            axes[1].plot(t_rel, ctrl_target_erpm,
                         label="ctrl_target_erpm (controller intent)",
                         linestyle="--", color="blue", alpha=0.8)
            axes[1].plot(t_rel, mux_target_erpm,
                         label="mux_target_erpm (cpp input, post-override)",
                         linestyle="--", color="purple", alpha=0.8)
            axes[1].plot(t_rel, erpm, label="erpm_actual",
                         color="red", linewidth=1.5)
            axes[1].set_ylabel("ERPM")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(loc="lower right", fontsize=8)
            _shade(axes[1])

            axes[2].plot(t_rel, duty, label="duty_cycle",
                         color="green", linewidth=1.5)
            axes[2].axhline(1.0, color="red", linestyle=":", alpha=0.5,
                            label="duty=1 saturate")
            axes[2].set_ylabel("Duty")
            axes[2].set_xlabel("t [s] (since trigger rising)")
            axes[2].set_ylim(-0.05, 1.1)
            axes[2].grid(True, alpha=0.3)
            axes[2].legend(loc="upper right", fontsize=8)
            _shade(axes[2])

            png_path = os.path.join(self._rec_plot_dir, base_no_ext + "_electrical.png")
            fig.tight_layout()
            fig.savefig(png_path, dpi=110)
            plt.close(fig)
            rospy.loginfo("[simple_mux] rec PNG saved: %s", png_path)
        except Exception as e:
            rospy.logwarn("[simple_mux] electrical PNG render failed: %s", e)

        # ----- mechanical PNG: Speed -----
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax.plot(t_rel, ctrl_v, label="ctrl_speed_cmd (controller intent)",
                    linestyle="--", color="blue", alpha=0.8)
            ax.plot(t_rel, mux_v, label="mux_speed_cmd (cpp input, post-override)",
                    linestyle="--", color="purple", alpha=0.8)
            ax.plot(t_rel, cur_v_vesc, label="cur_v_vesc (/vesc/odom)",
                    color="red", linewidth=1.5)
            # Plot fused only if any non-NaN
            if any(not math.isnan(v) for v in car_v):
                ax.plot(t_rel, car_v, label="car_state_v (fused)",
                        color="darkgreen", linewidth=1.5)
            ax.set_ylabel("Speed [m/s]")
            ax.set_xlabel("t [s] (since trigger rising)")
            ax.set_title(title_base + "  — mechanical")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="lower right", fontsize=8)
            _shade(ax)

            png_path = os.path.join(self._rec_plot_dir, base_no_ext + "_mechanical.png")
            fig.tight_layout()
            fig.savefig(png_path, dpi=110)
            plt.close(fig)
            rospy.loginfo("[simple_mux] rec PNG saved: %s", png_path)
        except Exception as e:
            rospy.logwarn("[simple_mux] mechanical PNG render failed: %s", e)

    def _rec_on_shutdown(self):
        # If trigger still active at shutdown, close + render so we don't lose the last session.
        if self._rec_csv_file is not None:
            rospy.loginfo("[simple_mux] rec: shutdown while session active — closing")
            self._rec_close_csv_and_render()



if __name__ == '__main__':
    simple_mux = SimpleMuxNode()
    rospy.on_shutdown(simple_mux._rec_on_shutdown)
    rospy.spin()
