#!/usr/bin/env python3
# Manual launch test rig with per-session CSV + auto PNG.
#
# What this node does:
#   * Replaces autonomy stack: publishes ackermann (steering=0, speed=rqt-tunable) at 50Hz
#     to /vesc/high_level/ackermann_cmd_mux/input/nav_1.
#   * Mirrors simple_mux launch debug + VESC sensor telemetry.
#   * Logs every 50Hz tick to CSV from RB rising (autodrive enter) to LB rising (human takeover).
#     Each RB->LB cycle = one CSV file.
#   * On CSV close, spawns a daemon thread that renders a PNG plot (first 2.5s) of:
#       - currents:  mux target_I  vs  actual current_motor   (key comparison)
#       - speeds:    cmd_speed_pub vs mux_speed_cmd vs actual v_mps
#       - duty:      duty_cycle    (saturation indicator)
#
# Folder layout (one session per node start):
#   launch_logs/session_<YYYYMMDD_HHMMSS>/
#       csv/  launch_<YYYYMMDD_HHMMSS>_seqNNN.csv
#       plot/ launch_<YYYYMMDD_HHMMSS>_seqNNN.png
#
# Usage flow:
#   1. roslaunch stack_master test_launch_control.launch
#   2. rqt -> /test_control_publisher : cmd_speed_mps (post-launch target)
#   3. (optional) A button -> arm launch (toggle)
#   4. RB -> autodrive ON; CSV opens.
#       If armed, simple_mux fires launch (constant launch_current_A for launch_t_total seconds).
#   5. LB -> human takeover; CSV closes; PNG is generated in background.
#   6. Repeat steps 3-5 to collect more samples; each cycle creates a new CSV+PNG.
#
# Safety:
#   * cmd_speed_mps>0 + RB will drive the motor even WITHOUT arming launch (speed mode).
#     Keep wheels off ground until verified.
#   * LB is wired to simple_mux humandrive path: instant abort.
import csv
import datetime as _dt
import json
import math
import os
import threading
import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Joy
from sensor_msgs.msg import Imu
from std_msgs.msg import Empty, Float64, String
from vesc_msgs.msg import VescStateStamped
from dynamic_reconfigure.server import Server as DynRecServer
from stack_master.cfg import TestCtrlConfig


PUB_TOPIC = "/vesc/high_level/ackermann_cmd_mux/input/nav_1"
CORE_TOPIC = "/vesc/sensors/core"
LAUNCH_DBG_TOPIC = "/launch_controller/debug"
JOY_TOPIC = "/joy"
# commanded current cpp publishes when jerk==512 path active. 0 (not published) in speed mode.
VESC_CMD_CURRENT_TOPIC = "/vesc/commands/motor/current"
# VESC built-in IMU. m/s^2 in standard frame. Use acc_x for longitudinal launch accel.
VESC_IMU_TOPIC = "/vesc/sensors/imu/raw"

# in the container, $HOME/catkin_ws/src/race_stack is the mounted repository
DEFAULT_LOG_ROOT = os.path.expanduser("~/catkin_ws/src/race_stack/launch_logs")

PLOT_HORIZON_SEC = 2.5  # how much of a CSV to render in the PNG
RB_BUTTON_DEFAULT = 5
LB_BUTTON_DEFAULT = 4


class TestControlPublisher:
    def __init__(self):
        rospy.init_node("test_control_publisher", anonymous=False)

        # rqt-tunable
        self.cmd_speed = 0.0
        self.cmd_steering = 0.0
        self.cmd_duration = 2.0  # seconds, overridden by dyn_reconfigure

        # VESC telemetry
        self.current_motor = 0.0
        self.current_input = 0.0
        self.duty_cycle = 0.0
        self.measured_erpm = 0.0
        self.voltage_input = 0.0
        self.temperature_pcb = 0.0
        # commanded current as cpp pushes to /vesc/commands/motor/current (0 when not in jerk==512 path)
        self.vesc_cmd_current = 0.0
        # VESC IMU (m/s^2). acc_x typically forward depending on mount; verify with test publish.
        self.imu_acc_x = 0.0
        self.imu_acc_y = 0.0
        self.imu_acc_z = 0.0

        # erpm -> m/s conversion (for log readability)
        self.s2e_gain = float(rospy.get_param("/vesc/speed_to_erpm_gain", 1875.0))
        self.s2e_offset = float(rospy.get_param("/vesc/speed_to_erpm_offset", 0.0))

        # Launch state mirrored from simple_mux debug
        self.launch_active = False
        self.launch_armed = False
        self.launch_t = 0.0
        self.launch_mode = "INIT"
        self.launch_jerk = 0
        self.launch_speed_cmd = 0.0
        self.launch_accel_cmd = 0.0
        self.launch_target_I = 0.0
        self.launch_window_until = rospy.Time(0)  # for high-freq print only

        # Joy edge tracking
        self.rb_button = int(rospy.get_param("~rb_button", RB_BUTTON_DEFAULT))
        self.lb_button = int(rospy.get_param("~lb_button", LB_BUTTON_DEFAULT))
        self._prev_rb = 0
        self._prev_lb = 0

        # CSV / session
        self.session_active = False  # True between RB rising and LB rising (or auto-cutoff)
        self._rb_rising_t = None     # ROS time when RB rising opened the session (for duration cutoff)
        self._launch_seq = 0
        self._csv_file = None
        self._csv_writer = None
        self._csv_path = None
        self._csv_lock = threading.Lock()

        # Per-session folder
        log_root = rospy.get_param("~log_dir", DEFAULT_LOG_ROOT)
        session_ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(log_root, "session_{}".format(session_ts))
        self.csv_dir = os.path.join(self.session_dir, "csv")
        self.plot_dir = os.path.join(self.session_dir, "plot")
        try:
            os.makedirs(self.csv_dir, exist_ok=True)
            os.makedirs(self.plot_dir, exist_ok=True)
            rospy.loginfo("[test_ctrl] session dir: %s", self.session_dir)
        except OSError as e:
            rospy.logwarn("[test_ctrl] cannot create session dirs (%s): %s -- CSV disabled", self.session_dir, e)
            self.csv_dir = None
            self.plot_dir = None

        # Publisher (replaces autonomy stack)
        self.pub = rospy.Publisher(PUB_TOPIC, AckermannDriveStamped, queue_size=10)
        # abort trigger for simple_mux (sent on duration cutoff to switch to humandrive idle)
        self.abort_pub = rospy.Publisher("/launch_controller/abort", Empty, queue_size=1)
        # Subscribers
        rospy.Subscriber(CORE_TOPIC, VescStateStamped, self._core_cb, queue_size=10)
        rospy.Subscriber(LAUNCH_DBG_TOPIC, String, self._launch_dbg_cb, queue_size=10)
        rospy.Subscriber(JOY_TOPIC, Joy, self._joy_cb, queue_size=10)
        rospy.Subscriber(VESC_CMD_CURRENT_TOPIC, Float64, self._vesc_cmd_current_cb, queue_size=10)
        rospy.Subscriber(VESC_IMU_TOPIC, Imu, self._vesc_imu_cb, queue_size=10)

        # rqt
        self.dyn_srv = DynRecServer(TestCtrlConfig, self._dyn_cb)

        # 50Hz tick (publish + log)
        rospy.Timer(rospy.Duration(1.0 / 50.0), self._tick)

        # Ensure CSV closes cleanly on shutdown
        rospy.on_shutdown(self._on_shutdown)

        rospy.loginfo("[test_ctrl] up. publishing %s @ 50Hz. WHEELS OFF GROUND until verified.", PUB_TOPIC)

    # ---------- callbacks ----------
    def _dyn_cb(self, config, level):
        self.cmd_speed = float(config.cmd_speed_mps)
        self.cmd_steering = float(config.cmd_steering_rad)
        self.cmd_duration = float(config.cmd_duration_sec)
        rospy.loginfo("[test_ctrl] tuned: speed=%.2fm/s steer=%.3frad duration=%.2fs",
                      self.cmd_speed, self.cmd_steering, self.cmd_duration)
        return config

    def _core_cb(self, msg):
        self.current_motor = float(msg.state.current_motor)
        self.current_input = float(msg.state.current_input)
        self.duty_cycle = float(msg.state.duty_cycle)
        self.measured_erpm = float(msg.state.speed)
        self.voltage_input = float(msg.state.voltage_input)
        self.temperature_pcb = float(msg.state.temperature_pcb)

    def _vesc_cmd_current_cb(self, msg):
        # cpp publishes here only when jerk==512 path runs. In speed mode this stays at last value.
        self.vesc_cmd_current = float(msg.data)

    def _vesc_imu_cb(self, msg):
        self.imu_acc_x = float(msg.linear_acceleration.x)
        self.imu_acc_y = float(msg.linear_acceleration.y)
        self.imu_acc_z = float(msg.linear_acceleration.z)

    def _launch_dbg_cb(self, msg):
        try:
            d = json.loads(msg.data)
        except Exception:
            return
        self.launch_armed = bool(d.get("armed", False))
        self.launch_active = bool(d.get("active", False))
        self.launch_t = float(d.get("t", 0.0))
        self.launch_mode = str(d.get("mode", "INIT"))
        self.launch_jerk = int(d.get("jerk", 0))
        self.launch_speed_cmd = float(d.get("speed_cmd", 0.0))
        self.launch_accel_cmd = float(d.get("accel_cmd", 0.0))
        self.launch_target_I = float(d.get("target_I_A", 0.0))
        if self.launch_active:
            # extend high-freq print window 0.5s past DONE
            self.launch_window_until = rospy.Time.now() + rospy.Duration(0.5)

    def _joy_cb(self, msg):
        if len(msg.buttons) <= max(self.rb_button, self.lb_button):
            return
        rb = int(msg.buttons[self.rb_button])
        lb = int(msg.buttons[self.lb_button])
        rb_rising = rb and not self._prev_rb
        lb_rising = lb and not self._prev_lb
        self._prev_rb = rb
        self._prev_lb = lb

        # RB rising -> open new CSV (start logging session). Idempotent if already active.
        if rb_rising and not self.session_active:
            self.session_active = True
            self._rb_rising_t = rospy.Time.now()
            self._open_csv()

        # LB rising -> close CSV + spawn PNG generation in background
        if lb_rising and self.session_active:
            self.session_active = False
            self._rb_rising_t = None
            self._close_csv_and_plot()

    # ---------- CSV ----------
    def _open_csv(self):
        if self.csv_dir is None:
            return
        with self._csv_lock:
            self._launch_seq += 1
            ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._csv_path = os.path.join(self.csv_dir,
                                          "launch_{}_seq{:03d}.csv".format(ts, self._launch_seq))
            try:
                self._csv_file = open(self._csv_path, "w", newline="")
                self._csv_writer = csv.writer(self._csv_file)
                self._csv_writer.writerow([
                    "wall_t",
                    "launch_t",
                    "active", "armed", "mode",
                    "cmd_speed_pub", "cmd_steer_pub",
                    "mux_jerk", "mux_speed_cmd", "mux_accel_cmd", "mux_target_I_A",
                    "vesc_cmd_current_A",
                    "I_motor_actual", "I_input_actual", "duty_actual",
                    "erpm_actual", "v_mps_actual", "V_input_actual",
                    "temperature_pcb",
                    "imu_acc_x", "imu_acc_y", "imu_acc_z",
                ])
                self._csv_file.flush()
                rospy.loginfo("[test_ctrl] CSV opened: %s", self._csv_path)
            except OSError as e:
                rospy.logwarn("[test_ctrl] CSV open failed: %s", e)
                self._csv_file = None
                self._csv_writer = None

    def _write_csv_row(self, now, v_mps):
        if self._csv_writer is None:
            return
        try:
            self._csv_writer.writerow([
                "{:.4f}".format(now.to_sec()),
                "{:.4f}".format(self.launch_t),
                int(self.launch_active),
                int(self.launch_armed),
                self.launch_mode,
                "{:.3f}".format(self.cmd_speed),
                "{:.4f}".format(self.cmd_steering),
                self.launch_jerk,
                "{:.3f}".format(self.launch_speed_cmd),
                "{:.4f}".format(self.launch_accel_cmd),
                "{:.2f}".format(self.launch_target_I),
                "{:.3f}".format(self.vesc_cmd_current),
                "{:.3f}".format(self.current_motor),
                "{:.3f}".format(self.current_input),
                "{:.4f}".format(self.duty_cycle),
                "{:.0f}".format(self.measured_erpm),
                "{:.3f}".format(v_mps),
                "{:.2f}".format(self.voltage_input),
                "{:.1f}".format(self.temperature_pcb),
                "{:.3f}".format(self.imu_acc_x),
                "{:.3f}".format(self.imu_acc_y),
                "{:.3f}".format(self.imu_acc_z),
            ])
        except Exception as e:
            rospy.logwarn_throttle(1.0, "[test_ctrl] CSV write error: %s", e)

    def _close_csv_and_plot(self):
        with self._csv_lock:
            if self._csv_file is None:
                return
            path = self._csv_path
            try:
                self._csv_file.flush()
                self._csv_file.close()
                rospy.loginfo("[test_ctrl] CSV closed: %s", path)
            except Exception as e:
                rospy.logwarn("[test_ctrl] CSV close error: %s", e)
            self._csv_file = None
            self._csv_writer = None
        # Off-thread PNG render so 50Hz tick is not blocked
        threading.Thread(target=self._render_png, args=(path,), daemon=True).start()

    def _render_png(self, csv_path):
        if csv_path is None or self.plot_dir is None:
            return
        try:
            # Defer heavy imports until needed
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import csv as _csv
        except Exception as e:
            rospy.logwarn("[test_ctrl] matplotlib import failed: %s", e)
            return

        try:
            rows = []
            with open(csv_path, "r") as f:
                reader = _csv.DictReader(f)
                for r in reader:
                    rows.append(r)
            if not rows:
                rospy.logwarn("[test_ctrl] empty CSV, skipping PNG: %s", csv_path)
                return
            t0 = float(rows[0]["wall_t"])
            t_rel, target_I, vesc_cmd_I, I_act, I_in, duty = [], [], [], [], [], []
            cmd_v, mux_v, v_act, motor_v = [], [], [], []
            erpm_act = []
            acc_x, acc_y, acc_z = [], [], []
            mode_marks = []
            s2e = self.s2e_gain if abs(self.s2e_gain) > 1e-6 else 1.0
            for r in rows:
                tr = float(r["wall_t"]) - t0
                if tr > PLOT_HORIZON_SEC:
                    break
                t_rel.append(tr)
                target_I.append(float(r["mux_target_I_A"]))
                vesc_cmd_I.append(float(r.get("vesc_cmd_current_A", 0.0)))
                I_act.append(float(r["I_motor_actual"]))
                I_in.append(float(r["I_input_actual"]))
                duty.append(float(r["duty_actual"]))
                cmd_v.append(float(r["cmd_speed_pub"]))
                mux_v.append(float(r["mux_speed_cmd"]))
                v_act.append(float(r["v_mps_actual"]))
                erpm = float(r["erpm_actual"])
                erpm_act.append(erpm)
                m_v = (erpm - self.s2e_offset) / s2e
                motor_v.append(m_v)
                acc_x.append(float(r.get("imu_acc_x", 0.0)))
                acc_y.append(float(r.get("imu_acc_y", 0.0)))
                acc_z.append(float(r.get("imu_acc_z", 0.0)))
                mode_marks.append(r.get("mode", "INIT"))

            # IMU forward axis auto-select:
            #   1) Subtract first-100ms mean from each axis = mount-tilt bias removal.
            #      (NOT gravity removal -- gravity is on the vertical axis whose RANGE is small,
            #       so it is not selected as forward anyway. The bias term only zeroes a small
            #       constant offset on horizontal axes when the IMU is mounted slightly tilted.)
            #   2) Pick axis with the largest range over the plotted window = the axis where
            #      acceleration is actually changing (longitudinal during launch).
            #   3) If its mean during ACTIVE phase is negative, flip sign so forward = positive.
            n_bias = max(1, sum(1 for tr in t_rel if tr < 0.10))
            bx = sum(acc_x[:n_bias]) / n_bias
            by = sum(acc_y[:n_bias]) / n_bias
            bz = sum(acc_z[:n_bias]) / n_bias
            ax_c = [a - bx for a in acc_x]
            ay_c = [a - by for a in acc_y]
            az_c = [a - bz for a in acc_z]

            def _rng(arr):
                return (max(arr) - min(arr)) if arr else 0.0

            ranges = {"acc_x": _rng(ax_c), "acc_y": _rng(ay_c), "acc_z": _rng(az_c)}
            axis_data = {"acc_x": ax_c, "acc_y": ay_c, "acc_z": az_c}
            fwd_axis = max(ranges, key=lambda k: ranges[k])
            fwd_raw = axis_data[fwd_axis]

            # sign: prefer positive during ACTIVE
            active_vals = [v for v, m in zip(fwd_raw, mode_marks) if m == "ACTIVE"]
            sign_flip = False
            if active_vals and (sum(active_vals) / len(active_vals)) < 0.0:
                sign_flip = True
                fwd_raw = [-v for v in fwd_raw]

            # other two axes (raw bias-corrected, for context only)
            other_axes = [k for k in ("acc_x", "acc_y", "acc_z") if k != fwd_axis]
            other1_data = axis_data[other_axes[0]]
            other2_data = axis_data[other_axes[1]]

            if len(t_rel) < 2:
                rospy.logwarn("[test_ctrl] < 2 rows within %ss, skipping PNG: %s", PLOT_HORIZON_SEC, csv_path)
                return

            # Find ACTIVE/POST boundaries to shade phases
            phase_changes = []
            cur = mode_marks[0]
            phase_start = t_rel[0]
            for tr, m in zip(t_rel, mode_marks):
                if m != cur:
                    phase_changes.append((cur, phase_start, tr))
                    cur = m
                    phase_start = tr
            phase_changes.append((cur, phase_start, t_rel[-1]))

            def _shade(ax):
                colors = {"ACTIVE": "#ffeeaa", "POST": "#cceeff", "DONE": "#eeeeee"}
                for m, s, e in phase_changes:
                    if m in colors:
                        ax.axvspan(s, e, alpha=0.4, color=colors[m], zorder=0)

            # Peak forward acceleration time (using auto-selected, sign-corrected forward axis)
            try:
                peak_idx = max(range(len(fwd_raw)), key=lambda i: fwd_raw[i])
                peak_t = t_rel[peak_idx]
                peak_ax = fwd_raw[peak_idx]
            except ValueError:
                peak_t, peak_ax = None, None

            fig, axes = plt.subplots(5, 1, figsize=(10, 14), sharex=True)
            # Plot 1: currents — intent (mux target) vs cpp commanded vs actual + battery
            axes[0].plot(t_rel, target_I, label="mux target_I (intent)", linestyle="--", color="black")
            axes[0].plot(t_rel, vesc_cmd_I, label="vesc cmd_current (cpp->driver)", color="blue", linewidth=1.0, alpha=0.7)
            axes[0].plot(t_rel, I_act, label="I_motor (actual)", color="red", linewidth=1.5)
            axes[0].plot(t_rel, I_in, label="I_input (battery)", color="orange", alpha=0.6)
            axes[0].set_ylabel("Current [A]")
            axes[0].set_title(os.path.basename(csv_path))
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(loc="upper right", fontsize=8)
            _shade(axes[0])

            # Plot 2: speeds (m/s)
            axes[1].plot(t_rel, cmd_v, label="cmd_speed (publisher)", linestyle="--", color="blue", alpha=0.7)
            axes[1].plot(t_rel, mux_v, label="mux_speed_cmd (-> vesc)", linestyle="-", color="purple", linewidth=1.5)
            axes[1].plot(t_rel, v_act, label="v_odom (vesc-derived)", color="red", linewidth=1.5)
            axes[1].plot(t_rel, motor_v, label="v_motor (from erpm)", color="darkgreen", linewidth=1.0, alpha=0.6)
            axes[1].set_ylabel("Speed [m/s]")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(loc="lower right", fontsize=8)
            _shade(axes[1])

            # Plot 3: ERPM (motor RPM domain)
            axes[2].plot(t_rel, erpm_act, label="erpm_actual", color="darkgreen", linewidth=1.5)
            axes[2].set_ylabel("ERPM")
            axes[2].grid(True, alpha=0.3)
            axes[2].legend(loc="lower right", fontsize=8)
            _shade(axes[2])

            # Plot 4: IMU accel (m/s²). Forward axis auto-selected (largest range).
            #   bias-corrected with first 100ms, sign flipped if mean(ACTIVE)<0.
            fwd_label = "imu {} (forward, auto{})".format(fwd_axis, ", flipped" if sign_flip else "")
            axes[3].plot(t_rel, fwd_raw, label=fwd_label, color="darkred", linewidth=1.5)
            axes[3].plot(t_rel, other1_data, label="imu {} (other)".format(other_axes[0]),
                         color="gray", linewidth=1.0, alpha=0.4)
            axes[3].plot(t_rel, other2_data, label="imu {} (other)".format(other_axes[1]),
                         color="lightgray", linewidth=1.0, alpha=0.4)
            axes[3].axhline(0.0, color="black", linestyle="-", alpha=0.2)
            if peak_t is not None:
                axes[3].axvline(peak_t, color="darkred", linestyle=":", alpha=0.6)
                axes[3].annotate(
                    "peak ax={:.2f}m/s² @ t={:.2f}s".format(peak_ax, peak_t),
                    xy=(peak_t, peak_ax),
                    xytext=(peak_t + 0.05, peak_ax),
                    fontsize=8, color="darkred",
                )
            axes[3].set_ylabel("Accel [m/s²]\n(bias-corrected)")
            axes[3].grid(True, alpha=0.3)
            axes[3].legend(loc="upper right", fontsize=8)
            _shade(axes[3])

            # Plot 5: duty
            axes[4].plot(t_rel, duty, label="duty_cycle", color="green", linewidth=1.5)
            axes[4].axhline(1.0, color="red", linestyle=":", alpha=0.5, label="duty=1 saturate")
            axes[4].set_ylabel("Duty")
            axes[4].set_xlabel("t [s] (since RB)")
            axes[4].set_ylim(-0.05, 1.1)
            axes[4].grid(True, alpha=0.3)
            axes[4].legend(loc="upper right", fontsize=8)
            _shade(axes[4])

            png_name = os.path.splitext(os.path.basename(csv_path))[0] + ".png"
            png_path = os.path.join(self.plot_dir, png_name)
            fig.tight_layout()
            fig.savefig(png_path, dpi=110)
            plt.close(fig)
            rospy.loginfo("[test_ctrl] PNG saved: %s", png_path)
        except Exception as e:
            rospy.logwarn("[test_ctrl] PNG render failed: %s", e)

    def _on_shutdown(self):
        if self.session_active:
            rospy.loginfo("[test_ctrl] shutting down with session still active -- closing CSV")
            self.session_active = False
            self._close_csv_and_plot()

    # ---------- helpers ----------
    def _measured_v_mps(self):
        if abs(self.s2e_gain) < 1e-6:
            return 0.0
        return (self.measured_erpm - self.s2e_offset) / self.s2e_gain

    # ---------- main 50Hz tick ----------
    def _tick(self, _evt):
        now = rospy.Time.now()

        # duration cutoff -- after cmd_duration seconds since RB rising,
        # publish abort to simple_mux so it switches to humandrive idle (same as LB), then
        # auto-close CSV/PNG. Once aborted, simple_mux ignores our subsequent ackermann anyway.
        speed_to_pub = self.cmd_speed
        if self.session_active and self._rb_rising_t is not None:
            elapsed = (now - self._rb_rising_t).to_sec()
            if elapsed > self.cmd_duration:
                speed_to_pub = 0.0
                self.session_active = False
                rb_t_for_log = self._rb_rising_t
                self._rb_rising_t = None
                rospy.loginfo("[test_ctrl] duration cutoff at %.2fs (RB at %.3f) -> abort + auto-close",
                              elapsed, rb_t_for_log.to_sec())
                # Tell simple_mux to drop autodrive (same as LB press)
                try:
                    self.abort_pub.publish(Empty())
                except Exception as e:
                    rospy.logwarn("[test_ctrl] abort publish failed: %s", e)
                self._close_csv_and_plot()

        # 1. Publish ackermann (50Hz, constant unless rqt-changed)
        m = AckermannDriveStamped()
        m.header.stamp = now
        m.drive.steering_angle = self.cmd_steering
        m.drive.speed = speed_to_pub
        m.drive.acceleration = 0.0
        m.drive.jerk = 0.0
        self.pub.publish(m)

        # 2. CSV row (every tick during session)
        v_mps = self._measured_v_mps()
        if self.session_active:
            self._write_csv_row(now, v_mps)

        # 3. Print: 10Hz inside print-window (launch + 0.5s post), 1Hz idle
        in_print_window = now < self.launch_window_until
        if in_print_window:
            tag = "[LAUNCH]" if self.launch_active else "[POST  ]"
            rospy.loginfo_throttle(
                0.10,
                "%s t=%.2f mode=%s cmd_pub=%.2fm/s | mux: jerk=%d sp=%.2f acc=%.3f tgtI=%5.2fA "
                "| ACTUAL: I=%6.2fA Iin=%5.2fA duty=%5.3f erpm=%6.0f v=%5.2fm/s V=%4.1fV",
                tag, self.launch_t, self.launch_mode, self.cmd_speed,
                self.launch_jerk, self.launch_speed_cmd, self.launch_accel_cmd, self.launch_target_I,
                self.current_motor, self.current_input, self.duty_cycle,
                self.measured_erpm, v_mps, self.voltage_input,
            )
        elif self.session_active:
            rospy.loginfo_throttle(
                0.5,
                "[REC   ] cmd_pub=%.2fm/s | I=%6.2fA duty=%5.3f erpm=%6.0f v=%5.2fm/s V=%4.1fV armed=%s",
                self.cmd_speed,
                self.current_motor, self.duty_cycle,
                self.measured_erpm, v_mps, self.voltage_input,
                self.launch_armed,
            )
        else:
            rospy.loginfo_throttle(
                1.0,
                "[idle  ] cmd_pub=%.2fm/s | I=%6.2fA duty=%5.3f erpm=%6.0f v=%5.2fm/s V=%4.1fV armed=%s",
                self.cmd_speed,
                self.current_motor, self.duty_cycle,
                self.measured_erpm, v_mps, self.voltage_input,
                self.launch_armed,
            )


if __name__ == "__main__":
    try:
        TestControlPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
