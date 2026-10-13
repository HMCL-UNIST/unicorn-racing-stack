#!/usr/bin/env python3
# servo calibration recorder.
# Strategy: piggy-back on existing ackermann pipeline with the algebraic trick:
#   delta_cmd = (target_servo - offset) / gain
# so the linear map produces the desired raw servo value.
# Requires linear mapping active; 3d_base_system.launch forces
# /vesc/enable_nonlinear_servo_gain=false when calibration:=true.
#
# Flow (step-by-step, operator-paced):
#   1. Operator sees "[WAITING] Press RB for step N/M: delta=+0.15" in terminal
#   2. Operator uses LB + joy to position the car (enough room)
#   3. Operator presses RB -> recorder commands the servo + constant speed
#      warmup (settling) -> record (bag) -> stop -> back to step 1 for next point
#   4. After all points (or operator abort), fitter runs automatically.
#   5. Node stays alive (rospy.spin) so roslaunch and low_level keep running.

import os
import sys
import signal
import subprocess
import threading
import numpy as np
import rospy
import rosbag
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from sensor_msgs.msg import Joy


NAV_TOPIC   = "/vesc/high_level/ackermann_cmd_mux/input/nav_1"
ODOM_TOPIC  = "/car_state/odom"
SERVO_TOPIC = "/vesc/commands/servo/position"
JOY_TOPIC   = "/joy"

PUBLISH_RATE_HZ = 50.0


class ServoCalibrationRecorder:
    def __init__(self):
        rospy.init_node("servo_calibration_recorder")

        # --- safety: refuse to run if nonlinear mapping is active ---
        if rospy.get_param("/vesc/enable_nonlinear_servo_gain", False):
            rospy.logfatal("enable_nonlinear_servo_gain is true. Calibration requires the "
                           "linear map to be active for the algebraic trick to work. "
                           "Re-run with calibration:=true (which forces it off) or set it "
                           "false in vesc.yaml.")
            sys.exit(2)

        # --- params ---
        self.racecar    = rospy.get_param("~racecar_version")
        self.delta_max  = float(rospy.get_param("~delta_max",  0.6))
        self.delta_step = float(rospy.get_param("~delta_step", 0.05))
        self.v_const    = float(rospy.get_param("~v_const",    1.0))
        self.warmup_sec = float(rospy.get_param("~warmup_sec", 2.5))
        self.record_sec = float(rospy.get_param("~record_sec", 5.0))
        self.bag_dir    = rospy.get_param("~bag_dir")
        self.vesc_yaml  = rospy.get_param("~vesc_yaml")
        self.fitter     = rospy.get_param("~fitter_script")

        self.gain   = float(rospy.get_param("/vesc/steering_angle_to_servo_gain"))
        self.offset = float(rospy.get_param("/vesc/steering_angle_to_servo_offset"))

        # usable servo range
        vesc_min = float(rospy.get_param("/vesc/vesc_driver/servo_min", 0.0))
        vesc_max = float(rospy.get_param("/vesc/vesc_driver/servo_max", 1.0))
        self.servo_clip_min = max(0.02, vesc_min)
        self.servo_clip_max = min(0.98, vesc_max)

        os.makedirs(self.bag_dir, exist_ok=True)

        # --- joy state ---
        self._rb_pressed = threading.Event()   # set once when RB pressed, cleared after consumed
        self._lb_pressed = False               # true while LB held
        rospy.Subscriber(JOY_TOPIC, Joy, self._joy_cb, queue_size=10)

        # --- publisher ---
        self.pub = rospy.Publisher(NAV_TOPIC, AckermannDriveStamped, queue_size=10)

        rospy.loginfo("ServoCalibrationRecorder: waiting 1.0s for subscribers...")
        rospy.sleep(1.0)

    def _joy_cb(self, msg):
        if len(msg.buttons) < 6:
            return
        # LB (button 4) = manual override
        self._lb_pressed = bool(msg.buttons[4])
        # RB (button 5) = advance to next step (edge trigger)
        if msg.buttons[5]:
            self._rb_pressed.set()

    def _wait_for_rb(self, prompt_msg):
        """Block until operator presses RB. Clears the event before waiting."""
        self._rb_pressed.clear()
        rospy.loginfo(prompt_msg)
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self._rb_pressed.is_set():
                self._rb_pressed.clear()
                return True
            rate.sleep()
        return False

    def _delta_to_servo(self, delta_rad):
        return self.offset + self.gain * delta_rad

    def _servo_to_delta(self, target_servo):
        return (target_servo - self.offset) / self.gain

    def _build_sweep(self):
        # sweep one side fully then the other — operator only turns in one direction
        # at a time, no back-and-forth. Order: 0 -> +step -> ... -> +max -> -step -> ... -> -max
        n = int(np.floor(self.delta_max / self.delta_step + 1e-9))
        deltas = [0.0]
        for i in range(1, n + 1):
            deltas.append(round(+i * self.delta_step, 4))
        for i in range(1, n + 1):
            deltas.append(round(-i * self.delta_step, 4))
        filtered = []
        for d in deltas:
            s = self._delta_to_servo(d)
            if self.servo_clip_min <= s <= self.servo_clip_max:
                filtered.append(d)
            else:
                rospy.logwarn("Skipping delta=%.3f (raw servo %.3f outside [%.3f, %.3f])",
                              d, s, self.servo_clip_min, self.servo_clip_max)
        return filtered

    def _publish_cmd(self, delta_cmd, speed):
        msg = AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.drive.steering_angle = delta_cmd
        msg.drive.speed = speed
        self.pub.publish(msg)

    def _sweep_point(self, delta_target):
        target_servo = self._delta_to_servo(delta_target)
        delta_cmd    = self._servo_to_delta(target_servo)

        rate = rospy.Rate(PUBLISH_RATE_HZ)

        # --- warmup phase (operator should keep RB held or have just pressed it) ---
        rospy.loginfo("[sweep] WARMUP delta=%+.3f rad  target_servo=%.3f  (%.1fs)",
                      delta_target, target_servo, self.warmup_sec)
        t_end = rospy.Time.now() + rospy.Duration(self.warmup_sec)
        while not rospy.is_shutdown() and rospy.Time.now() < t_end:
            if self._lb_pressed:
                rospy.logwarn("[sweep] LB pressed during warmup — aborting this point.")
                return False
            self._publish_cmd(delta_cmd, self.v_const)
            rate.sleep()

        # --- record phase ---
        # trial-aware naming. If a bag for this delta already exists,
        # increment trial number so repeated measurements are kept, not overwritten.
        trial = 0
        while True:
            if trial == 0:
                bag_name = "servo_calib_delta_{:+.2f}.bag".format(delta_target)
            else:
                bag_name = "servo_calib_delta_{:+.2f}_t{:d}.bag".format(delta_target, trial)
            if not os.path.exists(os.path.join(self.bag_dir, bag_name)):
                break
            trial += 1
        bag_path = os.path.join(self.bag_dir, bag_name)
        bag = rosbag.Bag(bag_path, "w")
        bag_lock = threading.Lock()
        subs = []

        def make_cb(topic):
            def _cb(msg):
                with bag_lock:
                    try:
                        bag.write(topic, msg)
                    except Exception:
                        pass
            return _cb

        # only record odom + servo. NAV_TOPIC excluded because its AckermannDriveStamped
        # caused deserialization errors in bags (thread-safety issue), and fitter doesn't need it —
        # target delta is derived from the bag filename.
        subs.append(rospy.Subscriber(ODOM_TOPIC,  Odometry, make_cb(ODOM_TOPIC),  queue_size=100))
        subs.append(rospy.Subscriber(SERVO_TOPIC, Float64,  make_cb(SERVO_TOPIC), queue_size=100))

        rospy.loginfo("[sweep] RECORDING delta=%+.3f  (%.1fs)", delta_target, self.record_sec)
        t_end = rospy.Time.now() + rospy.Duration(self.record_sec)
        aborted = False
        while not rospy.is_shutdown() and rospy.Time.now() < t_end:
            if self._lb_pressed:
                aborted = True
                break
            self._publish_cmd(delta_cmd, self.v_const)
            rate.sleep()

        for s in subs:
            s.unregister()
        bag.close()

        if aborted:
            rospy.logwarn("[sweep] delta=%+.3f aborted by LB.", delta_target)
            return False

        # validate recorded bag — if odom samples or duration is too low,
        # the data is unusable (corruption, bad timing, etc). Treat as failed so
        # the run loop retries this point instead of silently skipping it.
        # Duration threshold is relative to record_sec with a small tolerance:
        # first/last odom bag timestamps never span the full record window
        # (subscribe latency + rate.sleep gap), so a hard 5.0s threshold rejected
        # every bag when record_sec itself was 5.0.
        MIN_VALID_ODOM  = 200   # at ~90 Hz odom, 200 ≈ 2.2s — well above trim margins
        MIN_VALID_SECS  = max(1.0, self.record_sec - 0.5)
        n_odom = 0
        t_first, t_last = None, None
        try:
            check_bag = rosbag.Bag(bag_path, "r")
            for topic, msg, t in check_bag.read_messages(topics=[ODOM_TOPIC]):
                n_odom += 1
                ts = t.to_sec()
                if t_first is None:
                    t_first = ts
                t_last = ts
            check_bag.close()
        except Exception as e:
            rospy.logwarn("[sweep] delta=%+.3f bag read-back error: %s", delta_target, e)
            return False

        duration = (t_last - t_first) if (t_first is not None and t_last is not None) else 0.0
        if n_odom < MIN_VALID_ODOM or duration < MIN_VALID_SECS:
            rospy.logwarn("[sweep] delta=%+.3f INVALID: %d odom samples, %.1fs duration "
                          "(need >=%d samples, >=%.1fs). Will retry.",
                          delta_target, n_odom, duration, MIN_VALID_ODOM, MIN_VALID_SECS)
            return False

        rospy.loginfo("[sweep] delta=%+.3f OK (%d odom, %.1fs) -> %s",
                      delta_target, n_odom, duration, bag_path)
        return True

    def run(self):
        deltas = self._build_sweep()
        total = len(deltas)
        rospy.loginfo("Sweep plan: %d points, delta range [%+.2f, %+.2f], step %.2f, v=%.2f m/s",
                      total, min(deltas), max(deltas), self.delta_step, self.v_const)
        rospy.loginfo("=== Operator controls ===")
        rospy.loginfo("  RB (button 5) = proceed to next step")
        rospy.loginfo("  LB (button 4) = take manual control, abort current step")
        rospy.loginfo("  After LB abort: reposition with joy, then RB to retry")
        rospy.loginfo("=========================")

        ok_points = 0
        i = 0
        while i < total and not rospy.is_shutdown():
            d = deltas[i]
            prompt = "[WAITING] Press RB for step %d/%d: delta=%+.3f rad  (servo=%.3f)" % (
                i + 1, total, d, self._delta_to_servo(d))
            if not self._wait_for_rb(prompt):
                break  # shutdown

            success = self._sweep_point(d)
            if success:
                ok_points += 1
                i += 1  # advance to next point
            else:
                # LB abort — stay on the same point, operator repositions, press RB to retry
                rospy.loginfo("[RETRY] Same point delta=%+.3f will be re-attempted on next RB.", d)
                # don't increment i

        rospy.loginfo("Sweep finished. %d/%d points recorded.", ok_points, total)

        if ok_points < 5:
            rospy.logwarn("Only %d points — need >=5 for fitting. Fit skipped.", ok_points)
        else:
            rospy.loginfo("Running fitter...")
            cmd = [
                sys.executable,
                self.fitter,
                "--bag-dir",  self.bag_dir,
                "--vesc-yaml", self.vesc_yaml,
                "--poly-degree", str(int(rospy.get_param("~poly_degree", 3))),
                "--wheelbase", str(float(rospy.get_param("/vesc/wheelbase", 0.33))),
            ]
            rospy.loginfo("fit cmd: %s", " ".join(cmd))
            rc = subprocess.call(cmd)
            if rc == 0:
                rospy.loginfo("Fitter done. vesc.yaml updated. Set enable_nonlinear_servo_gain: true "
                              "in vesc.yaml to activate for normal runs.")
            else:
                rospy.logerr("Fitter exited with code %d. vesc.yaml NOT updated.", rc)

        rospy.loginfo("Calibration node staying alive (low_level still running). Ctrl+C to exit.")
        rospy.spin()


def _install_sigint():
    signal.signal(signal.SIGINT, lambda *a: rospy.signal_shutdown("sigint"))


if __name__ == "__main__":
    _install_sigint()
    try:
        ServoCalibrationRecorder().run()
    except rospy.ROSInterruptException:
        pass
