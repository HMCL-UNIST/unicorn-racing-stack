#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ELRS Joy Node (ROS1)
- Reads CRSF packets from ELRS receiver via USB-TTL serial
- Publishes sensor_msgs/Joy topic (Xbox-compatible layout)
- No CRC check - uses channel value range validation instead
  (for USB-TTL chips that can't hit exact 420000 baud)
"""

import rospy
from sensor_msgs.msg import Joy
from std_msgs.msg import Int32MultiArray
import serial
import time


class ELRSJoyNode:
    CRSF_SYNC = 0xC8
    CRSF_FRAMETYPE_RC_CHANNELS = 0x16
    CRSF_NUM_CHANNELS = 16

    # Valid channel range (with margin)
    CH_MIN = 100
    CH_MAX = 1900
    CH_MID = 992

    # For normalization
    NORM_MIN = 172
    NORM_MAX = 1811

    def __init__(self):
        rospy.init_node('elrs_joy_node', anonymous=False)

        self.port = rospy.get_param('~port', '/dev/ttyUSB0')
        self.baud_rate = rospy.get_param('~baud_rate', 416666)
        self.frame_id = rospy.get_param('~frame_id', 'elrs_joy')
        self.publish_rate = rospy.get_param('~publish_rate', 100)

        # Xbox-compatible Joy message layout
        # axes_map: {joy_index: crsf_channel}  buttons_map: {joy_index: crsf_channel}
        self.num_axes = rospy.get_param('~num_axes', 8)
        self.num_buttons = rospy.get_param('~num_buttons', 11)
        self.axes_joy_indices = rospy.get_param('~axes_joy_indices', [1, 3])
        self.axes_crsf_channels = rospy.get_param('~axes_crsf_channels', [0, 2])
        self.button_joy_indices = rospy.get_param('~button_joy_indices', [4, 5])
        self.button_crsf_channels = rospy.get_param('~button_crsf_channels', [5, 6])
        # per-button polarity invert (1 = pressed when value > threshold)
        self.button_invert = rospy.get_param('~button_invert', [0, 0])
        self.button_threshold = rospy.get_param('~button_threshold', 992)
        self.axes_invert = rospy.get_param('~axes_invert', [1.0, 1.0])
        # per-axis (min, mid, max) calibration in CRSF raw units. Each
        # entry is for the corresponding axes_crsf_channels[i]. Used by
        # normalize_axis to map raw -> [-1, +1] with asymmetric stops so
        # that hitting the real mechanical max yields exactly +1.0 even
        # when the transmitter end-point isn't the canonical 172/1811.
        # Falls back to (172, 992, 1811) per axis if length mismatches.
        self.axes_cal_min = rospy.get_param('~axes_cal_min', [172, 172])
        self.axes_cal_mid = rospy.get_param('~axes_cal_mid', [992, 992])
        self.axes_cal_max = rospy.get_param('~axes_cal_max', [1811, 1811])
        self.deadzone = rospy.get_param('~deadzone', 0.05)
        self.failsafe_timeout = rospy.get_param('~failsafe_timeout', 2.0)

        self.joy_pub = rospy.Publisher('joy', Joy, queue_size=10)
        # raw CRSF channels for live debug (Int32MultiArray of 16 ch values)
        self.debug_ch_pub = rospy.Publisher('~debug_channels', Int32MultiArray, queue_size=10)

        self.channels = [self.CH_MID] * self.CRSF_NUM_CHANNELS
        self.last_valid_time = time.time()
        self.connected = False
        self.serial_port = None
        self.buffer = bytearray()

        # Stats (print once then stop)
        self.accept_count = 0
        self.reject_count = 0
        self.last_stats_time = time.time()
        self.stats_printed = False

        # settling window after self.connected flips False -> True (i.e. first
        # valid packet following a gap: USB EIO reconnect, failsafe recovery,
        # or initial startup). For settling_sec we keep parsing into
        # self.channels but DO NOT publish to /joy. This drops any post-
        # reconnect burst of bit-corrupted-but-validation-passing frames before
        # they can reach simple_mux and flip buttons[4] (LB) into a fake
        # humandrive switch — the path that breaks autodrive_latched mid-run
        # when the car jolts over a bridge and the USB blips.
        self.settling_sec = float(rospy.get_param('~settling_sec', 0.2))
        self.settling_until = 0.0  # epoch seconds; publish when time.time() >= this

        # LB-specific safety guard. LB (humandrive takeover) is the single
        # most dangerous bit: if it spuriously flips to 1 mid-autodrive,
        # simple_mux drops the autodrive latch and pushes raw stick value
        # to VESC — observed once as a sudden full-throttle event after a
        # bump shook the USB-TTL.
        # LB is a 3-position switch with healthy resting values:
        #   pos1 ≈ 191  (pressed / humandrive ON)
        #   pos2 ≈ 992  (mid, IDLE — user's "released")
        #   pos3 ≈ 1792 (high, also treated as released)
        # Any sample outside these three bands is untrusted (bit
        # corruption / transient) and lb_state is held. Asymmetric
        # debounce: 0 -> 1 (humandrive entry) requires N consecutive
        # trusted "pressed" samples — this is the dangerous direction.
        # 1 -> 0 (release) commits immediately on first trusted
        # "released" sample so the user is never trapped in humandrive.
        # Released = pos2 band OR pos3 band; mid-band gaps (between
        # positions) remain untrusted.
        self.lb_pressed_max  = int(rospy.get_param('~lb_pressed_max',  350))
        self.lb_idle_min     = int(rospy.get_param('~lb_idle_min',     700))
        self.lb_idle_max     = int(rospy.get_param('~lb_idle_max',    1300))
        self.lb_released_min = int(rospy.get_param('~lb_released_min', 1600))
        self.lb_debounce_frames = int(rospy.get_param('~lb_debounce_frames', 5))
        # State held across publish_joy calls
        self.lb_state = 0          # last published LB button value (0=released, 1=pressed)
        self.lb_pending = 0        # candidate value currently accumulating
        self.lb_pending_count = 0  # consecutive trusted samples agreeing with lb_pending

        # A button (joy_idx==0, momentary/latched switch on CH7, invert=1)
        # also gets an asymmetric N-frame debounce so a transient high
        # sample cannot fake an A press. 0 -> 1 requires N consecutive
        # "pressed" samples; 1 -> 0 commits immediately on first
        # "released" sample (user is never trapped in a stuck press).
        self.a_debounce_frames = int(rospy.get_param('~a_debounce_frames', 5))
        self.a_state = 0
        self.a_pending_count = 0

    def normalize_axis(self, value, cal_min=None, cal_mid=None, cal_max=None):
        # asymmetric normalization around per-axis calibrated mid.
        # below mid -> scale by (mid - min); above mid -> scale by
        # (max - mid). Guarantees full ±1.0 reach at each side's real
        # end-point, independent of transmitter end-point asymmetry.
        if cal_min is None:
            cal_min = self.NORM_MIN
        if cal_mid is None:
            cal_mid = self.CH_MID
        if cal_max is None:
            cal_max = self.NORM_MAX
        if value >= cal_mid:
            span = cal_max - cal_mid
            normalized = (value - cal_mid) / span if span > 0 else 0.0
        else:
            span = cal_mid - cal_min
            normalized = (value - cal_mid) / span if span > 0 else 0.0
        normalized = max(-1.0, min(1.0, normalized))
        if abs(normalized) < self.deadzone:
            normalized = 0.0
        return normalized

    def channel_to_button(self, value, invert=0):
        # invert=1 means "pressed when value > threshold" (for CH polarity flip)
        if invert:
            return 1 if value > self.button_threshold else 0
        return 1 if value < self.button_threshold else 0

    def validate_channels(self, channels):
        """Validate by checking channel values are in sane range"""
        # At least first 4 channels must be in valid range
        for i in range(min(4, len(channels))):
            if channels[i] < self.CH_MIN or channels[i] > self.CH_MAX:
                return False

        # Must have some variation (not all identical)
        if len(set(channels[:4])) < 2:
            # Exception: all centered is ok (sticks neutral)
            if not all(abs(ch - self.CH_MID) < 50 for ch in channels[:4]):
                return False

        return True

    def parse_rc_channels(self, payload):
        if len(payload) < 22:
            return False

        channels = []
        bit_offset = 0
        for i in range(self.CRSF_NUM_CHANNELS):
            byte_offset = bit_offset // 8
            bit_shift = bit_offset % 8

            if byte_offset + 1 < len(payload):
                value = payload[byte_offset] >> bit_shift
                bits_from_first = 8 - bit_shift
                if bits_from_first < 11 and byte_offset + 1 < len(payload):
                    value |= payload[byte_offset + 1] << bits_from_first
                    bits_from_second = 11 - bits_from_first
                    if bits_from_second > 8 and byte_offset + 2 < len(payload):
                        value |= payload[byte_offset + 2] << (bits_from_first + 8)
                value &= 0x7FF
                channels.append(value)
            else:
                channels.append(self.CH_MID)

            bit_offset += 11

        if self.validate_channels(channels):
            self.channels = channels
            self.last_valid_time = time.time()
            self.accept_count += 1
            return True
        else:
            self.reject_count += 1
            return False

    def parse_crsf_frame(self):
        while len(self.buffer) > 2:
            # Find sync byte
            sync_idx = -1
            for i in range(len(self.buffer)):
                if self.buffer[i] == self.CRSF_SYNC:
                    sync_idx = i
                    break

            if sync_idx == -1:
                self.buffer.clear()
                return

            if sync_idx > 0:
                self.buffer = self.buffer[sync_idx:]

            if len(self.buffer) < 3:
                return

            frame_length = self.buffer[1]

            # RC channels frame: length should be 24 (type + 22 payload + crc)
            if frame_length < 2 or frame_length > 64:
                self.buffer = self.buffer[1:]
                continue

            total_size = 2 + frame_length
            if len(self.buffer) < total_size:
                return

            frame_type = self.buffer[2]

            if frame_type == self.CRSF_FRAMETYPE_RC_CHANNELS:
                payload = self.buffer[3:total_size - 1]
                if self.parse_rc_channels(payload):
                    if not self.connected:
                        self.connected = True
                        # arm settling window so simple_mux is shielded from any
                        # burst noise immediately following reconnect/failsafe.
                        self.settling_until = time.time() + self.settling_sec
                        rospy.loginfo("CRSF receiver connected! (settling %.0fms)",
                                      self.settling_sec * 1000.0)

            self.buffer = self.buffer[total_size:]

        # Prevent unbounded growth
        if len(self.buffer) > 512:
            self.buffer = self.buffer[-256:]

    def publish_joy(self):
        msg = Joy()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.frame_id
        msg.axes = [0.0] * self.num_axes
        msg.buttons = [0] * self.num_buttons
        for i, (joy_idx, crsf_ch) in enumerate(zip(self.axes_joy_indices, self.axes_crsf_channels)):
            sign = self.axes_invert[i] if i < len(self.axes_invert) else 1.0
            cal_min = self.axes_cal_min[i] if i < len(self.axes_cal_min) else self.NORM_MIN
            cal_mid = self.axes_cal_mid[i] if i < len(self.axes_cal_mid) else self.CH_MID
            cal_max = self.axes_cal_max[i] if i < len(self.axes_cal_max) else self.NORM_MAX
            msg.axes[joy_idx] = sign * self.normalize_axis(
                self.channels[crsf_ch], cal_min, cal_mid, cal_max)
        for i, (joy_idx, crsf_ch) in enumerate(zip(self.button_joy_indices, self.button_crsf_channels)):
            inv = self.button_invert[i] if i < len(self.button_invert) else 0
            # LB (joy_idx==4) uses dead-band + N-frame debounce. Mid-band
            # samples are noise; on either side, require lb_debounce_frames
            # consecutive trusted samples before flipping lb_state.
            if joy_idx == 4:
                msg.buttons[joy_idx] = self._lb_filtered_button(self.channels[crsf_ch], inv)
            elif joy_idx == 0:
                # A button asymmetric N-frame debounce (see __init__)
                msg.buttons[joy_idx] = self._a_filtered_button(self.channels[crsf_ch], inv)
            else:
                msg.buttons[joy_idx] = self.channel_to_button(self.channels[crsf_ch], invert=inv)
        self.joy_pub.publish(msg)
        # also publish raw CRSF channel snapshot for live diagnosis
        dbg = Int32MultiArray()
        dbg.data = [int(v) for v in self.channels]
        self.debug_ch_pub.publish(dbg)

    def _lb_filtered_button(self, raw_value, invert):
        # map raw CRSF value to a candidate (0/1/None=untrusted) using the
        # 3-position band model, then run an N-frame agreement filter
        # against lb_state. For the default (invert=0) LB switch, pos1
        # (≈191) is pressed and both pos2 (≈992) and pos3 (≈1792) count
        # as released — any value outside those bands is untrusted noise.
        if invert:
            # invert=1: pressed when value is HIGH (near 1811), released when LOW.
            # Kept for non-LB channels that reuse this filter via legacy config.
            if raw_value >= self.lb_released_min:
                candidate = 1
            elif raw_value <= self.lb_pressed_max:
                candidate = 0
            else:
                candidate = None
        else:
            # 3-position model
            if raw_value <= self.lb_pressed_max:
                candidate = 1  # pos1: pressed
            elif self.lb_idle_min <= raw_value <= self.lb_idle_max:
                candidate = 0  # pos2: idle = released
            elif raw_value >= self.lb_released_min:
                candidate = 0  # pos3: also released
            else:
                candidate = None  # between-position transient / bit corruption

        if candidate is None:
            # mid-band sample - untrusted. Hold lb_state, reset pending.
            self.lb_pending_count = 0
            return self.lb_state

        if candidate == self.lb_state:
            # sample agrees with current state - nothing to flip.
            self.lb_pending_count = 0
            return self.lb_state

        # asymmetric debounce. Releasing (1 -> 0) is the safe direction:
        # commit immediately on the first trusted "released" sample so
        # the user is never trapped in humandrive after intentionally
        # letting go. Entering (0 -> 1) is the dangerous direction:
        # require N consecutive trusted "pressed" samples.
        if self.lb_state == 1 and candidate == 0:
            self.lb_state = 0
            self.lb_pending_count = 0
            rospy.loginfo("[elrs_joy] LB 1 -> 0 (released, immediate)")
            return self.lb_state

        # 0 -> 1 path: accumulate trusted "pressed" samples.
        if candidate == self.lb_pending:
            self.lb_pending_count += 1
        else:
            self.lb_pending = candidate
            self.lb_pending_count = 1

        if self.lb_pending_count >= self.lb_debounce_frames:
            # N consecutive trusted "pressed" samples - commit entry.
            self.lb_state = 1
            self.lb_pending_count = 0
            rospy.loginfo("[elrs_joy] LB 0 -> 1 (debounced over %d frames)",
                          self.lb_debounce_frames)

        return self.lb_state

    def _a_filtered_button(self, raw_value, invert):
        # asymmetric N-frame debounce for the A button. Uses the same
        # threshold-based instantaneous decision as channel_to_button to
        # pick a candidate, then requires a_debounce_frames consecutive
        # "pressed" samples to commit 0 -> 1, but releases (1 -> 0)
        # immediately on the first "released" sample.
        candidate = self.channel_to_button(raw_value, invert=invert)

        if candidate == self.a_state:
            self.a_pending_count = 0
            return self.a_state

        if self.a_state == 1 and candidate == 0:
            self.a_state = 0
            self.a_pending_count = 0
            rospy.loginfo("[elrs_joy] A 1 -> 0 (released, immediate)")
            return self.a_state

        # 0 -> 1 path: accumulate consecutive "pressed" samples.
        self.a_pending_count += 1
        if self.a_pending_count >= self.a_debounce_frames:
            self.a_state = 1
            self.a_pending_count = 0
            rospy.loginfo("[elrs_joy] A 0 -> 1 (debounced over %d frames)",
                          self.a_debounce_frames)

        return self.a_state

    def check_failsafe(self):
        elapsed = time.time() - self.last_valid_time
        if elapsed > self.failsafe_timeout:
            if self.connected:
                rospy.logwarn("CRSF signal lost! (no valid packet for %.1fs)", elapsed)
                self.connected = False
                # on signal loss, reset LB filter to released so post-recovery
                # doesn't carry a stale "pressed" state across the gap.
                self.lb_state = 0
                self.lb_pending = 0
                self.lb_pending_count = 0
                # same idea for A — don't carry a stale press across a gap.
                self.a_state = 0
                self.a_pending_count = 0
                msg = Joy()
                msg.header.stamp = rospy.Time.now()
                msg.header.frame_id = self.frame_id
                msg.axes = [0.0] * self.num_axes
                msg.buttons = [0] * self.num_buttons
                self.joy_pub.publish(msg)

    def print_stats(self):
        now = time.time()
        if now - self.last_stats_time > 10.0:
            total = self.accept_count + self.reject_count
            if total > 0:
                rate = 100.0 * self.accept_count / total
                rospy.loginfo("CRSF frames: %d accepted, %d rejected (%.1f%% accept rate)",
                              self.accept_count, self.reject_count, rate)
            self.accept_count = 0
            self.reject_count = 0
            self.last_stats_time = now
            self.stats_printed = True

    def connect_serial(self):
        try:
            self.serial_port = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.01
            )
            rospy.loginfo("Serial port %s opened at %d baud", self.port, self.baud_rate)
            return True
        except serial.SerialException as e:
            rospy.logerr("Failed to open serial port: %s", str(e))
            return False

    def run(self):
        if not self.connect_serial():
            return

        rate = rospy.Rate(self.publish_rate)
        rospy.loginfo("ELRS Joy Node started (no-CRC mode)")
        rospy.loginfo("  Port: %s @ %d baud", self.port, self.baud_rate)
        rospy.loginfo("  Failsafe timeout: %.1fs", self.failsafe_timeout)

        while not rospy.is_shutdown():
            try:
                if self.serial_port.in_waiting > 0:
                    data = self.serial_port.read(self.serial_port.in_waiting)
                    self.buffer.extend(data)
                    self.parse_crsf_frame()

                # honor settling window. connected=True but publish suppressed
                # until settling_until elapses — simple_mux sees no /joy update
                # during this gap and keeps autodrive_latched from the pre-glitch
                # state, so the car keeps running.
                if self.connected and time.time() >= self.settling_until:
                    self.publish_joy()

                self.check_failsafe()
                if not self.stats_printed:
                    self.print_stats()
                rate.sleep()

            # also catch raw OSError (e.g. EIO from in_waiting ioctl on USB drop)
            except (serial.SerialException, OSError) as e:
                rospy.logerr("Serial error: %s. Reconnecting...", str(e))
                self.connected = False
                try:
                    if self.serial_port and self.serial_port.is_open:
                        self.serial_port.close()
                except Exception:
                    pass
                self.serial_port = None
                self.buffer.clear()
                while not rospy.is_shutdown():
                    time.sleep(1.0)
                    if self.connect_serial():
                        break
            except KeyboardInterrupt:
                break

        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()


if __name__ == '__main__':
    try:
        node = ELRSJoyNode()
        node.run()
    except rospy.ROSInterruptException:
        pass