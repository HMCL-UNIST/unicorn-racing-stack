#!/usr/bin/env python3

import json

import rospy

from std_msgs.msg import String
from visualization_msgs.msg import Marker
from blink1.msg import Blink1msg

BL_FADE      = 1  # fade to the RGB color, t is the time of fading in ms
BL_ON        = 2  # turn on to the RGB color, t is ignored
BL_BLINK     = 3  # blink the RGB color, t is the period in ms
BL_RANDBLINK = 4  # blink at random RGB colors, t is the period in ms

# current (launch) mode override colour — cyan.
LAUNCH_OVERRIDE_RGB = (0, 255, 255)


class StateIndicatorNode:
    """
    This class implements a ROS node that handles the terminal utilities of the state indicator.
    """

    def __init__(self):
        """
        Initialize the node, subscribe to topics, create publishers
        """

        # Initialize the node
        self.name = "state_indicator_node"
        rospy.init_node(self.name, anonymous=True)

        # launch (current-injection) mode flag, driven by simple_mux's
        # /launch_controller/debug heartbeat (50Hz, present whenever the mux runs).
        # The LED is repainted on the active EDGE (not just on /state_marker) so the
        # override works even when state_machine / headtohead is NOT running and no
        # /state_marker ever arrives. last_state_rgb caches the most recent normal
        # state colour; it starts OFF so that without state_machine the LED simply
        # goes cyan during launch and back to OFF afterwards.
        self.launch_active = False
        self.last_state_rgb = (0, 0, 0)  # OFF until a real /state_marker is seen

        # Subscribe to the topics
        rospy.Subscriber('/state_marker', Marker, self.state_callback)
        # current-mode source. Topic name overridable to match simple_mux.
        launch_topic = rospy.get_param('~launch_debug_topic', '/launch_controller/debug')
        rospy.Subscriber(launch_topic, String, self.launch_debug_callback)
        # Publish the topics
        self.blink_pub = rospy.Publisher('blink1/blink', Blink1msg, queue_size=10)

    #############
    # CALLBACKS #
    #############

    # push an RGB triple to the blink1 LED.
    def _publish_rgb(self, rgb):
        blink_msg = Blink1msg()
        blink_msg.function = BL_ON
        blink_msg.t = 10  # milliseconds
        blink_msg.r, blink_msg.g, blink_msg.b = rgb
        self.blink_pub.publish(blink_msg)

    # track launch (current) mode and repaint the LED every heartbeat. simple_mux
    # publishes /launch_controller/debug at 50Hz (active/armed flags both present in
    # idle and live messages), so we just paint the current intent each tick — no
    # edge detection, no timer. Override is on while EITHER armed or active is true:
    # armed is the prerequisite for active in simple_mux, so this covers the full
    # "user intends current mode" window — A-toggle press (armed True) right through
    # fire (active True) until completion/abort. This is also the single repaint
    # source that makes the override robust: if a frame is lost or a concurrent
    # state_callback briefly clobbers the colour, the next heartbeat (20ms) repaints
    # it. When both flags are false we repaint last_state_rgb, which is the real
    # state colour if state_machine is running, or the OFF default if it is not (so
    # the LED goes dark after launch without state_machine).
    def launch_debug_callback(self, msg):
        try:
            payload = json.loads(msg.data)
            self.launch_active = bool(payload.get("active", False) or payload.get("armed", False))
        except (ValueError, TypeError):
            # Malformed payload — leave the last known flag untouched.
            return
        if self.launch_active:
            self._publish_rgb(LAUNCH_OVERRIDE_RGB)          # cyan while in current mode
        else:
            self._publish_rgb(self.last_state_rgb)          # restore (OFF if no state_machine)

    def state_callback(self, state_msg):
        """
        Callback function for the state of the racecar.

        Args:
        - state_msg (String): The received message containing the state.
        """
        # cache the normal state colour; only paint it while NOT in launch
        # mode so the cyan override is not clobbered by incoming state markers.
        self.last_state_rgb = (int(255 * state_msg.color.r),
                               int(255 * state_msg.color.g),
                               int(255 * state_msg.color.b))
        if not self.launch_active:
            self._publish_rgb(self.last_state_rgb)


if __name__ == '__main__':
    state_indicator = StateIndicatorNode()
    rospy.spin()
    # state_indicator.run()

