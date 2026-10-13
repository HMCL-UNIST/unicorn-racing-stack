#!/usr/bin/env python3
# Bridge sector marker publisher — RViz visualisation of bridge zones
# defined in <map>/bridge_sectors.yaml. Mirrors the established
# 3d_sector_tuner / friction_sector_tuner publisher pattern so RViz
# configs treat bridges the same as other sector types.
#
#      Topic   : /bridge_sector_markers (visualization_msgs/MarkerArray)
#      Source  : <map>/bridge_sectors.yaml (Bridge{i}: {start, end})
#      Subscr. : /global_waypoints (WpntArray) — used purely to map yaml
#                row indices to map-frame xyz.
#
#      Bridge yaml stores *smoothed-CSV row indices*, but the waypoints
#      received on /global_waypoints are *raceline-resampled*. They share
#      the same s_m axis though, so we resample yaml indices onto the
#      waypoint grid by ratio (same trick velopt uses for friction sectors).
#      Missing yaml ⇒ publish empty MarkerArray (silent no-op).
import os
import sys
import rospy
import rospkg
import numpy as np
from f110_msgs.msg import WpntArray
from tf.transformations import quaternion_from_euler
from visualization_msgs.msg import Marker, MarkerArray

# shared yaml helper lives next to this file.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
import bridge_yaml_io


class BridgeSectorPublisher:
    def __init__(self):
        self.glb_waypoints = None
        pkg_path = rospkg.RosPack().get_path('stack_master')
        map_name = rospy.get_param('/map')
        self.map_dir = os.path.join(pkg_path, 'maps', map_name)

        self.sector_pub = rospy.Publisher(
            '/bridge_sector_markers', MarkerArray, queue_size=10, latch=True)
        rospy.Subscriber('/global_waypoints', WpntArray, self.glb_wpnts_cb)

    def glb_wpnts_cb(self, data):
        self.glb_waypoints = [
            [w.x_m, w.y_m, w.z_m, w.s_m] for w in data.wpnts
        ]

    def _heading_at(self, idx):
        n = len(self.glb_waypoints)
        i_next = (idx + 1) % n
        dx = self.glb_waypoints[i_next][0] - self.glb_waypoints[idx][0]
        dy = self.glb_waypoints[i_next][1] - self.glb_waypoints[idx][1]
        return float(np.arctan2(dy, dx))

    def _make_arrow(self, idx, marker_id, label_color):
        wp = self.glb_waypoints[idx]
        theta = self._heading_at(idx)
        q = quaternion_from_euler(0, 0, theta)
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = rospy.Time.now()
        m.type = Marker.ARROW
        m.scale.x = 0.5
        m.scale.y = 0.05
        m.scale.z = 0.15
        m.color.r, m.color.g, m.color.b, m.color.a = label_color
        m.pose.position.x = wp[0]
        m.pose.position.y = wp[1]
        m.pose.position.z = wp[2]
        m.pose.orientation.x = q[0]
        m.pose.orientation.y = q[1]
        m.pose.orientation.z = q[2]
        m.pose.orientation.w = q[3]
        m.id = marker_id
        return m

    def _make_text(self, idx, marker_id, text, color):
        wp = self.glb_waypoints[idx]
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = rospy.Time.now()
        m.type = Marker.TEXT_VIEW_FACING
        m.text = text
        m.scale.z = 0.4
        m.color.r, m.color.g, m.color.b, m.color.a = color
        m.pose.position.x = wp[0]
        m.pose.position.y = wp[1]
        m.pose.position.z = wp[2] + 1.5
        m.pose.orientation.w = 1.0
        m.id = marker_id
        return m

    def pub_loop(self):
        rate = rospy.Rate(1.0)
        # pink/magenta to match the matplotlib GUI's bridge highlight
        # and to distinguish from speed (red), overtake, friction.
        start_color = (1.0, 0.0, 1.0, 1.0)
        end_color   = (0.6, 0.0, 0.6, 1.0)
        text_color  = (0.4, 0.0, 0.4, 1.0)

        while not rospy.is_shutdown():
            if self.glb_waypoints is None:
                rate.sleep()
                continue
            # single source of truth — bridge_yaml_io returns the
            # master enable flag plus a fully-defaulted bridge list,
            # so the server just renders what the rest of the stack
            # will actually use. Missing yaml ⇒ empty bridges.
            enable, bridges = bridge_yaml_io.load_bridges(self.map_dir)
            arr = MarkerArray()
            # DELETEALL prefix so removed bridges disappear from RViz
            # without leaving stale markers.
            clear = Marker()
            clear.header.frame_id = 'map'
            clear.action = Marker.DELETEALL
            arr.markers.append(clear)

            n_wp = len(self.glb_waypoints)
            if enable and bridges:
                # yaml indices are on the smoothed-CSV grid, waypoints
                # are on the raceline-resampled grid — they share s_m
                # so we project via ratio. Same trick velopt uses for
                # friction sectors so the visualization matches what
                # the optimizer actually constrains.
                total_yaml_idx = max((b['end'] for b in bridges), default=0) + 1
                marker_id = 1  # 0 reserved for DELETEALL safety
                for i, b in enumerate(bridges):
                    s_i, e_i = b['start'], b['end']
                    if total_yaml_idx > 0:
                        wp_start = int(round(s_i / total_yaml_idx * n_wp))
                        wp_end   = int(round(e_i / total_yaml_idx * n_wp))
                    else:
                        wp_start, wp_end = s_i, e_i
                    wp_start = max(0, min(wp_start, n_wp - 1))
                    wp_end   = max(0, min(wp_end,   n_wp - 1))

                    arr.markers.append(
                        self._make_arrow(wp_start, marker_id, start_color))
                    marker_id += 1
                    arr.markers.append(
                        self._make_arrow(wp_end, marker_id, end_color))
                    marker_id += 1
                    arr.markers.append(
                        self._make_text(wp_start, marker_id,
                                        f'Bridge {i} start', text_color))
                    marker_id += 1
                    arr.markers.append(
                        self._make_text(wp_end, marker_id,
                                        f'Bridge {i} end', text_color))
                    marker_id += 1

            self.sector_pub.publish(arr)
            rate.sleep()


if __name__ == '__main__':
    rospy.init_node('bridge_sector_server', anonymous=False)
    print('Bridge Sector Server Launched...')
    BridgeSectorPublisher().pub_loop()
