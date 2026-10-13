#!/usr/bin/env python3
# Safety-distance sector server — clone of friction_sector_server.py.
# - Reads <map>/safety_sectors.yaml on startup.
# - Spawns a dynamic_reconfigure Server bound to dyn_safety_tunerConfig
# (auto-generated from <pkg>/cfg/safety_sectors.yaml at build time).
# - On every reconfigure callback, mirrors the current sliders into the
# /safety_sector_params/... rosparam tree so any downstream consumer
# (gen_global_racing_line.py at NLP setup time) sees the live values.
# - `save_params` trigger overwrites <map>/safety_sectors.yaml with the
# current slider values. `load_yaml` trigger reloads sliders from the
# map yaml (useful after hand-editing).
#
#      Rosparam tree (consumed by gen_global_racing_line.py):
#        /safety_sector_params/safety_distance_default  -> float
#        /safety_sector_params/n_sectors                -> int
#        /safety_sector_params/Sector{i}/start          -> int
#        /safety_sector_params/Sector{i}/end            -> int
#        /safety_sector_params/Sector{i}/safety_distance -> float
#        /safety_sector_params/Sector{i}/pre_m          -> float
#        /safety_sector_params/Sector{i}/post_m         -> float
import numpy as np
import rospy
import rospkg
import yaml
from dynamic_reconfigure.server import Server
from safety_sector_tuner_3d.cfg import dyn_safety_tunerConfig
from f110_msgs.msg import WpntArray
from tf.transformations import quaternion_from_euler
from visualization_msgs.msg import Marker, MarkerArray

_PARAM_NS = '/safety_sector_params'


class SafetySectorPublisher:
    def __init__(self):
        self.sectors = None
        self.glb_waypoints = None
        pkg_path = rospkg.RosPack().get_path("stack_master")
        # The map name comes from /map (set by stack_master launches).
        # Fallback to '~map' private param so the server can also run
        # standalone via safety_tuner.launch with map:=<name>.
        map_name = (rospy.get_param('/map', None)
                    or rospy.get_param('~map', ''))
        if not map_name:
            rospy.logwarn("[SafetySector] no /map or ~map; "
                          "yaml IO disabled, sliders run on built-in defaults")
            self.yaml_file_path = None
            self.yaml_data = self._fallback_yaml()
        else:
            self.yaml_file_path = (pkg_path + "/maps/" + map_name
                                   + "/safety_sectors.yaml")
            self.yaml_data = self.get_yaml_values(self.yaml_file_path)

        self.default_config = self.decode_yaml(self.yaml_data)
        self.srv = Server(dyn_safety_tunerConfig, self.callback)
        self.srv.update_configuration(self.default_config)

        # RViz marker output — friction_sector_server.pub_sector_markers
        # clone. Per-sector arrow at the sector start point + text label
        # with the live safety_distance value. pre_m / post_m are NOT
        # shown on the marker by design (only the core sector value).
        self.sector_pub = rospy.Publisher(
            '/safety_sector_markers', MarkerArray, queue_size=10)
        rospy.Subscriber('/global_waypoints', WpntArray, self.glb_wpnts_cb)

    def callback(self, config, level):
        if config.save_params:
            self.save_yaml(config)
            config.save_params = False
        if config.load_yaml:
            self._reload_yaml(config)
            config.load_yaml = False
        self.update_safety_params(config)
        return config

    def _reload_yaml(self, config):
        """Reload safety params from yaml and update rqt sliders."""
        if self.yaml_file_path is None:
            return
        self.yaml_data = self.get_yaml_values(self.yaml_file_path)
        self.default_config = self.decode_yaml(self.yaml_data)
        for key, val in self.default_config.items():
            setattr(config, key, val)
        rospy.loginfo(
            f"[SafetySector] Reloaded from {self.yaml_file_path}")

    def update_safety_params(self, config):
        """Mirror current slider state to rosparam tree. gen_global_racing_line
        reads from this tree at NLP setup time."""
        if self.yaml_data is None:
            return
        n_sectors = int(self.yaml_data['n_sectors'])
        default_d = float(config.safety_distance_default)
        rospy.set_param(f'{_PARAM_NS}/safety_distance_default', default_d)
        rospy.set_param(f'{_PARAM_NS}/n_sectors', n_sectors)
        for i in range(n_sectors):
            sec_key = f"Sector{i}"
            d_i  = float(getattr(config, sec_key, default_d))
            pre  = float(getattr(config, f"{sec_key}_pre_m",  0.0))
            post = float(getattr(config, f"{sec_key}_post_m", 0.0))
            start = int(self.yaml_data[sec_key]['start'])
            end   = int(self.yaml_data[sec_key]['end'])
            rospy.set_param(f'{_PARAM_NS}/{sec_key}/start', start)
            rospy.set_param(f'{_PARAM_NS}/{sec_key}/end',   end)
            rospy.set_param(f'{_PARAM_NS}/{sec_key}/safety_distance', d_i)
            rospy.set_param(f'{_PARAM_NS}/{sec_key}/pre_m',  pre)
            rospy.set_param(f'{_PARAM_NS}/{sec_key}/post_m', post)
        rospy.loginfo(
            f"[SafetySector] rosparam updated: default={default_d:.3f}m, "
            f"{n_sectors} sector(s)")

    def save_yaml(self, config):
        if self.yaml_file_path is None:
            rospy.logwarn("[SafetySector] save skipped: no yaml path")
            return
        try:
            self.yaml_data['safety_distance_default'] = float(
                config.safety_distance_default)
            for key in self.sectors:
                self.yaml_data[key]['safety_distance'] = float(
                    getattr(config, key, 0.20))
                self.yaml_data[key]['pre_m'] = float(
                    getattr(config, f"{key}_pre_m", 0.0))
                self.yaml_data[key]['post_m'] = float(
                    getattr(config, f"{key}_post_m", 0.0))
            with open(self.yaml_file_path, "w") as f:
                yaml.dump(self.yaml_data, f, default_flow_style=False,
                          sort_keys=False)
            rospy.loginfo(
                f"[SafetySector] saved to {self.yaml_file_path}")
        except Exception as e:
            rospy.logerr(f"[SafetySector] save failed: {e}")

    def get_yaml_values(self, yaml_file_path):
        try:
            with open(yaml_file_path, "r") as f:
                data = yaml.safe_load(f)
            return data
        except FileNotFoundError:
            rospy.logwarn(
                f"[SafetySector] yaml not found: {yaml_file_path}; "
                f"falling back to single-sector default")
            return self._fallback_yaml()

    @staticmethod
    def _fallback_yaml():
        return {
            'safety_distance_default': 0.20,
            'n_sectors': 1,
            'Sector0': {'start': 0, 'end': 1,
                        'safety_distance': 0.20,
                        'pre_m': 0.0, 'post_m': 0.0},
        }

    def glb_wpnts_cb(self, data):
        # cache (x, y, z) tuples for marker placement. Same shape as
        # friction_sector_server.glb_waypoints (sans s_m since markers
        # don't need s for positioning).
        self.glb_waypoints = [(w.x_m, w.y_m, w.z_m) for w in data.wpnts]

    def pub_sector_markers(self):
        # 1Hz arrow + text marker per sector. Sector start/end in the
        # yaml are smoothed-CSV row indices (pre-opt grid), but
        # /global_waypoints is raceline-resampled (coarser step). The
        # two grids share the same s axis though, so we project yaml
        # idx → raceline idx by ratio — same trick bridge_sector_server
        # uses. Text shows the live safety_distance read from rosparam
        # so slider edits reflect immediately. Cyan distinguishes from
        # friction (orange) and bridge (magenta).
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            if self.glb_waypoints is None or self.yaml_data is None:
                rate.sleep()
                continue
            n_sectors = int(self.yaml_data['n_sectors'])
            n_wpts = len(self.glb_waypoints)
            sec_markers = MarkerArray()
            # DELETEALL prefix so a shrinking n_sectors doesn't leave
            # stale markers behind in RViz (matches bridge pattern).
            clear = Marker()
            clear.header.frame_id = 'map'
            clear.action = Marker.DELETEALL
            sec_markers.markers.append(clear)

            # yaml indices partition the full smoothed-CSV (closed loop),
            # so max(end)+1 == N_smoothed. Use that as the divisor for
            # the ratio projection onto the n_wpts raceline grid.
            total_yaml_idx = max(
                (int(self.yaml_data[f"Sector{i}"]['end'])
                 for i in range(n_sectors)),
                default=0) + 1

            marker_id = 1   # 0 reserved for DELETEALL hygiene
            for i in range(n_sectors):
                s_yaml = int(self.yaml_data[f"Sector{i}"]['start'])
                if total_yaml_idx > 0:
                    start = int(round(s_yaml / total_yaml_idx * n_wpts))
                else:
                    start = s_yaml
                start = max(0, min(start, n_wpts - 1))

                nxt = (start + 1) % n_wpts
                dx = self.glb_waypoints[nxt][0] - self.glb_waypoints[start][0]
                dy = self.glb_waypoints[nxt][1] - self.glb_waypoints[start][1]
                theta = float(np.arctan2(dy, dx))
                quaternions = quaternion_from_euler(0, 0, theta)
                d_val = float(rospy.get_param(
                    f'{_PARAM_NS}/Sector{i}/safety_distance', 0.20))

                arrow = Marker()
                arrow.header.frame_id = "map"
                arrow.header.stamp = rospy.Time.now()
                arrow.type = arrow.ARROW
                arrow.scale.x = 0.3
                arrow.scale.y = 0.05
                arrow.scale.z = 0.05
                arrow.color.r = 0.0
                arrow.color.g = 0.7
                arrow.color.b = 1.0
                arrow.color.a = 1.0
                arrow.pose.position.x = self.glb_waypoints[start][0]
                arrow.pose.position.y = self.glb_waypoints[start][1]
                arrow.pose.position.z = self.glb_waypoints[start][2]
                arrow.pose.orientation.x = quaternions[0]
                arrow.pose.orientation.y = quaternions[1]
                arrow.pose.orientation.z = quaternions[2]
                arrow.pose.orientation.w = quaternions[3]
                arrow.id = marker_id
                sec_markers.markers.append(arrow)
                marker_id += 1

                txt = Marker()
                txt.header.frame_id = "map"
                txt.header.stamp = rospy.Time.now()
                txt.type = txt.TEXT_VIEW_FACING
                txt.text = f"Safety {i} ({d_val:.2f}m)"
                txt.scale.z = 0.4
                txt.color.r = 0.0
                txt.color.g = 0.7
                txt.color.b = 1.0
                txt.color.a = 1.0
                txt.pose.position.x = self.glb_waypoints[start][0]
                txt.pose.position.y = self.glb_waypoints[start][1]
                # offset text above the arrow but below the friction
                # label (friction uses +1.5) so the two stack instead
                # of overlapping on bridge maps.
                txt.pose.position.z = self.glb_waypoints[start][2] + 2.5
                txt.pose.orientation.w = 1.0
                txt.id = marker_id
                sec_markers.markers.append(txt)
                marker_id += 1

            self.sector_pub.publish(sec_markers)
            rate.sleep()

    def decode_yaml(self, yaml_data):
        default_config = {
            'safety_distance_default': float(yaml_data.get(
                'safety_distance_default', 0.20)),
        }
        self.sectors = {k: v for k, v in yaml_data.items()
                        if k.startswith('Sector')}
        for key, item in self.sectors.items():
            default_config[key] = float(item.get('safety_distance', 0.20))
            default_config[f"{key}_pre_m"]  = float(item.get('pre_m',  0.0))
            default_config[f"{key}_post_m"] = float(item.get('post_m', 0.0))
        return default_config


if __name__ == "__main__":
    rospy.init_node("safety_sector_tuner_3d", anonymous=False)
    print('Safety Sector Server Launched...')
    pub = SafetySectorPublisher()
    pub.pub_sector_markers()
