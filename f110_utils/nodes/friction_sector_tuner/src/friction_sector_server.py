#!/usr/bin/env python3
# Friction sector tuner — per-sector friction scaling for 3D velocity planning
import rospy
import rospkg
import yaml
import numpy as np
from f110_msgs.msg import Wpnt, WpntArray
from std_msgs.msg import Float32MultiArray
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from dynamic_reconfigure.server import Server
from friction_sector_tuner.cfg import dyn_sect_tunerConfig
from visualization_msgs.msg import Marker, MarkerArray

class FrictionSectorPublisher:
    def __init__(self):
        self.sectors = None
        self.glb_waypoints = None
        # one-way sync Sector -> mu_scale_x/y
        self._prev_sector_vals = {}
        self._init_done = False
        pkg_path = rospkg.RosPack().get_path("stack_master")
        map_name = rospy.get_param('/map')
        self.yaml_file_path = pkg_path + "/maps/" + map_name + "/friction_scaling.yaml"
        self.yaml_data = self.get_yaml_values(self.yaml_file_path)
        self.default_config = self.decode_yaml(self.yaml_data)
        self.srv = Server(dyn_sect_tunerConfig, self.callback)
        self.srv.update_configuration(self.default_config)
        self.sector_pub = rospy.Publisher('/friction_sector_markers', MarkerArray, queue_size=10)
        self.friction_pub = rospy.Publisher('/friction_map', Float32MultiArray, queue_size=10)
        rospy.Subscriber('/global_waypoints', WpntArray, self.glb_wpnts_cb)

    def callback(self, config, level):
        if config.save_params:
            self.save_yaml(config)
            config.save_params = False
        if config.load_yaml:
            self._reload_yaml(config)
            config.load_yaml = False
        else:
            # detect Sector slider moves, mirror to mu_scale_x/y of the
            # same sector. Skipped on the first callback (seed only)
            # and on yaml reload (handled in _reload_yaml).
            self._sync_sector_to_xy(config)
        self.update_friction_params(config)
        return config

    def _sync_sector_to_xy(self, config):
        if self.yaml_data is None:
            return
        n_sectors = self.yaml_data.get('n_sectors', 0)
        for i in range(n_sectors):
            key = f"Sector{i}"
            cur = float(getattr(config, key, 1.0))
            if not self._init_done:
                self._prev_sector_vals[key] = cur
                continue
            prev = self._prev_sector_vals.get(key, cur)
            if cur != prev:
                setattr(config, f"{key}_mu_scale_x", cur)
                setattr(config, f"{key}_mu_scale_y", cur)
            self._prev_sector_vals[key] = cur
        if not self._init_done:
            self._init_done = True

    def _reload_yaml(self, config):
        """Reload friction params from yaml and update rqt sliders"""
        self.yaml_data = self.get_yaml_values(self.yaml_file_path)
        self.default_config = self.decode_yaml(self.yaml_data)
        for key, val in self.default_config.items():
            setattr(config, key, val)
        # reseed prev Sector values so reload does not trigger sync
        n_sectors = self.yaml_data.get('n_sectors', 0)
        for i in range(n_sectors):
            k = f"Sector{i}"
            self._prev_sector_vals[k] = float(getattr(config, k, 1.0))
        rospy.loginfo(f"[FrictionSector] Reloaded from {self.yaml_file_path}")

    def update_friction_params(self, config):
        """Update rosparam with current friction sector values (clamp by global_friction_limit)"""
        if self.yaml_data is None:
            return

        n_sectors = self.yaml_data['n_sectors']
        friction_limit = config.global_friction_limit
        rospy.set_param('/friction_map_params/global_friction_limit', friction_limit)
        rospy.set_param('/friction_map_params/n_sectors', n_sectors)

        for i in range(n_sectors):
            sector_key = f"Sector{i}"
            sector_friction = min(getattr(config, sector_key, 1.0), friction_limit)
            start = self.yaml_data[sector_key]['start']
            end = self.yaml_data[sector_key]['end']
            rospy.set_param(f'/friction_map_params/{sector_key}/start', start)
            rospy.set_param(f'/friction_map_params/{sector_key}/end', end)
            rospy.set_param(f'/friction_map_params/{sector_key}/friction', float(sector_friction))
            # per-sector mu_scale_x / mu_scale_y (on top of sector friction).
            # Consumed by 3d_optimized_vel_planner per-sector arrays.
            mu_scale_x = float(getattr(config, f"{sector_key}_mu_scale_x", 1.0))
            mu_scale_y = float(getattr(config, f"{sector_key}_mu_scale_y", 1.0))
            rospy.set_param(f'/friction_map_params/{sector_key}/mu_scale_x', mu_scale_x)
            rospy.set_param(f'/friction_map_params/{sector_key}/mu_scale_y', mu_scale_y)
            # s-based lookup (for planners with local paths)
            if self.glb_waypoints is not None:
                s_start = self.glb_waypoints[min(start, len(self.glb_waypoints)-1)][3]
                s_end = self.glb_waypoints[min(end, len(self.glb_waypoints)-1)][3]
                rospy.set_param(f'/friction_map_params/{sector_key}/s_start', float(s_start))
                rospy.set_param(f'/friction_map_params/{sector_key}/s_end', float(s_end))

        rospy.loginfo(f"[FrictionSector] Updated rosparam: limit={friction_limit}, {n_sectors} sectors")

    def save_yaml(self, config):
        try:
            self.yaml_data["global_friction_limit"] = config.global_friction_limit
            for key in self.sectors:
                self.yaml_data[key]['friction'] = float(getattr(config, key, None))
                # persist per-sector mu_scale_x / mu_scale_y to yaml.
                self.yaml_data[key]['mu_scale_x'] = float(
                    getattr(config, f"{key}_mu_scale_x", 1.0))
                self.yaml_data[key]['mu_scale_y'] = float(
                    getattr(config, f"{key}_mu_scale_y", 1.0))
            with open(self.yaml_file_path, "w") as yaml_file:
                yaml.dump(self.yaml_data, yaml_file, default_flow_style=False)
            rospy.loginfo("Friction config saved to: %s", self.yaml_file_path)
        except Exception as e:
            rospy.logerr("Failed to save friction config: %s", str(e))

    def get_yaml_values(self, yaml_file_path):
        try:
            with open(yaml_file_path, "r") as file:
                data = yaml.safe_load(file)
            return data
        except FileNotFoundError:
            rospy.logwarn("friction_scaling.yaml not found: %s", yaml_file_path)
            return {'global_friction': 1.0, 'n_sectors': 1,
                    'Sector0': {'start': 0, 'end': 600, 'friction': 1.0}}

    def decode_yaml(self, yaml_data):
        default_config = {"global_friction_limit": float(yaml_data.get("global_friction_limit", 1.0))}
        self.sectors = {k: v for k, v in yaml_data.items() if k.startswith('Sector')}
        for key, item in self.sectors.items():
            default_config[key] = float(item['friction'])
            # per-sector mu_scale_x / mu_scale_y default = 1.0 if missing
            # (back-compat with old yamls lacking the field).
            default_config[f"{key}_mu_scale_x"] = float(item.get('mu_scale_x', 1.0))
            default_config[f"{key}_mu_scale_y"] = float(item.get('mu_scale_y', 1.0))
        return default_config

    def glb_wpnts_cb(self, data):
        self.glb_waypoints = []
        for waypoint in data.wpnts:
            self.glb_waypoints.append([waypoint.x_m, waypoint.y_m, waypoint.z_m, waypoint.s_m])

    def pub_sector_markers(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            if self.glb_waypoints is None:
                rate.sleep()
                continue

            n_sectors = self.yaml_data['n_sectors']
            sec_markers = MarkerArray()

            for i in range(n_sectors):
                s = self.yaml_data[f"Sector{i}"]['start']
                if s >= len(self.glb_waypoints):
                    continue
                if s == (len(self.glb_waypoints) - 1):
                    theta = np.arctan2((self.glb_waypoints[0][1] - self.glb_waypoints[s][1]),
                                      (self.glb_waypoints[0][0] - self.glb_waypoints[s][0]))
                else:
                    theta = np.arctan2((self.glb_waypoints[s+1][1] - self.glb_waypoints[s][1]),
                                      (self.glb_waypoints[s+1][0] - self.glb_waypoints[s][0]))
                quaternions = quaternion_from_euler(0, 0, theta)

                # Read live friction value from rosparam
                friction_val = rospy.get_param(f'/friction_map_params/Sector{i}/friction', 1.0)
                # per-sector mu_scale_x / mu_scale_y for RViz label
                mu_scale_x_val = rospy.get_param(f'/friction_map_params/Sector{i}/mu_scale_x', 1.0)
                mu_scale_y_val = rospy.get_param(f'/friction_map_params/Sector{i}/mu_scale_y', 1.0)

                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = rospy.Time.now()
                marker.type = marker.ARROW
                marker.scale.x = 0.3
                marker.scale.y = 0.05
                marker.scale.z = 0.05
                marker.color.r = 1.0
                marker.color.g = 0.5
                marker.color.b = 0.0
                marker.color.a = 1.0
                marker.pose.position.x = self.glb_waypoints[s][0]
                marker.pose.position.y = self.glb_waypoints[s][1]
                marker.pose.position.z = self.glb_waypoints[s][2]
                marker.pose.orientation.x = quaternions[0]
                marker.pose.orientation.y = quaternions[1]
                marker.pose.orientation.z = quaternions[2]
                marker.pose.orientation.w = quaternions[3]
                marker.id = i
                sec_markers.markers.append(marker)

                marker_text = Marker()
                marker_text.header.frame_id = "map"
                marker_text.header.stamp = rospy.Time.now()
                marker_text.type = marker_text.TEXT_VIEW_FACING
                marker_text.text = f"Friction {i} ({friction_val:.2f}, {mu_scale_x_val:.2f}, {mu_scale_y_val:.2f})"
                marker_text.scale.z = 0.4
                marker_text.color.r = 1.0
                marker_text.color.g = 0.5
                marker_text.color.b = 0.0
                marker_text.color.a = 1.0
                marker_text.pose.position.x = self.glb_waypoints[s][0]
                marker_text.pose.position.y = self.glb_waypoints[s][1]
                marker_text.pose.position.z = self.glb_waypoints[s][2] + 1.5
                marker_text.pose.orientation.w = 1.0
                marker_text.id = i + n_sectors
                sec_markers.markers.append(marker_text)

            self.sector_pub.publish(sec_markers)
            rate.sleep()

if __name__ == "__main__":
    rospy.init_node("friction_sector_tuner", anonymous=False)
    print('Friction Sector Server Launched...')
    sec_pub = FrictionSectorPublisher()
    sec_pub.pub_sector_markers()
