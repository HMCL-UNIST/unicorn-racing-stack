#!/usr/bin/env python3
# fake odom publisher for testing 3D frenet pipeline without real hardware
"""
Reads global_waypoints.json and publishes odom along the raceline trajectory.
Simulates a car driving along the waypoints at the planned speed.

Usage:
  rosrun stack_master fake_odom_publisher.py _map:=gazebo_wall_3d_rc_car_10th_timeoptimal
"""
import rospy
import json
import numpy as np
import os
import tf
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion

class FakeOdomPublisher:
    def __init__(self):
        rospy.init_node("fake_odom_publisher", anonymous=True)

        map_name = rospy.get_param("~map", "gazebo_wall_3d_rc_car_10th_timeoptimal")
        speed_scale = rospy.get_param("~speed_scale", 1.0)
        rate_hz = rospy.get_param("~rate", 50.0)

        # load waypoints
        json_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "maps", map_name, "global_waypoints.json"
        )
        with open(json_path) as f:
            data = json.load(f)

        self.wpnts = data["global_traj_wpnts_sp"]["wpnts"]
        self.n_wpnts = len(self.wpnts)
        self.speed_scale = speed_scale
        rospy.loginfo("Loaded %d waypoints from %s" % (self.n_wpnts, map_name))

        self.pub = rospy.Publisher("/glim_ros/base_odom", Odometry, queue_size=10)
        self.rate = rospy.Rate(rate_hz)
        self.dt = 1.0 / rate_hz

        self.run()

    def run(self):
        s_current = 0.0
        total_s = self.wpnts[-1]["s_m"]

        while not rospy.is_shutdown():
            # find current waypoint by s
            idx = 0
            for i in range(self.n_wpnts - 1):
                if self.wpnts[i + 1]["s_m"] > s_current:
                    idx = i
                    break
            else:
                idx = self.n_wpnts - 1

            w = self.wpnts[idx]
            w_next = self.wpnts[(idx + 1) % self.n_wpnts]

            # interpolate between waypoints
            ds = w_next["s_m"] - w["s_m"]
            if ds <= 0:
                ds = total_s - w["s_m"] + w_next["s_m"]
            t = (s_current - w["s_m"]) / ds if ds > 1e-6 else 0.0
            t = np.clip(t, 0.0, 1.0)

            x = w["x_m"] + t * (w_next["x_m"] - w["x_m"])
            y = w["y_m"] + t * (w_next["y_m"] - w["y_m"])
            z = w["z_m"] + t * (w_next["z_m"] - w["z_m"])
            psi = w["psi_rad"]
            vx = w["vx_mps"] * self.speed_scale

            # compute vz from slope (dz/ds * v)
            dz = w_next["z_m"] - w["z_m"]
            ds_3d = np.sqrt((w_next["x_m"] - w["x_m"])**2 +
                            (w_next["y_m"] - w["y_m"])**2 +
                            (w_next["z_m"] - w["z_m"])**2)
            slope = dz / ds_3d if ds_3d > 1e-6 else 0.0
            vz = vx * slope

            # publish odom
            odom = Odometry()
            odom.header.stamp = rospy.Time.now()
            odom.header.frame_id = "map"
            odom.child_frame_id = "base_link"
            odom.pose.pose.position.x = x
            odom.pose.pose.position.y = y
            odom.pose.pose.position.z = z
            q = tf.transformations.quaternion_from_euler(0, 0, psi)
            odom.pose.pose.orientation = Quaternion(*q)
            odom.twist.twist.linear.x = vx  # body frame forward
            odom.twist.twist.linear.y = 0.0
            odom.twist.twist.linear.z = vz
            self.pub.publish(odom)

            # advance s
            s_current += vx * self.dt
            if s_current >= total_s:
                s_current -= total_s

            self.rate.sleep()

if __name__ == "__main__":
    try:
        FakeOdomPublisher()
    except rospy.ROSInterruptException:
        pass
