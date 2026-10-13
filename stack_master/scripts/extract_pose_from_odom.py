#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry

def extract_pose(msg):
    # build the PoseWithCovarianceStamped message
    pose = PoseStamped()
    pose.header = msg.header
    pose.pose = msg.pose.pose
    
    # covariance (simply initialized to 0 here)
    # in real use it may need to be set to meaningful values.
    # pose_with_cov.pose.covariance = [
    #     0.01, 0, 0, 0, 0, 0,    # x axis (0.1m)
    #     0, 0.01, 0, 0, 0, 0,    # y axis (0.1m)
    #     0, 0, 0.01, 0, 0, 0,    # z axis (0.1m)
    #     0, 0, 0, 0.0001, 0, 0,  # roll axis (0.01 rad)
    #     0, 0, 0, 0, 0.0001, 0,  # pitch axis (0.01 rad)
    #     0, 0, 0, 0, 0, 0.0001   # yaw axis (0.01 rad)
    # ]
    return pose

def callback(msg):
    # convert the PoseStamped message to PoseWithCovarianceStamped
    pose = extract_pose(msg)

    # publish the converted message
    pub.publish(pose)

def listener():
    rospy.init_node('odom_to_pose', anonymous=True)
    in_topic  = rospy.get_param("~in_topic", "car_state/odom")
    out_topic = rospy.get_param("~out_topic", "car_state/pose")
    
    rospy.Subscriber(in_topic, Odometry, callback)

    global pub
    pub = rospy.Publisher(out_topic, PoseStamped, queue_size=10)

    rospy.spin()

if __name__ == '__main__':
    listener()
