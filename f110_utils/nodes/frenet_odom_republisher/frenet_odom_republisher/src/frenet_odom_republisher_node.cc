#include <frenet_odom_republisher_node.h>
#include <tf/tf.h>
#include <string> 

namespace frenet_odom_republisher{

FrenetRepublisher::FrenetRepublisher(ros::NodeHandle& nh):
  nh_(nh){
  InitSubscribersPublishers();
  ros::spin();
}

FrenetRepublisher::~FrenetRepublisher() {}

void FrenetRepublisher::InitSubscribersPublishers() {
  global_trajectory_sub_ = nh_.subscribe<f110_msgs::WpntArray>
      ("/global_waypoints", 10, &FrenetRepublisher::GlobalTrajectoryCallback, this);

  fixed_path_trajectory_sub_ = nh_.subscribe<f110_msgs::OTWpntArray>
      ("/planner/avoidance/smart_static_otwpnts", 10, &FrenetRepublisher::FixedPathTrajectoryCallback, this);

  odom_sub_ = nh_.subscribe<nav_msgs::Odometry>
      ("/odom", 10, &FrenetRepublisher::OdomCallback, this);

  frenet_odom_pub_ = nh_.advertise<nav_msgs::Odometry>
      ("/odom_frenet", 1);

  frenet_odom_fixed_pub_ = nh_.advertise<nav_msgs::Odometry>
      ("/odom_frenet_fixed", 1);

  // subscribe to trackbounds once for wall-crossing detection
  trackbounds_sub_ = nh_.subscribe<visualization_msgs::MarkerArray>
      ("/trackbounds/markers", 1, &FrenetRepublisher::TrackBoundsCallback, this);

  // interactive marker button for forcing full search
  im_server_ = std::make_shared<interactive_markers::InteractiveMarkerServer>("frenet_full_search");
  CreateFullSearchButton();
}

void FrenetRepublisher::GlobalTrajectoryCallback(
    const f110_msgs::WpntArrayConstPtr &wpt_array){
  wpt_array_ = wpt_array->wpnts;
  bool enable_wrapping = true;
  frenet_converter_.SetGlobalTrajectory(&wpt_array_, enable_wrapping);
  has_global_trajectory_ = true;
}

// receive trackbounds once, then unsubscribe
void FrenetRepublisher::TrackBoundsCallback(
    const visualization_msgs::MarkerArrayConstPtr &bounds_msg){
  frenet_converter_.SetTrackBounds(bounds_msg);
  frenet_converter_fixed_.SetTrackBounds(bounds_msg);
  trackbounds_sub_.shutdown();
  ROS_WARN("[FrenetRepublisher] Track bounds received and stored, unsubscribed.");
}

void FrenetRepublisher::FixedPathTrajectoryCallback(
    const f110_msgs::OTWpntArrayConstPtr &wpt_array){
  fixed_path_wpt_array_ = wpt_array->wpnts;
  bool enable_wrapping = true;
  frenet_converter_fixed_.SetGlobalTrajectory(&fixed_path_wpt_array_, enable_wrapping);
  has_fixed_path_trajectory_ = true;
}

void FrenetRepublisher::OdomCallback(const nav_msgs::OdometryConstPtr &msg){
  // Get quaternion and yaw for both conversions
  tf::Quaternion q(
        msg->pose.pose.orientation.x,
        msg->pose.pose.orientation.y,
        msg->pose.pose.orientation.z,
        msg->pose.pose.orientation.w);
  tf::Matrix3x3 m(q);
  double roll, pitch, yaw;
  m.getRPY(roll, pitch, yaw);

  // pass z for 3D closest-point search
  // Publish GB path based frenet odom
  if (has_global_trajectory_) {
    nav_msgs::Odometry frenet_odom = *msg;
    frenet_converter_.GetFrenetOdometry(msg->pose.pose.position.x,
                                        msg->pose.pose.position.y,
                                        msg->pose.pose.position.z,
                                        yaw, msg->twist.twist.linear.x,
                                        msg->twist.twist.linear.y,
                                        &frenet_odom.pose.pose.position.x,
                                        &frenet_odom.pose.pose.position.y,
                                        &frenet_odom.twist.twist.linear.x,
                                        &frenet_odom.twist.twist.linear.y,
                                        &closest_wpt_index_);
    // abuse child frame id to pass closest wapoint index
    frenet_odom.child_frame_id = std::to_string(closest_wpt_index_);
    frenet_odom_pub_.publish(frenet_odom);
  }

  // Publish fixed path based frenet odom
  if (has_fixed_path_trajectory_) {
    nav_msgs::Odometry frenet_odom_fixed = *msg;
    frenet_converter_fixed_.GetFrenetOdometry(msg->pose.pose.position.x,
                                              msg->pose.pose.position.y,
                                              msg->pose.pose.position.z,
                                              yaw, msg->twist.twist.linear.x,
                                              msg->twist.twist.linear.y,
                                              &frenet_odom_fixed.pose.pose.position.x,
                                              &frenet_odom_fixed.pose.pose.position.y,
                                              &frenet_odom_fixed.twist.twist.linear.x,
                                              &frenet_odom_fixed.twist.twist.linear.y,
                                              &closest_wpt_index_fixed_);
    // abuse child frame id to pass closest wapoint index
    frenet_odom_fixed.child_frame_id = std::to_string(closest_wpt_index_fixed_);
    frenet_odom_fixed_pub_.publish(frenet_odom_fixed);
  }
}

// interactive marker — full search button
void FrenetRepublisher::CreateFullSearchButton() {
  visualization_msgs::InteractiveMarker int_marker;
  int_marker.header.frame_id = "map";
  int_marker.name = "full_search_button";
  int_marker.description = "Force Full Search";
  int_marker.pose.position.x = 10.0;
  int_marker.pose.position.y = 10.0;
  int_marker.pose.position.z = 0.0;
  int_marker.scale = 0.8;

  // yellow box marker
  visualization_msgs::Marker box;
  box.type = visualization_msgs::Marker::CUBE;
  box.scale.x = 0.8;
  box.scale.y = 0.4;
  box.scale.z = 0.15;
  box.color.r = 1.0;
  box.color.g = 1.0;
  box.color.b = 0.0;
  box.color.a = 1.0;

  // button control — click to trigger
  visualization_msgs::InteractiveMarkerControl button_control;
  button_control.interaction_mode = visualization_msgs::InteractiveMarkerControl::BUTTON;
  button_control.always_visible = true;
  button_control.markers.push_back(box);
  int_marker.controls.push_back(button_control);

  im_server_->insert(int_marker,
      boost::bind(&FrenetRepublisher::FullSearchButtonCallback, this, _1));
  im_server_->applyChanges();
  ROS_INFO("[FrenetRepublisher] Full Search interactive marker created");
}

void FrenetRepublisher::FullSearchButtonCallback(
    const visualization_msgs::InteractiveMarkerFeedbackConstPtr &feedback) {
  if (feedback->event_type == visualization_msgs::InteractiveMarkerFeedback::BUTTON_CLICK) {
    frenet_converter_.ForceFullSearch();
    frenet_converter_fixed_.ForceFullSearch();
    ROS_WARN("[FrenetRepublisher] Full Search triggered by interactive marker!");
  }
}

}// end namespace frenet_odom_republisher

// launch node
int main(int argc, char** argv)
{
  ros::init(argc, argv, "frenet_odom_republisher");
  ros::NodeHandle nh;

  frenet_odom_republisher::FrenetRepublisher node(nh);
  return 0;
}