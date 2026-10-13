#ifndef FRENET_ODOM_REPUBLISHER_H_
#define FRENET_ODOM_REPUBLISHER_H_

#include <ros/ros.h>
#include <vector>

#include <nav_msgs/Odometry.h>
#include <nav_msgs/OccupancyGrid.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/InteractiveMarker.h>
#include <visualization_msgs/InteractiveMarkerControl.h>
#include <visualization_msgs/InteractiveMarkerFeedback.h>
#include <interactive_markers/interactive_marker_server.h>
#include <f110_msgs/WpntArray.h>
#include <f110_msgs/OTWpntArray.h>
#include <f110_msgs/Wpnt.h>
#include <std_msgs/Bool.h>

#include "frenet_conversion.h"

namespace frenet_odom_republisher{
    
class FrenetRepublisher {
 public:
  FrenetRepublisher(ros::NodeHandle& nh);
  ~FrenetRepublisher();
 private:
  void InitSubscribersPublishers();
  void GlobalTrajectoryCallback(const f110_msgs::WpntArrayConstPtr &wpt_array);
  void FixedPathTrajectoryCallback(const f110_msgs::OTWpntArrayConstPtr &wpt_array);
  void OdomCallback(const nav_msgs::OdometryConstPtr &msg);
  void MapCallback(const nav_msgs::OccupancyGridConstPtr &map_msg);
  // receive trackbounds once for wall-crossing detection
  void TrackBoundsCallback(const visualization_msgs::MarkerArrayConstPtr &bounds_msg);

  // interactive marker for forcing full search
  void FullSearchButtonCallback(const visualization_msgs::InteractiveMarkerFeedbackConstPtr &feedback);
  void CreateFullSearchButton();

  ros::NodeHandle nh_;
  ros::Subscriber global_trajectory_sub_;
  ros::Subscriber fixed_path_trajectory_sub_;
  ros::Subscriber odom_sub_;
  ros::Subscriber map_sub_;
  ros::Subscriber trackbounds_sub_;  // trackbounds once
  ros::Publisher frenet_odom_pub_;
  ros::Publisher frenet_odom_fixed_pub_;

  std::vector<f110_msgs::Wpnt> wpt_array_;
  std::vector<f110_msgs::Wpnt> fixed_path_wpt_array_;
  int closest_wpt_index_{0};
  int closest_wpt_index_fixed_{0};

  frenet_conversion::FrenetConverter frenet_converter_;
  frenet_conversion::FrenetConverter frenet_converter_fixed_;

  bool has_global_trajectory_{false};
  bool has_fixed_path_trajectory_{false};

  // full search interactive marker
  std::shared_ptr<interactive_markers::InteractiveMarkerServer> im_server_;
  bool force_full_search_{false};
};
   
}// end namespace frenet_odom_republisher

#endif /* FRENET_ODOM_REPUBLISHER_H_ */