#include "frenet_conversion.h"
#include <math.h>
#include <ros/ros.h>

namespace frenet_conversion{

  FrenetConverter::FrenetConverter() {}

  FrenetConverter::~FrenetConverter() {}

  /*** -------------- PUBLIC ----------------- ***/

  void FrenetConverter::SetGlobalTrajectory(
                                    const std::vector<f110_msgs::Wpnt> *wptns,
                                    const bool is_closed_contour) {
    // deep copy wptns
    wpt_array_ = *wptns;
    global_trajectory_length_ = wpt_array_.back().s_m;
    is_closed_contour_ = is_closed_contour;
    // remove the first element -> no negative s
    if (is_closed_contour_) {
      wpt_array_.erase(wpt_array_.begin());
    }
    has_global_trajectory_ = true;
  }

  // parse trackbounds markers (even index=left, odd index=right)
  void FrenetConverter::SetTrackBounds(const visualization_msgs::MarkerArrayConstPtr& bounds_msg) {
    left_bounds_.clear();
    right_bounds_.clear();
    // markers alternate: left(id=0), right(id=1), left(id=2), right(id=3), ...
    for (size_t i = 0; i < bounds_msg->markers.size(); i++) {
      const auto& m = bounds_msg->markers[i];
      BoundPoint bp{m.pose.position.x, m.pose.position.y, m.pose.position.z};
      if (i % 2 == 0) {
        left_bounds_.push_back(bp);
      } else {
        right_bounds_.push_back(bp);
      }
    }
    has_track_bounds_ = true;
    ROS_INFO("[FrenetConverter] Track bounds loaded: %zu left, %zu right",
             left_bounds_.size(), right_bounds_.size());
  }

  // added z for 3D frenet conversion
  void FrenetConverter::GetFrenetPoint(const double x, const double y, const double z,
                                       double* s, double* d, int* idx, bool full_search) {

    if (!has_global_trajectory_) {
      ROS_ERROR("[FrenetConverter] No global trajectory set!");
      return;
    }

    UpdateClosestIndex(x, y, z, idx, full_search);

    CalcFrenetPoint(x, y, z, s, d);
  }

  // add z for 3D closest-point search + first_call full search
  // CalcFrenetPoint uses 3D tangent, CalcFrenetVelocity stays 2D (body frame twist)
  void FrenetConverter::GetFrenetOdometry(const double x, const double y,
                                          const double z, const double theta,
                                          const double v_x, const double v_y,
                                          double* s, double* d, double* v_s,
                                          double* v_d, int* idx) {

    if (!has_global_trajectory_) {
      ROS_ERROR("[FrenetConverter] No global trajectory set!");
      return;
    }
    std::unique_lock<std::mutex> lock(mutexGlobalTrajectory_);

    // full search on first call to correctly initialize from any start position
    UpdateClosestIndex(x, y, z, idx, first_call_);
    if (first_call_) first_call_ = false;

    CalcFrenetPoint(x, y, z, s, d);

    CalcFrenetVelocity(v_x, v_y, theta, v_s, v_d);
    lock.unlock();
  }

  // added z output for 3D
  void FrenetConverter::GetGlobalPoint(const double s, const double d,
                                       double* x, double* y, double* z) {

    if (!has_global_trajectory_) {
      ROS_ERROR("[FrenetConverter] No global trajectory set!");
      return;
    }
    std::unique_lock<std::mutex> lock(mutexGlobalTrajectory_);

    UpdateClosestIndex(s);

    CalcGlobalPoint(s, d, x, y);
    *z = wpt_array_.at(closest_idx_).z_m;
    lock.unlock();
  }

  void FrenetConverter::GetClosestIndex(const double s, int* idx) {
    if (!has_global_trajectory_) {
      ROS_ERROR("[FrenetConverter] No global trajectory set!");
      return;
    }
    UpdateClosestIndex(s);
    *idx = (closest_idx_ + 1); // account for removing the first element
  }

  void FrenetConverter::GetClosestIndex(const double x, const double y, 
                                         int* idx) {
    if (!has_global_trajectory_) {
      ROS_ERROR("[FrenetConverter] No global trajectory set!");
      return;
    }
    UpdateClosestIndex(x, y, 0.0, idx, false);
  }

  /*** -------------- PRIVATE ----------------- ***/

  // 3D tangent projection for slopes — use adjacent waypoints to compute 3D tangent
  void FrenetConverter::CalcFrenetPoint(const double x, const double y,
                                             const double z, double* s, double* d) {
    double d_x = x - wpt_array_.at(closest_idx_).x_m;
    double d_y = y - wpt_array_.at(closest_idx_).y_m;
    double d_z = z - wpt_array_.at(closest_idx_).z_m;

    // compute 3D tangent from adjacent waypoints
    int next_idx = (closest_idx_ + 1) % wpt_array_.size();
    double tx = wpt_array_.at(next_idx).x_m - wpt_array_.at(closest_idx_).x_m;
    double ty = wpt_array_.at(next_idx).y_m - wpt_array_.at(closest_idx_).y_m;
    double tz = wpt_array_.at(next_idx).z_m - wpt_array_.at(closest_idx_).z_m;
    double t_norm = std::sqrt(tx*tx + ty*ty + tz*tz);
    if (t_norm > 1e-9) { tx /= t_norm; ty /= t_norm; tz /= t_norm; }

    // lateral direction: perpendicular to tangent in xy plane, then normalize
    // (lateral is horizontal — vehicle doesn't slide sideways off the slope)
    double nx = -ty;
    double ny = tx;
    double n_norm_xy = std::sqrt(nx*nx + ny*ny);
    if (n_norm_xy > 1e-9) { nx /= n_norm_xy; ny /= n_norm_xy; }

    // s = projection onto 3D tangent + waypoint s_m
    *s = (d_x * tx + d_y * ty + d_z * tz) + wpt_array_.at(closest_idx_).s_m;
    if (is_closed_contour_) {
      *s = std::fmod(*s, global_trajectory_length_);
      if (*s < 0) *s += global_trajectory_length_;  // wrap negative s
    }
    // d = projection onto lateral (horizontal normal)
    *d = d_x * nx + d_y * ny;
  }

  void FrenetConverter::CalcGlobalPoint(const double s, const double d,
                                        double* x, double* y) {
    double d_s = s - wpt_array_.at(closest_idx_).s_m;
    *x = wpt_array_.at(closest_idx_).x_m 
        + d_s * std::cos(wpt_array_.at(closest_idx_).psi_rad) 
        - d * std::sin(wpt_array_.at(closest_idx_).psi_rad);
    *y = wpt_array_.at(closest_idx_).y_m 
        + d * std::cos(wpt_array_.at(closest_idx_).psi_rad) 
        + d_s * std::sin(wpt_array_.at(closest_idx_).psi_rad);
  }

  // GLIL base_odom twist is in body frame — body linear.x is already on-surface speed
  void FrenetConverter::CalcFrenetVelocity(const double v_x, const double v_y,
                                          const double theta, double* v_s,
                                          double* v_d) {
    double delta_psi = theta - wpt_array_.at(closest_idx_).psi_rad;
    *v_s = v_x * std::cos(delta_psi) - v_y * std::sin(delta_psi);
    *v_d = v_x * std::sin(delta_psi) + v_y * std::cos(delta_psi);
  }

  // use 3D distance (dx²+dy²+dz²) to disambiguate bridge/slope overlaps
  // TODO speed up by using intelligent search (binary search)
  void FrenetConverter::UpdateClosestIndex(const double x, const double y,
                                           const double z, int* idx,
                                           bool full_search) {
    // get the closest waypoint using 3D distance
    double min_dist = INFINITY;
    int search_ahead_radius = 20; // TODO don't hardcode this
    int start = (wpt_array_.size() + *idx - 1) % wpt_array_.size();
    int end_idx = (start + search_ahead_radius) % wpt_array_.size();

    // -- proximity search: --
    if (!full_search) {
      if (end_idx < start) {
        for (int i = start; i < (int)wpt_array_.size(); i++) {
          double d_squared = std::pow(x - wpt_array_[i].x_m, 2) +
                             std::pow(y - wpt_array_[i].y_m, 2) +
                             std::pow(z - wpt_array_[i].z_m, 2);
          if (d_squared < min_dist) { min_dist = d_squared; closest_idx_ = i; }
        }
        for (int i = 0; i < end_idx; i++) {
          double d_squared = std::pow(x - wpt_array_[i].x_m, 2) +
                             std::pow(y - wpt_array_[i].y_m, 2) +
                             std::pow(z - wpt_array_[i].z_m, 2);
          if (d_squared < min_dist) { min_dist = d_squared; closest_idx_ = i; }
        }
      } else {
        for (int i = start; i < end_idx; i++) {
          double d_squared = std::pow(x - wpt_array_[i].x_m, 2) +
                             std::pow(y - wpt_array_[i].y_m, 2) +
                             std::pow(z - wpt_array_[i].z_m, 2);
          if (d_squared < min_dist) { min_dist = d_squared; closest_idx_ = i; }
        }
      }
    }
    // check if proximity result crosses boundary → trigger full search
    bool boundary_crossed = false;
    if (!full_search && has_track_bounds_ && min_dist < 1.0) {
      boundary_crossed = IsLineCrossingBoundary(x, y,
          wpt_array_[closest_idx_].x_m, wpt_array_[closest_idx_].y_m, z);
    }

    // full search with d_height filter + boundary raycast
    if (min_dist > 1.0 || full_search || boundary_crossed) {
      min_dist = INFINITY;
      for (int i = 0; i < (int)wpt_array_.size(); i++) {
        // d_height filter: skip waypoints on different surface layer
        double d_height = CalcHeightOffset(x, y, z, i);
        if (std::abs(d_height) > height_filter_threshold_) continue;

        double d_squared = std::pow(x - wpt_array_[i].x_m, 2) +
                           std::pow(y - wpt_array_[i].y_m, 2) +
                           std::pow(z - wpt_array_[i].z_m, 2);
        if (d_squared < min_dist) {
          // boundary raycast: skip if line crosses track wall
          if (has_track_bounds_ &&
              IsLineCrossingBoundary(x, y, wpt_array_[i].x_m, wpt_array_[i].y_m, z)) {
            continue;
          }
          min_dist = d_squared;
          closest_idx_ = i;
        }
      }

      // rotational search — if full search result crosses wall, try 90/180/270°
      if (has_track_bounds_ && min_dist < INFINITY &&
          IsLineCrossingBoundary(x, y, wpt_array_[closest_idx_].x_m,
                                wpt_array_[closest_idx_].y_m, z)) {
        double vec_x = wpt_array_[closest_idx_].x_m - x;
        double vec_y = wpt_array_[closest_idx_].y_m - y;

        int angles[] = {90, 180, 270};
        for (int angle_deg : angles) {
          double angle_rad = angle_deg * M_PI / 180.0;
          double cos_a = std::cos(angle_rad);
          double sin_a = std::sin(angle_rad);

          double target_x = x + vec_x * cos_a - vec_y * sin_a;
          double target_y = y + vec_x * sin_a + vec_y * cos_a;

          // find nearest height-filtered waypoint to rotated target
          double target_min_dist = INFINITY;
          int candidate_idx = -1;
          for (int i = 0; i < (int)wpt_array_.size(); i++) {
            double dh = CalcHeightOffset(x, y, z, i);
            if (std::abs(dh) > height_filter_threshold_) continue;
            double d_sq = std::pow(target_x - wpt_array_[i].x_m, 2) +
                          std::pow(target_y - wpt_array_[i].y_m, 2);
            if (d_sq < target_min_dist) {
              target_min_dist = d_sq;
              candidate_idx = i;
            }
          }

          if (candidate_idx < 0) continue;
          if (IsLineCrossingBoundary(x, y, wpt_array_[candidate_idx].x_m,
                                    wpt_array_[candidate_idx].y_m, z)) continue;

          // search s-direction from candidate for closest non-wall waypoint
          int best_idx = candidate_idx;
          double best_dist = std::pow(x - wpt_array_[candidate_idx].x_m, 2) +
                             std::pow(y - wpt_array_[candidate_idx].y_m, 2) +
                             std::pow(z - wpt_array_[candidate_idx].z_m, 2);
          int n = (int)wpt_array_.size();

          // forward
          for (int off = 1; off < n; off++) {
            int ti = (candidate_idx + off) % n;
            double dh = CalcHeightOffset(x, y, z, ti);
            if (std::abs(dh) > height_filter_threshold_) break;
            if (IsLineCrossingBoundary(x, y, wpt_array_[ti].x_m,
                                      wpt_array_[ti].y_m, z)) break;
            double d_sq = std::pow(x - wpt_array_[ti].x_m, 2) +
                          std::pow(y - wpt_array_[ti].y_m, 2) +
                          std::pow(z - wpt_array_[ti].z_m, 2);
            if (d_sq < best_dist) { best_dist = d_sq; best_idx = ti; }
          }

          // backward
          for (int off = 1; off < n; off++) {
            int ti = (candidate_idx - off + n) % n;
            double dh = CalcHeightOffset(x, y, z, ti);
            if (std::abs(dh) > height_filter_threshold_) break;
            if (IsLineCrossingBoundary(x, y, wpt_array_[ti].x_m,
                                      wpt_array_[ti].y_m, z)) break;
            double d_sq = std::pow(x - wpt_array_[ti].x_m, 2) +
                          std::pow(y - wpt_array_[ti].y_m, 2) +
                          std::pow(z - wpt_array_[ti].z_m, 2);
            if (d_sq < best_dist) { best_dist = d_sq; best_idx = ti; }
          }

          closest_idx_ = best_idx;
          ROS_DEBUG_THROTTLE(1.0, "[FrenetConverter] Wall avoided via %d rotation", angle_deg);
          break;
        }
      }
      // fallback — if all filters failed or result too far, use simple 3D nearest
      double max_valid_dist_sq = 2.0 * 2.0;  // 2m threshold
      if (min_dist == INFINITY || min_dist > max_valid_dist_sq) {
        ROS_WARN_THROTTLE(1.0, "[FrenetConverter] Filters failed or dist>2m (%.2fm), using simple 3D nearest",
                          std::sqrt(min_dist));
        for (int i = 0; i < (int)wpt_array_.size(); i++) {
          double d_squared = std::pow(x - wpt_array_[i].x_m, 2) +
                             std::pow(y - wpt_array_[i].y_m, 2) +
                             std::pow(z - wpt_array_[i].z_m, 2);
          if (d_squared < min_dist) {
            min_dist = d_squared;
            closest_idx_ = i;
          }
        }
      }
    }
    *idx = (closest_idx_ + 1); // account for removing the first element
  }

  void FrenetConverter::UpdateClosestIndex(const double s) {
    closest_idx_ = wpt_array_.size() - 1; // lazy don't handle wrapping
    for (int i = 1; i < wpt_array_.size(); i++) {
      // find first waypoint with s > s
      if (wpt_array_[i].s_m > s) {
        if (wpt_array_[i].s_m - s > s - wpt_array_[i-1].s_m) {
          closest_idx_ = i - 1;
          break;
        } else {
          closest_idx_ = i;
          break;
        }
      }
    }
  }

  // compute height offset from waypoint's local surface normal
  double FrenetConverter::CalcHeightOffset(const double x, const double y,
                                           const double z, int wpt_idx) {
    const auto& wpt = wpt_array_.at(wpt_idx);
    double dx = x - wpt.x_m;
    double dy = y - wpt.y_m;
    double dz = z - wpt.z_m;
    // normal = (cos(psi)*sin(mu), sin(psi)*sin(mu), cos(mu))
    double sin_mu = std::sin(wpt.mu_rad);
    double cos_mu = std::cos(wpt.mu_rad);
    double sin_psi = std::sin(wpt.psi_rad);
    double cos_psi = std::cos(wpt.psi_rad);
    return dx * cos_psi * sin_mu + dy * sin_psi * sin_mu + dz * cos_mu;
  }

  // check if line segment (x1,y1)→(x2,y2) crosses any z-filtered boundary
  bool FrenetConverter::IsLineCrossingBoundary(const double x1, const double y1,
                                               const double x2, const double y2,
                                               const double z_ref) {
    if (!has_track_bounds_) return false;

    // check left boundary segments
    for (size_t i = 0; i + 1 < left_bounds_.size(); i++) {
      // z filter: skip boundary segments not near z_ref
      double seg_z = (left_bounds_[i].z + left_bounds_[i+1].z) * 0.5;
      if (std::abs(seg_z - z_ref) > z_boundary_margin_) continue;
      if (SegmentsIntersect2D(x1, y1, x2, y2,
                              left_bounds_[i].x, left_bounds_[i].y,
                              left_bounds_[i+1].x, left_bounds_[i+1].y)) {
        return true;
      }
    }
    // check right boundary segments
    for (size_t i = 0; i + 1 < right_bounds_.size(); i++) {
      double seg_z = (right_bounds_[i].z + right_bounds_[i+1].z) * 0.5;
      if (std::abs(seg_z - z_ref) > z_boundary_margin_) continue;
      if (SegmentsIntersect2D(x1, y1, x2, y2,
                              right_bounds_[i].x, right_bounds_[i].y,
                              right_bounds_[i+1].x, right_bounds_[i+1].y)) {
        return true;
      }
    }
    return false;
  }

  // 2D line segment intersection (cross product method)
  bool FrenetConverter::SegmentsIntersect2D(double ax, double ay, double bx, double by,
                                             double cx, double cy, double dx, double dy) {
    double d1 = (dx-cx)*(ay-cy) - (dy-cy)*(ax-cx);
    double d2 = (dx-cx)*(by-cy) - (dy-cy)*(bx-cx);
    double d3 = (bx-ax)*(cy-ay) - (by-ay)*(cx-ax);
    double d4 = (bx-ax)*(dy-ay) - (by-ay)*(dx-ax);
    if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
        ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) {
      return true;
    }
    return false;
  }

} // end of namespace frenet_conversion
