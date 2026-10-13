from typing import Union
import numpy as np
from scipy.interpolate import CubicSpline
import rospy
from visualization_msgs.msg import MarkerArray

# 3D frenet converter — added z support, height filter, boundary raycast
class FrenetConverter:
    def __init__(self, waypoints_x: np.array, waypoints_y: np.array, waypoints_z: np.array = None):
        self.waypoints_x = waypoints_x
        self.waypoints_y = waypoints_y
        self.waypoints_z = waypoints_z if waypoints_z is not None else np.zeros_like(waypoints_x)
        self.waypoints_s = None
        self.spline_x = None
        self.spline_y = None
        self.spline_z = None
        self.raceline_length = None
        self.waypoints_distance_m = None  # computed from actual waypoint spacing in build_raceline
        self.iter_max = 3
        self.has_z = waypoints_z is not None

        # track boundary data for wall-crossing detection (mirrors C++ FrenetConverter)
        self.left_bounds = None   # Nx3 array (x, y, z)
        self.right_bounds = None  # Nx3 array (x, y, z)
        self.has_track_bounds = False
        self.height_filter_threshold = 0.10  # [m] same as C++
        self.z_boundary_margin = 0.10        # [m] same as C++

        # precomputed waypoint psi and mu for height offset calculation
        self.waypoints_psi = None
        self.waypoints_mu = None

        self.build_raceline()
        self._load_track_bounds()

    def build_raceline(self):
        # use 3D distance for s calculation
        self.waypoints_s = [0.0]
        prev_x = self.waypoints_x[0]
        prev_y = self.waypoints_y[0]
        prev_z = self.waypoints_z[0]
        for wx, wy, wz in zip(self.waypoints_x[1:], self.waypoints_y[1:], self.waypoints_z[1:]):
            dist = np.linalg.norm([wx - prev_x, wy - prev_y, wz - prev_z])
            prev_x, prev_y, prev_z = wx, wy, wz
            self.waypoints_s.append(self.waypoints_s[-1] + dist)
        self.waypoints_s = np.array(self.waypoints_s)
        self.spline_x = CubicSpline(self.waypoints_s, self.waypoints_x)
        self.spline_y = CubicSpline(self.waypoints_s, self.waypoints_y)
        self.spline_z = CubicSpline(self.waypoints_s, self.waypoints_z)
        self.raceline_length = self.waypoints_s[-1]
        # compute actual median waypoint spacing
        self.waypoints_distance_m = float(np.median(np.diff(self.waypoints_s)))

        # precompute psi and mu at each waypoint for height offset
        dx = self.spline_x(self.waypoints_s, 1)
        dy = self.spline_y(self.waypoints_s, 1)
        dz = self.spline_z(self.waypoints_s, 1)
        self.waypoints_psi = np.arctan2(dy, dx)
        ds_xy = np.sqrt(dx**2 + dy**2)
        self.waypoints_mu = np.arctan2(dz, ds_xy)  # pitch angle

    # auto-load track bounds from /trackbounds/markers topic (once)
    def _load_track_bounds(self):
        try:
            msg = rospy.wait_for_message('/trackbounds/markers', MarkerArray, timeout=5.0)
            left = []
            right = []
            for i, m in enumerate(msg.markers):
                pt = [m.pose.position.x, m.pose.position.y, m.pose.position.z]
                if i % 2 == 0:
                    left.append(pt)
                else:
                    right.append(pt)
            self.set_track_bounds(left, right)
            rospy.logwarn(f"[FrenetConverter] Track bounds loaded: {len(left)} left, {len(right)} right")
        except rospy.ROSException:
            rospy.logwarn("[FrenetConverter] No trackbounds received (timeout 5s), boundary check disabled")

    # set track bounds — precompute segment arrays for vectorized boundary check
    def set_track_bounds(self, left_bounds, right_bounds):
        """
        Set track boundary data for wall-crossing detection.
        Precomputes segment start/end arrays and z averages for fast vectorized checks.
        left_bounds, right_bounds: Nx3 arrays [[x, y, z], ...]
        """
        self.left_bounds = np.array(left_bounds)
        self.right_bounds = np.array(right_bounds)

        # precompute segment arrays: start xy, end xy, z average
        self.left_seg_start = self.left_bounds[:-1, :2]
        self.left_seg_end = self.left_bounds[1:, :2]
        self.left_seg_z_avg = (self.left_bounds[:-1, 2] + self.left_bounds[1:, 2]) * 0.5

        self.right_seg_start = self.right_bounds[:-1, :2]
        self.right_seg_end = self.right_bounds[1:, :2]
        self.right_seg_z_avg = (self.right_bounds[:-1, 2] + self.right_bounds[1:, 2]) * 0.5

        self.has_track_bounds = True

    # set track bounds from ROS MarkerArray dict (from global_waypoints.json)
    def set_track_bounds_from_markers(self, markers):
        """
        Parse trackbounds from MarkerArray (alternating left/right markers).
        markers: list of marker dicts with pose.position {x, y, z}
        """
        left = []
        right = []
        for i, m in enumerate(markers):
            pos = m['pose']['position']
            pt = [pos['x'], pos['y'], pos['z']]
            if i % 2 == 0:
                left.append(pt)
            else:
                right.append(pt)
        self.set_track_bounds(left, right)

    # height offset calculation — mirrors C++ CalcHeightOffset
    def _calc_height_offset(self, x, y, z, wpt_idx):
        """
        Calculate height offset from track surface normal at waypoint.
        Positive = above surface, negative = below.
        """
        dx = x - self.waypoints_x[wpt_idx]
        dy = y - self.waypoints_y[wpt_idx]
        dz = z - self.waypoints_z[wpt_idx]
        psi = self.waypoints_psi[wpt_idx]
        mu = self.waypoints_mu[wpt_idx]
        sin_mu = np.sin(mu)
        cos_mu = np.cos(mu)
        sin_psi = np.sin(psi)
        cos_psi = np.cos(psi)
        return dx * cos_psi * sin_mu + dy * sin_psi * sin_mu + dz * cos_mu

    # vectorized boundary crossing check — z filter + cross product, no for loop
    def _is_line_crossing_boundary(self, x1, y1, x2, y2, z_ref):
        if not self.has_track_bounds:
            return False

        for seg_start, seg_end, seg_z_avg in [
            (self.left_seg_start, self.left_seg_end, self.left_seg_z_avg),
            (self.right_seg_start, self.right_seg_end, self.right_seg_z_avg),
        ]:
            # z filter: only check segments near z_ref
            z_mask = np.abs(seg_z_avg - z_ref) <= self.z_boundary_margin
            if not np.any(z_mask):
                continue

            # extract filtered segment endpoints
            cx = seg_start[z_mask, 0]
            cy = seg_start[z_mask, 1]
            dx = seg_end[z_mask, 0]
            dy = seg_end[z_mask, 1]

            # vectorized cross product intersection test
            d1 = (dx - cx) * (y1 - cy) - (dy - cy) * (x1 - cx)
            d2 = (dx - cx) * (y2 - cy) - (dy - cy) * (x2 - cx)
            d3 = (x2 - x1) * (cy - y1) - (y2 - y1) * (cx - x1)
            d4 = (x2 - x1) * (dy - y1) - (y2 - y1) * (dx - x1)

            intersects = ((d1 > 0) != (d2 > 0)) & ((d3 > 0) != (d4 > 0))
            if np.any(intersects):
                return True

        return False

    def get_frenet(self, x, y, s=None) -> np.array:
        # Compute Frenet coordinates for a given (x, y) point
        if s is None:
            s = self.get_approx_s(x, y)
            s, d = self.get_frenet_coord(x, y, s)
        else:
            s, d = self.get_frenet_coord(x, y, s)

        return np.array([s, d])

    def get_approx_s(self, x, y) -> float:
        """
        Finds the s-coordinate of the given point by finding the nearest waypoint.
        """
        lenx = len(x)
        dist_x = x - np.tile(self.waypoints_x, (lenx, 1)).T
        dist_y = y - np.tile(self.waypoints_y, (lenx, 1)).T
        dist_2d = np.linalg.norm([dist_x.T, dist_y.T], axis=0)
        # use actual s values instead of index * fixed_distance
        return self.waypoints_s[np.argmin(dist_2d, axis=1)]

    # 3D nearest search with height filter + boundary raycast + rotational search
    def get_approx_s_3d(self, x, y, z) -> float:
        """
        3D nearest waypoint search with height filter, boundary raycast,
        and rotational search fallback. Mirrors C++ UpdateClosestIndex logic.
        """
        lenx = len(x)
        result_indices = np.zeros(lenx, dtype=int)

        for qi in range(lenx):
            qx, qy, qz = x[qi], y[qi], z[qi]

            # vectorized height filter
            dx_all = qx - self.waypoints_x
            dy_all = qy - self.waypoints_y
            dz_all = qz - self.waypoints_z
            d_height = (dx_all * np.cos(self.waypoints_psi) * np.sin(self.waypoints_mu) +
                        dy_all * np.sin(self.waypoints_psi) * np.sin(self.waypoints_mu) +
                        dz_all * np.cos(self.waypoints_mu))
            height_mask = np.abs(d_height) <= self.height_filter_threshold

            # 3D distance for candidates that pass height filter
            d_sq_all = dx_all**2 + dy_all**2 + dz_all**2
            d_sq = d_sq_all.copy()
            d_sq[~height_mask] = np.inf

            # find nearest waypoint among height-filtered candidates
            nearest_idx = np.argmin(d_sq)

            # fallback if all filtered out OR nearest > 2m
            max_valid_dist_sq = 2.0 * 2.0  # 2m threshold
            if d_sq[nearest_idx] == np.inf or d_sq[nearest_idx] > max_valid_dist_sq:
                # fallback: use simple 3D nearest
                result_indices[qi] = np.argmin(d_sq_all)
                continue

            # check boundary crossing for nearest candidate
            if not self.has_track_bounds or \
               not self._is_line_crossing_boundary(qx, qy,
                                                    self.waypoints_x[nearest_idx],
                                                    self.waypoints_y[nearest_idx], qz):
                # no wall crossing — use this waypoint
                result_indices[qi] = nearest_idx
                continue

            # wall detected — rotational search (90°, 180°, 270°)
            vec_x = self.waypoints_x[nearest_idx] - qx
            vec_y = self.waypoints_y[nearest_idx] - qy
            found = False

            for angle_deg in [90, 180, 270]:
                angle_rad = np.radians(angle_deg)
                cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                target_x = qx + vec_x * cos_a - vec_y * sin_a
                target_y = qy + vec_x * sin_a + vec_y * cos_a

                # find nearest waypoint to rotated target (height-filtered)
                d_sq_target = (self.waypoints_x - target_x)**2 + (self.waypoints_y - target_y)**2
                d_sq_target[~height_mask] = np.inf
                candidate_idx = int(np.argmin(d_sq_target))

                if d_sq_target[candidate_idx] == np.inf:
                    continue

                if self._is_line_crossing_boundary(qx, qy,
                                                    self.waypoints_x[candidate_idx],
                                                    self.waypoints_y[candidate_idx], qz):
                    continue

                # found valid candidate — search s-direction for closest
                best_idx = candidate_idx
                best_dist = d_sq_all[candidate_idx]
                n_wpts = len(self.waypoints_x)

                # forward along s
                for s_off in range(1, n_wpts):
                    test_idx = (candidate_idx + s_off) % n_wpts
                    if not height_mask[test_idx]:
                        break
                    if self._is_line_crossing_boundary(qx, qy,
                                                        self.waypoints_x[test_idx],
                                                        self.waypoints_y[test_idx], qz):
                        break
                    if d_sq_all[test_idx] < best_dist:
                        best_dist = d_sq_all[test_idx]
                        best_idx = test_idx

                # backward along s
                for s_off in range(1, n_wpts):
                    test_idx = (candidate_idx - s_off) % n_wpts
                    if not height_mask[test_idx]:
                        break
                    if self._is_line_crossing_boundary(qx, qy,
                                                        self.waypoints_x[test_idx],
                                                        self.waypoints_y[test_idx], qz):
                        break
                    if d_sq_all[test_idx] < best_dist:
                        best_dist = d_sq_all[test_idx]
                        best_idx = test_idx

                result_indices[qi] = best_idx
                found = True
                break

            if not found:
                # all directions blocked — use nearest (ignoring walls)
                result_indices[qi] = nearest_idx

        # use actual s values instead of index * fixed_distance (3D spacing is non-uniform)
        return self.waypoints_s[result_indices]

    def get_frenet_3d(self, x, y, z, s=None) -> np.array:
        """
        3D frenet conversion — uses z for nearest search with height filter + boundary
        """
        if s is None:
            s = self.get_approx_s_3d(x, y, z)
            s, d = self.get_frenet_coord(x, y, s)
        else:
            s, d = self.get_frenet_coord(x, y, s)
        return np.array([s, d])

    def get_frenet_coord(self, x, y, s, eps_m=0.01) -> float:
        """
        Finds the s-coordinate of the given point, considering the perpendicular
        projection of the point on the track.
        """
        _, projection, d = self.check_perpendicular(x, y, s, eps_m)
        for i in range(self.iter_max):
            cand_s = (s + projection)%self.raceline_length
            _, cand_projection, cand_d = self.check_perpendicular(x, y, cand_s, eps_m)
            cand_projection = np.clip(cand_projection, -self.waypoints_distance_m/(2*self.iter_max), self.waypoints_distance_m/(2*self.iter_max))
            updated_idxs = np.abs(cand_projection) <= np.abs(projection)
            d[updated_idxs] = cand_d[updated_idxs]
            s[updated_idxs] = cand_s[updated_idxs]
            projection[updated_idxs] = cand_projection[updated_idxs]

        return s, d

    def check_perpendicular(self, x, y, s, eps_m=0.01) -> Union[bool, float]:
        # obtain unit vector parallel to the track
        dx_ds, dy_ds = self.get_derivative(s)
        tangent = np.array([dx_ds, dy_ds])
        if np.any(np.isnan(s)):
            raise ValueError("BUB FRENET CONVERTER: S is nan")
        tangent /= np.linalg.norm(tangent, axis=0)

        # obtain vector from the track to the point
        x_vec = x - self.spline_x(s)
        y_vec = y - self.spline_y(s)
        point_to_track = np.array([x_vec, y_vec])

        # computes the projection of point_to_track on tangent
        proj = np.einsum('ij,ij->j', tangent, point_to_track)
        perps = np.array([-tangent[1, :], tangent[0, :]])
        d = np.einsum('ij,ij->j', perps, point_to_track)

        check_perpendicular = None

        return check_perpendicular, proj, d

    def get_derivative(self, s) -> np.array:
        """
        Returns the derivative of the point corresponding to s on the chosen line.
        """
        s = s%self.raceline_length
        der = [self.spline_x(s, 1), self.spline_y(s, 1)]
        return der


    def get_cartesian(self, s: float, d: float) -> np.array:
        """
        Convert Frenet coordinates to Cartesian coordinates
        """
        x = self.spline_x(s)
        y = self.spline_y(s)
        psi = self.get_derivative(s)
        psi = np.arctan2(psi[1], psi[0])
        x += d * np.cos(psi + np.pi / 2)
        y += d * np.sin(psi + np.pi / 2)

        return np.array([x, y])

    # 3D cartesian conversion
    def get_cartesian_3d(self, s: float, d: float) -> np.array:
        """
        Convert Frenet coordinates to 3D Cartesian coordinates
        """
        x = self.spline_x(s)
        y = self.spline_y(s)
        z = self.spline_z(s)
        psi = self.get_derivative(s)
        psi = np.arctan2(psi[1], psi[0])
        x += d * np.cos(psi + np.pi / 2)
        y += d * np.sin(psi + np.pi / 2)
        return np.array([x, y, z])

    def get_e_psi(self, x: float, y:float, yaw:float) -> float:
        """
        Calculate E_psi: the heading error between vehicle yaw and track direction.
        """
        s = self.get_approx_s(np.array([x]), np.array([y]))[0]
        psi = np.arctan2(*self.get_derivative(s)[::-1])  # dy/ds, dx/ds → arctan2(dy, dx)

        e_psi = yaw - psi
        e_psi = (e_psi + np.pi) % (2 * np.pi) - np.pi  # normalize to [-pi, pi]

        return e_psi
