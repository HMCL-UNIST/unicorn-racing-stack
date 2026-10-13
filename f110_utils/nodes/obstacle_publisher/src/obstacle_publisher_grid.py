#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
import yaml
import os
from geometry_msgs.msg import PointStamped
from f110_msgs.msg import ObstacleArray, Obstacle, WpntArray, Wpnt, OpponentTrajectory, OppWpnt
from visualization_msgs.msg import Marker, MarkerArray, InteractiveMarkerFeedback
from nav_msgs.msg import OccupancyGrid, Odometry
from frenet_conversion.srv import Glob2FrenetArr, Frenet2GlobArr
from std_msgs.msg import Bool, String


class ObstaclePublisherGrid:
    """Publish a dynamic obstacle by modifying occupancy grid in real-time

    This node publishes a dynamic obstacle by drawing pixels on the occupancy grid,
    which allows LiDAR-based detection to naturally handle FOV limitations.
    Unlike ground-truth tracking, obstacles are only detected when visible to sensors.

    Features:
    - Rectangular obstacle (65cm length x 35cm width) following GB heading
    - Switches between GB raceline and Fixed path based on Smart Static mode
    - Realistic FOV simulation (only detects visible obstacles)

    Attributes
    ----------
        speed_scaler: float
            Speed multiplier for the obstacle (follows raceline speed * scaler)
        starting_s_gb: float
            Initial starting position in GB Frenet s coordinate
        starting_s_fixed: float
            Initial starting position in Fixed Frenet s coordinate
        obstacle_length_m: float
            Length of obstacle in heading direction (default: 0.65m)
        obstacle_width_m: float
            Width of obstacle in normal direction (default: 0.35m)
        update_rate: int
            Rate at which to update occupancy grid (Hz)
    """
    def __init__(self):
        # ===== Parameters =====
        self.update_rate = rospy.get_param("~update_rate", 20)  # Hz
        self.rate = rospy.Rate(self.update_rate)
        self.looptime = 1.0 / self.update_rate

        self.speed_scaler = rospy.get_param("~speed_scaler", 1.0)
        self.constant_speed = rospy.get_param("~constant_speed", False)
        self.starting_s_gb = rospy.get_param("~start_s_gb", 0.0)
        self.starting_s_fixed = rospy.get_param("~start_s_fixed", 0.0)

        # ===== Smart Static mode enable/disable =====
        self.enable_smart = rospy.get_param("~enable_smart", False)  # Enable Smart Static trajectory

        # ===== Reactive mode parameters =====
        self.reactive = rospy.get_param("~reactive", False)  # Enable reactive mode

        # Reactive mode state variables
        self.reactive_phase = 0.0  # Phase of sine wave (radians)
        self.reactive_amplitude = 0.0  # Current amplitude (randomly varies per cycle)
        self.reactive_frequency = 0.6  # Frequency (cycles per second in time domain) - lower = slower oscillation
        self.prev_phase = 0.0  # Track phase to detect cycle completion
        self.max_amplitude_limit = 0.5  # Maximum allowed amplitude in meters
        self.prev_d_perturbation = 0.0  # Previous d perturbation for derivative calculation

        # Rectangular obstacle dimensions
        self.obstacle_length_m = rospy.get_param("~obstacle_length_m", 0.65)  # Heading direction
        self.obstacle_width_m = rospy.get_param("~obstacle_width_m", 0.35)    # Normal direction

        # ===== Map data =====
        self.map_name = rospy.get_param('/map', 'default_map')
        # Use catkin workspace path to find maps directory
        # This file is in: catkin_ws/src/race_stack/f110_utils/nodes/obstacle_publisher/src/
        # We need to go to: catkin_ws/src/race_stack/stack_master/maps/
        pkg_path = os.path.dirname(os.path.abspath(__file__))
        # Go up 4 levels: src -> obstacle_publisher -> nodes -> f110_utils -> race_stack
        race_stack_path = os.path.abspath(os.path.join(pkg_path, '../../../..'))
        self.map_dir = os.path.join(race_stack_path, 'stack_master', 'maps', self.map_name)

        self.original_map = None  # Original occupancy grid (grayscale image)
        self.current_map = None   # Current occupancy grid with obstacle
        self.map_resolution = None
        self.map_origin_x = None
        self.map_origin_y = None
        self.map_height = None

        # ===== Obstacle state =====
        self.current_s_gb = self.starting_s_gb
        self.current_s_fixed = self.starting_s_fixed
        self.current_speed = 0.0
        self.prev_obstacle_pixels = []  # Track pixels to erase

        # Smart Static mode switching
        self.smart_static_active = False

        # Only subscribe to Smart Static topics if enabled
        if self.enable_smart:
            rospy.Subscriber("/planner/avoidance/smart_static_active", Bool, self.smart_static_cb, queue_size=1)
            rospy.Subscriber("/planner/avoidance/smart_static_avoidance_wpnts", WpntArray, self.smart_trajectory_cb, queue_size=1)
            rospy.loginfo("[ObstaclePublisherGrid] Smart Static mode ENABLED")
        else:
            rospy.loginfo("[ObstaclePublisherGrid] Smart Static mode DISABLED - using GB only")

        # Subscribe to simulator clear obstacles button (reload original map without static obstacles)
        rospy.Subscriber("/racecar_sim/feedback", InteractiveMarkerFeedback, self.clear_obstacles_cb, queue_size=1)

        # Store both GB and Fixed trajectories
        # Each trajectory is a list of dicts: {'x_m', 'y_m', 's_m', 'd_m', 'vx_mps', 'heading', 'd_left', 'd_right'}
        self.trajectory_gb = None
        self.trajectory_fixed = None
        self.trajectory_gb_s_array = None
        self.trajectory_fixed_s_array = None
        self.max_s_gb = None
        self.max_s_fixed = None

        # ===== ROS Services =====
        rospy.wait_for_service("convert_glob2frenetarr_service")
        rospy.wait_for_service("convert_frenet2globarr_service")
        self.glob2frenet = rospy.ServiceProxy("convert_glob2frenetarr_service", Glob2FrenetArr)
        self.frenet2glob = rospy.ServiceProxy("convert_frenet2globarr_service", Frenet2GlobArr)

        # ===== ROS Publishers =====
        self.map_pub = rospy.Publisher("/map", OccupancyGrid, queue_size=1)
        self.obstacle_marker_pub = rospy.Publisher("/dynamic_obstacle_marker", MarkerArray, queue_size=1)

        rospy.loginfo("[ObstaclePublisherGrid] Initialized with params:")
        rospy.loginfo(f"  - Speed scaler: {self.speed_scaler}")
        rospy.loginfo(f"  - Constant speed: {self.constant_speed}")
        rospy.loginfo(f"  - Starting s (GB): {self.starting_s_gb}")
        rospy.loginfo(f"  - Starting s (Fixed): {self.starting_s_fixed}")
        rospy.loginfo(f"  - Obstacle size: {self.obstacle_length_m}m x {self.obstacle_width_m}m")
        rospy.loginfo(f"  - Update rate: {self.update_rate}Hz")
        rospy.loginfo(f"  - Reactive mode: {self.reactive}")
        if self.reactive:
            rospy.loginfo(f"    - Frequency: {self.reactive_frequency} Hz")
        rospy.loginfo(f"  - Map: {self.map_name}")

    def smart_static_cb(self, msg: Bool):
        """Callback for Smart Static mode switching"""
        prev_state = self.smart_static_active
        self.smart_static_active = msg.data

        if prev_state != self.smart_static_active:
            mode_name = "FIXED" if self.smart_static_active else "GB"
            rospy.loginfo(f"[ObstaclePublisherGrid] Switched to {mode_name} trajectory")

    def smart_trajectory_cb(self, msg: WpntArray):
        """Callback for Smart Static trajectory updates (dynamically loaded)"""
        if len(msg.wpnts) == 0:
            rospy.logwarn("[ObstaclePublisherGrid] Received empty Smart Static trajectory")
            return

        # Extract data from waypoints
        s_array = np.array([wpnt.s_m for wpnt in msg.wpnts])
        x_array = np.array([wpnt.x_m for wpnt in msg.wpnts])
        y_array = np.array([wpnt.y_m for wpnt in msg.wpnts])
        heading_array = np.array([wpnt.psi_rad for wpnt in msg.wpnts])
        d_array = np.array([wpnt.d_m for wpnt in msg.wpnts])
        d_left_array = np.array([wpnt.d_left for wpnt in msg.wpnts])
        d_right_array = np.array([wpnt.d_right for wpnt in msg.wpnts])

        self.max_s_fixed = s_array[-1]

        # Use GB speed profile (resampled to Fixed s values) if available
        if self.trajectory_gb is not None:
            gb_speeds = np.array([wpnt['vx_mps'] for wpnt in self.trajectory_gb])
            speed_array = np.interp(s_array, self.trajectory_gb_s_array, gb_speeds)
        else:
            # Fallback: use speed from waypoints or constant speed
            if self.constant_speed:
                speed_array = np.ones(len(msg.wpnts)) * self.speed_scaler
            else:
                speed_array = np.array([wpnt.vx_mps * self.speed_scaler for wpnt in msg.wpnts])

        # Store trajectory
        self.trajectory_fixed = []
        for i in range(len(s_array)):
            wpnt = {
                'x_m': x_array[i],
                'y_m': y_array[i],
                's_m': s_array[i],
                'd_m': d_array[i],
                'vx_mps': speed_array[i],
                'heading': heading_array[i],
                'd_left': d_left_array[i],
                'd_right': d_right_array[i]
            }
            self.trajectory_fixed.append(wpnt)

        self.trajectory_fixed_s_array = s_array

        rospy.loginfo(f"[ObstaclePublisherGrid] Updated Fixed trajectory: {len(self.trajectory_fixed)} waypoints")
        rospy.loginfo(f"  - Max s: {self.max_s_fixed:.2f}m")
        rospy.loginfo(f"  - Speed range: [{speed_array.min():.2f}, {speed_array.max():.2f}] m/s")

    def clear_obstacles_cb(self, msg: InteractiveMarkerFeedback):
        """Callback for simulator clear obstacles button - reload original map without static obstacles"""
        # Check if this is the clear_obstacles button being clicked
        if msg.marker_name == "clear_obstacles" and msg.event_type == 3:  # event_type 3 = button click
            rospy.loginfo("[ObstaclePublisherGrid] Clear obstacles button clicked - reloading original map")

            # Reload original map without static obstacles
            map_png_path = os.path.join(self.map_dir, f'{self.map_name}.png')

            if os.path.exists(map_png_path):
                new_map = cv2.imread(map_png_path, cv2.IMREAD_GRAYSCALE)
                if new_map is not None:
                    self.original_map = new_map
                    self.current_map = self.original_map.copy()
                    rospy.loginfo(f"[ObstaclePublisherGrid] Reloaded original map: {map_png_path}")
                else:
                    rospy.logerr(f"[ObstaclePublisherGrid] Failed to read map: {map_png_path}")
            else:
                rospy.logerr(f"[ObstaclePublisherGrid] Original map not found: {map_png_path}")

    def load_map(self):
        """Load map with static obstacles from PNG file"""
        rospy.loginfo("[ObstaclePublisherGrid] Loading map with static obstacles...")

        # Try to load map with static obstacles (e.g., gym_1025_obstacles_only_nearby.png)
        map_with_obstacles_path = os.path.join(self.map_dir, f'{self.map_name}_obstacles_only_nearby.png')
        map_yaml_path = os.path.join(self.map_dir, f'{self.map_name}.yaml')

        # Check if obstacles PNG exists, otherwise fallback to original
        if os.path.exists(map_with_obstacles_path):
            map_png_path = map_with_obstacles_path
            rospy.loginfo(f"[ObstaclePublisherGrid] Using map with static obstacles: {map_with_obstacles_path}")
        else:
            map_png_path = os.path.join(self.map_dir, f'{self.map_name}.png')
            rospy.logwarn(f"[ObstaclePublisherGrid] Static obstacles PNG not found, using original: {map_png_path}")

        if not os.path.exists(map_png_path) or not os.path.exists(map_yaml_path):
            rospy.logerr(f"[ObstaclePublisherGrid] Map files not found: {map_png_path}, {map_yaml_path}")
            return False

        # Load map image (grayscale) with static obstacles
        self.original_map = cv2.imread(map_png_path, cv2.IMREAD_GRAYSCALE)
        if self.original_map is None:
            rospy.logerr(f"[ObstaclePublisherGrid] Failed to load map image: {map_png_path}")
            return False

        # Load map metadata
        with open(map_yaml_path, 'r') as f:
            map_data = yaml.safe_load(f)

        self.map_resolution = map_data['resolution']
        self.map_origin_x = map_data['origin'][0]
        self.map_origin_y = map_data['origin'][1]
        self.map_height = self.original_map.shape[0]

        # Create working copy for dynamic obstacle overlay
        self.current_map = self.original_map.copy()

        rospy.loginfo(f"[ObstaclePublisherGrid] Loaded base map: {self.original_map.shape}")
        rospy.loginfo(f"  - Resolution: {self.map_resolution} m/pixel")
        rospy.loginfo(f"  - Origin: ({self.map_origin_x}, {self.map_origin_y})")
        rospy.loginfo("[ObstaclePublisherGrid] Static obstacles preserved in base map")

        return True

    def load_trajectory_gb(self):
        """Load GB trajectory waypoints (once) and prepare speed profile"""
        rospy.loginfo("[ObstaclePublisherGrid] Loading GB trajectory...")

        # Read GB waypoints for speed profile
        global_wpnts_msg = rospy.wait_for_message("/global_waypoints", WpntArray, timeout=10.0)
        global_wpnts = global_wpnts_msg.wpnts[:-1]  # Exclude last point (same as first)

        self.max_s_gb = global_wpnts[-1].s_m
        s_array = np.array([wpnt.s_m for wpnt in global_wpnts])

        # Apply speed scaler
        if self.constant_speed:
            speed_array = np.ones(len(global_wpnts)) * self.speed_scaler
        else:
            speed_array = np.array([wpnt.vx_mps * self.speed_scaler for wpnt in global_wpnts])

        # Extract x, y, heading, d_left, d_right
        x_array = np.array([wpnt.x_m for wpnt in global_wpnts])
        y_array = np.array([wpnt.y_m for wpnt in global_wpnts])
        heading_array = np.array([wpnt.psi_rad for wpnt in global_wpnts])
        d_left_array = np.array([wpnt.d_left for wpnt in global_wpnts])
        d_right_array = np.array([wpnt.d_right for wpnt in global_wpnts])

        # Convert to Frenet for d values
        frenet_result = self.glob2frenet(x_array.tolist(), y_array.tolist())
        d_array = np.array(frenet_result.d)

        # Store trajectory
        self.trajectory_gb = []
        for i in range(len(s_array)):
            wpnt = {
                'x_m': x_array[i],
                'y_m': y_array[i],
                's_m': s_array[i],
                'd_m': d_array[i],
                'vx_mps': speed_array[i],
                'heading': heading_array[i],
                'd_left': d_left_array[i],
                'd_right': d_right_array[i]
            }
            self.trajectory_gb.append(wpnt)

        self.trajectory_gb_s_array = s_array

        rospy.loginfo(f"[ObstaclePublisherGrid] Loaded GB trajectory: {len(self.trajectory_gb)} waypoints")
        rospy.loginfo(f"  - Max s: {self.max_s_gb:.2f}m")
        rospy.loginfo(f"  - Speed range: [{speed_array.min():.2f}, {speed_array.max():.2f}] m/s")

        return True

    def meters_to_pixels(self, x_m, y_m):
        """Convert meters to pixel coordinates"""
        x_px = int((x_m - self.map_origin_x) / self.map_resolution)
        y_px = int((y_m - self.map_origin_y) / self.map_resolution)
        # Flip y (image coordinates are top-down, map coordinates are bottom-up)
        y_px = self.map_height - y_px
        return x_px, y_px

    def get_current_trajectory_and_s(self):
        """Get current trajectory and s value based on Smart Static mode"""
        # Only use Fixed trajectory if Smart mode is enabled
        if self.enable_smart and self.smart_static_active and self.trajectory_fixed is not None:
            return self.trajectory_fixed, self.trajectory_fixed_s_array, self.current_s_fixed, self.max_s_fixed, 'fixed'
        else:
            # Debug: log why we're using GB (only if Smart is enabled)
            if self.enable_smart and self.smart_static_active and self.trajectory_fixed is None:
                rospy.logwarn_throttle(5.0, "[ObstaclePublisherGrid] Smart Static active but trajectory_fixed is None - using GB")
            return self.trajectory_gb, self.trajectory_gb_s_array, self.current_s_gb, self.max_s_gb, 'gb'

    def get_obstacle_state(self):
        """Get obstacle position, heading, and speed at current s"""
        trajectory, s_array, current_s, max_s, mode = self.get_current_trajectory_and_s()

        # Find nearest waypoint index
        idx = np.abs(s_array - current_s).argmin()
        wpnt = trajectory[idx]

        x_m = wpnt['x_m']
        y_m = wpnt['y_m']
        heading = wpnt['heading']

        # Apply reactive mode perturbation if enabled
        if self.reactive:
            # Get track boundaries at current waypoint
            d_left = wpnt['d_left']   # Positive (left side)
            d_right = wpnt['d_right']  # Negative (right side)

            # Update phase based on time
            phase_increment = 2.0 * np.pi * self.reactive_frequency * self.looptime
            new_phase = self.reactive_phase + phase_increment

            # Detect cycle completion (phase wraps from 2π to 0)
            if new_phase >= 2.0 * np.pi:
                # Start new cycle with random amplitude
                # Limit to smaller of boundaries and max 1.0 meter
                max_amplitude = min(abs(d_left), abs(d_right), self.max_amplitude_limit)
                self.reactive_amplitude = np.random.uniform(0.3 * max_amplitude, max_amplitude)
                new_phase = new_phase % (2.0 * np.pi)
                rospy.loginfo_throttle(5.0, f"[ObstaclePublisherGrid] New cycle: amplitude={self.reactive_amplitude:.3f}m, range=[{d_right:.3f}, {d_left:.3f}]")

            self.reactive_phase = new_phase

            # Calculate d perturbation using sine wave
            d_perturbation = self.reactive_amplitude * np.sin(self.reactive_phase)

            # Apply perturbation in Frenet frame: convert (s, d) to (x, y)
            # Normal vector is perpendicular to heading (rotated by 90 degrees)
            normal_x = -np.sin(heading)  # Normal direction (left is positive)
            normal_y = np.cos(heading)

            x_m += d_perturbation * normal_x
            y_m += d_perturbation * normal_y

            # Adjust heading based on rate of change of lateral position
            # Calculate derivative of d perturbation (how fast we're moving laterally)
            d_dot = (d_perturbation - self.prev_d_perturbation) / self.looptime
            self.prev_d_perturbation = d_perturbation

            # Add small heading adjustment proportional to lateral velocity
            # Positive d_dot (moving left) should rotate heading left (counterclockwise)
            # The adjustment is small to keep motion smooth
            heading_adjustment = np.arctan2(d_dot, wpnt['vx_mps']) if wpnt['vx_mps'] > 0.1 else 0.0
            heading += heading_adjustment

        return x_m, y_m, heading, wpnt['vx_mps'], mode

    def update_obstacle_on_map(self, x_m, y_m, heading):
        """Update occupancy grid with rectangular obstacle at new position

        Args:
            x_m, y_m: Center position in meters
            heading: Heading angle in radians (for rectangle orientation)
        """
        # Step 1: Restore original map at previous obstacle pixels
        if len(self.prev_obstacle_pixels) > 0:
            # rospy.loginfo(f"[ObstaclePublisherGrid] Restoring {len(self.prev_obstacle_pixels)} previous pixels")
            for px, py in self.prev_obstacle_pixels:
                if 0 <= px < self.current_map.shape[1] and 0 <= py < self.current_map.shape[0]:
                    self.current_map[py, px] = self.original_map[py, px]

        # Step 2: Draw rectangular obstacle at new position
        # Draw rotated rectangle following heading
        center_px = self.meters_to_pixels(x_m, y_m)

        # Convert dimensions to pixels
        length_px = int(self.obstacle_length_m / self.map_resolution)  # Heading direction
        width_px = int(self.obstacle_width_m / self.map_resolution)    # Normal direction

        # Create rectangle points (centered at origin, then rotate and translate)
        # Rectangle: half_length in ±heading direction, half_width in ±normal direction
        half_length = length_px / 2.0
        half_width = width_px / 2.0

        # Define rectangle corners in local frame (before rotation)
        # Swapped: Y-axis is heading direction (forward), X-axis is width (lateral)
        rect_corners_local = np.array([
            [-half_width, -half_length],  # Back-left
            [-half_width, half_length],   # Front-left
            [half_width, half_length],    # Front-right
            [half_width, -half_length]    # Back-right
        ])

        # Rotation matrix (heading angle - 90 degrees to align with map frame)
        # Reversed rotation direction (counter-clockwise in image coordinates)
        heading_adjusted = heading - np.pi / 2.0
        cos_h = np.cos(heading_adjusted)
        sin_h = np.sin(heading_adjusted)
        rotation_matrix = np.array([[cos_h, sin_h], [-sin_h, cos_h]])

        # Rotate and translate corners
        rect_corners_px = (rotation_matrix @ rect_corners_local.T).T + np.array(center_px)
        rect_corners_px = rect_corners_px.astype(np.int32)

        # Step 3: Track pixels BEFORE drawing - save map state
        map_before = self.current_map.copy()

        # Draw filled rectangle (black = occupied in grayscale, will be 100 in OccupancyGrid)
        # Gym PNG: 0 (black) = walls/obstacles, 255 (white) = free space
        cv2.fillPoly(self.current_map, [rect_corners_px], 0)

        # Step 4: Find exactly which pixels changed - these are the ones we need to restore
        self.prev_obstacle_pixels = []
        diff = (self.current_map != map_before)
        changed_pixels = np.argwhere(diff)
        for px_coords in changed_pixels:
            py, px = px_coords  # argwhere returns [row, col] = [y, x]
            self.prev_obstacle_pixels.append((px, py))

        # rospy.loginfo(f"[ObstaclePublisherGrid] Drew obstacle at ({x_m:.2f}, {y_m:.2f}), heading={heading:.2f}, tracked {len(self.prev_obstacle_pixels)} pixels")

    def publish_map(self):
        """Publish updated occupancy grid"""
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = rospy.Time.now()
        grid_msg.header.frame_id = "map"

        grid_msg.info.resolution = self.map_resolution
        grid_msg.info.width = self.current_map.shape[1]
        grid_msg.info.height = self.current_map.shape[0]
        grid_msg.info.origin.position.x = self.map_origin_x
        grid_msg.info.origin.position.y = self.map_origin_y
        grid_msg.info.origin.position.z = 0.0
        grid_msg.info.origin.orientation.w = 1.0

        # Convert grayscale image to occupancy grid data
        # Flip vertically (image origin is top-left, map origin is bottom-left)
        flipped_img = np.flipud(self.current_map)
        occupancy_data = np.zeros(self.current_map.shape, dtype=np.int8)
        # Correct mapping for gym maps
        # Gym maps: 0 (black) = walls/obstacles, 255 (white) = free space
        occupancy_data[flipped_img < 128] = 100  # Dark pixels (0-127) -> occupied
        occupancy_data[flipped_img >= 128] = 0   # Light pixels (128-255) -> free

        grid_msg.data = occupancy_data.flatten().tolist()

        self.map_pub.publish(grid_msg)

    def publish_marker(self, x_m, y_m, heading, mode):
        """Publish visualization marker for obstacle

        Args:
            x_m, y_m: Position in meters
            heading: Heading angle in radians
            mode: 'gb' or 'fixed'
        """
        marker_array = MarkerArray()

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position.x = x_m
        marker.pose.position.y = y_m
        marker.pose.position.z = 0.0

        # Set orientation based on heading
        marker.pose.orientation.z = np.sin(heading / 2.0)
        marker.pose.orientation.w = np.cos(heading / 2.0)

        marker.scale.x = self.obstacle_length_m  # Length in heading direction
        marker.scale.y = self.obstacle_width_m   # Width in normal direction
        marker.scale.z = 0.5

        # Color based on mode
        if mode == 'fixed':
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0  # Blue for Fixed mode
        else:
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0  # Red for GB mode
        marker.color.a = 0.8

        marker_array.markers.append(marker)
        self.obstacle_marker_pub.publish(marker_array)

    def shutdown(self):
        """Cleanup on shutdown - restore original map"""
        rospy.loginfo("[ObstaclePublisherGrid] Shutting down, restoring original map...")
        if self.original_map is not None:
            self.current_map = self.original_map.copy()
            self.publish_map()

    def ros_loop(self):
        """Main loop"""
        # Load map
        if not self.load_map():
            rospy.logerr("[ObstaclePublisherGrid] Failed to load map")
            return

        # Load GB trajectory
        if not self.load_trajectory_gb():
            rospy.logerr("[ObstaclePublisherGrid] Failed to load GB trajectory")
            return

        # Fixed trajectory will be loaded dynamically via smart_trajectory_cb when available
        rospy.loginfo("[ObstaclePublisherGrid] Starting with GB trajectory (Fixed trajectory loaded dynamically)")
        rospy.loginfo("[ObstaclePublisherGrid] Starting main loop...")
        rospy.sleep(0.5)

        while not rospy.is_shutdown():
            # Get obstacle state (position, heading, speed, mode)
            x_m, y_m, heading, speed_mps, mode = self.get_obstacle_state()

            # Update occupancy grid
            self.update_obstacle_on_map(x_m, y_m, heading)

            # Publish updated map
            self.publish_map()

            # Publish visualization marker
            self.publish_marker(x_m, y_m, heading, mode)

            # Update s for next iteration (based on current mode)
            if mode == 'fixed':
                self.current_s_fixed = (self.current_s_fixed + speed_mps * self.looptime) % self.max_s_fixed
            else:
                self.current_s_gb = (self.current_s_gb + speed_mps * self.looptime) % self.max_s_gb

            self.current_speed = speed_mps

            self.rate.sleep()


if __name__ == "__main__":
    rospy.init_node("obstacle_publisher_grid", anonymous=False, log_level=rospy.INFO)
    obstacle_publisher = ObstaclePublisherGrid()
    rospy.on_shutdown(obstacle_publisher.shutdown)
    obstacle_publisher.ros_loop()
