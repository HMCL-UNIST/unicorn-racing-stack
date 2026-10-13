#!/usr/bin/env python3
"""
Smart Static State Helper

Provides Fixed Frenet-based state checking by inheriting StateMachine.
Uses Fixed path (Smart Static) as reference instead of GB raceline.

Usage:
    checker = SmartStaticChecker(state_machine)
    checker.update()  # Extract Fixed Frenet from parent's obstacles
    close = checker._check_close_to_raceline()  # Fixed Frenet based check
"""

import numpy as np
import rospy
from nav_msgs.msg import Odometry
from f110_msgs.msg import ObstacleArray, OTWpntArray
from state_machine_node import StateMachine, debug_log_on_change, DEBUG_LOGGING_ENABLED


class SmartStaticChecker(StateMachine):
    """StateMachine functions but with Fixed Frenet reference

    Inherits all check functions from StateMachine.
    Overrides instance variables (cur_s, cur_d, obstacles) with Fixed Frenet values.
    Subscribes only to Fixed Frenet odom, reuses parent's obstacles and waypoints.
    """

    def __init__(self, parent_state_machine):
        """Initialize with Fixed Frenet reference

        Args:
            parent_state_machine: Main StateMachine instance (GB raceline based)
        """
        self.parent = parent_state_machine

        # Copy ALL parent attributes, then override specific ones
        # Copy all parent's attributes by reference (shallow copy of __dict__)
        # This ensures all inherited methods have access to all needed variables
        self.__dict__.update(parent_state_machine.__dict__)

        # Now override only the Fixed Frenet specific variables
        self.cur_s = 0.0  # Progress along Fixed path (override)
        self.cur_d = 0.0  # Lateral offset from Fixed path (override)
        self.cur_vs = 0.0  # Velocity along Fixed path (override)
        self.cur_vd = 0.0  # Lateral velocity along Fixed path (override)

        # Fixed path has its own obstacles_in_interest (override)
        # Calculated in update() by filtering self.obstacles using Fixed Frenet
        self.obstacles_in_interest = []  # Will be calculated based on Fixed Frenet s
        self.cur_obstacles_in_interest = []  # Current obstacles in interest

        # obstacles populated in update() from parent
        # self.obstacles will be populated by update() with Fixed Frenet coordinates
        # Reads parent's obstacles and copies _fixed fields to primary fields
        self.obstacles = []  # Will be populated in update()

        # Reset mode flags to avoid sharing with parent
        # Note: Debug logging now uses global debug_log_on_change() with its own cache
        self.static_overtaking_mode = False  # Independent overtaking flag

        # Override waypoint-related variables for Fixed Frenet
        # Waypoint data - use Smart Static waypoints as "gb_wpnts" (override)
        self.cur_gb_wpnts = parent_state_machine.cur_smart_static_avoidance_wpnts
        self.num_glb_wpnts = 0  # Will be updated in update()
        self.waypoints_dist = 0.0  # Will be updated in update()
        # track_length will be updated in update() to match Smart Static path length

        # Dynamic waypoint references - always use parent's latest values
        # These properties ensure we always get parent's latest waypoints (which are updated by callbacks)
        # DO NOT create static copies - waypoints must dynamically reference parent
        # The parent's callbacks update these, and we need live access to them

        # Fixed Frenet odom subscriber (only this one needed!)
        rospy.Subscriber('/car_state/odom_frenet_fixed', Odometry, self._odom_fixed_cb)
        rospy.loginfo("[SmartStaticChecker] Initialized with Fixed Frenet odom subscription")

        # No timer needed - parent calls update() synchronously
        # Parent's update_waypoints() calls self.smart_helper.update() directly
        # This ensures perfect synchronization with parent's loop() timing

    # Properties to dynamically access parent's callback-updated data
    # Waypoint properties (for transitions and check functions)
    @property
    def static_avoidance_wpnts(self):
        """Always return parent's latest static_avoidance_wpnts"""
        return self.parent.static_avoidance_wpnts

    @property
    def avoidance_wpnts(self):
        """Always return parent's latest avoidance_wpnts"""
        return self.parent.avoidance_wpnts

    @property
    def recovery_wpnts(self):
        """Always return parent's latest recovery_wpnts"""
        return self.parent.recovery_wpnts

    # Prediction properties (for _check_free_frenet)
    @property
    def obstacles_prediction(self):
        """Always return parent's latest obstacle predictions"""
        return self.parent.obstacles_prediction

    @property
    def obstacles_prediction_id(self):
        """Always return parent's latest obstacle prediction ID"""
        return self.parent.obstacles_prediction_id

    @property
    def ego_prediction(self):
        """Always return parent's latest ego prediction"""
        return self.parent.ego_prediction

    # Position property (for _check_close_to_raceline_heading, _check_on_spline)
    @property
    def current_position(self):
        """Always return parent's latest cartesian position [x, y, heading]"""
        return self.parent.current_position

    # Zone properties (for _check_only_ftg_zone, _check_ot_sector)
    @property
    def only_ftg_zones(self):
        """Always return parent's latest FTG-only zones"""
        return self.parent.only_ftg_zones

    @property
    def overtake_zones(self):
        """Always return parent's latest overtake zones"""
        return self.parent.overtake_zones

    # Dynamic config parameters (for check functions)
    @property
    def ftg_speed_mps(self):
        """Always return parent's latest FTG speed threshold"""
        return self.parent.ftg_speed_mps

    @property
    def ftg_timer_sec(self):
        """Always return parent's latest FTG timer threshold"""
        return self.parent.ftg_timer_sec

    @property
    def ftg_disabled(self):
        """Always return parent's latest FTG disabled flag"""
        return self.parent.ftg_disabled

    @property
    def gb_ego_width_m(self):
        """Always return parent's latest ego vehicle width"""
        return self.parent.gb_ego_width_m

    @property
    def lateral_width_gb_m(self):
        """Always return parent's latest lateral width for GB"""
        return self.parent.lateral_width_gb_m

    @property
    def lateral_width_ot_m(self):
        """Always return parent's latest lateral width for overtaking"""
        return self.parent.lateral_width_ot_m

    @property
    def overtaking_ttl_sec(self):
        """Always return parent's latest overtaking TTL seconds"""
        return self.parent.overtaking_ttl_sec

    @property
    def overtaking_ttl_count_threshold(self):
        """Always return parent's latest overtaking TTL count threshold"""
        return self.parent.overtaking_ttl_count_threshold

    @property
    def force_gbtrack_state(self):
        """Always return parent's latest force GB track flag"""
        return self.parent.force_gbtrack_state

    @property
    def use_force_trailing(self):
        """Always return parent's latest use force trailing flag"""
        return self.parent.use_force_trailing

    @property
    def emergency_break_horizon(self):
        """Always return parent's latest emergency break horizon"""
        return self.parent.emergency_break_horizon

    @property
    def splini_ttl(self):
        """Always return parent's latest splini TTL"""
        return self.parent.splini_ttl

    @property
    def splini_ttl_counter(self):
        """Always return parent's latest splini TTL counter"""
        return self.parent.splini_ttl_counter

    @property
    def splini_hyst_timer_sec(self):
        """Always return parent's latest splini hysteresis timer"""
        return self.parent.splini_hyst_timer_sec

    def _odom_fixed_cb(self, data):
        """Fixed Frenet odom callback

        Updates cur_s, cur_d, cur_vs, cur_vd from Fixed Frenet odom.
        Overrides parent's GB Frenet values.
        """
        self.cur_s = data.pose.pose.position.x
        self.cur_d = data.pose.pose.position.y
        self.cur_vs = data.twist.twist.linear.x
        self.cur_vd = data.twist.twist.linear.y

    def update(self):
        """Update waypoint metadata and obstacles in interest

        Called synchronously by parent's update_waypoints() every iteration.
        Reads parent's obstacles, copies _fixed fields to primary fields,
        and filters to get obstacles in interest.
        """
        if len(self.parent.cur_smart_static_avoidance_wpnts.list) == 0:
            # No Smart Static path available
            self.num_glb_wpnts = 0
            self.obstacles = []
            self.obstacles_in_interest = []
            self.cur_obstacles_in_interest = []
            return

        # Update waypoint reference and metadata
        self.cur_gb_wpnts = self.parent.cur_smart_static_avoidance_wpnts
        self.num_glb_wpnts = len(self.cur_gb_wpnts.list)

        # For closed loop, the last waypoint's s_m is the total track length
        if self.num_glb_wpnts > 0:
            self.track_length = self.cur_gb_wpnts.list[-1].s_m
            # Calculate average waypoint distance (needed for _check_close_to_raceline_heading)
            self.waypoints_dist = self.track_length / self.num_glb_wpnts

            # Override max_s with Fixed Frenet track length
            # _check_free_frenet() and _check_free_cartesian() use self.max_s for modulo operations
            # In Smart mode, self.cur_s is Fixed Frenet, so max_s MUST be Fixed path length
            # Otherwise: gap = (obs_s - self.cur_s) % self.max_s uses wrong track length!
            self.max_s = self.track_length
        else:
            self.track_length = 0.0
            self.waypoints_dist = 0.0
            self.max_s = 0.0  # Reset when no path available

        # Copy parent's obstacles and replace primary fields with _fixed fields
        import copy
        self.obstacles = []
        for obs in self.parent.obstacles:
            # Shallow copy to avoid modifying parent's obstacle
            obs_copy = copy.copy(obs)

            # Replace primary Frenet fields with Fixed Frenet fields
            obs_copy.s_start = obs.s_start_fixed
            obs_copy.s_end = obs.s_end_fixed
            obs_copy.s_center = obs.s_center_fixed
            obs_copy.d_center = obs.d_center_fixed
            obs_copy.d_right = obs.d_right_fixed
            obs_copy.d_left = obs.d_left_fixed
            obs_copy.vs = obs.vs_fixed
            obs_copy.vd = obs.vd_fixed
            obs_copy.s_var = obs.s_var_fixed
            obs_copy.d_var = obs.d_var_fixed
            obs_copy.vs_var = obs.vs_var_fixed
            obs_copy.vd_var = obs.vd_var_fixed

            self.obstacles.append(obs_copy)

        # Filter obstacles to get obstacles in interest
        self._update_obstacles_in_interest()

    def _update_obstacles_in_interest(self):
        """Filter obstacles based on Fixed Frenet coordinates

        Only includes obstacles within interest_horizon_m ahead on Fixed path.
        Uses self.obstacles which was populated in update() with Fixed Frenet coordinates.
        """
        obstacles_in_interest = []

        # Filter obstacles based on proximity to current Fixed Frenet s position
        # Note: self.obstacles already has _fixed fields copied to primary fields
        for obs in self.obstacles:
            gap = (obs.s_start - self.cur_s) % self.track_length
            if gap < self.interest_horizon_m:
                obstacles_in_interest.append(obs)

        self.obstacles_in_interest = obstacles_in_interest
        self.cur_obstacles_in_interest = obstacles_in_interest
