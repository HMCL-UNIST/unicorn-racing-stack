# Controller
This controller package implements the following controllers, with more details in the respective READMEs:
- [Combined controller](./combined/README.md)
- [Follow the Gap controller](./ftg/README.md)

The `control_node` implemented in `controller_manager.py` initializes the needed controllers. It runs at a specified loop rate where in each cycle the next control inputs are calculated via the choosen controller.

## Input/Output Topic Signature
This node subscribes to:
- `/behavior_strategy`: Local waypoints, opponent data, and current behavior state; updates trajectory, opponent info, and replanning flags.
- `/global_waypoints`: Global raceline, loaded once at startup.
- `/car_state/odom`: Ego speed and yaw rate.
- `/car_state/pose`: Ego position and heading.
- `/car_state/odom_frenet`: Ego state in Frenet coordinates (s, d, vs, vd).
- `/imu/data`: IMU measurements (acceleration and yaw rate).
- `/dyn_controller/parameter_updates`: Dynamic reconfigure parameter updates.
- `/scan`: LiDAR scans.
- `/save_start_traj`: Trigger for the START state.

The node publishes to:
- `/vesc/high_level/ackermann_cmd_mux/input/nav_1`: Publishes the control commands (topic name is configurable).
- `lookahead_point`: Lookahead point marker.
- `trailing_opponent_marker`: Trailing opponent marker.
- `future_position`: Future predicted position marker.
- `l1_distance`: L1 distance, nearest waypoint index and curvature.
- `/controller/latency`: Controller cycle latency (when latency measurement is enabled).
- `/vesc/commands/motor/brake`, `/vesc/commands/servo/position`: Direct brake current and servo position (only used when brake control runs in direct-brake mode).

