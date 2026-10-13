# Global Optimizer
Global trajectory package of the 2D stack, based on the TUM [global_racetrajectory_optimization](https://github.com/TUMFTM/global_racetrajectory_optimization) (in `src/global_racetrajectory_optimization`).

In the 3D stack it provides `global_trajectory_publisher.py`, started by `3d_base_system.launch`: it loads `stack_master/maps/<map>/global_waypoints.json` (written by the [3D global planner](../3d_gb_optimizer/README.md)) and publishes it:
- `/global_waypoints`, `/global_waypoints/shortest_path`, `/centerline_waypoints`: waypoints.
- `/global_waypoints/markers`, `/trackbounds/markers`, ...: visualization.
- `/map_infos`, `/estimated_lap_time`: track information.

The 2D global planner (`global_planner_node.py`, minimum-curvature line from an occupancy grid map) and the trajectory tuner are kept from the 2D stack and are not used by the 3D pipeline.
